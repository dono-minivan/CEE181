"""Single-lane freeway segment: arrivals -> upstream entry queue -> IDM road -> free downstream exit.

This is the verified baseline the merge/bottleneck and ramp controller build on.

Design choices (each traced in docs/source_to_design.md):
  * Arrivals: piecewise-constant demand profile, Poisson (seeded) or uniform headways.
  * Upstream boundary: a point ("vertical") queue. A queued vehicle enters at x = 0 only when the gap
    to the last vehicle on the road is at least s*(v_ins, 0) = s0 + v_ins*T, with v_ins = speed of
    that last vehicle (v0 if the road is empty). At that gap the entering vehicle's IDM deceleration
    is at most a, so insertion never forces a hard brake. Vehicles that cannot enter WAIT; they are
    never dropped. Waiting time counts toward travel time.
  * Downstream boundary: free exit (the most downstream vehicle has no leader). The exit time is
    interpolated within the step so travel times do not depend on dt resolution.
  * Simultaneous update: all accelerations are computed from the state at t, then all vehicles move.

Units: SI internally. Position = front bumper, measured from the segment entrance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tmc.idm import IDMParams, acceleration, ballistic_update, load_params


# --------------------------------------------------------------------------- configuration

@dataclass
class ScenarioConfig:
    scenario_id: str
    demand_id: str
    param_set: str
    seed: int | None
    length_m: float
    dt_s: float
    record_every_s: float
    max_time_s: float
    arrival_process: str                 # "poisson" or "uniform"
    demand_profile: list                 # [[start_s, end_s, flow_veh_per_h], ...]
    lane_id: int = 0
    initial_vehicles: list = field(default_factory=list)  # [[position_m, speed_mps], ...] (tests)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScenarioConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls(
            scenario_id=d["scenario_id"],
            demand_id=d["demand_id"],
            param_set=d.get("param_set", "baseline"),
            seed=d.get("seed"),
            length_m=float(d["road"]["length_m"]),
            lane_id=int(d["road"].get("lane_id", 0)),
            dt_s=float(d["sim"]["dt_s"]),
            record_every_s=float(d["sim"]["record_every_s"]),
            max_time_s=float(d["sim"]["max_time_s"]),
            arrival_process=d["arrivals"]["process"],
            demand_profile=[list(map(float, row)) for row in d["arrivals"]["profile"]],
        )


def generate_arrivals(profile, process: str, seed: int | None) -> np.ndarray:
    """Arrival times (s) at the upstream boundary for a piecewise-constant demand profile."""
    rng = np.random.default_rng(seed)
    times = []
    for start, end, q_vph in profile:
        q = q_vph / 3600.0
        if q <= 0:
            continue
        if process == "uniform":
            h = 1.0 / q
            times.extend(np.arange(start + 0.5 * h, end, h))
        elif process == "poisson":
            t = start + rng.exponential(1.0 / q)
            while t < end:
                times.append(t)
                t += rng.exponential(1.0 / q)
        else:
            raise ValueError(f"unknown arrival process {process!r}")
    return np.sort(np.asarray(times, dtype=float))


# --------------------------------------------------------------------------- results

@dataclass
class SimResult:
    config: ScenarioConfig
    params: IDMParams
    trajectories: pd.DataFrame   # VehicleState rows: time_s, vehicle_id, lane_id, position_m, speed_mps, acceleration_mps2
    vehicles: pd.DataFrame       # vehicle_id, arrival_s, entry_s, exit_s (NaN = did not happen)
    counts: pd.DataFrame         # time_s, n_arrived, n_queued, n_on_road, n_exited
    end_time_s: float
    min_gap_m: float
    cleared: bool                # all arrived vehicles exited before max_time_s

    def summary(self) -> dict:
        # Only vehicles that arrived before the run ended; later scheduled arrivals are reported separately.
        n_not_arrived = int((self.vehicles["arrival_s"] > self.end_time_s).sum())
        v = self.vehicles[self.vehicles["arrival_s"] <= self.end_time_s]
        exited = v["exit_s"].notna()
        # Unfinished vehicles accrue time until the end of the run (never silently dropped).
        tt = np.where(exited, v["exit_s"] - v["arrival_s"], self.end_time_s - v["arrival_s"])
        wait = np.where(v["entry_s"].notna(), v["entry_s"] - v["arrival_s"], self.end_time_s - v["arrival_s"])
        c = self.counts
        return {
            "scenario_id": self.config.scenario_id,
            "demand_id": self.config.demand_id,
            "dt_s": self.config.dt_s,
            "n_arrived": int(len(v)),
            "n_exited": int(exited.sum()),
            "unmet_demand_veh": int((~v["entry_s"].notna()).sum()),
            "on_road_at_end_veh": int((v["entry_s"].notna() & ~exited).sum()),
            "demand_not_yet_arrived_veh": n_not_arrived,
            "cleared": self.cleared,
            "end_time_s": self.end_time_s,
            "total_travel_time_veh_h": float(tt.sum() / 3600.0),
            "mean_travel_time_s": float(tt.mean()) if len(v) else float("nan"),
            "total_entry_wait_veh_h": float(wait.sum() / 3600.0),
            "peak_entry_queue_veh": int(c["n_queued"].max()) if len(c) else 0,
            "min_gap_m": self.min_gap_m,
            "min_speed_mps": float(self.trajectories["speed_mps"].min()) if len(self.trajectories) else float("nan"),
        }


# --------------------------------------------------------------------------- simulator

class SingleLaneSegment:
    def __init__(self, params: IDMParams, cfg: ScenarioConfig, arrivals: np.ndarray | None = None):
        self.p = params
        self.cfg = cfg
        self.arrivals = generate_arrivals(cfg.demand_profile, cfg.arrival_process, cfg.seed) \
            if arrivals is None else np.sort(np.asarray(arrivals, dtype=float))
        n_arr = len(self.arrivals)
        n_init = len(cfg.initial_vehicles)
        # Vehicle IDs: initial vehicles first (0..n_init-1, downstream-most first), then arrivals in order.
        self.entry_s = np.full(n_init + n_arr, np.nan)
        self.exit_s = np.full(n_init + n_arr, np.nan)
        self.arrival_s = np.concatenate([np.zeros(n_init), self.arrivals])
        init = sorted(cfg.initial_vehicles, key=lambda r: -r[0])
        self.ids = np.arange(n_init, dtype=int)                 # on-road, ordered downstream -> upstream
        self.x = np.array([r[0] for r in init], dtype=float)
        self.v = np.array([r[1] for r in init], dtype=float)
        self.entry_s[:n_init] = 0.0
        self.next_arrival = 0          # index into self.arrivals of next vehicle not yet arrived
        self.queue: list[int] = []     # vehicle IDs waiting at the entrance (FIFO)
        self.n_init = n_init
        self.min_gap = np.inf

    # -- helpers
    def gaps(self) -> np.ndarray:
        """Bumper-to-bumper gap to the leader; inf for the most downstream vehicle."""
        s = np.full(len(self.x), np.inf)
        if len(self.x) > 1:
            s[1:] = self.x[:-1] - self.x[1:] - self.p.length
        return s

    def accelerations(self) -> np.ndarray:
        s = self.gaps()
        dv = np.zeros(len(self.v))
        if len(self.v) > 1:
            dv[1:] = self.v[1:] - self.v[:-1]
        return acceleration(self.v, s, dv, self.p)

    def _try_insert(self, t: float) -> None:
        if not self.queue:
            return
        if len(self.x) == 0:
            v_ins, gap = self.p.v0, np.inf
        else:
            v_ins = float(self.v[-1])
            gap = float(self.x[-1] - self.p.length)
        if gap >= self.p.s0 + v_ins * self.p.T:
            vid = self.queue.pop(0)
            self.ids = np.append(self.ids, vid)
            self.x = np.append(self.x, 0.0)
            self.v = np.append(self.v, v_ins)
            self.entry_s[vid] = t

    # -- main loop
    def run(self) -> SimResult:
        cfg, p = self.cfg, self.p
        dt = cfg.dt_s
        rec_every = max(1, int(round(cfg.record_every_s / dt)))
        max_steps = int(np.ceil(cfg.max_time_s / dt))
        last_arrival = self.arrivals[-1] if len(self.arrivals) else 0.0
        traj_rows, count_rows = [], []
        k = 0
        while True:
            t = k * dt
            # 1) arrivals up to and including t join the entry queue
            while self.next_arrival < len(self.arrivals) and self.arrivals[self.next_arrival] <= t + 1e-9:
                self.queue.append(self.n_init + self.next_arrival)
                self.next_arrival += 1
            # 2) entry from the queue (state at t)
            self._try_insert(t)
            acc = self.accelerations()
            gaps = self.gaps()
            if len(gaps) > 1:
                self.min_gap = min(self.min_gap, float(gaps[1:].min()))
                if gaps[1:].min() <= 0:
                    raise RuntimeError(f"collision/overlap at t={t:.2f}s (min gap {gaps[1:].min():.3f} m)")
            # 3) record state at t
            if k % rec_every == 0:
                n = len(self.x)
                traj_rows.append(pd.DataFrame({
                    "time_s": np.full(n, t), "vehicle_id": self.ids.copy(), "lane_id": np.full(n, cfg.lane_id),
                    "position_m": self.x.copy(), "speed_mps": self.v.copy(), "acceleration_mps2": acc,
                }))
                count_rows.append((t, self.next_arrival + self.n_init, len(self.queue), len(self.x),
                                   int(np.isfinite(self.exit_s).sum())))
            # 4) stop when demand is over and everything has cleared, or at the time limit
            all_arrived = self.next_arrival == len(self.arrivals)
            empty = len(self.x) == 0 and not self.queue
            if (all_arrived and empty and t >= last_arrival) or k >= max_steps:
                break
            # 5) move everyone simultaneously
            x_old = self.x
            self.x, self.v = ballistic_update(self.x, self.v, acc, dt)
            # 6) exits at the downstream boundary (interpolate the crossing time inside the step)
            out = self.x > cfg.length_m
            if np.any(out):
                frac = (cfg.length_m - x_old[out]) / (self.x[out] - x_old[out])
                self.exit_s[self.ids[out]] = t + frac * dt
                keep = ~out
                self.ids, self.x, self.v = self.ids[keep], self.x[keep], self.v[keep]
            k += 1

        end_t = k * dt
        vehicles = pd.DataFrame({
            "vehicle_id": np.arange(len(self.arrival_s)),
            "arrival_s": self.arrival_s, "entry_s": self.entry_s, "exit_s": self.exit_s,
        })
        counts = pd.DataFrame(count_rows, columns=["time_s", "n_arrived", "n_queued", "n_on_road", "n_exited"])
        traj = pd.concat(traj_rows, ignore_index=True) if traj_rows else pd.DataFrame()
        cleared = bool(np.isfinite(self.exit_s).all())  # False also if the time limit cut off demand
        return SimResult(cfg, p, traj, vehicles, counts, end_t,
                         float(self.min_gap), cleared)


def run_scenario(path: str | Path, dt_s: float | None = None) -> SimResult:
    cfg = ScenarioConfig.from_yaml(path)
    if dt_s is not None:
        cfg.dt_s = dt_s
    return SingleLaneSegment(load_params(cfg.param_set), cfg).run()
