"""Freeway segment: mainline lane + optional on-ramp with acceleration lane (merge bottleneck).

    mainline arrivals -> entry queue -> mainline lane (x = 0 .. L) -> free exit at x = L
    ramp arrivals -> ramp queue -> ramp entrance (stop line, future meter) -> approach -> acceleration lane
                  -> merge into the mainline when safe

Design choices (each traced in docs/source_to_design.md):
  * Arrivals: piecewise-constant demand, Poisson (seeded) or uniform headways. Mainline and ramp use
    independent random streams (ramp seed = seed + 1000 unless set explicitly).
  * Entry rule (mainline entrance and ramp entrance alike): a queued vehicle enters when the gap to
    the last vehicle on that lane is at least s0 + v_ins*T, at v_ins = that vehicle's speed (v0 if the
    lane is empty). Vehicles that cannot enter WAIT in a point queue; they are never dropped.
  * Ramp geometry: ramp positions use the mainline x axis. The ramp entrance is at
    x = merge_start - approach_length. The acceleration lane runs merge_start .. merge_end.
  * Acceleration-lane driving (FreewaySegment._ramp_accelerations): IDM toward the ramp leader and a
    kinematic lane-end stop (brake only once v^2/(2d) >= b). Optional (config, default off): IDM speed
    adaptation to the adjacent mainline leader, and cooperative yielding by the mainline follower.
    PROVISIONAL: the merge capacity this produces (~1430-1470 veh/h) is below q_c; see experiment log E1.
  * Merge rule (MOBIL safety criterion, mandatory lane change, Kesting-Treiber-Helbing 2007): a ramp
    vehicle with x >= merge_start merges if, at its position, the gap to the mainline leader and
    follower is > 0 and neither the merging vehicle nor its new follower would need to decelerate
    harder than b_safe. Ramp vehicles are checked downstream-first; each merge is seen by the next check.
  * Downstream boundary: free exit; exit time interpolated within the step.
  * Simultaneous update: accelerations from the state at t, then everyone moves.

Lane IDs: 0 = mainline, 1 = ramp / acceleration lane. Units: SI. Position = front bumper.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tmc.idm import IDMParams, acceleration, ballistic_update, load_params

MAIN, RAMP = 0, 1
STOP_MARGIN_M = 1.0  # ramp vehicles that cannot merge stop this far before the end of the acceleration lane


# --------------------------------------------------------------------------- configuration

@dataclass
class RampConfig:
    merge_start_m: float                 # start of acceleration lane (mainline coordinate)
    merge_end_m: float                   # end of acceleration lane (vehicles stop before it)
    approach_length_m: float             # ramp entrance (stop line) to merge_start
    demand_profile: list                 # [[start_s, end_s, flow_veh_per_h], ...]
    arrival_process: str = "poisson"
    seed: int | None = None              # None -> scenario seed + 1000
    b_safe_mps2: float = 4.0             # MOBIL safe braking limit
    cooperative_yield: bool = False      # mainline follower eases off (<= b) to let a ramp vehicle in
    yield_min_speed_ratio: float = 0.0   # yield only if v_ramp >= ratio * v_follower (0 = always eligible)
    lane_end_decel_mps2: float | None = None  # braking rate that triggers the lane-end stop (None -> IDM b)
    speed_adaptation: bool = False       # in the merge zone, also follow the adjacent mainline leader

    @property
    def entrance_m(self) -> float:
        return self.merge_start_m - self.approach_length_m


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
    lane_id: int = MAIN
    initial_vehicles: list = field(default_factory=list)  # [[position_m, speed_mps], ...] on the mainline
    ramp: RampConfig | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ScenarioConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        ramp = None
        if d.get("ramp"):
            r = d["ramp"]
            ramp = RampConfig(
                merge_start_m=float(r["merge_start_m"]),
                merge_end_m=float(r["merge_end_m"]),
                approach_length_m=float(r["approach_length_m"]),
                demand_profile=[list(map(float, row)) for row in r["arrivals"]["profile"]],
                arrival_process=r["arrivals"].get("process", "poisson"),
                seed=r.get("seed"),
                b_safe_mps2=float(r.get("b_safe_mps2", 4.0)),
                cooperative_yield=bool(r.get("cooperative_yield", False)),
                yield_min_speed_ratio=float(r.get("yield_min_speed_ratio", 0.0)),
                lane_end_decel_mps2=r.get("lane_end_decel_mps2"),
                speed_adaptation=bool(r.get("speed_adaptation", False)),
            )
        return cls(
            scenario_id=d["scenario_id"],
            demand_id=d["demand_id"],
            param_set=d.get("param_set", "baseline"),
            seed=d.get("seed"),
            length_m=float(d["road"]["length_m"]),
            lane_id=int(d["road"].get("lane_id", MAIN)),
            dt_s=float(d["sim"]["dt_s"]),
            record_every_s=float(d["sim"]["record_every_s"]),
            max_time_s=float(d["sim"]["max_time_s"]),
            arrival_process=d["arrivals"]["process"],
            demand_profile=[list(map(float, row)) for row in d["arrivals"]["profile"]],
            ramp=ramp,
        )


def generate_arrivals(profile, process: str, seed: int | None) -> np.ndarray:
    """Arrival times (s) at a boundary for a piecewise-constant demand profile."""
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
    vehicles: pd.DataFrame       # vehicle_id, origin, arrival_s, entry_s, merge_s, exit_s (NaN = did not happen)
    counts: pd.DataFrame         # time_s, n_arrived, n_queued(_main/_ramp), n_on_road, n_on_main, n_on_ramp, n_exited
    end_time_s: float
    min_gap_m: float
    cleared: bool                # all scheduled vehicles exited before max_time_s

    def summary(self) -> dict:
        # Only vehicles that arrived before the run ended; later scheduled arrivals are reported separately.
        n_not_arrived = int((self.vehicles["arrival_s"] > self.end_time_s).sum())
        v = self.vehicles[self.vehicles["arrival_s"] <= self.end_time_s]
        exited = v["exit_s"].notna()
        entered = v["entry_s"].notna()
        # Unfinished vehicles accrue time until the end of the run (never silently dropped).
        tt = np.where(exited, v["exit_s"] - v["arrival_s"], self.end_time_s - v["arrival_s"])
        wait = np.where(entered, v["entry_s"] - v["arrival_s"], self.end_time_s - v["arrival_s"])
        is_ramp = (v["origin"] == "ramp").to_numpy()
        c, tr = self.counts, self.trajectories
        main_speeds = tr.loc[tr["lane_id"] == MAIN, "speed_mps"] if len(tr) else pd.Series(dtype=float)
        return {
            "scenario_id": self.config.scenario_id,
            "demand_id": self.config.demand_id,
            "dt_s": self.config.dt_s,
            "n_arrived": int(len(v)),
            "n_arrived_ramp": int(is_ramp.sum()),
            "n_exited": int(exited.sum()),
            "n_merged": int(v["merge_s"].notna().sum()),
            "unmet_demand_veh": int((~entered).sum()),
            "on_road_at_end_veh": int((entered & ~exited).sum()),
            "demand_not_yet_arrived_veh": n_not_arrived,
            "cleared": self.cleared,
            "end_time_s": self.end_time_s,
            "total_travel_time_veh_h": float(tt.sum() / 3600.0),
            "total_travel_time_mainline_origin_veh_h": float(tt[~is_ramp].sum() / 3600.0),
            "total_travel_time_ramp_origin_veh_h": float(tt[is_ramp].sum() / 3600.0),
            "mean_travel_time_s": float(tt.mean()) if len(v) else float("nan"),
            "total_entry_wait_veh_h": float(wait[~is_ramp].sum() / 3600.0),
            "total_ramp_wait_veh_h": float(wait[is_ramp].sum() / 3600.0),
            "peak_entry_queue_veh": int(c["n_queued_main"].max()) if len(c) else 0,
            "peak_ramp_queue_veh": int(c["n_queued_ramp"].max()) if len(c) else 0,
            "min_gap_m": self.min_gap_m,
            "min_speed_mps": float(tr["speed_mps"].min()) if len(tr) else float("nan"),
            "min_mainline_speed_mps": float(main_speeds.min()) if len(main_speeds) else float("nan"),
        }


# --------------------------------------------------------------------------- simulator

class _Lane:
    """Vehicles on one lane, ordered downstream -> upstream (index 0 = most downstream)."""

    def __init__(self):
        self.ids = np.empty(0, dtype=int)
        self.x = np.empty(0)
        self.v = np.empty(0)

    def __len__(self):
        return len(self.x)

    def insert(self, i: int, vid: int, x: float, v: float):
        self.ids = np.insert(self.ids, i, vid)
        self.x = np.insert(self.x, i, x)
        self.v = np.insert(self.v, i, v)

    def remove(self, mask):
        keep = ~mask
        self.ids, self.x, self.v = self.ids[keep], self.x[keep], self.v[keep]

    def gaps_and_dv(self, length: float):
        """Gap to leader and approach rate; the most downstream vehicle has an infinite gap."""
        n = len(self.x)
        s = np.full(n, np.inf)
        dv = np.zeros(n)
        if n > 1:
            s[1:] = self.x[:-1] - self.x[1:] - length
            dv[1:] = self.v[1:] - self.v[:-1]
        return s, dv


class FreewaySegment:
    def __init__(self, params: IDMParams, cfg: ScenarioConfig,
                 arrivals: np.ndarray | None = None, ramp_arrivals: np.ndarray | None = None):
        self.p = params
        self.cfg = cfg
        self.ramp = cfg.ramp
        main_arr = generate_arrivals(cfg.demand_profile, cfg.arrival_process, cfg.seed) \
            if arrivals is None else np.sort(np.asarray(arrivals, dtype=float))
        if self.ramp is None:
            ramp_arr = np.empty(0)
        elif ramp_arrivals is not None:
            ramp_arr = np.sort(np.asarray(ramp_arrivals, dtype=float))
        else:
            rseed = self.ramp.seed if self.ramp.seed is not None else \
                (None if cfg.seed is None else cfg.seed + 1000)
            ramp_arr = generate_arrivals(self.ramp.demand_profile, self.ramp.arrival_process, rseed)

        # Vehicle IDs: initial mainline vehicles (downstream-most first), then mainline arrivals, then ramp.
        init = sorted(cfg.initial_vehicles, key=lambda r: -r[0])
        n_init, n_m, n_r = len(init), len(main_arr), len(ramp_arr)
        n = n_init + n_m + n_r
        self.origin = np.array(["main"] * (n_init + n_m) + ["ramp"] * n_r)
        self.arrival_s = np.concatenate([np.zeros(n_init), main_arr, ramp_arr])
        self.entry_s = np.full(n, np.nan)
        self.merge_s = np.full(n, np.nan)
        self.exit_s = np.full(n, np.nan)
        self.entry_s[:n_init] = 0.0

        self.main, self.rlane = _Lane(), _Lane()
        for i, (x0, v0) in enumerate(init):
            self.main.insert(len(self.main), i, float(x0), float(v0))
        # Pending arrivals per boundary: (time, vehicle_id), in time order.
        self._pending = {
            MAIN: deque(zip(main_arr, range(n_init, n_init + n_m))),
            RAMP: deque(zip(ramp_arr, range(n_init + n_m, n))),
        }
        self.queue = {MAIN: deque(), RAMP: deque()}
        self.n_arrived = n_init
        self.min_gap = np.inf
        self.last_arrival = float(max(main_arr.max(initial=0.0), ramp_arr.max(initial=0.0)))

    # -- boundary helpers
    def _try_insert(self, lane: _Lane, queue: deque, x_entry: float, t: float) -> None:
        if not queue:
            return
        if len(lane) == 0:
            v_ins, gap = self.p.v0, np.inf
        else:
            v_ins = float(lane.v[-1])
            gap = float(lane.x[-1] - self.p.length - x_entry)
        if gap >= self.p.s0 + v_ins * self.p.T:
            vid = queue.popleft()
            lane.insert(len(lane), vid, x_entry, v_ins)
            self.entry_s[vid] = t

    def _merges(self, t: float) -> None:
        """Mandatory merge from the acceleration lane, MOBIL safety criterion only."""
        r, m, p = self.rlane, self.main, self.p
        b_safe = self.ramp.b_safe_mps2
        i = 0
        while i < len(r):
            xr, vr = float(r.x[i]), float(r.v[i])
            if xr < self.ramp.merge_start_m:
                break  # lane is ordered downstream-first; nobody further back is in the merge zone
            # first mainline index with x <= xr (the would-be follower); leader is the one before it
            j = int(np.searchsorted(-m.x, -xr, side="left"))
            ok = True
            if j > 0:
                s_lead = m.x[j - 1] - xr - p.length
                ok = s_lead > 0 and acceleration(vr, s_lead, vr - m.v[j - 1], p) >= -b_safe
            if ok and j < len(m):
                s_fol = xr - m.x[j] - p.length
                ok = s_fol > 0 and acceleration(m.v[j], s_fol, m.v[j] - vr, p) >= -b_safe
            if ok:
                vid = int(r.ids[i])
                m.insert(j, vid, xr, vr)
                mask = np.zeros(len(r), dtype=bool)
                mask[i] = True
                r.remove(mask)
                self.merge_s[vid] = t
            else:
                i += 1

    def _ramp_accelerations(self, s_r, dv_r) -> np.ndarray:
        """Ramp-lane accelerations = min of three demands:
        (1) IDM toward the ramp leader;
        (2) lane end (most downstream ramp vehicle only): no reaction until the kinematic stopping
            deceleration v^2 / (2 d) reaches b, then brake at exactly that rate so the vehicle stops
            STOP_MARGIN_M before merge_end. This replaces IDM's virtual standing obstacle, whose
            approach-rate term makes vehicles brake ~v^2/(2 sqrt(ab)) ahead (~180 m at 20 m/s), i.e.
            for the whole acceleration lane;
        (3) speed adaptation (x >= merge_start): IDM toward the adjacent mainline leader, with braking
            capped at b, so ramp vehicles arrive at the merge at mainline speed.
        """
        p, r, m, rc = self.p, self.rlane, self.main, self.ramp
        acc = acceleration(r.v, s_r, dv_r, p)
        if len(r) == 0:
            return acc
        # (2) lane end
        d = rc.merge_end_m - STOP_MARGIN_M - r.x[0]
        v = r.v[0]
        if d <= 0.5:
            acc[0] = min(acc[0], -max(p.b, v * v / (2 * max(d, 1e-3))))
        else:
            b_req = v * v / (2 * d)
            if b_req >= (rc.lane_end_decel_mps2 or p.b):
                acc[0] = min(acc[0], -b_req)
        # (3) adapt to the mainline vehicle just ahead
        if not rc.speed_adaptation:
            return acc
        for i in np.flatnonzero(r.x >= rc.merge_start_m):
            j = int(np.searchsorted(-m.x, -r.x[i], side="left"))  # first mainline vehicle with x <= x_i
            if j == 0:
                continue
            s = m.x[j - 1] - r.x[i] - p.length
            a_sync = -p.b if s <= 0 else max(float(acceleration(r.v[i], s, r.v[i] - m.v[j - 1], p)), -p.b)
            acc[i] = min(acc[i], a_sync)
        return acc

    def _yield(self, acc_m: np.ndarray) -> None:
        """Cooperative yielding (after Hidas 2005): for each ramp vehicle in the merge zone, the mainline
        vehicle directly behind it treats it as an extra leader, but only if the IDM response needs no
        more than comfortable braking b. Otherwise it does not yield. Modifies acc_m in place."""
        p, r, m, rc = self.p, self.rlane, self.main, self.ramp
        for i in np.flatnonzero(r.x >= rc.merge_start_m):
            j = int(np.searchsorted(-m.x, -r.x[i], side="left"))  # would-be follower
            if j >= len(m):
                continue
            s = r.x[i] - m.x[j] - p.length
            if s <= 0 or r.v[i] < rc.yield_min_speed_ratio * m.v[j]:
                continue
            a_y = float(acceleration(m.v[j], s, m.v[j] - r.v[i], p))
            if a_y >= -p.b:
                acc_m[j] = min(acc_m[j], a_y)

    # -- main loop
    def run(self) -> SimResult:
        cfg, p = self.cfg, self.p
        dt = cfg.dt_s
        rec_every = max(1, int(round(cfg.record_every_s / dt)))
        max_steps = int(np.ceil(cfg.max_time_s / dt))
        traj_rows, count_rows = [], []
        k = 0
        while True:
            t = k * dt
            # 1) arrivals up to and including t join their queues
            for b in (MAIN, RAMP):
                pend = self._pending[b]
                while pend and pend[0][0] <= t + 1e-9:
                    self.queue[b].append(pend.popleft()[1])
                    self.n_arrived += 1
            # 2) boundary entries and merges (state at t)
            self._try_insert(self.main, self.queue[MAIN], 0.0, t)
            if self.ramp:
                self._try_insert(self.rlane, self.queue[RAMP], self.ramp.entrance_m, t)
                self._merges(t)
            # 3) accelerations + safety audit
            s_m, dv_m = self.main.gaps_and_dv(p.length)
            s_r, dv_r = self.rlane.gaps_and_dv(p.length)
            acc_m = acceleration(self.main.v, s_m, dv_m, p)
            acc_r = self._ramp_accelerations(s_r, dv_r) if self.ramp else np.empty(0)
            if self.ramp and self.ramp.cooperative_yield:
                self._yield(acc_m)
            finite = np.concatenate([s_m[np.isfinite(s_m)], s_r[1:]])
            if len(finite):
                g = float(finite.min())
                self.min_gap = min(self.min_gap, g)
                if g <= 0:
                    raise RuntimeError(f"collision/overlap at t={t:.2f}s (min gap {g:.3f} m)")
            # 4) record state at t
            if k % rec_every == 0:
                for lane_id, ln, acc in ((MAIN, self.main, acc_m), (RAMP, self.rlane, acc_r)):
                    if len(ln):
                        traj_rows.append(pd.DataFrame({
                            "time_s": np.full(len(ln), t), "vehicle_id": ln.ids.copy(),
                            "lane_id": np.full(len(ln), lane_id), "position_m": ln.x.copy(),
                            "speed_mps": ln.v.copy(), "acceleration_mps2": acc,
                        }))
                qm, qr = len(self.queue[MAIN]), len(self.queue[RAMP])
                count_rows.append((t, self.n_arrived, qm + qr, qm, qr, len(self.main) + len(self.rlane),
                                   len(self.main), len(self.rlane), int(np.isfinite(self.exit_s).sum())))
            # 5) stop when demand is over and everything has cleared, or at the time limit
            done = (not self._pending[MAIN] and not self._pending[RAMP] and not self.queue[MAIN]
                    and not self.queue[RAMP] and len(self.main) == 0 and len(self.rlane) == 0)
            if (done and t >= self.last_arrival) or k >= max_steps:
                break
            # 6) move everyone simultaneously
            x_old = self.main.x
            self.main.x, self.main.v = ballistic_update(self.main.x, self.main.v, acc_m, dt)
            if len(self.rlane):
                self.rlane.x, self.rlane.v = ballistic_update(self.rlane.x, self.rlane.v, acc_r, dt)
            # 7) exits at the downstream boundary (interpolate crossing time inside the step)
            out = self.main.x > cfg.length_m
            if np.any(out):
                frac = (cfg.length_m - x_old[out]) / (self.main.x[out] - x_old[out])
                self.exit_s[self.main.ids[out]] = t + frac * dt
                self.main.remove(out)
            k += 1

        end_t = k * dt
        vehicles = pd.DataFrame({
            "vehicle_id": np.arange(len(self.arrival_s)), "origin": self.origin,
            "arrival_s": self.arrival_s, "entry_s": self.entry_s, "merge_s": self.merge_s,
            "exit_s": self.exit_s,
        })
        counts = pd.DataFrame(count_rows, columns=["time_s", "n_arrived", "n_queued", "n_queued_main",
                                                   "n_queued_ramp", "n_on_road", "n_on_main", "n_on_ramp",
                                                   "n_exited"])
        traj = pd.concat(traj_rows, ignore_index=True) if traj_rows else pd.DataFrame(
            columns=["time_s", "vehicle_id", "lane_id", "position_m", "speed_mps", "acceleration_mps2"])
        cleared = bool(np.isfinite(self.exit_s).all())  # False also if the time limit cut off demand
        return SimResult(cfg, p, traj, vehicles, counts, end_t, float(self.min_gap), cleared)


SingleLaneSegment = FreewaySegment  # backward-compatible name: a segment with no ramp configured


def run_scenario(path: str | Path, dt_s: float | None = None) -> SimResult:
    cfg = ScenarioConfig.from_yaml(path)
    if dt_s is not None:
        cfg.dt_s = dt_s
    return FreewaySegment(load_params(cfg.param_set), cfg).run()
