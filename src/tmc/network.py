"""Freeway segment: 1+ mainline lanes, optional lane changing, optional on-ramp with acceleration lane.

    mainline arrivals -> entry queue -> mainline lanes (x = 0 .. L) -> free exit at x = L
    ramp arrivals -> ramp queue -> ramp entrance (stop line, future meter) -> approach -> acceleration lane
                  -> merge into the right mainline lane (lane 0) when safe

Design choices (each traced in docs/source_to_design.md):
  * Arrivals: piecewise-constant demand, Poisson (seeded) or uniform headways. Mainline and ramp use
    independent random streams (ramp seed = seed + 1000 unless set explicitly).
  * Entry rule (mainline entrance and ramp entrance alike): a queued vehicle enters a lane when the gap
    to the last vehicle on that lane is at least s0 + v_ins*T, at v_ins = that vehicle's speed (v0 if the
    lane is empty). With several mainline lanes, the one queue feeds every lane that passes this test,
    largest gap first (ties -> rightmost lane), at most one vehicle per lane per step. Vehicles that
    cannot enter WAIT in a point queue; they are never dropped.
  * Lane changing (n_lanes >= 2): symmetric MOBIL (Kesting, Treiber & Helbing 2007), no keep-right bias.
    Safety: gaps > 0, and the changer and its new follower decelerate no harder than b_safe. Incentive:
    a_c_new - a_c + p * (da_new_follower + da_old_follower) > threshold. Decisions use the state at t;
    candidates are applied largest-gain first, each re-checked for safety against the lanes as already
    updated in this step. A cooldown (min time between changes of one vehicle) prevents ping-pong.
    Lane changes are instantaneous (no lateral motion), a stated limitation.
  * Ramp geometry: ramp positions use the mainline x axis. The ramp entrance is at
    x = merge_start - approach_length. The acceleration lane runs merge_start .. merge_end.
  * Acceleration-lane driving (FreewaySegment._ramp_accelerations): IDM toward the ramp leader and a
    kinematic lane-end stop (brake only once v^2/(2d) >= b). Optional (config, default off): IDM speed
    adaptation to the adjacent mainline leader, and cooperative yielding by the mainline follower.
  * Merge rule (MOBIL safety criterion, mandatory lane change): a ramp vehicle with x >= merge_start
    merges into lane 0 if the same safety test as above holds (with the ramp's b_safe). Ramp vehicles
    are checked downstream-first; each merge is seen by the next check.
  * Downstream boundary: free exit; exit time interpolated within the step.
  * Simultaneous update: accelerations from the state at t, then everyone moves.

Lane IDs: 0 = rightmost mainline lane (next to the ramp), 1, 2, ... to the left; -1 = ramp /
acceleration lane. Units: SI. Position = front bumper.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tmc.detectors import DetectorBank, DetectorSpec
from tmc.idm import IDMParams, acceleration, ballistic_update, load_params

MAIN, RAMP = 0, -1   # MAIN = rightmost mainline lane; further mainline lanes are 1, 2, ...
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
    gap_seeking: bool = False            # in the merge zone, steer toward the center of an adjacent lane-0 gap
    gap_kx: float = 0.1                  # gap-seeking position gain, 1/s^2
    gap_kv: float = 0.6                  # gap-seeking speed gain, 1/s
    anticipation_m: float = 0.0          # lane-0 drivers from merge_start - this to merge_end count the nearest
                                         # ramp vehicle ahead as a leader in lane-change decisions (0 = off)
    courtesy: str = "none"               # "zipper": lane-0 would-be follower yields (<= courtesy_b) to the ramp
                                         # vehicle ahead of it, and each mainline driver lets in at most one
    courtesy_b_mps2: float | None = None # max courtesy braking (None -> IDM b)

    @property
    def entrance_m(self) -> float:
        return self.merge_start_m - self.approach_length_m


@dataclass
class LaneChangeConfig:
    """Discretionary lane changes between mainline lanes (symmetric MOBIL). Used only if n_lanes >= 2."""
    enabled: bool = True
    politeness: float = 0.2              # p: weight on the followers' acceleration changes
    threshold_mps2: float = 0.1          # delta a_th: minimum net gain to change
    b_safe_mps2: float = 4.0             # safety: max deceleration imposed on self and new follower
    cooldown_s: float = 3.0              # min time between two changes of one vehicle (anti ping-pong)


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
    initial_vehicles: list = field(default_factory=list)  # [[position_m, speed_mps(, lane)], ...] mainline
    ramp: RampConfig | None = None
    n_lanes: int = 1
    lane_change: LaneChangeConfig = field(default_factory=LaneChangeConfig)
    detectors: list = field(default_factory=list)   # [DetectorSpec, ...] (tmc.detectors)
    control: dict | None = None                     # ramp-meter controller spec (tmc.control.make_controller)

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
                gap_seeking=bool(r.get("gap_seeking", False)),
                gap_kx=float(r.get("gap_kx", 0.1)),
                gap_kv=float(r.get("gap_kv", 0.6)),
                anticipation_m=float(r.get("anticipation_m", 0.0)),
                courtesy=str(r.get("courtesy", "none")),
                courtesy_b_mps2=r.get("courtesy_b_mps2"),
            )
            if ramp.courtesy not in ("none", "zipper"):
                raise ValueError(f"ramp.courtesy must be 'none' or 'zipper', got {ramp.courtesy!r}")
        lc = d.get("lane_change") or {}
        lane_change = LaneChangeConfig(
            enabled=bool(lc.get("enabled", True)),
            politeness=float(lc.get("politeness", 0.2)),
            threshold_mps2=float(lc.get("threshold_mps2", 0.1)),
            b_safe_mps2=float(lc.get("b_safe_mps2", 4.0)),
            cooldown_s=float(lc.get("cooldown_s", 3.0)),
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
            n_lanes=int(d["road"].get("n_lanes", 1)),
            lane_change=lane_change,
            detectors=[DetectorSpec.from_dict(x) for x in (d.get("detectors") or [])],
            control=d.get("control"),
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

LC_COLUMNS = ["time_s", "vehicle_id", "from_lane", "to_lane", "position_m", "speed_mps", "kind"]


@dataclass
class SimResult:
    config: ScenarioConfig
    params: IDMParams
    trajectories: pd.DataFrame   # VehicleState rows: time_s, vehicle_id, lane_id, position_m, speed_mps, acceleration_mps2
    vehicles: pd.DataFrame       # vehicle_id, origin, arrival_s, entry_s, entry_lane, merge_s, exit_s, n_lane_changes
    counts: pd.DataFrame         # time_s, n_arrived, n_queued(_main/_ramp), n_on_road, n_on_main, n_on_ramp, n_exited, n_on_lane_<k>
    end_time_s: float
    min_gap_m: float
    cleared: bool                # all scheduled vehicles exited before max_time_s
    lane_changes: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=LC_COLUMNS))
    # lane_changes: one row per lane change, kind = "discretionary" (mainline MOBIL) or "merge" (ramp -> lane 0)
    observations: pd.DataFrame = field(default_factory=pd.DataFrame)
    control_trace: pd.DataFrame = field(default_factory=pd.DataFrame)   # one row per controller update
    # observations: one row per detector interval (interfaces.Observation fields), empty if no detectors

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
        main_speeds = tr.loc[tr["lane_id"] >= MAIN, "speed_mps"] if len(tr) else pd.Series(dtype=float)
        lc = self.lane_changes
        return {
            "scenario_id": self.config.scenario_id,
            "demand_id": self.config.demand_id,
            "dt_s": self.config.dt_s,
            "n_lanes": self.config.n_lanes,
            "n_arrived": int(len(v)),
            "n_arrived_ramp": int(is_ramp.sum()),
            "n_exited": int(exited.sum()),
            "n_merged": int(v["merge_s"].notna().sum()),
            "n_lane_changes": int((lc["kind"] == "discretionary").sum()) if len(lc) else 0,
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

    def remove_at(self, i: int):
        self.ids, self.x, self.v = np.delete(self.ids, i), np.delete(self.x, i), np.delete(self.v, i)

    def slot(self, x: float) -> int:
        """Index a vehicle at x would take: first vehicle with position <= x (its would-be follower)."""
        return int(np.searchsorted(-self.x, -x, side="left"))

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
                 arrivals: np.ndarray | None = None, ramp_arrivals: np.ndarray | None = None,
                 controller=None):
        if cfg.n_lanes < 1:
            raise ValueError("n_lanes must be >= 1")
        self.p = params
        self.cfg = cfg
        self.ramp = cfg.ramp
        self.lc = cfg.lane_change
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
        self.entry_lane = np.full(n, np.nan)
        self.merge_s = np.full(n, np.nan)
        self.exit_s = np.full(n, np.nan)
        self.last_lc = np.full(n, -np.inf)   # time of each vehicle's last lane change or merge
        self.let_in = np.zeros(n, dtype=bool)  # zipper: this mainline vehicle already let a ramp vehicle in
        self.entry_s[:n_init] = 0.0

        self.lanes = [_Lane() for _ in range(cfg.n_lanes)]
        self.rlane = _Lane()
        for i, row in enumerate(init):
            lane = int(row[2]) if len(row) > 2 else MAIN
            if not 0 <= lane < cfg.n_lanes:
                raise ValueError(f"initial vehicle {i}: lane {lane} outside 0..{cfg.n_lanes - 1}")
            ln = self.lanes[lane]
            ln.insert(len(ln), i, float(row[0]), float(row[1]))
            self.entry_lane[i] = lane
        # Pending arrivals per boundary: (time, vehicle_id), in time order.
        self._pending = {
            MAIN: deque(zip(main_arr, range(n_init, n_init + n_m))),
            RAMP: deque(zip(ramp_arr, range(n_init + n_m, n))),
        }
        self.queue = {MAIN: deque(), RAMP: deque()}
        self.n_arrived = n_init
        self.min_gap = np.inf
        self.lc_log: list[tuple] = []
        self.last_arrival = float(max(main_arr.max(initial=0.0), ramp_arr.max(initial=0.0)))
        # Detectors are passive: they read the state and never change it.
        self.det = DetectorBank(cfg.detectors, cfg.n_lanes, params.length, cfg.dt_s) if cfg.detectors else None
        # Ramp meter at the ramp entrance (stop line). meter_rate None = dark (no signal).
        if controller is None and cfg.control:
            from tmc.control import make_controller
            controller = make_controller(cfg.control)
        if controller is not None and (self.ramp is None or self.det is None):
            raise ValueError("a ramp controller needs a ramp and detectors (it may only read detectors)")
        self.controller = controller
        self.meter_rate: float | None = None
        self.next_release = 0.0
        self._next_ctrl = 0.0
        if controller is not None:
            from tmc.estimation import DetectorView
            self._view = DetectorView(self.det)

    @property
    def main(self) -> _Lane:
        """Rightmost mainline lane (the only one when n_lanes = 1)."""
        return self.lanes[MAIN]

    # -- boundary helpers
    def _entry_gap(self, lane: _Lane, x_entry: float):
        """(v_ins, gap) for a vehicle entering `lane` at x_entry."""
        if len(lane) == 0:
            return self.p.v0, np.inf
        return float(lane.v[-1]), float(lane.x[-1] - self.p.length - x_entry)

    def _try_insert(self, lane: _Lane, queue: deque, x_entry: float, t: float) -> None:
        if not queue:
            return
        v_ins, gap = self._entry_gap(lane, x_entry)
        if gap >= self.p.s0 + v_ins * self.p.T:
            vid = queue.popleft()
            lane.insert(len(lane), vid, x_entry, v_ins)
            self.entry_s[vid] = t

    def _ramp_entry(self, t: float) -> None:
        """Ramp stop line. Dark meter: the usual entry rule. Metered: one vehicle per green, at most one
        every 3600 / rate seconds; it starts from a stop (v = 0), so it needs only s0 of clear space."""
        q = self.queue[RAMP]
        x_e = self.ramp.entrance_m
        if self.meter_rate is None:
            self._try_insert(self.rlane, q, x_e, t)
            return
        if not q or t + 1e-9 < self.next_release:
            return
        _, gap = self._entry_gap(self.rlane, x_e)
        if gap >= self.p.s0:
            vid = q.popleft()
            self.rlane.insert(len(self.rlane), vid, x_e, 0.0)
            self.entry_s[vid] = t
            self.next_release = t + 3600.0 / self.meter_rate

    def _enter_mainline(self, t: float) -> None:
        """One queue feeds all mainline lanes: every lane passing the entry test takes the next queued
        vehicle, largest gap first (ties -> rightmost lane), at most one per lane per step."""
        q = self.queue[MAIN]
        if not q:
            return
        options = []
        for k, ln in enumerate(self.lanes):
            v_ins, gap = self._entry_gap(ln, 0.0)
            if gap >= self.p.s0 + v_ins * self.p.T:
                options.append((-gap, k, v_ins))
        for _, k, v_ins in sorted(options):
            if not q:
                break
            vid = q.popleft()
            ln = self.lanes[k]
            ln.insert(len(ln), vid, 0.0, v_ins)
            self.entry_s[vid] = t
            self.entry_lane[vid] = k

    def _safe_slot(self, lane: _Lane, x: float, v: float, b_safe: float):
        """MOBIL safety criterion for a vehicle (x, v) moving into `lane`: gaps to the new leader and
        follower > 0, and neither the mover nor its new follower must brake harder than b_safe.
        Returns the insertion index if safe, else None."""
        p = self.p
        j = lane.slot(x)
        if j > 0:
            s_lead = lane.x[j - 1] - x - p.length
            if not (s_lead > 0 and acceleration(v, s_lead, v - lane.v[j - 1], p) >= -b_safe):
                return None
        if j < len(lane):
            s_fol = x - lane.x[j] - p.length
            if not (s_fol > 0 and acceleration(lane.v[j], s_fol, lane.v[j] - v, p) >= -b_safe):
                return None
        return j

    def _merges(self, t: float) -> None:
        """Mandatory merge from the acceleration lane into lane 0, MOBIL safety criterion only."""
        r, m = self.rlane, self.main
        i = 0
        while i < len(r):
            xr, vr = float(r.x[i]), float(r.v[i])
            if xr < self.ramp.merge_start_m:
                break  # lane is ordered downstream-first; nobody further back is in the merge zone
            j = self._safe_slot(m, xr, vr, self.ramp.b_safe_mps2)
            if j is not None:
                vid = int(r.ids[i])
                m.insert(j, vid, xr, vr)
                r.remove_at(i)
                if j + 1 < len(m):
                    self.let_in[m.ids[j + 1]] = True   # its new follower has now let one in
                self.merge_s[vid] = t
                self.last_lc[vid] = t
                self.lc_log.append((t, vid, RAMP, MAIN, xr, vr, "merge"))
            else:
                i += 1

    def _lane_changes(self, t: float, accs: list) -> None:
        """Discretionary lane changes between mainline lanes (symmetric MOBIL).

        Step 1 (state at t, vectorized): for each vehicle i in lane A and each neighbor lane B, with
            a_c       = i's current acceleration,             a_c_new = i's acceleration behind B's leader,
            n         = i's would-be follower in B,           da_n    = n's acceleration change,
            o         = i's current follower in A,            da_o    = o's change if i leaves,
        candidate if: cooldown over, gaps > 0, a_c_new >= -b_safe, n's new acceleration >= -b_safe, and
            gain = a_c_new - a_c + p * (da_n + da_o) > threshold.
        Step 2: apply candidates largest gain first; re-check safety against the updated lanes, skip any
        vehicle that has already changed in this step.
        """
        lc, p = self.lc, self.p
        bs = lc.b_safe_mps2
        cands = []
        for ia, A in enumerate(self.lanes):
            n = len(A)
            if n == 0:
                continue
            ready = (t - self.last_lc[A.ids]) >= lc.cooldown_s - 1e-9
            if not ready.any():
                continue
            a_c = accs[ia]
            if ia == MAIN:
                a_c = np.minimum(a_c, self._ramp_anticipation(A.x, A.v))
            # old follower o = i+1 would follow i-1 (or a free road if i = 0) once i leaves
            da_o = np.zeros(n)
            if n > 1:
                fo = np.arange(1, n)
                lead = fo - 2                       # i - 1, where i = fo - 1
                s_o = np.full(n - 1, np.inf)
                dv_o = np.zeros(n - 1)
                h = lead >= 0
                s_o[h] = A.x[lead[h]] - A.x[fo[h]] - p.length
                dv_o[h] = A.v[fo[h]] - A.v[lead[h]]
                da_o[:-1] = acceleration(A.v[fo], s_o, dv_o, p) - a_c[fo]
            for ib in (ia - 1, ia + 1):
                if not 0 <= ib < len(self.lanes):
                    continue
                B, a_b = self.lanes[ib], accs[ib]
                j = np.searchsorted(-B.x, -A.x, side="left")   # would-be follower index in B
                hl, hf = j > 0, j < len(B)
                s_l = np.full(n, np.inf)
                dv_l = np.zeros(n)
                s_l[hl] = B.x[j[hl] - 1] - A.x[hl] - p.length
                dv_l[hl] = A.v[hl] - B.v[j[hl] - 1]
                a_c_new = acceleration(A.v, s_l, dv_l, p)
                if ib == MAIN:
                    a_c_new = np.minimum(a_c_new, self._ramp_anticipation(A.x, A.v))
                s_f = np.full(n, np.inf)
                a_n_new = np.zeros(n)
                da_n = np.zeros(n)
                if hf.any():
                    jf = j[hf]
                    s_f[hf] = A.x[hf] - B.x[jf] - p.length
                    a_n_new[hf] = acceleration(B.v[jf], s_f[hf], B.v[jf] - A.v[hf], p)
                    da_n[hf] = a_n_new[hf] - a_b[jf]
                with np.errstate(invalid="ignore"):
                    safe = (s_l > 0) & (s_f > 0) & (a_c_new >= -bs) & (~hf | (a_n_new >= -bs))
                gain = a_c_new - a_c + lc.politeness * (da_n + da_o)
                for i in np.flatnonzero(ready & safe & (gain > lc.threshold_mps2)):
                    cands.append((-float(gain[i]), int(A.ids[i]), ia, ib))
        moved = set()
        for _, vid, ia, ib in sorted(cands):
            if vid in moved:
                continue
            A, B = self.lanes[ia], self.lanes[ib]
            i = int(np.flatnonzero(A.ids == vid)[0])
            x, v = float(A.x[i]), float(A.v[i])
            j = self._safe_slot(B, x, v, bs)
            if j is None:
                continue
            A.remove_at(i)
            B.insert(j, vid, x, v)
            self.last_lc[vid] = t
            moved.add(vid)
            self.lc_log.append((t, vid, ia, ib, x, v, "discretionary"))

    def _ramp_anticipation(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Perceived lane-0 acceleration limit from ramp traffic, for lane-change decisions only.
        A lane-0 position in [merge_start - anticipation_m, merge_end] counts the nearest acceleration-lane
        vehicle ahead (x_r > x, x_r >= merge_start) as a leader; returns IDM toward it, floored at -b_safe
        so an overlapping ramp vehicle gives a strong but finite incentive. +inf where it does not apply."""
        out = np.full(len(x), np.inf)
        rc = self.ramp
        if rc is None or rc.anticipation_m <= 0 or len(self.rlane) == 0 or len(x) == 0:
            return out
        r = self.rlane
        k = int(np.searchsorted(-r.x, -rc.merge_start_m, side="right"))  # r.x[:k] are in the merge zone
        if k == 0:
            return out
        zx, zv = r.x[:k], r.v[:k]
        zone = (x >= rc.merge_start_m - rc.anticipation_m) & (x <= rc.merge_end_m)
        j = np.searchsorted(-zx, -x, side="left") - 1     # nearest merge-zone ramp vehicle with x_r > x
        h = zone & (j >= 0)
        if h.any():
            s = zx[j[h]] - x[h] - self.p.length
            with np.errstate(invalid="ignore", divide="ignore"):
                a = np.where(s > 0, acceleration(v[h], np.where(s > 0, s, 1.0), v[h] - zv[j[h]], self.p),
                             -self.lc.b_safe_mps2)
            out[h] = np.maximum(a, -self.lc.b_safe_mps2)
        return out

    def _gap_seeking(self, i: int) -> float | None:
        """Gap-seeking acceleration for ramp vehicle i in the merge zone: steer front bumper toward the
        center of the larger of the two adjacent lane-0 gaps (beside it, or the one behind), matching that
        gap's mean speed. PD law a = kx (x* - x) + kv (v* - v), clipped to [-b, a]. None if no gap exists."""
        p, r, m, rc = self.p, self.rlane, self.main, self.ramp
        x, v = r.x[i], r.v[i]
        j = m.slot(x)                               # m[j-1] ahead, m[j] behind (if present)
        best = None
        for lead, fol in ((j - 1, j), (j, j + 1)):
            if lead < 0 or fol >= len(m):
                continue
            size = m.x[lead] - p.length - m.x[fol]
            if best is None or size > best[0]:
                best = (size, lead, fol)
        if best is None:
            return None
        _, lead, fol = best
        x_star = 0.5 * (m.x[lead] + m.x[fol])       # front bumper here centers the car in the gap
        v_star = 0.5 * (m.v[lead] + m.v[fol])
        return float(np.clip(rc.gap_kx * (x_star - x) + rc.gap_kv * (v_star - v), -p.b, p.a))

    def _ramp_accelerations(self, s_r, dv_r) -> np.ndarray:
        """Ramp-lane accelerations = min of three demands:
        (1) IDM toward the ramp leader;
        (2) lane end (most downstream ramp vehicle only): no reaction until the kinematic stopping
            deceleration v^2 / (2 d) reaches b, then brake at exactly that rate so the vehicle stops
            STOP_MARGIN_M before merge_end. This replaces IDM's virtual standing obstacle, whose
            approach-rate term makes vehicles brake ~v^2/(2 sqrt(ab)) ahead (~180 m at 20 m/s), i.e.
            for the whole acceleration lane;
        (3) speed adaptation (x >= merge_start): IDM toward the adjacent lane-0 leader, with braking
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
        # (3b) gap seeking (replaces speed adaptation when on): still never above IDM / lane-end limits
        if rc.gap_seeking:
            for i in np.flatnonzero(r.x >= rc.merge_start_m):
                a_g = self._gap_seeking(int(i))
                if a_g is not None:
                    acc[i] = min(acc[i], a_g)
            return acc
        # (3) adapt to the lane-0 vehicle just ahead
        if not rc.speed_adaptation:
            return acc
        for i in np.flatnonzero(r.x >= rc.merge_start_m):
            j = m.slot(r.x[i])  # first lane-0 vehicle with x <= x_i
            if j == 0:
                continue
            s = m.x[j - 1] - r.x[i] - p.length
            a_sync = -p.b if s <= 0 else max(float(acceleration(r.v[i], s, r.v[i] - m.v[j - 1], p)), -p.b)
            acc[i] = min(acc[i], a_sync)
        return acc

    def _yield(self, acc_m: np.ndarray) -> None:
        """Cooperative yielding (after Hidas 2005): for each ramp vehicle in the merge zone, the lane-0
        vehicle directly behind it treats it as an extra leader, but only if the IDM response needs no
        more than comfortable braking b. Otherwise it does not yield. Modifies acc_m in place."""
        p, r, m, rc = self.p, self.rlane, self.main, self.ramp
        for i in np.flatnonzero(r.x >= rc.merge_start_m):
            j = m.slot(r.x[i])  # would-be follower
            if j >= len(m):
                continue
            s = r.x[i] - m.x[j] - p.length
            if s <= 0 or r.v[i] < rc.yield_min_speed_ratio * m.v[j]:
                continue
            a_y = float(acceleration(m.v[j], s, m.v[j] - r.v[i], p))
            if a_y >= -p.b:
                acc_m[j] = min(acc_m[j], a_y)

    def _courtesy(self, acc_m: np.ndarray) -> None:
        """Zipper courtesy (cooperative merge after Hidas 2005), ramp.courtesy == "zipper". For each ramp
        vehicle in the merge zone, its would-be lane-0 follower treats it as an extra leader if the gap is
        > 0, the IDM response needs braking no harder than courtesy_b (default b), and that follower has not
        already let a ramp vehicle in. Each mainline driver yields to at most one merging vehicle, so
        courtesy braking is bounded and saturated merging alternates. Modifies acc_m in place."""
        p, r, m, rc = self.p, self.rlane, self.main, self.ramp
        b_c = rc.courtesy_b_mps2 if rc.courtesy_b_mps2 is not None else p.b
        for i in np.flatnonzero(r.x >= rc.merge_start_m):
            j = m.slot(r.x[i])  # would-be follower
            if j >= len(m) or self.let_in[m.ids[j]]:
                continue
            s = r.x[i] - m.x[j] - p.length
            if s <= 0:
                continue
            a_y = float(acceleration(m.v[j], s, m.v[j] - r.v[i], p))
            if a_y >= -b_c:
                acc_m[j] = min(acc_m[j], a_y)

    def _accelerations(self):
        """Accelerations of every mainline lane and the ramp lane from the current state.
        Returns (list of per-lane arrays, ramp array, list of per-lane gaps, ramp gaps)."""
        p = self.p
        accs, gaps = [], []
        for ln in self.lanes:
            s, dv = ln.gaps_and_dv(p.length)
            accs.append(acceleration(ln.v, s, dv, p))
            gaps.append(s)
        s_r, dv_r = self.rlane.gaps_and_dv(p.length)
        acc_r = self._ramp_accelerations(s_r, dv_r) if self.ramp else np.empty(0)
        if self.ramp and self.ramp.cooperative_yield:
            self._yield(accs[MAIN])
        if self.ramp and self.ramp.courtesy == "zipper":
            self._courtesy(accs[MAIN])
        return accs, acc_r, gaps, s_r

    # -- main loop
    def run(self) -> SimResult:
        cfg, p = self.cfg, self.p
        dt = cfg.dt_s
        rec_every = max(1, int(round(cfg.record_every_s / dt)))
        max_steps = int(np.ceil(cfg.max_time_s / dt))
        do_lc = cfg.n_lanes >= 2 and self.lc.enabled
        lane_id_of = {id(ln): i for i, ln in enumerate(self.lanes)}
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
                    if self.det:
                        self.det.arrival(b)
            if self.det:
                self.det.advance(t)   # close intervals ending at t (arrivals up to t belong to them)
            if self.controller is not None and t + 1e-9 >= self._next_ctrl:
                cmd = self.controller.update(t, self._view)       # sees detector data only
                self.meter_rate = None if cmd.action == "dark" else float(cmd.value)
                self._next_ctrl += self.controller.interval_s
            # 2) boundary entries, merges, lane changes (state at t)
            self._enter_mainline(t)
            if self.ramp:
                self._ramp_entry(t)
                self._merges(t)
            if do_lc:
                self._lane_changes(t, self._accelerations()[0])
            if self.det:
                states = {i: ln.x for i, ln in enumerate(self.lanes)}
                states[RAMP] = self.rlane.x
                self.det.sample(states)
            # 3) accelerations + safety audit
            accs, acc_r, gaps, s_r = self._accelerations()
            finite = np.concatenate([g[np.isfinite(g)] for g in gaps] + [s_r[1:]])
            if len(finite):
                g = float(finite.min())
                self.min_gap = min(self.min_gap, g)
                if g <= 0:
                    raise RuntimeError(f"collision/overlap at t={t:.2f}s (min gap {g:.3f} m)")
            # 4) record state at t
            if k % rec_every == 0:
                for lane_id, ln, acc in [*((i, ln, a) for i, (ln, a) in enumerate(zip(self.lanes, accs))),
                                         (RAMP, self.rlane, acc_r)]:
                    if len(ln):
                        traj_rows.append(pd.DataFrame({
                            "time_s": np.full(len(ln), t), "vehicle_id": ln.ids.copy(),
                            "lane_id": np.full(len(ln), lane_id), "position_m": ln.x.copy(),
                            "speed_mps": ln.v.copy(), "acceleration_mps2": acc,
                        }))
                qm, qr = len(self.queue[MAIN]), len(self.queue[RAMP])
                on_lanes = [len(ln) for ln in self.lanes]
                count_rows.append((t, self.n_arrived, qm + qr, qm, qr, sum(on_lanes) + len(self.rlane),
                                   sum(on_lanes), len(self.rlane), int(np.isfinite(self.exit_s).sum()),
                                   *on_lanes))
            # 5) stop when demand is over and everything has cleared, or at the time limit
            done = (not self._pending[MAIN] and not self._pending[RAMP] and not self.queue[MAIN]
                    and not self.queue[RAMP] and all(len(ln) == 0 for ln in self.lanes)
                    and len(self.rlane) == 0)
            if (done and t >= self.last_arrival) or k >= max_steps:
                break
            # 6) move everyone simultaneously, 7) exits at the downstream boundary (interpolated in the step)
            for ln, acc in zip(self.lanes, accs):
                if len(ln) == 0:
                    continue
                x_old, v_old = ln.x, ln.v
                ln.x, ln.v = ballistic_update(ln.x, ln.v, acc, dt)
                if self.det:
                    self.det.crossings(lane_id_of[id(ln)], x_old, ln.x, v_old, ln.v)
                out = ln.x > cfg.length_m
                if np.any(out):
                    frac = (cfg.length_m - x_old[out]) / (ln.x[out] - x_old[out])
                    self.exit_s[ln.ids[out]] = t + frac * dt
                    ln.remove(out)
            if len(self.rlane):
                x_old, v_old = self.rlane.x, self.rlane.v
                self.rlane.x, self.rlane.v = ballistic_update(self.rlane.x, self.rlane.v, acc_r, dt)
                if self.det:
                    self.det.crossings(RAMP, x_old, self.rlane.x, v_old, self.rlane.v)
            k += 1

        end_t = k * dt
        lc = pd.DataFrame(self.lc_log, columns=LC_COLUMNS)
        n_changes = np.bincount(lc.loc[lc["kind"] == "discretionary", "vehicle_id"].astype(int),
                                minlength=len(self.arrival_s)) if len(lc) else np.zeros(len(self.arrival_s), int)
        vehicles = pd.DataFrame({
            "vehicle_id": np.arange(len(self.arrival_s)), "origin": self.origin,
            "arrival_s": self.arrival_s, "entry_s": self.entry_s, "entry_lane": self.entry_lane,
            "merge_s": self.merge_s, "exit_s": self.exit_s, "n_lane_changes": n_changes,
        })
        counts = pd.DataFrame(count_rows, columns=["time_s", "n_arrived", "n_queued", "n_queued_main",
                                                   "n_queued_ramp", "n_on_road", "n_on_main", "n_on_ramp",
                                                   "n_exited"] + [f"n_on_lane_{i}" for i in range(cfg.n_lanes)])
        traj = pd.concat(traj_rows, ignore_index=True) if traj_rows else pd.DataFrame(
            columns=["time_s", "vehicle_id", "lane_id", "position_m", "speed_mps", "acceleration_mps2"])
        cleared = bool(np.isfinite(self.exit_s).all())  # False also if the time limit cut off demand
        obs = pd.DataFrame()
        if self.det:
            self.det.finish(end_t)
            obs = self.det.to_frame()
        trace = self.controller.trace_frame() if self.controller is not None else pd.DataFrame()
        return SimResult(cfg, p, traj, vehicles, counts, end_t, float(self.min_gap), cleared, lc, obs, trace)


SingleLaneSegment = FreewaySegment  # backward-compatible name: a segment with no ramp configured


def run_scenario(path: str | Path, dt_s: float | None = None) -> SimResult:
    cfg = ScenarioConfig.from_yaml(path)
    if dt_s is not None:
        cfg.dt_s = dt_s
    return FreewaySegment(load_params(cfg.param_set), cfg).run()
