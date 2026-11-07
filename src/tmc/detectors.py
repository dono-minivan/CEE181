"""Virtual detectors: the only traffic data a controller may see (CLAUDE.md rule 2).

    python -m tmc.detectors [scenario.yaml]      # small example: runs the scenario, prints observations

Detector kinds (configured under `detectors:` in a scenario YAML):
  * loop     -- point detector at position_m on one lane (lane_id = 0, 1, ... mainline, -1 = ramp),
                or on every mainline lane at once (lane_id omitted/None = a "station").
                Per interval: count, mean speed, occupancy.
  * arrivals -- counts vehicles joining a boundary queue (boundary "main" or "ramp"), i.e. a demand
                detector upstream of the entry / ramp queue. Count only; speed and occupancy are None.

Measurement definitions (traced in docs/source_to_design.md):
  * Crossing: a vehicle's front bumper passes x_d during a step (x_old < x_d <= x_new). Crossing time and
    spot speed are interpolated linearly within the step.
  * Count: crossings in the interval [k I, (k+1) I).
  * Mean speed: harmonic mean of the spot speeds, an estimate of the space-mean speed (so q = k v holds
    for the derived quantities). None when count = 0 (unmeasurable, never 0). Spot speeds are floored at
    V_FLOOR_MPS so a vehicle crossing at the instant it stops does not make the harmonic mean zero.
  * Occupancy: fraction of time steps in the interval in which any vehicle body [x - l, x] overlaps the
    detector field [x_d, x_d + field_length]. Sampled every step, so a vehicle standing on the loop
    counts as occupied (crossing-based occupancy would miss it). Station occupancy = mean over lanes.
  * Intervals must be a whole number of simulation steps. The last interval, if cut short by the end of
    the run, carries quality_flag "partial".

Units: seconds, meters, m/s internally. Output rows follow interfaces.Observation.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from tmc.interfaces import Observation

V_FLOOR_MPS = 0.1
BOUNDARIES = {"main": 0, "ramp": -1}


@dataclass(frozen=True)
class DetectorSpec:
    sensor_id: str
    kind: str = "loop"                  # "loop" or "arrivals"
    position_m: float | None = None     # loop only
    lane_id: int | None = None          # loop: lane (None = all mainline lanes); unused for arrivals
    boundary: str | None = None         # arrivals only: "main" or "ramp"
    interval_s: float = 30.0
    field_length_m: float = 2.0         # loop length along the road

    @classmethod
    def from_dict(cls, d: dict) -> "DetectorSpec":
        lane = d.get("lane_id", d.get("lane"))
        return cls(
            sensor_id=str(d["id"]),
            kind=d.get("kind", "loop"),
            position_m=None if d.get("position_m") is None else float(d["position_m"]),
            lane_id=None if lane is None else int(lane),
            boundary=d.get("boundary"),
            interval_s=float(d.get("interval_s", 30.0)),
            field_length_m=float(d.get("field_length_m", 2.0)),
        )

    def validate(self, n_lanes: int) -> None:
        if self.kind == "loop":
            if self.position_m is None:
                raise ValueError(f"detector {self.sensor_id}: loop needs position_m")
            if self.lane_id is not None and not -1 <= self.lane_id < n_lanes:
                raise ValueError(f"detector {self.sensor_id}: lane {self.lane_id} outside -1..{n_lanes - 1}")
        elif self.kind == "arrivals":
            if self.boundary not in BOUNDARIES:
                raise ValueError(f"detector {self.sensor_id}: arrivals needs boundary 'main' or 'ramp'")
        else:
            raise ValueError(f"detector {self.sensor_id}: unknown kind {self.kind!r}")
        if self.interval_s <= 0:
            raise ValueError(f"detector {self.sensor_id}: interval_s must be > 0")


class _Acc:
    """Accumulator for one detector's current interval."""

    def __init__(self, n_lanes_covered: int):
        self.k = 0                                   # current interval index
        self.count = 0
        self.inv_speed_sum = 0.0
        self.occ_steps = np.zeros(n_lanes_covered)   # occupied steps per covered lane
        self.n_steps = 0

    def reset(self) -> None:
        self.count = 0
        self.inv_speed_sum = 0.0
        self.occ_steps[:] = 0
        self.n_steps = 0


class DetectorBank:
    """All detectors of one run. The simulator calls arrival(), advance(), sample() and crossings();
    controllers read finished intervals through latest() / observations."""

    def __init__(self, specs: list[DetectorSpec], n_lanes: int, vehicle_length_m: float, dt_s: float):
        self.specs = list(specs)
        ids = [s.sensor_id for s in self.specs]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate detector ids in {ids}")
        self.l = vehicle_length_m
        self.dt = dt_s
        self.observations: list[Observation] = []
        self._acc: dict[str, _Acc] = {}
        self._lanes: dict[str, list[int]] = {}       # lanes each loop covers
        for s in self.specs:
            s.validate(n_lanes)
            steps = s.interval_s / dt_s
            if abs(steps - round(steps)) > 1e-6:
                raise ValueError(f"detector {s.sensor_id}: interval_s {s.interval_s} is not a multiple of dt {dt_s}")
            lanes = [] if s.kind == "arrivals" else \
                (list(range(n_lanes)) if s.lane_id is None else [s.lane_id])
            self._lanes[s.sensor_id] = lanes
            self._acc[s.sensor_id] = _Acc(max(1, len(lanes)))
        self._by_lane: dict[int, list[DetectorSpec]] = {}
        for s in self.specs:
            for ln in self._lanes[s.sensor_id]:
                self._by_lane.setdefault(ln, []).append(s)

    # -- simulator hooks
    def arrival(self, boundary_lane: int) -> None:
        """One vehicle joined the queue at boundary MAIN (0) or RAMP (-1)."""
        for s in self.specs:
            if s.kind == "arrivals" and BOUNDARIES[s.boundary] == boundary_lane:
                self._acc[s.sensor_id].count += 1

    def advance(self, t: float) -> None:
        """Close every interval that ends at or before t."""
        for s in self.specs:
            a = self._acc[s.sensor_id]
            while (a.k + 1) * s.interval_s <= t + 1e-9:
                self._emit(s, a, (a.k + 1) * s.interval_s, "ok")
                a.k += 1
                a.reset()

    def sample(self, lane_states: dict[int, np.ndarray]) -> None:
        """Occupancy sample for the step starting now. lane_states: lane_id -> front positions."""
        for s in self.specs:
            if s.kind != "loop":
                continue
            a = self._acc[s.sensor_id]
            a.n_steps += 1
            lo, hi = s.position_m, s.position_m + s.field_length_m
            for j, ln in enumerate(self._lanes[s.sensor_id]):
                x = lane_states.get(ln)
                if x is not None and len(x) and np.any((x >= lo) & (x - self.l <= hi)):
                    a.occ_steps[j] += 1

    def crossings(self, lane_id: int, x_old: np.ndarray, x_new: np.ndarray,
                  v_old: np.ndarray, v_new: np.ndarray) -> None:
        """Front bumpers that passed a loop on `lane_id` while moving from x_old to x_new this step."""
        for s in self._by_lane.get(lane_id, ()):
            hit = (x_old < s.position_m) & (x_new >= s.position_m)
            if not np.any(hit):
                continue
            frac = (s.position_m - x_old[hit]) / (x_new[hit] - x_old[hit])
            v_c = np.maximum(v_old[hit] + frac * (v_new[hit] - v_old[hit]), V_FLOOR_MPS)
            a = self._acc[s.sensor_id]
            a.count += int(hit.sum())
            a.inv_speed_sum += float(np.sum(1.0 / v_c))

    def finish(self, t_end: float) -> None:
        """Close all completed intervals and the final partial one (if it holds any time)."""
        self.advance(t_end)
        for s in self.specs:
            a = self._acc[s.sensor_id]
            if t_end > a.k * s.interval_s + 1e-9 or a.count:
                self._emit(s, a, t_end, "partial")
                a.reset()

    # -- reading
    def _emit(self, s: DetectorSpec, a: _Acc, t_end: float, flag: str) -> None:
        loop = s.kind == "loop"
        speed = a.count / a.inv_speed_sum if loop and a.count > 0 else None
        occ = float(a.occ_steps.mean() / a.n_steps) if loop and a.n_steps > 0 else None
        self.observations.append(Observation(
            sensor_id=s.sensor_id, interval_start_s=a.k * s.interval_s, interval_end_s=float(t_end),
            count=a.count, mean_speed_mps=speed, quality_flag=flag, occupancy=occ))

    def latest(self, sensor_id: str, n: int = 1) -> list[Observation]:
        """The last n finished intervals of one detector, oldest first (empty if none yet)."""
        out = [o for o in reversed(self.observations) if o.sensor_id == sensor_id][:n]
        return out[::-1]

    def to_frame(self) -> pd.DataFrame:
        cols = list(Observation.__dataclass_fields__)
        if not self.observations:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame([asdict(o) for o in self.observations], columns=cols)
        df["mean_speed_mps"] = df["mean_speed_mps"].astype(float)   # None -> NaN (explicitly missing)
        df["occupancy"] = df["occupancy"].astype(float)
        return df


def flow_vph(obs: Observation) -> float:
    """Count converted to a flow rate over the observation's own interval length."""
    return obs.count * 3600.0 / (obs.interval_end_s - obs.interval_start_s)


def main(argv: list[str]) -> None:
    from pathlib import Path

    from tmc.network import run_scenario
    path = Path(argv[0]) if argv else Path(__file__).resolve().parents[2] / "configs" / "scenario_s1_onramp.yaml"
    res = run_scenario(path)
    df = res.observations.copy()
    if df.empty:
        raise SystemExit(f"{path} defines no detectors")
    df["flow_vph"] = df["count"] * 3600.0 / (df["interval_end_s"] - df["interval_start_s"])
    df["speed_mph"] = df["mean_speed_mps"] * 2.23694
    pd.set_option("display.width", 160)
    pv = df.pivot_table(index="interval_start_s", columns="sensor_id", values="flow_vph")
    print(f"{res.config.scenario_id}: flow (veh/h) per detector, every 10th interval")
    print(pv.iloc[::10].round(0).to_string())
    print("\ncount totals:", df.groupby("sensor_id")["count"].sum().to_dict())


if __name__ == "__main__":
    main(sys.argv[1:])
