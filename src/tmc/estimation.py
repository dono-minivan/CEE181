"""Estimation from detector data only (CLAUDE.md rule 2).

    python -m tmc.estimation [scenario.yaml]     # small example: estimates every 30 s for a scenario

DetectorView is the ONLY handle a controller gets on traffic: it exposes finished detector intervals
(interfaces.Observation) and the detector layout, never the simulator state.

Estimates (traced in docs/source_to_design.md):
  * flow q = count / interval (veh/h).
  * density from occupancy, k = occ / (l + L_loop) per lane (veh/m), using the nominal vehicle length.
  * ramp queue by vehicle counting: N = cumulative(R_dem) - cumulative(R_pass), clipped at 0. R_dem
    counts vehicles joining the ramp queue, R_pass the loop just past the stop line, so N is the number
    waiting upstream of the stop line (exact at interval ends, up to vehicles in the 10 m between the stop
    line and the loop).
  Missing inputs give None (never 0): a missing interval makes the queue estimate None until the next one.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from tmc.interfaces import Observation


class DetectorView:
    """Read-only window on a DetectorBank: finished intervals and the layout, nothing else."""

    def __init__(self, bank):
        self._bank = bank
        self.specs = {s.sensor_id: s for s in bank.specs}

    def latest(self, sensor_id: str, n: int = 1) -> list[Observation]:
        if sensor_id not in self.specs:
            raise KeyError(f"no detector {sensor_id!r} in this scenario")
        return self._bank.latest(sensor_id, n)


def flow_vph(obs: Observation | None) -> float | None:
    if obs is None:
        return None
    return obs.count * 3600.0 / (obs.interval_end_s - obs.interval_start_s)


def density_vpm(obs: Observation | None, vehicle_length_m: float, loop_length_m: float) -> float | None:
    """Per-lane density (veh/m) from occupancy: k = occ / (l + L_loop)."""
    if obs is None or obs.occupancy is None:
        return None
    return obs.occupancy / (vehicle_length_m + loop_length_m)


@dataclass
class RampQueueEstimator:
    """Count-in / count-out queue estimate. Feed it each new pair of intervals once."""
    in_id: str = "R_dem"
    out_id: str = "R_pass"
    cum_in: int = 0
    cum_out: int = 0
    last_end_s: float = -1.0
    valid: bool = True

    def update(self, view: DetectorView) -> float | None:
        a = view.latest(self.in_id)
        b = view.latest(self.out_id)
        if not a or not b:
            return None
        a, b = a[0], b[0]
        if a.interval_end_s != b.interval_end_s:      # out of step: do not guess
            self.valid = False
            return None
        if a.interval_end_s > self.last_end_s:
            if a.quality_flag not in ("ok", "partial") or b.quality_flag not in ("ok", "partial"):
                self.valid = False
                return None
            self.cum_in += a.count
            self.cum_out += b.count
            self.last_end_s = a.interval_end_s
        return float(max(0, self.cum_in - self.cum_out)) if self.valid else None


def main(argv: list[str]) -> None:
    from pathlib import Path

    from tmc.network import run_scenario
    path = Path(argv[0]) if argv else Path(__file__).resolve().parents[2] / "configs" / "scenario_s3_onramp_zipgap.yaml"
    res = run_scenario(path)
    obs = res.observations
    # replay the observations through the estimator, then compare with simulator truth (evaluation only)
    est = RampQueueEstimator()

    class _Replay:
        specs = []

        def __init__(self, upto):
            self.upto = upto

        def latest(self, sid, n=1):
            o = obs[(obs["sensor_id"] == sid) & (obs["interval_end_s"] <= self.upto + 1e-9)]
            if o.empty:
                return []
            r = o.iloc[-1]
            return [Observation(r.sensor_id, r.interval_start_s, r.interval_end_s, int(r["count"]),
                                None, r.quality_flag, None)]
    truth = res.counts.set_index("time_s")["n_queued_ramp"]
    print("t_s  queue_est  queue_true")
    for i, t in enumerate(sorted(obs["interval_end_s"].unique())):
        q = est.update(_Replay(t))            # every interval must be fed; print every 4th
        if i % 4 == 0:
            tt = truth.index[truth.index <= t].max()
            print(f"{t:6.0f} {q!s:>9} {int(truth.loc[tt]):>10}")


if __name__ == "__main__":
    main(sys.argv[1:])
