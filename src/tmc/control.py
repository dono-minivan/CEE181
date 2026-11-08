"""Ramp-meter controllers: no signal, pre-timed, actuated (ALINEA feedback).

    python -m tmc.control [scenario.yaml] [mode] [param]   # small example, prints the decision trace
      e.g.  python -m tmc.control configs/scenario_s3_onramp_zipgap.yaml alinea 0.12

Every controller sees ONLY a DetectorView (finished detector intervals; CLAUDE.md rule 2) and runs every
interval_s seconds, right after a detector interval closes. Each update:
  1. estimate: ramp queue N by vehicle counting (R_dem in, R_pass out); ramp demand lambda by
     exponential smoothing of the R_dem flow; occupancy o at the control detector.
  2. forecast: queue N_fc = N + (lambda - r) h over the horizon h (r = the rate about to be applied).
  3. control law -> rate r_law (veh/h), clipped to [r_min, r_max].
  4. queue override: if N or N_fc >= n_max the meter goes dark (no signal) to flush the queue, so the
     ramp queue stays within its storage.
  5. fallback: if a needed measurement is missing, hold the last command (ControlCommand.fallback).
  6. a decision-trace row records every input, intermediate, and the final command.

Laws:
  * none       -- meter always dark (the S3 no-control merge).
  * pretimed   -- fixed rate r (best rate chosen by a sweep in tmc.evaluate, CLAUDE.md rule 4).
  * alinea     -- r(k) = r(k-1) + K_R (o_set - o(k)), Papageorgiou et al. 1991; K_R = 70 veh/h per
                  percent occupancy. Measured just UPSTREAM of the merge (detector M_mrg), because in
                  this model congestion forms upstream of the merge, so a downstream detector (classic
                  placement) would see occupancy fall at breakdown and raise the rate. If r_law reaches
                  r_max the meter goes dark (meters off when not needed).
A metered vehicle is released from a stop at the stop line, one vehicle per green (tmc.network).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field

import pandas as pd

from tmc.estimation import DetectorView, RampQueueEstimator, flow_vph
from tmc.forecast import ExpSmoothing, queue_forecast
from tmc.interfaces import ControlCommand

ACTUATOR = "meter_R"


@dataclass
class MeterLimits:
    r_min_vph: float = 240.0      # 15 s per vehicle
    r_max_vph: float = 900.0      # 4 s per vehicle, one-car-per-green single lane
    n_max_veh: float = 35.0       # ramp storage upstream of the stop line (~250 m at ~7 m/veh)
    horizon_s: float = 60.0       # queue forecast horizon


@dataclass
class Controller:
    mode: str = "none"
    rate_vph: float | None = None           # pretimed rate
    o_set: float | None = None              # alinea setpoint (fraction, 0..1)
    k_r: float = 70.0                       # alinea gain, veh/h per percent occupancy
    occ_sensor: str = "M_mrg"
    interval_s: float = 30.0
    limits: MeterLimits = field(default_factory=MeterLimits)
    alpha: float = 0.3                      # ramp-demand smoothing
    trace: list = field(default_factory=list)

    def __post_init__(self):
        if self.mode not in ("none", "pretimed", "alinea"):
            raise ValueError(f"unknown control mode {self.mode!r}")
        if self.mode == "pretimed" and self.rate_vph is None:
            raise ValueError("pretimed needs rate_vph")
        if self.mode == "alinea" and self.o_set is None:
            raise ValueError("alinea needs o_set")
        self._queue = RampQueueEstimator()
        self._lam = ExpSmoothing(self.alpha)
        self._r_law = self.limits.r_max_vph       # ALINEA state r(k-1), starts open
        self._last: ControlCommand | None = None

    @property
    def label(self) -> str:
        if self.mode == "pretimed":
            return f"pretimed_{self.rate_vph:.0f}"
        if self.mode == "alinea":
            return f"alinea_{self.o_set:.3f}"
        return "none"

    def _command(self, t: float, rate: float | None, fallback: str) -> ControlCommand:
        lim = self.limits
        return ControlCommand(issue_time_s=t, apply_time_s=t, actuator_id=ACTUATOR,
                              action="dark" if rate is None else "set_rate",
                              value=float("nan") if rate is None else rate, units="veh/h",
                              min_value=lim.r_min_vph, max_value=lim.r_max_vph, fallback=fallback)

    def update(self, t: float, view: DetectorView) -> ControlCommand:
        lim = self.limits
        n = self._queue.update(view)
        dem = view.latest("R_dem")
        lam_meas = flow_vph(dem[0]) if dem else None
        lam = self._lam.update(lam_meas)
        occ_obs = view.latest(self.occ_sensor) if self.mode == "alinea" else []
        occ = occ_obs[0].occupancy if occ_obs else None

        missing = []
        if self.mode == "none":
            r_law = None
        elif self.mode == "pretimed":
            r_law = float(self.rate_vph)
        else:
            if occ is None:
                missing.append(self.occ_sensor)
                r_law = self._r_law
            else:
                self._r_law = min(max(self._r_law + self.k_r * 100.0 * (self.o_set - occ),
                                      lim.r_min_vph), lim.r_max_vph)
                r_law = self._r_law
        if r_law is not None:
            r_law = min(max(r_law, lim.r_min_vph), lim.r_max_vph)

        rate, reason = r_law, self.mode
        if self.mode == "alinea" and r_law is not None and r_law >= lim.r_max_vph:
            rate, reason = None, "alinea_at_max_dark"
        n_fc = queue_forecast(n, lam, rate if rate is not None else lim.r_max_vph, lim.horizon_s)
        if self.mode != "none" and n is None:
            missing.append("ramp_queue")
        override = self.mode != "none" and ((n is not None and n >= lim.n_max_veh) or
                                            (n_fc is not None and n_fc >= lim.n_max_veh))
        if override:
            rate, reason = None, "queue_override_dark"
        if missing and self._last is not None and not override:
            cmd = self._last                                  # hold last command
            reason = "missing:" + ",".join(missing) + " -> hold_last"
        else:
            cmd = self._command(t, rate, "hold_last")
        self._last = cmd
        self.trace.append({
            "time_s": t, "mode": self.mode, "occ_sensor": self.occ_sensor, "occ": occ,
            "ramp_demand_meas_vph": lam_meas, "ramp_demand_smoothed_vph": lam,
            "queue_est_veh": n, "queue_fc_veh": n_fc, "queue_fc_target_s": t + lim.horizon_s,
            "r_law_vph": r_law, "override": override, "action": cmd.action,
            "rate_cmd_vph": None if cmd.action == "dark" else cmd.value, "reason": reason,
        })
        return cmd

    def trace_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.trace)


def make_controller(spec: dict | None) -> Controller | None:
    """Build from a YAML `control:` block, e.g. {mode: alinea, o_set: 0.12}. None -> no controller."""
    if not spec:
        return None
    spec = dict(spec)
    lim = MeterLimits(**spec.pop("limits", {}))
    return Controller(limits=lim, **spec)


def main(argv: list[str]) -> None:
    from pathlib import Path

    from tmc.idm import load_params
    from tmc.network import FreewaySegment, ScenarioConfig
    path = Path(argv[0]) if argv else Path(__file__).resolve().parents[2] / "configs" / "scenario_s3_onramp_zipgap.yaml"
    mode = argv[1] if len(argv) > 1 else "alinea"
    cfg = ScenarioConfig.from_yaml(path)
    if mode == "alinea":
        ctrl = Controller("alinea", o_set=float(argv[2]) if len(argv) > 2 else 0.12)
    elif mode == "pretimed":
        ctrl = Controller("pretimed", rate_vph=float(argv[2]) if len(argv) > 2 else 400.0)
    else:
        ctrl = Controller("none")
    res = FreewaySegment(load_params(cfg.param_set), cfg, controller=ctrl).run()
    pd.set_option("display.width", 200)
    tr = ctrl.trace_frame()
    print(tr.iloc[::4].round(3).to_string(index=False))
    s = res.summary()
    print({k: s[k] for k in ("total_travel_time_veh_h", "total_ramp_wait_veh_h", "peak_ramp_queue_veh", "cleared")})


if __name__ == "__main__":
    main(sys.argv[1:])
