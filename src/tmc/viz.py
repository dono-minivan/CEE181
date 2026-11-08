"""Animated viewer:
    python -m tmc.viz [scenario.yaml] [--control MODE[:PARAM]] [--open]
    python -m tmc.viz --e7 X_PEAK [--seed N] [--control MODE[:PARAM]] [--open]

  e.g. python -m tmc.viz --e7 1.35 --control alinea:0.20 --open
       python -m tmc.viz --e7 1.10 --control pretimed:800 --open
       python -m tmc.viz configs/scenario_s3_onramp_zipgap.yaml --control none --open

Runs a scenario and writes results/<scenario_id>[_<controller>]_viewer.html, a self-contained page (no
server, no internet) that animates the mainline lanes, lane changes, on-ramp, acceleration lane, and queues
with vehicles colored by speed. With a ramp controller it also shows the meter signal (dark / red, green
flash at each release), the detectors, the controller's decision trace at the current time, and a timeline
of the ramp queue and meter rate. --e7 builds the E7 demand profile (tmc.evaluate) at the given X_peak.

Display-only conventions (not model behavior):
  * Vehicles in the point queues (mainline entrance, ramp entrance) are drawn as grey cars stacked
    upstream of the entry point at s0 + l spacing. In the model they occupy no road space.
  * Positions are recorded every record_every_s and linearly interpolated between records for drawing.
    A lane change (or merge) is drawn as a sideways move spread over the record interval in which it
    happened; in the model it is instantaneous.
"""
from __future__ import annotations

import json
import sys
import webbrowser
from pathlib import Path

import numpy as np

from tmc.idm import capacity, load_params
from tmc.network import FreewaySegment, ScenarioConfig, SimResult

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).with_name("viewer_template.html")
DEFAULT_SCENARIO = ROOT / "configs" / "scenario_s1_onramp.yaml"


def _r(x, nd=1):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


TRACE_KEYS = ["time_s", "occ", "ramp_demand_meas_vph", "ramp_demand_smoothed_vph", "queue_est_veh",
              "queue_fc_veh", "r_law_vph", "override", "action", "rate_cmd_vph", "reason"]


def _clean(x):
    if x is None or isinstance(x, (bool, np.bool_, str)):
        return bool(x) if isinstance(x, np.bool_) else x
    x = float(x)
    return None if not np.isfinite(x) else round(x, 4)


def export_control(res: SimResult, label: str | None) -> dict | None:
    tr = res.control_trace
    if tr is None or tr.empty:
        return None
    rows = [[_clean(r[k]) for k in TRACE_KEYS] for r in tr.to_dict("records")]
    return {"label": label or str(tr["mode"].iloc[0]), "mode": str(tr["mode"].iloc[0]),
            "occ_sensor": str(tr["occ_sensor"].iloc[0]), "keys": TRACE_KEYS, "rows": rows}


def export(res: SimResult, control_label: str | None = None) -> dict:
    cfg, p = res.config, res.params
    dt_f = cfg.record_every_s
    tr = res.trajectories.sort_values(["vehicle_id", "time_s"])
    info = res.vehicles.set_index("vehicle_id")
    veh = []
    for vid, g in tr.groupby("vehicle_id", sort=False):
        frames = np.rint(g["time_s"].to_numpy() / dt_f).astype(int)
        if len(frames) > 1 and np.any(np.diff(frames) != 1):
            raise ValueError(f"vehicle {vid}: non-consecutive records")
        lanes = g["lane_id"].to_numpy()
        # lane history as runs [[first_frame_rel, lane_id], ...]; -1 = ramp / acceleration lane
        starts = np.concatenate([[0], np.flatnonzero(np.diff(lanes) != 0) + 1])
        lane_runs = [[int(i), int(lanes[i])] for i in starts]
        row = info.loc[vid]
        veh.append([
            int(vid), 1 if row["origin"] == "ramp" else 0, int(frames[0]), lane_runs,
            np.rint(g["position_m"].to_numpy() * 10).astype(int).tolist(),   # decimeters
            np.rint(g["speed_mps"].to_numpy() * 10).astype(int).tolist(),    # dm/s
            [_r(row["arrival_s"]), _r(row["entry_s"]), _r(row["merge_s"]), _r(row["exit_s"])],
        ])
    c = res.counts
    ramp = cfg.ramp
    return {
        "meta": {
            "scenario_id": cfg.scenario_id, "demand_id": cfg.demand_id, "param_set": cfg.param_set,
            "seed": cfg.seed, "dt_s": cfg.dt_s, "frame_dt": dt_f, "n_frames": int(len(c)),
            "L": cfg.length_m, "n_lanes": cfg.n_lanes, "veh_len": p.length, "s0": p.s0, "v0": p.v0,
            "q_c_vph": round(capacity(p)["q_c_veh_per_h"]),
            "ramp": None if ramp is None else {
                "merge_start": ramp.merge_start_m, "merge_end": ramp.merge_end_m,
                "entrance": ramp.entrance_m, "approach": ramp.approach_length_m,
            },
        },
        "veh": veh,
        "counts": {
            "qm": c["n_queued_main"].astype(int).tolist(), "qr": c["n_queued_ramp"].astype(int).tolist(),
            "onm": c["n_on_main"].astype(int).tolist(), "onr": c["n_on_ramp"].astype(int).tolist(),
            "exited": c["n_exited"].astype(int).tolist(), "arrived": c["n_arrived"].astype(int).tolist(),
        },
        "summary": {k: (_r(v, 2) if isinstance(v, float) else v) for k, v in res.summary().items()},
        "control": export_control(res, control_label),
        "detectors": [{"id": d.sensor_id, "x": d.position_m, "lane": d.lane_id}
                      for d in cfg.detectors if d.kind == "loop"],
    }


def write_viewer(res: SimResult, out_path: Path, control_label: str | None = None) -> Path:
    data = json.dumps(export(res, control_label), separators=(",", ":"))
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", data).replace("__TITLE__", f"{res.config.scenario_id} viewer")
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _opt(argv: list[str], name: str):
    return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None


def parse_control(spec: str | None):
    """'none' | 'pretimed:RATE' | 'alinea:O_SET' -> Controller (None if spec is None)."""
    if spec is None:
        return None
    from tmc.control import Controller
    mode, _, param = spec.partition(":")
    if mode == "none":
        return Controller("none")
    if mode == "pretimed":
        return Controller("pretimed", rate_vph=float(param or 400))
    if mode == "alinea":
        return Controller("alinea", o_set=float(param or 0.20))
    raise SystemExit(f"unknown --control {spec!r}; use none, pretimed:RATE or alinea:O_SET")


def main(argv: list[str]) -> None:
    vals = {_opt(argv, n) for n in ("--control", "--e7", "--seed")}
    args = [a for a in argv if not a.startswith("--") and a not in vals]
    ctrl = parse_control(_opt(argv, "--control"))
    if _opt(argv, "--e7") is not None:
        from tmc.evaluate import build_config, merge_capacity
        cfg = build_config(float(_opt(argv, "--e7")), int(_opt(argv, "--seed") or 181), merge_capacity())
        cfg.record_every_s = 1.0
    else:
        cfg = ScenarioConfig.from_yaml(Path(args[0]) if args else DEFAULT_SCENARIO)
        if _opt(argv, "--seed") is not None:
            cfg.seed = int(_opt(argv, "--seed"))
    if ctrl is not None:
        cfg.control = None
    res = FreewaySegment(load_params(cfg.param_set), cfg, controller=ctrl).run()
    label = ctrl.label if ctrl is not None else (cfg.control or {}).get("mode")
    suffix = f"_{ctrl.label}" if ctrl is not None else ""
    out = ROOT / "results" / f"{res.config.scenario_id}_s{cfg.seed}{suffix}_viewer.html" if ctrl is not None \
        else ROOT / "results" / f"{res.config.scenario_id}_viewer.html"
    out.parent.mkdir(exist_ok=True)
    write_viewer(res, out, label)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if "--open" in argv:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main(sys.argv[1:])
