"""Animated viewer: python -m tmc.viz [scenario.yaml] [--open]

Runs a scenario and writes results/<scenario_id>_viewer.html, a self-contained page (no server, no
internet) that animates the mainline lanes, lane changes, on-ramp, acceleration lane, and queues with
vehicles colored by speed.

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

from tmc.idm import capacity
from tmc.network import SimResult, run_scenario

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).with_name("viewer_template.html")
DEFAULT_SCENARIO = ROOT / "configs" / "scenario_s1_onramp.yaml"


def _r(x, nd=1):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def export(res: SimResult) -> dict:
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
    }


def write_viewer(res: SimResult, out_path: Path) -> Path:
    data = json.dumps(export(res), separators=(",", ":"))
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", data).replace("__TITLE__", f"{res.config.scenario_id} viewer")
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main(argv: list[str]) -> None:
    args = [a for a in argv if not a.startswith("--")]
    scenario = Path(args[0]) if args else DEFAULT_SCENARIO
    res = run_scenario(scenario)
    out = ROOT / "results" / f"{res.config.scenario_id}_viewer.html"
    out.parent.mkdir(exist_ok=True)
    write_viewer(res, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if "--open" in argv:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main(sys.argv[1:])
