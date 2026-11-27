"""Animated viewer:
    python -m tmc.viz [scenario.yaml] [--control MODE[:PARAM]] [--open]
    python -m tmc.viz --e7 X_PEAK [--seed N] [--control MODE[:PARAM]] [--open]
    python -m tmc.viz --e9b X_PEAK [--mode human|acc|cacc|cacc_eff] [--p P] [--seed N] [--open]
    python -m tmc.viz --e9a nominal|strong [--mode human|acc|cacc] [--p P] [--seed N] [--open]
    python -m tmc.viz --examples          # regenerate results/viewer.html (all example runs)

  e.g. python -m tmc.viz --e7 1.35 --control alinea:0.20 --open
       python -m tmc.viz --e7 1.10 --control pretimed:800 --open
       python -m tmc.viz configs/scenario_s3_onramp_zipgap.yaml --control none --open

Runs a scenario and writes results/<scenario_id>[_<controller>]_viewer.html, a self-contained page (no
server, no internet) that animates the mainline lanes, lane changes, on-ramp, acceleration lane, and queues
with vehicles colored by speed. With a ramp controller it also shows the meter signal (dark / red, green
flash at each release), the detectors, the controller's decision trace at the current time, and a timeline
of the ramp queue and meter rate. --e7 builds the E7 demand profile (tmc.evaluate) at the given X_peak.
--e9b builds an E9 setting B run (tmc.setting_b: S3r2 merge, humans with T_r = 0.6 s restarted at leader
changes, dt 0.05 s, C_m from E9b2); mode cacc_eff = CACC broadcasting its effective command. --e9a builds an
E9 setting A platoon (tmc.disturbance) and opens following the scripted lead. Equipped vehicles carry a
roof marker (blue; orange while a CACC vehicle is in fallback), and the selected-vehicle panel shows the
driver type and where the leader-acceleration input comes from at that moment.

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


def fallback_runs(res: SimResult, tick_s: float = 0.1) -> dict[int, list]:
    """CACC fallback log compressed per vehicle into [start_s, end_s, reason] runs of consecutive ticks."""
    ev = res.cav_events
    out: dict[int, list] = {}
    if ev is None or ev.empty:
        return out
    for vid, g in ev.sort_values(["vehicle_id", "time_s"]).groupby("vehicle_id"):
        runs = []
        for ts, reason in zip(g["time_s"].to_numpy(), g["reason"].to_numpy()):
            if runs and runs[-1][2] == reason and ts - runs[-1][1] <= tick_s * 1.5:
                runs[-1][1] = ts
            else:
                runs.append([ts, ts, reason])
        out[int(vid)] = [[round(a, 1), round(b + tick_s, 1), r] for a, b, r in runs]
    return out


def driver_meta(cfg) -> dict:
    c = cfg.cav or {}
    p_eq = float(c.get("p_equipped", 0.0)) if c else 0.0
    mode = str(c.get("mode", "acc")) if p_eq > 0 else None
    return {
        "T_r": cfg.reaction_time_s, "leader_change": cfg.leader_change_delay if cfg.reaction_time_s > 0 else None,
        "cav": None if not p_eq else {
            "p": p_eq, "mode": mode, "tau_s": float(c.get("tau_s", 0.1)),
            "delay_s": float(c.get("delay_s", 0.15)) if mode == "cacc" else None,
            "broadcast": str(c.get("broadcast", "command")) if mode == "cacc" else None,
            "controller": str(c.get("controller", "eq1126")),
        },
    }


def export(res: SimResult, control_label: str | None = None, follow: int | None = None) -> dict:
    cfg, p = res.config, res.params
    dt_f = cfg.record_every_s
    tr = res.trajectories.sort_values(["vehicle_id", "time_s"])
    info = res.vehicles.set_index("vehicle_id")
    fb = fallback_runs(res)
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
            1 if bool(row.get("equipped", False)) else 0, fb.get(int(vid), []),
        ])
    c = res.counts
    ramp = cfg.ramp
    # draw length: the configured segment, or the recorded extent for an open road (setting A, 1e7 m)
    L = cfg.length_m if cfg.length_m < 1e5 else float(np.ceil((tr["position_m"].max() + 60) / 100) * 100)
    summ = {k: (_r(v, 2) if isinstance(v, float) else v) for k, v in res.summary().items()}
    summ["n_equipped"] = int(res.vehicles["equipped"].sum()) if "equipped" in res.vehicles else 0
    summ["fallback_ticks"] = int(len(res.cav_events)) if res.cav_events is not None else 0
    gs = res.gap_stats or {}
    for k in ("min_gap_m", "min_time_gap_s", "max_decel_mps2"):
        for typ in ("eq", "hh", "he"):
            v = gs.get(f"{k}_{typ}")
            summ[f"{k}_{typ}"] = _r(v, 2) if v is not None and np.isfinite(v) and (k != "max_decel_mps2" or v > 0) else None
    return {
        "meta": {
            "scenario_id": cfg.scenario_id, "demand_id": cfg.demand_id, "param_set": cfg.param_set,
            "seed": cfg.seed, "dt_s": cfg.dt_s, "frame_dt": dt_f, "n_frames": int(len(c)),
            "L": L, "n_lanes": cfg.n_lanes, "drivers": driver_meta(cfg), "follow": follow,
            "open_road": bool(cfg.length_m >= 1e5), "veh_len": p.length, "s0": p.s0, "v0": p.v0,
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
        "summary": summ,
        "control": export_control(res, control_label),
        "detectors": [{"id": d.sensor_id, "x": d.position_m, "lane": d.lane_id}
                      for d in cfg.detectors if d.kind == "loop"],
    }


def write_viewer(res: SimResult, out_path: Path, control_label: str | None = None,
                 follow: int | None = None) -> Path:
    data = json.dumps(export(res, control_label, follow), separators=(",", ":"))
    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", data).replace("__TITLE__", f"{res.config.scenario_id} viewer")
    out_path.write_text(html, encoding="utf-8")
    return out_path


def write_combined(entries: list[tuple[str, str, str, dict]], out_path: Path) -> Path:
    """One page holding several runs, switched with a dropdown. entries: (key, group, label, export dict).

    Each run is an inert <script type="application/json"> block; the page decodes only the selected one
    and reloads on a switch, carrying time, view and selected vehicle in the URL hash."""
    keys = [k for k, *_ in entries]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate run keys")
    blocks = "\n".join(
        f'<script type="application/json" id="run-{k}">'
        + json.dumps(d, separators=(",", ":")).replace("</", r"<\/") + "</script>"
        for k, _, _, d in entries)
    runs = json.dumps([{"key": k, "group": g, "label": lab} for k, g, lab, _ in entries], separators=(",", ":"))
    html = TEMPLATE.read_text(encoding="utf-8")
    html = (html.replace("<!--__RUNDATA__-->", blocks).replace("/*__RUNS__*/null", runs)
            .replace("/*__DATA__*/null", 'JSON.parse(document.getElementById("run-" + RUN.key).textContent)')
            .replace("__TITLE__", "CEE 181 viewers"))
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


def run_e9b(x_peak: float, mode: str = "human", p_eq: float = 0.0, seed: int = 181) -> tuple[SimResult, str]:
    """One E9 setting B run as in tmc.setting_b (dt 0.05 s, C_m from E9b2), recorded every 1 s."""
    from tmc import setting_b as sb
    if mode == "human":
        p_eq = 0.0
    elif p_eq <= 0:
        raise SystemExit("--p must be > 0 for an equipped mode")
    bc = "effective" if mode == "cacc_eff" else "command"
    cfg = sb.build_config(x_peak, seed, sb.merge_capacity_s3r(), "cacc" if mode == "cacc_eff" else mode,
                          p_eq, broadcast=bc)
    cfg.record_every_s = 1.0
    cfg.scenario_id = f"E9B_Xp{x_peak:.2f}_{mode}" + (f"_p{p_eq:.2f}" if mode != "human" else "")
    res = FreewaySegment(load_params(cfg.param_set), cfg).run()
    return res, f"{cfg.scenario_id}_s{seed}"


def run_e9a(dist_name: str = "nominal", mode: str = "human", p_eq: float = 0.0, seed: int = 181) -> tuple[SimResult, str]:
    """One E9 setting A platoon run as in tmc.disturbance (T_r = 0.6 s humans), recorded every 0.5 s."""
    from tmc.disturbance import ScriptedLead, load_setting, platoon_config
    s = load_setting()
    dist = s["disturbances"][dist_name]
    p = load_params(s["param_set"])
    cav = None if mode == "human" else {"p_equipped": p_eq, "mode": mode, "placement_seed": seed,
                                        "tau_s": 0.1, "delay_s": 0.15}
    cfg = platoon_config(s, p, 0.6, None, cav)
    cfg.record_every_s = 0.5
    cfg.scenario_id = f"E9A_{dist_name}_{mode}" + (f"_p{p_eq:.2f}" if cav else "")
    cfg.demand_id = f"platoon of {int(s['n_followers'])} at {s['v_eq_mps']} m/s, lead brakes to {dist.v_low} m/s"
    cfg.seed = seed if cav else None
    res = FreewaySegment(p, cfg, lead_accel=ScriptedLead(dist, float(s["v_eq_mps"]), p.a, cfg.dt_s)).run()
    return res, cfg.scenario_id + (f"_s{seed}" if cav else "")


# Runs in the committed combined viewer results/viewer.html (regenerate with --examples), in dropdown
# order: (group, label, viz argv).
EXAMPLES = [
    ("Basic scenarios", "S0 single lane", ["configs/scenario_s0_single_lane.yaml"]),
    ("Basic scenarios", "S1 on-ramp, no control", ["configs/scenario_s1_onramp.yaml"]),
    ("Basic scenarios", "S2 two lanes + on-ramp, no control", ["configs/scenario_s2_twolane_onramp.yaml"]),
    ("E7 ramp metering (seed 181)", "X 1.35 · no control", ["--e7", "1.35", "--control", "none"]),
    ("E7 ramp metering (seed 181)", "X 1.35 · ALINEA o_set 0.20", ["--e7", "1.35", "--control", "alinea:0.20"]),
    ("E7 ramp metering (seed 181)", "X 1.10 · pre-timed 800 veh/h", ["--e7", "1.10", "--control", "pretimed:800"]),
    ("E9 setting B (S3r2 merge, X 1.35, seed 181)", "human", ["--e9b", "1.35", "--mode", "human"]),
    ("E9 setting B (S3r2 merge, X 1.35, seed 181)", "ACC, p = 1.00", ["--e9b", "1.35", "--mode", "acc", "--p", "1.0"]),
    ("E9 setting B (S3r2 merge, X 1.35, seed 181)", "CACC (effective broadcast), p = 0.50",
     ["--e9b", "1.35", "--mode", "cacc_eff", "--p", "0.5"]),
    ("E9 setting A (platoon, nominal braking)", "human", ["--e9a", "nominal", "--mode", "human"]),
    ("E9 setting A (platoon, nominal braking)", "CACC, p = 0.50", ["--e9a", "nominal", "--mode", "cacc", "--p", "0.5"]),
]


def examples() -> None:
    """Run every example and write the combined results/viewer.html holding all of them."""
    entries = []
    for group, label, argv in EXAMPLES:
        res, out, ctrl_label, follow = run_from_argv(argv)
        print(f"ran {out.name.removesuffix('_viewer.html')}")
        entries.append((out.name.removesuffix("_viewer.html"), group, label, export(res, ctrl_label, follow)))
    out = write_combined(entries, ROOT / "results" / "viewer.html")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(entries)} runs)")


def run_from_argv(argv: list[str]) -> tuple[SimResult, Path, str | None, int | None]:
    """Run what a viz command line asks for: (result, viewer path, control label, vehicle to follow)."""
    vals = {_opt(argv, n) for n in ("--control", "--e7", "--seed", "--e9b", "--e9a", "--mode", "--p")}
    args = [a for a in argv if not a.startswith("--") and a not in vals]
    if _opt(argv, "--e9b") is not None or _opt(argv, "--e9a") is not None:
        mode = _opt(argv, "--mode") or "human"
        p_eq = float(_opt(argv, "--p") or (0.0 if mode == "human" else 1.0))
        seed = int(_opt(argv, "--seed") or 181)
        if _opt(argv, "--e9b") is not None:
            res, name = run_e9b(float(_opt(argv, "--e9b")), mode, p_eq, seed)
            follow = None
        else:
            res, name = run_e9a(_opt(argv, "--e9a"), mode, p_eq, seed)
            follow = 0
        return res, ROOT / "results" / f"{name}_viewer.html", None, follow
    ctrl = parse_control(_opt(argv, "--control"))
    if _opt(argv, "--e7") is not None:
        from tmc.evaluate import build_config, merge_capacity
        cfg = build_config(float(_opt(argv, "--e7")), int(_opt(argv, "--seed") or 181), merge_capacity())
        cfg.record_every_s = 1.0
    else:
        path = Path(args[0]) if args else DEFAULT_SCENARIO
        cfg = ScenarioConfig.from_yaml(path if path.exists() else ROOT / path)   # repo-relative paths work too
        if _opt(argv, "--seed") is not None:
            cfg.seed = int(_opt(argv, "--seed"))
    if ctrl is not None:
        cfg.control = None
    res = FreewaySegment(load_params(cfg.param_set), cfg, controller=ctrl).run()
    label = ctrl.label if ctrl is not None else (cfg.control or {}).get("mode")
    out = ROOT / "results" / (f"{res.config.scenario_id}_s{cfg.seed}_{ctrl.label}_viewer.html" if ctrl is not None
                              else f"{res.config.scenario_id}_viewer.html")
    return res, out, label, None


def main(argv: list[str]) -> None:
    if "--examples" in argv:
        examples()
        return
    res, out, label, follow = run_from_argv(argv)
    out.parent.mkdir(exist_ok=True)
    write_viewer(res, out, label, follow)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if "--open" in argv:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main(sys.argv[1:])
