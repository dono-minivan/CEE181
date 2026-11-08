"""Control-mode comparison across demand levels (experiment log E7).

    python -m tmc.evaluate --quick          # small example: 2 demand levels, 1 seed, few settings (~3 min)
    python -m tmc.evaluate                  # full E7 sweep (~1 h on 6 cores), writes results/E7_*.csv/png
    python -m tmc.evaluate --from-csv       # re-summarize results/E7_runs.csv without re-simulating
    python -m tmc.evaluate --extend         # run only settings missing from results/E7_runs.csv, then summarize

Fairness (CLAUDE.md rule 4): every run uses the S3 geometry, merge model, detector layout and seeds; only
the peak demand level and the controller change. Demand (ramp share fixed at r = 0.28, where C_m is
well-defined, E6), as X_m = q_tot / C_m:
    0-600 s: 0.85    600-1500 s: X_peak    1500-2400 s: 0.60
Runs go until every vehicle has exited (unmet demand reported, never dropped).
Baselines: pre-timed uses the best rate per demand level from a sweep (by mean TTT over the seeds, an
optimistic "oracle" plan). ALINEA uses ONE setpoint for all levels (chosen by mean TTT over everything),
and its per-level best is reported too.

Metrics use simulator truth, which is allowed for evaluation (not for control):
  TTT (veh*h, incl. entry and ramp waiting), ramp wait, served flow 600-2100 s, peak ramp queue, time with
  ramp queue > n_max, min 30-s mean speed at M_up / M_mrg (mph), unmet demand, clear time.
Forecast skill (from the decision traces of metered runs): MAE of the 60 s queue forecast and of the
smoothed ramp-demand forecast, each against persistence.
"""
from __future__ import annotations

import copy
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from tmc.control import Controller
from tmc.idm import load_params
from tmc.network import FreewaySegment, ScenarioConfig

ROOT = Path(__file__).resolve().parents[2]
S3 = ROOT / "configs" / "scenario_s3_onramp_zipgap.yaml"
CAP_CSV = ROOT / "results" / "S3_onramp_zipgap_nocontrol_merge_capacity.csv"
RAMP_SHARE = 0.28
PROFILE = [(0.0, 600.0, 0.85), (600.0, 1500.0, None), (1500.0, 2400.0, 0.60)]   # None -> X_peak
X_PEAKS = [0.85, 0.95, 1.00, 1.05, 1.10, 1.20, 1.35]
PRETIMED_RATES = [240, 300, 360, 420, 500, 600, 700, 800, 900]   # up to r_max (E7: 600 was a boundary optimum)
ALINEA_SETPOINTS = [0.08, 0.10, 0.12, 0.15, 0.20, 0.25]           # E7: 0.15 was a boundary optimum
SEEDS = [181, 182, 183, 184]
MPS_TO_MPH = 2.23694


def merge_capacity(share: float = RAMP_SHARE) -> float:
    """C_m(share) from the E6 sweep (saturated discharge), read from results, not copied."""
    est = pd.read_csv(CAP_CSV)
    row = est.loc[np.isclose(est["ramp_share"], share)]
    if row.empty:
        raise ValueError(f"no measured capacity for ramp share {share}")
    return float(row["sat_discharge_vph"].iloc[0])


def build_config(x_peak: float, seed: int, c_m: float) -> ScenarioConfig:
    cfg = ScenarioConfig.from_yaml(S3)
    cfg.seed = seed
    cfg.ramp.seed = None
    xs = [(a, b, x_peak if x is None else x) for a, b, x in PROFILE]
    cfg.demand_profile = [[a, b, x * c_m * (1 - RAMP_SHARE)] for a, b, x in xs]
    cfg.ramp.demand_profile = [[a, b, x * c_m * RAMP_SHARE] for a, b, x in xs]
    cfg.record_every_s = 5.0
    cfg.control = None
    cfg.scenario_id = f"E7_Xp{x_peak:.2f}"
    cfg.demand_id = f"D7_Xp{x_peak:.2f}_r{RAMP_SHARE:.2f}"
    return cfg


def make(mode: str, param) -> Controller:
    if mode == "none":
        return Controller("none")
    if mode == "pretimed":
        return Controller("pretimed", rate_vph=float(param))
    return Controller("alinea", o_set=float(param))


def forecast_skill(trace: pd.DataFrame, horizon_s: float = 60.0) -> dict:
    """Queue and ramp-demand forecast errors vs persistence, from one decision trace."""
    out = {}
    tr = trace.set_index("time_s")
    q = tr["queue_est_veh"].astype(float)
    fc = tr["queue_fc_veh"].astype(float)
    tgt = q.reindex(q.index + horizon_s).to_numpy()
    ok = np.isfinite(tgt) & np.isfinite(fc.to_numpy()) & np.isfinite(q.to_numpy())
    if ok.sum():
        out["queue_fc_mae_veh"] = float(np.mean(np.abs(fc.to_numpy()[ok] - tgt[ok])))
        out["queue_persist_mae_veh"] = float(np.mean(np.abs(q.to_numpy()[ok] - tgt[ok])))
    lam = tr["ramp_demand_smoothed_vph"].astype(float).to_numpy()
    meas = tr["ramp_demand_meas_vph"].astype(float).to_numpy()
    ok = np.isfinite(lam[:-1]) & np.isfinite(meas[1:]) & np.isfinite(meas[:-1])
    if ok.sum():
        out["demand_fc_mae_vph"] = float(np.mean(np.abs(lam[:-1][ok] - meas[1:][ok])))
        out["demand_persist_mae_vph"] = float(np.mean(np.abs(meas[:-1][ok] - meas[1:][ok])))
    return out


def run_case(args) -> dict:
    x_peak, mode, param, seed, c_m = args
    cfg = build_config(x_peak, seed, c_m)
    ctrl = make(mode, param)
    res = FreewaySegment(load_params(cfg.param_set), cfg, controller=ctrl).run()
    s = res.summary()
    v, c, o = res.vehicles, res.counts, res.observations
    ex = v["exit_s"].to_numpy()
    served = np.sum((ex > 600) & (ex <= 2100)) / 1500.0 * 3600.0
    n_max = ctrl.limits.n_max_veh
    dt_rec = float(c["time_s"].diff().median()) if len(c) > 1 else 0.0
    speeds = o[o["sensor_id"].isin(["M_up", "M_mrg"])]["mean_speed_mps"].dropna()
    tr = res.control_trace
    row = {
        "x_peak": x_peak, "mode": mode, "param": float("nan") if param is None else float(param),
        "label": ctrl.label, "seed": seed,
        "ttt_veh_h": s["total_travel_time_veh_h"], "ramp_wait_veh_h": s["total_ramp_wait_veh_h"],
        "entry_wait_veh_h": s["total_entry_wait_veh_h"],
        "served_600_2100_vph": served, "peak_ramp_queue_veh": s["peak_ramp_queue_veh"],
        "time_ramp_queue_over_nmax_s": float((c["n_queued_ramp"] > n_max).sum() * dt_rec),
        "min_det_speed_mph": float(speeds.min() * MPS_TO_MPH) if len(speeds) else float("nan"),
        "unmet_demand_veh": s["unmet_demand_veh"], "cleared": s["cleared"], "end_time_s": s["end_time_s"],
        "n_overrides": int(tr["override"].sum()) if len(tr) else 0,
        "frac_metered": float((tr["action"] == "set_rate").mean()) if len(tr) else 0.0,
        "min_gap_m": s["min_gap_m"],
    }
    if mode != "none" and len(tr):
        row.update(forecast_skill(tr))
    return row


def sweep(x_peaks, seeds, pretimed_rates, alinea_sets, processes: int = 6, include_none: bool = True) -> pd.DataFrame:
    c_m = merge_capacity()
    jobs = []
    for x in x_peaks:
        for s in seeds:
            if include_none:
                jobs.append((x, "none", None, s, c_m))
            jobs += [(x, "pretimed", r, s, c_m) for r in pretimed_rates]
            jobs += [(x, "alinea", o, s, c_m) for o in alinea_sets]
    jobs.sort(key=lambda j: -j[0])   # slow (congested) runs first for better load balance
    with Pool(processes) as pool:
        rows = pool.map(run_case, jobs, chunksize=1)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per X_peak: no control, best pre-timed rate (per level), ALINEA at the global setpoint, ALINEA best
    per level. Means and SDs over seeds."""
    m = df.groupby(["x_peak", "mode", "label"]).agg(
        ttt=("ttt_veh_h", "mean"), ttt_sd=("ttt_veh_h", "std"), ramp_wait=("ramp_wait_veh_h", "mean"),
        served=("served_600_2100_vph", "mean"), peak_q=("peak_ramp_queue_veh", "max"),
        q_over_nmax_s=("time_ramp_queue_over_nmax_s", "mean"), min_speed_mph=("min_det_speed_mph", "mean"),
        unmet=("unmet_demand_veh", "max"), overrides=("n_overrides", "mean"),
        cleared=("cleared", "all")).reset_index()
    al = m[m["mode"] == "alinea"]
    global_label = al.groupby("label")["ttt"].mean().idxmin() if len(al) else None
    out = []
    for x, g in m.groupby("x_peak"):
        picks = [("none", g[g["mode"] == "none"])]
        pt = g[g["mode"] == "pretimed"]
        if len(pt):
            picks.append(("pretimed_best", pt.loc[[pt["ttt"].idxmin()]]))
        if global_label is not None:
            picks.append(("alinea_global", g[g["label"] == global_label]))
            picks.append(("alinea_best", al[al["x_peak"] == x].loc[[al[al["x_peak"] == x]["ttt"].idxmin()]]))
        for name, r in picks:
            if len(r):
                d = r.iloc[0].to_dict()
                d["strategy"] = name
                out.append(d)
    res = pd.DataFrame(out)
    base = res[res["strategy"] == "none"].set_index("x_peak")["ttt"]
    res["ttt_vs_none_pct"] = 100.0 * (res["ttt"] / res["x_peak"].map(base) - 1.0)
    cols = ["x_peak", "strategy", "label", "ttt", "ttt_sd", "ttt_vs_none_pct", "ramp_wait", "served", "peak_q",
            "q_over_nmax_s", "min_speed_mph", "overrides", "unmet", "cleared"]
    return res[cols]


def plot(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    styles = {"none": ("No signal", "#555555", "o"), "pretimed_best": ("Pre-timed (best rate per level)", "#1f77b4", "s"),
              "alinea_global": ("ALINEA (one setpoint)", "#d62728", "^")}
    for k, (lab, col, mk) in styles.items():
        g = summary[summary["strategy"] == k].sort_values("x_peak")
        if len(g):
            ax.errorbar(g["x_peak"], g["ttt"], yerr=g["ttt_sd"], label=lab, color=col, marker=mk, capsize=3)
    ax.set_xlabel("peak demand ratio X_m = (q_in + lambda_r) / C_m   (C_m = merge discharge, r = 0.28)")
    ax.set_ylabel("total travel time (veh*h, incl. ramp wait)")
    ax.set_title("E7 | S3 geometry | mean +/- SD over seeds, runs until clear")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str]) -> None:
    out = ROOT / "results"
    runs_csv = out / "E7_runs.csv"
    if "--from-csv" in argv:
        df = pd.read_csv(runs_csv)
    elif "--extend" in argv:                   # add settings to an existing run file (no repeated runs)
        old = pd.read_csv(runs_csv)
        done_pt = set(old.loc[old["mode"] == "pretimed", "param"])
        done_al = set(old.loc[old["mode"] == "alinea", "param"])
        new = sweep(X_PEAKS, SEEDS, [r for r in PRETIMED_RATES if r not in done_pt],
                    [o for o in ALINEA_SETPOINTS if o not in done_al], include_none=False)
        df = pd.concat([old, new], ignore_index=True)
    elif "--quick" in argv:
        df = sweep([0.95, 1.10], [181], [300, 420], [0.10])
    else:
        df = sweep(X_PEAKS, SEEDS, PRETIMED_RATES, ALINEA_SETPOINTS)
    summ = summarize(df)
    pd.set_option("display.width", 220)
    print(f"C_m(r={RAMP_SHARE}) = {merge_capacity():.0f} veh/h")
    print(summ.round(2).to_string(index=False))
    fc = df[df["mode"] != "none"].filter(like="mae").mean()
    if len(fc):
        print("\nforecast skill (mean over metered runs):", fc.round(2).to_dict())
    if "--quick" not in argv:
        if "--from-csv" not in argv:
            df.to_csv(runs_csv, index=False)
        summ.to_csv(out / "E7_summary.csv", index=False)
        plot(summ, out / "E7_ttt_vs_X.png")
        print("wrote results/E7_runs.csv, E7_summary.csv, E7_ttt_vs_X.png")


if __name__ == "__main__":
    main(sys.argv[1:])
