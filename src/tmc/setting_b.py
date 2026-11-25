"""E9 setting B: human vs ACC vs CACC at the S3r merge (experiment log E9e, pre-registered 2026-11-25).

    python -m tmc.setting_b --quick          # small example: X 1.35, seed 181, human vs ACC p = 1 (~2 runs)
    python -m tmc.setting_b --dtcheck        # D3: human and ACC p = 1 at dt 0.05 vs 0.1 (X 1.35, seed 181)
    python -m tmc.setting_b --run 1.35       # every case at one X_peak, seeds 181-184 (appends to E9B_runs.csv)
    python -m tmc.setting_b --holdout 1.35 acc 1.0   # one case + the human baseline on held-out seeds 185-188
    python -m tmc.setting_b --judge          # judgement + figures from results/E9B_runs.csv (no simulation)
    dt is 0.05 s for all of setting B (D3 failed on S3r); --dt X overrides it for diagnostics only

Scenario (E9e): S3r (S3 merge, humans with T_r = 0.6 s), ramp share r = 0.28, E7 demand profile
0.85 / X_peak / 0.60 (0-600 / 600-1500 / 1500-2400 s) against C_m = the S3r2 saturated discharge read from
results (E9b2), no ramp metering, run until clear, dt 0.05 s (D3). S3r2 = S3r with the reaction delay
restarted at leader changes (team decision 2026-11-25). Cases: human; ACC and CACC at p = 0.25 / 0.5 /
1.0 (tau 0.1 s, delay 0.15 s, fallback at age > 5 ticks). Placement seed = run seed.

Metrics use simulator truth (evaluation only, CLAUDE.md rule 2). Judgement rules (E9e D1, D2, D4):
  better        mean TTT >= 2% lower AND lower in >= 3 of 4 paired seeds (E7c)
  throughput    served flow 600-2100 s, mean over seeds, not more than 2% below human
  D1 reduced    CACC congested minutes at M_mrg lower than the same-seed human AND same-seed ACC run by
                >= 10% and >= 1.0 min, in >= 3 of 4 seeds (E7c congestion definition)
  D2 gaps       net gap >= 1 m and time gap >= 0.5 s at v > 5 m/s (every follower); if the same-seed human
                run itself goes below a bound, that seed's bound becomes the human run's minimum ("no worse
                than human"; amended for the 1 m bound 2026-11-25, before any CAV run), reported. Equipped
                max deceleration <= 4 m/s^2 stays absolute.
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from tmc.evaluate import MPS_TO_MPH, PROFILE, RAMP_SHARE
from tmc.idm import capacity, load_params
from tmc.network import GAP_TYPES, FreewaySegment, ScenarioConfig
from tmc.report_figures import congested_intervals, critical_occupancy

ROOT = Path(__file__).resolve().parents[2]
# S3r2 = S3r with the reaction delay restarted at leader changes (team 2026-11-25, E9e); C_m from E9b2.
S3R = ROOT / "configs" / "scenario_s3r2_onramp_zipgap_tr06_restart.yaml"
CAP_CSV = ROOT / "results" / "S3r2_onramp_zipgap_tr06_restart_nocontrol_merge_capacity.csv"
DT_B = 0.05                                  # E9e D3 failed on S3r: all of setting B runs at dt 0.05
RES = ROOT / "results"
RUNS_CSV = RES / "E9B_runs.csv"
SEEDS = [181, 182, 183, 184]
HOLDOUT = [185, 186, 187, 188]
X_PEAKS = [1.35, 1.20]                      # D5 order: 1.35 first
PENETRATIONS = [0.25, 0.5, 1.0]
CASES = [("human", 0.0)] + [(m, p) for p in PENETRATIONS for m in ("acc", "cacc")]
TAU_S, DELAY_S = 0.1, 0.15                  # nominal (design record section 7); tau = 0.5 s cut (E9e D5)
BETTER_PCT, SEEDS_NEEDED = 2.0, 3           # E7c "better" rule
THROUGHPUT_PCT = 2.0
D1_REL, D1_ABS_MIN = 0.10, 1.0              # E9e D1
GAP_MIN_M, TIME_GAP_MIN_S, DECEL_MAX = 1.0, 0.5, 4.0
FALLBACK_REASONS = ("leader_unequipped", "no_message", "stale", "comms_off")


def merge_capacity_s3r(share: float = RAMP_SHARE) -> float:
    """C_m of S3r2 (E9b2), read from results, not copied."""
    est = pd.read_csv(CAP_CSV)
    row = est.loc[np.isclose(est["ramp_share"], share)]
    if row.empty:
        raise ValueError(f"no measured S3r2 capacity for ramp share {share}")
    return float(row["sat_discharge_vph"].iloc[0])


def build_config(x_peak: float, seed: int, c_m: float, mode: str = "human", p: float = 0.0,
                 dt: float = DT_B) -> ScenarioConfig:
    cfg = ScenarioConfig.from_yaml(S3R)
    cfg.seed = seed
    cfg.ramp.seed = None
    cfg.dt_s = dt
    xs = [(a, b, x_peak if x is None else x) for a, b, x in PROFILE]
    cfg.demand_profile = [[a, b, x * c_m * (1 - RAMP_SHARE)] for a, b, x in xs]
    cfg.ramp.demand_profile = [[a, b, x * c_m * RAMP_SHARE] for a, b, x in xs]
    cfg.record_every_s = 5.0
    cfg.control = None
    cfg.cav = None if mode == "human" else {"p_equipped": p, "mode": mode, "placement_seed": seed,
                                            "tau_s": TAU_S, "delay_s": DELAY_S}
    cfg.scenario_id = f"E9B_Xp{x_peak:.2f}_{mode}_p{p:.2f}"
    cfg.demand_id = f"D9B_Xp{x_peak:.2f}_r{RAMP_SHARE:.2f}_Cm{c_m:.0f}"
    return cfg


def congestion_minutes(obs: pd.DataFrame, sensor: str, v_c: float, occ_c: float) -> tuple[float, pd.DataFrame]:
    o = congested_intervals(obs, sensor, v_c, occ_c)
    return float(o["congested"].sum() * 0.5), o


def merge_discharge(obs: pd.DataFrame, cong_mrg: pd.DataFrame) -> float:
    """Mean flow at M_down over the 30-s intervals in which M_mrg is congested; NaN (not 0) if none."""
    ends = cong_mrg.loc[cong_mrg["congested"].astype(bool), "interval_end_s"]
    d = obs[(obs["sensor_id"] == "M_down") & obs["interval_end_s"].isin(ends)]
    if d.empty:
        return float("nan")
    return float((d["count"] * 3600.0 / (d["interval_end_s"] - d["interval_start_s"])).mean())


def run_case(args) -> dict:
    x_peak, mode, p, seed, c_m, dt = args
    cfg = build_config(x_peak, seed, c_m, mode, p, dt)
    prm = load_params(cfg.param_set)
    cap = capacity(prm)
    v_c, occ_c = cap["v_c_mps"], critical_occupancy(cap, prm)
    res = FreewaySegment(prm, cfg).run()
    s = res.summary()
    v, o = res.vehicles, res.observations
    ex = v["exit_s"].to_numpy()
    served = np.sum((ex > 600) & (ex <= 2100)) / 1500.0 * 3600.0
    cm_mrg, cong = congestion_minutes(o, "M_mrg", v_c, occ_c)
    cm_up, _ = congestion_minutes(o, "M_up", v_c, occ_c)
    speeds = {d: o.loc[o["sensor_id"] == d, "mean_speed_mps"].dropna() for d in ("M_up", "M_mrg")}
    ev = res.cav_events
    row = {
        "x_peak": x_peak, "mode": mode, "p": p, "seed": seed, "dt_s": dt, "c_m_vph": c_m,
        "n_vehicles": int(len(v)), "n_equipped": int(v["equipped"].sum()),
        "ttt_veh_h": s["total_travel_time_veh_h"],
        "ttt_main_origin_veh_h": s["total_travel_time_mainline_origin_veh_h"],
        "ttt_ramp_origin_veh_h": s["total_travel_time_ramp_origin_veh_h"],
        "ramp_wait_veh_h": s["total_ramp_wait_veh_h"], "entry_wait_veh_h": s["total_entry_wait_veh_h"],
        "served_600_2100_vph": served, "peak_ramp_queue_veh": s["peak_ramp_queue_veh"],
        "congested_min_M_mrg": cm_mrg, "congested_min_M_up": cm_up,
        "merge_discharge_vph": merge_discharge(o, cong),
        "min_speed_M_up_mph": float(speeds["M_up"].min() * MPS_TO_MPH) if len(speeds["M_up"]) else float("nan"),
        "min_speed_M_mrg_mph": float(speeds["M_mrg"].min() * MPS_TO_MPH) if len(speeds["M_mrg"]) else float("nan"),
        "unmet_demand_veh": s["unmet_demand_veh"], "cleared": s["cleared"], "end_time_s": s["end_time_s"],
        "min_gap_m": s["min_gap_m"], "n_fallback_ticks": int(len(ev)),
    }
    for k, val in res.gap_stats.items():
        row[k] = val if np.isfinite(val) else float("nan")    # no follower of that type -> missing, not inf
    for reason in FALLBACK_REASONS:                         # tmc.cav fallback labels (tick counts)
        row[f"fallback_{reason}"] = int((ev["reason"] == reason).sum()) if len(ev) else 0
    return row


def sweep(jobs, processes: int = 6) -> pd.DataFrame:
    jobs = sorted(jobs, key=lambda j: (-j[0], j[2]))   # most congested (high X, low p) first: load balance
    with Pool(processes) as pool:
        rows = pool.map(run_case, jobs, chunksize=1)
    return pd.DataFrame(rows)


def jobs_for(x_peak: float, seeds, c_m: float, dt: float, cases=CASES):
    return [(x_peak, m, p, s, c_m, dt) for m, p in cases for s in seeds]


# ----------------------------------------------------------------------------------------- judgement

def _paired(df: pd.DataFrame, a: tuple, b: tuple, col: str) -> pd.DataFrame:
    """Rows of case a and case b, paired by seed (same x_peak and dt)."""
    ka = df[(df["mode"] == a[0]) & np.isclose(df["p"], a[1])].set_index("seed")[col]
    kb = df[(df["mode"] == b[0]) & np.isclose(df["p"], b[1])].set_index("seed")[col]
    return pd.concat({"a": ka, "b": kb}, axis=1).dropna()


def better(df: pd.DataFrame, a: tuple, b: tuple) -> dict:
    """Is case a better than case b by the E7c rule (TTT)?"""
    pr = _paired(df, a, b, "ttt_veh_h")
    if pr.empty:
        return {"n_seeds": 0}
    pct = 100.0 * (pr["a"].mean() / pr["b"].mean() - 1.0)
    n_lower = int((pr["a"] < pr["b"]).sum())
    return {"n_seeds": len(pr), "ttt_a": pr["a"].mean(), "ttt_b": pr["b"].mean(), "ttt_pct": pct,
            "n_seeds_lower": n_lower, "better": bool(pct <= -BETTER_PCT and n_lower >= SEEDS_NEEDED)}


def d1_reduced(df: pd.DataFrame, p: float) -> dict:
    """E9e D1: CACC congested minutes at M_mrg vs the same-seed human AND ACC run."""
    h = _paired(df, ("cacc", p), ("human", 0.0), "congested_min_M_mrg")
    a = _paired(df, ("cacc", p), ("acc", p), "congested_min_M_mrg")
    ok = []
    for seed in h.index.intersection(a.index):
        c = h.loc[seed, "a"]
        good = all(ref > 0 and c <= (1 - D1_REL) * ref and ref - c >= D1_ABS_MIN
                   for ref in (h.loc[seed, "b"], a.loc[seed, "b"]))
        ok.append(good)
    return {"d1_seeds_pass": int(sum(ok)), "d1_n_seeds": len(ok), "d1_reduced": bool(sum(ok) >= SEEDS_NEEDED)}


def gaps_ok(df: pd.DataFrame, mode: str, p: float) -> dict:
    """E9e D2 (amended 2026-11-25, before any CAV run), per seed: net gap >= min(1 m, same-seed human-run
    min), time gap >= min(0.5 s, same-seed human-run min), equipped max deceleration <= 4 m/s^2 (absolute)."""
    hum = df[df["mode"] == "human"].set_index("seed")
    rows = df[(df["mode"] == mode) & np.isclose(df["p"], p)].set_index("seed")
    gap_cols = [f"min_gap_m_{k}" for k in GAP_TYPES]
    tg_cols = [f"min_time_gap_s_{k}" for k in GAP_TYPES]
    out, relaxed_tg, relaxed_g = [], 0, 0
    for seed, r in rows.iterrows():
        b_tg, b_g = TIME_GAP_MIN_S, GAP_MIN_M
        if seed in hum.index and hum.loc[seed, "min_time_gap_s_hh"] < TIME_GAP_MIN_S:
            b_tg, relaxed_tg = float(hum.loc[seed, "min_time_gap_s_hh"]), relaxed_tg + 1
        if seed in hum.index and hum.loc[seed, "min_gap_m_hh"] < GAP_MIN_M:
            b_g, relaxed_g = float(hum.loc[seed, "min_gap_m_hh"]), relaxed_g + 1
        g = np.nanmin(r[gap_cols].to_numpy(float))
        tg = np.nanmin(r[tg_cols].to_numpy(float))
        dec = r.get("max_decel_mps2_eq", np.nan)
        out.append(bool(g >= b_g - 1e-12 and tg >= b_tg - 1e-12 and not (dec > DECEL_MAX)))
    return {"gaps_ok": bool(all(out)) if out else None, "gaps_seeds_pass": int(sum(out)),
            "time_gap_bound_relaxed_seeds": relaxed_tg, "net_gap_bound_relaxed_seeds": relaxed_g}


def judge(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (x, dt), g in df.groupby(["x_peak", "dt_s"]):
        hum = g[g["mode"] == "human"]
        served_h = hum["served_600_2100_vph"].mean()
        for mode, p in CASES[1:]:
            c = g[(g["mode"] == mode) & np.isclose(g["p"], p)]
            if c.empty:
                continue
            row = {"x_peak": x, "dt_s": dt, "mode": mode, "p": p, "n_seeds": len(c)}
            vh = better(g, (mode, p), ("human", 0.0))
            row.update({"ttt": vh.get("ttt_a"), "ttt_human": vh.get("ttt_b"), "ttt_vs_human_pct": vh.get("ttt_pct"),
                        "seeds_lower_than_human": vh.get("n_seeds_lower"), "better_than_human": vh.get("better")})
            row["served_vs_human_pct"] = 100.0 * (c["served_600_2100_vph"].mean() / served_h - 1.0)
            row["throughput_preserved"] = bool(row["served_vs_human_pct"] >= -THROUGHPUT_PCT)
            row["congested_min_M_mrg"] = c["congested_min_M_mrg"].mean()
            row["congested_min_M_mrg_human"] = hum["congested_min_M_mrg"].mean()
            row["merge_discharge_vph"] = c["merge_discharge_vph"].mean()
            row["merge_discharge_human_vph"] = hum["merge_discharge_vph"].mean()
            if mode == "cacc":
                va = better(g, ("cacc", p), ("acc", p))
                vr = better(g, ("acc", p), ("cacc", p))
                row.update({"ttt_vs_acc_pct": va.get("ttt_pct"), "seeds_lower_than_acc": va.get("n_seeds_lower"),
                            "cacc_better_than_acc": va.get("better"), "acc_better_than_cacc": vr.get("better")})
                row.update(d1_reduced(g, p))
            row.update(gaps_ok(g, mode, p))
            out.append(row)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------------- figures

def plot(df: pd.DataFrame) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    style = {"acc": ("ACC (sensors)", "#1f77b4", "o"), "cacc": ("CACC (V2V)", "#d62728", "^")}
    for col, ylab, name in [("ttt_veh_h", "total travel time (veh*h, incl. ramp wait)", "E9B_ttt_vs_p.png"),
                            ("congested_min_M_mrg", "congested minutes at M_mrg (E7c definition)",
                             "E9B_congestion_vs_p.png")]:
        xs = sorted(df["x_peak"].unique(), reverse=True)
        fig, axes = plt.subplots(1, len(xs), figsize=(5.2 * len(xs), 4), squeeze=False)
        for ax, x in zip(axes[0], xs):
            g = df[(df["x_peak"] == x) & np.isclose(df["dt_s"], df["dt_s"].min())]
            h = g[g["mode"] == "human"][col]
            for mode, (lab, color, mk) in style.items():
                pts = [(0.0, h)] + [(p, g[(g["mode"] == mode) & np.isclose(g["p"], p)][col]) for p in PENETRATIONS]
                pts = [(p, s) for p, s in pts if len(s)]
                off = -0.012 if mode == "acc" else 0.012
                for p, s in pts:
                    ax.scatter(np.full(len(s), p + off), s, color=color, alpha=0.35, s=14, marker=mk)
                ax.plot([p + off for p, _ in pts], [s.mean() for _, s in pts], color=color, marker=mk, label=lab)
            ax.set_xlabel("penetration p (share of equipped vehicles)")
            ax.set_ylabel(ylab)
            ax.set_title(f"E9 setting B | S3r2 merge, X_m peak {x:.2f} (C_m {g['c_m_vph'].iloc[0]:.0f} veh/h)\n"
                         f"dt {g['dt_s'].iloc[0]} s | mean line, dots = seeds 181-184")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = RES / name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


# ----------------------------------------------------------------------------------------- CLI

def _append(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if path.exists():
        old = pd.read_csv(path)
        key = ["x_peak", "mode", "p", "seed", "dt_s"]
        merged = old.merge(df[key], on=key, how="left", indicator=True)
        old = old[(merged["_merge"] == "left_only").to_numpy()]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def dt_check(c_m: float) -> pd.DataFrame:
    """E9e D3: human and ACC p = 1.0 at X 1.35, seed 181, dt 0.05 vs 0.1; 3% bound on TTT and served flow."""
    jobs = [(1.35, m, p, 181, c_m, dt) for m, p in [("human", 0.0), ("acc", 1.0)] for dt in (0.1, 0.05)]
    df = sweep(jobs, processes=4)
    rows = []
    for (m, p), g in df.groupby(["mode", "p"]):
        a, b = g[np.isclose(g["dt_s"], 0.1)].iloc[0], g[np.isclose(g["dt_s"], 0.05)].iloc[0]
        r = {"mode": m, "p": p}
        for col in ("ttt_veh_h", "served_600_2100_vph", "congested_min_M_mrg"):
            r[f"{col}_dt0.1"], r[f"{col}_dt0.05"] = a[col], b[col]
            r[f"{col}_dev_pct"] = 100.0 * (b[col] / a[col] - 1.0) if a[col] else float("nan")
        r["pass_3pct"] = bool(abs(r["ttt_veh_h_dev_pct"]) <= 3.0 and abs(r["served_600_2100_vph_dev_pct"]) <= 3.0)
        rows.append(r)
    out = pd.DataFrame(rows)
    tag = "_s3r2" if "s3r2" in S3R.name else ""   # E9B_dtcheck.csv itself is the S3r check (E9e)
    df.to_csv(RES / f"E9B_dtcheck{tag}_runs.csv", index=False)
    out.to_csv(RES / f"E9B_dtcheck{tag}.csv", index=False)
    return out


def main(argv: list[str]) -> None:
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    c_m = merge_capacity_s3r()
    dt = float(argv[argv.index("--dt") + 1]) if "--dt" in argv else DT_B
    print(f"C_m(S3r2, r={RAMP_SHARE}) = {c_m:.1f} veh/h, dt = {dt} s")
    if "--quick" in argv:
        df = sweep([(1.35, "human", 0.0, 181, c_m, dt), (1.35, "acc", 1.0, 181, c_m, dt)], processes=2)
        print(df[["mode", "p", "ttt_veh_h", "served_600_2100_vph", "congested_min_M_mrg",
                  "merge_discharge_vph", "min_gap_m", "cleared"]].round(2).to_string(index=False))
        return
    if "--dtcheck" in argv:
        print(dt_check(c_m).round(3).T.to_string())
        print("wrote results/E9B_dtcheck_s3r2.csv, E9B_dtcheck_s3r2_runs.csv")
        return
    if "--run" in argv:
        x = float(argv[argv.index("--run") + 1])
        df = _append(sweep(jobs_for(x, SEEDS, c_m, dt)), RUNS_CSV)
    elif "--holdout" in argv:
        i = argv.index("--holdout")
        x, mode, p = float(argv[i + 1]), argv[i + 2], float(argv[i + 3])
        cases = [("human", 0.0), (mode, p)] + ([("acc", p)] if mode == "cacc" else [])
        hd = sweep(jobs_for(x, HOLDOUT, c_m, dt, cases))
        path = RES / "E9B_holdout_runs.csv"
        hd = _append(hd, path)
        jd = judge(hd)
        jd.to_csv(RES / "E9B_holdout_judgement.csv", index=False)
        print(jd.round(3).T.to_string())
        return
    else:
        df = pd.read_csv(RUNS_CSV)
    if "--judge" in argv or "--run" in argv:
        jd = judge(df)
        jd.to_csv(RES / "E9B_judgement.csv", index=False)
        print(jd.round(3).T.to_string())
        for path in plot(df):
            print("wrote", path.relative_to(ROOT))
        print("wrote results/E9B_runs.csv, E9B_judgement.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
