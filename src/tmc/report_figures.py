"""Figures and numbers for the progress report (Sections 2-3).

    python -m tmc.report_figures                      # writes results/PR_*.png and results/PR_numbers.csv
    python -m tmc.report_figures congestion 1.35      # congestion minutes at M_mrg, 4 seeds, none vs ALINEA

Everything is recomputed from the package (idm.capacity, evaluate.build_config, the E6/E7 CSVs).
Congestion definition used for Fig. PR5 / PR_numbers (proposed for Section 1):
    a 30-s interval at detector M_mrg is congested when its mean speed < v_c (computed IDM critical
    speed, 18.6 m/s) OR its occupancy > the critical occupancy o_c = k_c (l + L_loop) (computed,
    ~18 %), and a congestion episode needs >= 2 consecutive such intervals (60 s).
    Team decision 2026-11-15 (replaces a provisional 15 % threshold), see docs/design_record.md.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tmc.evaluate import ROOT, build_config, make, merge_capacity, MPS_TO_MPH
from tmc.idm import capacity, equilibrium_flow, load_params
from tmc.network import FreewaySegment, ScenarioConfig

RES = ROOT / "results"
N_CONSEC = 2
MI = 1609.344


def critical_occupancy(cap: dict, p, loop_m: float = 2.0) -> float:
    """o_c = k_c (l + L_loop): fraction of time a loop is occupied at the IDM capacity state."""
    return cap["k_c_veh_per_m"] * (p.length + loop_m)


def congested_intervals(obs: pd.DataFrame, sensor: str, v_c: float, occ_c: float) -> pd.DataFrame:
    o = obs[obs["sensor_id"] == sensor].sort_values("interval_end_s").copy()
    slow = o["mean_speed_mps"].astype(float) < v_c          # NaN (no vehicles) -> False
    full = o["occupancy"].astype(float) > occ_c
    raw = (slow | full).to_numpy()
    flag = np.zeros_like(raw)
    run = 0
    for i, r in enumerate(raw):
        run = run + 1 if r else 0
        if run >= N_CONSEC:
            flag[i - run + 1:i + 1] = True
    o["congested"] = flag
    return o


def fig_schematic(cfg: ScenarioConfig, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 2.6))
    L = cfg.length_m
    r = cfg.ramp
    ax.plot([0, L], [0, 0], color="0.3", lw=6, solid_capstyle="butt")
    ax.plot([r.entrance_m, r.merge_start_m], [-0.9, -0.25], color="0.5", lw=4)
    ax.plot([r.merge_start_m, r.merge_end_m], [-0.25, -0.25], color="0.5", lw=4)
    ax.annotate("entry queue\n(point queue)", (0, 0), (0, 0.9), ha="center", fontsize=8,
                arrowprops=dict(arrowstyle="->"))
    ax.annotate("free exit", (L, 0), (L, 0.9), ha="center", fontsize=8, arrowprops=dict(arrowstyle="->"))
    ax.annotate("ramp queue + meter\n(stop line)", (r.entrance_m, -0.9), (r.entrance_m - 350, -1.6),
                ha="center", fontsize=8, arrowprops=dict(arrowstyle="->"))
    ax.text(r.merge_end_m + 40, -0.55, "acceleration lane, 250 m\n(zipper + gap seeking)",
            ha="left", va="center", fontsize=8)
    for d in cfg.detectors:
        if d.kind != "loop":
            continue
        y = 0 if d.lane_id >= 0 else (-0.25 if d.position_m >= r.merge_start_m else
                                      -0.9 + 0.65 * (d.position_m - r.entrance_m) / (r.merge_start_m - r.entrance_m))
        ax.plot(d.position_m, y, "s", color="tab:red", ms=7, zorder=5)
        ax.text(d.position_m, y + (0.35 if d.lane_id >= 0 else -0.4), d.sensor_id, ha="center",
                fontsize=8, color="tab:red", va="center")
    ax.set_xlim(-250, L + 250)
    ax.set_ylim(-2.0, 1.4)
    ax.set_yticks([])
    ax.set_xlabel("position x (m), mainline coordinate")
    ax.set_title(f"{cfg.scenario_id}: single-lane mainline, on-ramp and virtual detectors (red)")
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_fd(p, cap, c_m, path: Path) -> None:
    v = np.linspace(1e-3, p.v0 * 0.999, 4000)
    q = equilibrium_flow(v, p) * 3600
    k = q / v / 3600 * MI
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(k, q, color="k", label="IDM equilibrium q_e(k), baseline parameters")
    ax.plot(cap["k_c_veh_per_m"] * MI, cap["q_c_veh_per_h"], "o", color="tab:blue",
            label=f"computed capacity q_c = {cap['q_c_veh_per_h']:.0f} veh/h "
                  f"(v_c = {cap['v_c_mps'] * MPS_TO_MPH:.1f} mph)")
    ax.axhline(c_m, color="tab:red", ls="--",
               label=f"measured merge discharge C_m(r=0.28) = {c_m:.0f} veh/h (E6)")
    ax.set_xlabel("density k (veh/mi/lane)")
    ax.set_ylabel("flow q (veh/h/lane)")
    ax.set_xlim(0, cap["k_jam_veh_per_m"] * MI * 1.02)
    ax.set_ylim(0, 2000)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Equilibrium fundamental diagram (computed in tmc.idm)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_merge_capacity(path: Path) -> None:
    runs = pd.read_csv(RES / "S3_onramp_zipgap_nocontrol_merge_capacity_runs.csv")
    est = pd.read_csv(RES / "S3_onramp_zipgap_nocontrol_merge_capacity.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4))
    cols = {0.1: "tab:gray", 0.2: "tab:green", 0.28: "tab:red", 0.4: "tab:purple"}
    for share, g in runs.groupby("ramp_share"):
        c = cols.get(round(share, 2), "k")
        ax.scatter(g["arrived_vph"], g["served_vph"], s=12, color=c, alpha=0.7, label=f"r = {share:.2f}")
    m = est.loc[np.isclose(est["ramp_share"], 0.28)].iloc[0]
    ax.axhline(m["sat_discharge_vph"], color="tab:red", ls="--", lw=1,
               label=f"C_m(0.28) = {m['sat_discharge_vph']:.0f} veh/h, CV {100 * m['sat_discharge_cv']:.1f}%")
    lim = [800, max(runs["arrived_vph"].max(), runs["served_vph"].max()) + 50]
    ax.plot(lim, lim, color="0.6", lw=0.8, label="served = arrived")
    ax.set_xlabel("realized total arrival flow, mainline + ramp (veh/h)")
    ax.set_ylabel("served flow past the merge (veh/h)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("E6: S3 merge saturates near C_m (132 constant-demand runs)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(x_peak, mode, param, seed, c_m, record_s=2.0):
    cfg = build_config(x_peak, seed, c_m)
    cfg.record_every_s = record_s
    ctrl = make(mode, param)
    res = FreewaySegment(load_params(cfg.param_set), cfg, controller=ctrl).run()
    return cfg, res


def fig_compare(c_m, cap, path: Path, x_peak=1.35, seed=181) -> list[dict]:
    v_c = cap["v_c_mps"]
    cases = [("none", None, "No signal"), ("alinea", 0.20, "ALINEA, o_set = 20%")]
    out, rows = [], []
    for mode, param, title in cases:
        cfg, res = run(x_peak, mode, param, seed, c_m)
        out.append((title, cfg, res))
        s = res.summary()
        cong = congested_intervals(res.observations, "M_mrg", v_c, cap["occ_c"])
        rows.append({"case": f"X_m={x_peak} seed={seed} {title}", "ttt_veh_h": s["total_travel_time_veh_h"],
                     "ramp_wait_veh_h": s["total_ramp_wait_veh_h"], "peak_ramp_queue": s["peak_ramp_queue_veh"],
                     "congested_min_M_mrg": cong["congested"].sum() * 0.5,
                     "min_mainline_speed_mph": s["min_mainline_speed_mps"] * MPS_TO_MPH,
                     "cleared": s["cleared"], "unmet": s["unmet_demand_veh"]})
    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1, 1], hspace=0.45)
    tmax = max(r.end_time_s for _, _, r in out) / 60
    for j, (title, cfg, res) in enumerate(out):
        ax = fig.add_subplot(gs[0, j])
        tr = res.trajectories
        tr = tr[tr["lane_id"] >= 0]
        sc = ax.scatter(tr["time_s"] / 60, tr["position_m"], c=tr["speed_mps"] * MPS_TO_MPH, s=0.3,
                        cmap="RdYlGn", vmin=0, vmax=75, rasterized=True)
        ax.axhspan(cfg.ramp.merge_start_m, cfg.ramp.merge_end_m, color="k", alpha=0.08)
        ax.set_title(f"{title}: TTT {res.summary()['total_travel_time_veh_h']:.1f} veh*h")
        ax.set_xlabel("time (min)")
        ax.set_ylabel("mainline position x (m)")
        ax.set_xlim(0, tmax)
    fig.colorbar(sc, ax=fig.axes[:2], label="speed (mph)", fraction=0.03, pad=0.02)
    # ramp queue and meter rate (ALINEA run)
    title, cfg, res = out[1]
    ax = fig.add_subplot(gs[1, :])
    for t2, c2, r2 in out:
        ax.plot(r2.counts["time_s"] / 60, r2.counts["n_queued_ramp"], label=f"ramp queue, {t2}")
    ax.axhline(35, color="k", ls=":", lw=1, label="n_max = 35 veh")
    ax.set_ylabel("ramp queue (veh)")
    ax.set_xlim(0, tmax)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax2 = fig.add_subplot(gs[2, :], sharex=ax)
    tr = res.control_trace
    rate = tr["rate_cmd_vph"].astype(float)
    ax2.step(tr["time_s"] / 60, rate.fillna(900), where="post", color="tab:red",
             label="ALINEA command (900 = meter dark)")
    ax2b = ax2.twinx()
    ax2b.plot(tr["time_s"] / 60, 100 * tr["occ"].astype(float), color="tab:blue", lw=0.8,
              label="occupancy at M_mrg (%)")
    ax2b.axhline(20, color="tab:blue", ls=":", lw=1)
    ax2.set_ylabel("meter rate (veh/h)")
    ax2b.set_ylabel("occupancy (%)", color="tab:blue")
    ax2.set_xlabel("time (min)")
    ax2.grid(alpha=0.3)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    fig.suptitle(f"E7 example: X_m peak = {x_peak}, r = 0.28, seed {seed} (same demand both runs)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rows


def congestion_table(c_m, cap, seeds=(181, 182, 183, 184), xs=(1.10, 1.20, 1.35)) -> list[dict]:
    rows = []
    for x in xs:
        for mode, param in (("none", None), ("alinea", 0.20)):
            mins = []
            for sd in seeds:
                cfg = build_config(x, sd, c_m)
                cfg.record_every_s = 30.0
                res = FreewaySegment(load_params(cfg.param_set), cfg, controller=make(mode, param)).run()
                mins.append(float(congested_intervals(res.observations, "M_mrg", cap["v_c_mps"], cap["occ_c"])["congested"].sum() * 0.5))
            rows.append({"x_peak": x, "mode": mode if param is None else f"alinea_{param:.2f}",
                         "congested_min_mean": float(np.mean(mins)), "congested_min_by_seed": mins})
    return rows


def main(argv: list[str]) -> None:
    """No args: figures. `congestion X`: congestion minutes for one X_peak (appends to the CSV)."""
    p = load_params("baseline")
    cap = capacity(p)
    s3 = ScenarioConfig.from_yaml(ROOT / "configs" / "scenario_s3_onramp_zipgap.yaml")
    loop = next(d for d in s3.detectors if d.sensor_id == "M_mrg").field_length_m
    cap["occ_c"] = critical_occupancy(cap, p, loop)
    c_m = merge_capacity()
    if argv and argv[0] == "congestion":
        out = RES / "PR_congestion_minutes.csv"
        print(f"critical occupancy o_c = {cap['occ_c']:.4f}")
        new = pd.DataFrame(congestion_table(c_m, cap, xs=tuple(float(x) for x in argv[1:])))
        new["occ_threshold"] = round(cap["occ_c"], 4)
        if out.exists():
            new = pd.concat([pd.read_csv(out), new]).drop_duplicates(["x_peak", "mode"], keep="last")
        new.to_csv(out, index=False)
        print(new.to_string())
        return
    fig_schematic(s3, RES / "PR1_segment_layout.png")
    fig_fd(p, cap, c_m, RES / "PR2_fundamental_diagram.png")
    fig_merge_capacity(RES / "PR3_merge_capacity.png")
    rows = fig_compare(c_m, cap, RES / "PR5_none_vs_alinea_Xm1.35.png")
    pd.DataFrame(rows).to_csv(RES / "PR_numbers.csv", index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
