"""Measured merge capacity (design record section 2, option (c)).

    python -m tmc.merge_capacity [scenario.yaml] [--quick | --from-csv]

--from-csv re-estimates from results/<scenario_id>_merge_capacity_runs.csv without re-simulating
(the full sweep is 132 one-hour runs, ~20 min on 6 cores).

The merge, not the computed IDM q_c, is the binding bottleneck in S1 (experiment log E1, E4). This module
measures what the configured merge can actually serve, so that the demand ratio can be stated against it:

    X_m = (q_in + lambda_r) / q_dis(ramp share)

Method (same scenario geometry, merge model, and params as the YAML; only demand is replaced):
  * Steady demand for HORIZON_S at a total flow q_tot and a fixed ramp share r: mainline q_tot (1 - r),
    ramp q_tot r, Poisson, several seeds.
  * Served flow = exits during [WARMUP_S, HORIZON_S] / window length (downstream boundary count, i.e.
    what a detector at the exit would see).
  * Queued at the end of the window = point queues + vehicles physically on the ramp lane.
  * A run is "demand-limited" if the end-of-window queue is below QUEUE_TOL_VEH AND the served flow is
    at least SERVED_TOL x the flow that actually arrived in the window; otherwise it is "saturated".
    History: v1 compared served with the nominal q_tot (fails on Poisson shortfalls, E4); v2 used the
    queue only, which misses vehicles stored in a mainline jam (E5b). v3 (this) uses both, against the
    realized arrivals. Old run files without arrived_vph fall back to the queue-only test.
  * Breakdown threshold(r) = largest q_tot at which every seed, at that and every lower demand, is
    demand-limited.
  * E4 found that the saturated served flow does NOT level off: the mainline keeps priority and the ramp
    gets what is left. So the useful product is the ramp capacity as a function of the mainline flow,
    c_r(q_m) (ramp_capacity_curve), as for a minor stream at a give-way junction, not one q_dis.
    served_at_max_vph (mean served flow at the highest demand) is reported for reference only.

Simulator truth (exit times, queue lengths) is used here because this is model characterization, not
control. Controllers must still only see detector data (CLAUDE.md rule 2).
"""
from __future__ import annotations

import copy
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from tmc.idm import capacity, load_params
from tmc.network import FreewaySegment, ScenarioConfig

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO = ROOT / "configs" / "scenario_s1_onramp.yaml"
HORIZON_S = 3600.0
WARMUP_S = 1200.0
QUEUE_TOL_VEH = 10
SERVED_TOL = 0.97


def steady_run(base: ScenarioConfig, q_tot: float, share: float, seed: int,
               horizon: float = HORIZON_S, warmup: float = WARMUP_S) -> dict:
    """One steady-demand run. Returns served flow and end-of-window queue (veh)."""
    cfg = copy.deepcopy(base)
    cfg.seed = seed
    cfg.ramp.seed = None                       # ramp stream = seed + 1000
    cfg.demand_profile = [[0.0, horizon, q_tot * (1 - share)]]
    cfg.ramp.demand_profile = [[0.0, horizon, q_tot * share]]
    cfg.max_time_s = horizon                   # stop at the window end; leftovers are what we measure
    cfg.record_every_s = 10.0
    cfg.scenario_id = f"{base.scenario_id}_cap"
    res = FreewaySegment(load_params(cfg.param_set), cfg).run()
    ex = res.vehicles["exit_s"].to_numpy()
    served = np.sum((ex > warmup) & (ex <= horizon)) / (horizon - warmup) * 3600.0
    c = res.counts.iloc[-1]
    queued = int(c["n_queued"] + c["n_on_ramp"])
    m = res.vehicles["merge_s"].to_numpy()
    ramp_served = np.sum((m > warmup) & (m <= horizon)) / (horizon - warmup) * 3600.0
    a = res.vehicles["arrival_s"].to_numpy()
    arrived = np.sum((a > warmup) & (a <= horizon)) / (horizon - warmup) * 3600.0
    return {"q_tot_vph": q_tot, "ramp_share": share, "seed": seed, "served_vph": served,
            "arrived_vph": arrived, "ramp_served_vph": ramp_served, "queued_end_veh": queued,
            "on_road_end_veh": int(c["n_on_road"]),
            "min_mainline_speed_mps": res.summary()["min_mainline_speed_mps"]}


def _job(args):
    path, q, r, s = args
    return steady_run(ScenarioConfig.from_yaml(path), q, r, s)


def sweep(path: Path, demands, shares, seeds, processes: int = 6) -> pd.DataFrame:
    jobs = [(str(path), float(q), float(r), int(s)) for r in shares for q in demands for s in seeds]
    with Pool(processes) as pool:
        rows = pool.map(_job, jobs)
    df = pd.DataFrame(rows)
    return classify(df)


def classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ok = df["queued_end_veh"] < QUEUE_TOL_VEH
    if "arrived_vph" in df:
        ok &= df["served_vph"] >= SERVED_TOL * df["arrived_vph"]
    df["demand_limited"] = ok
    return df


def estimate(df: pd.DataFrame) -> pd.DataFrame:
    """Per ramp share: breakdown threshold, highest flow served without a queue (max_free_served_vph),
    the saturated discharge (mean and coefficient of variation of the served flow over saturated runs),
    the capacity drop 1 - discharge / max_free, and the served flow at the highest demand."""
    out = []
    for r, g in df.groupby("ramp_share"):
        sat = g[~g["demand_limited"]]
        ok = g.groupby("q_tot_vph")["demand_limited"].all()
        ok_q = ok[ok].index
        below = [q for q in ok_q if all(ok.loc[ok.index <= q])]   # threshold: no failure at or below q
        top = g[g["q_tot_vph"] == g["q_tot_vph"].max()]
        free = g[g["demand_limited"]]
        dis = float(sat["served_vph"].mean()) if len(sat) else np.nan
        mx = float(free["served_vph"].max()) if len(free) else np.nan
        out.append({
            "ramp_share": r,
            "breakdown_threshold_vph": float(max(below)) if below else np.nan,
            "n_saturated_runs": int(len(sat)),
            "max_free_served_vph": mx,
            "sat_discharge_vph": dis,
            "sat_discharge_cv": float(sat["served_vph"].std() / dis) if len(sat) > 1 else np.nan,
            "capacity_drop": 1.0 - dis / mx if len(sat) and len(free) else np.nan,
            "sat_ramp_share": float((sat["ramp_served_vph"] / sat["served_vph"]).mean()) if len(sat) else np.nan,
            "served_at_max_vph": float(top["served_vph"].mean()),
            "ramp_served_at_max_vph": float(top["ramp_served_vph"].mean()),
        })
    return pd.DataFrame(out)


def ramp_capacity_curve(df: pd.DataFrame, bin_vph: float = 100.0, min_runs: int = 3) -> pd.DataFrame:
    """c_r(q_m): ramp flow served when the ramp is saturated, against the mainline flow actually served.
    Uses saturated runs in which the mainline got >= 95% of its demand (the queue is on the ramp, so the
    ramp flow is the merge's leftover capacity). Bins with fewer than min_runs runs are dropped."""
    d = df[~df["demand_limited"]].copy()
    d["main_served_vph"] = d["served_vph"] - d["ramp_served_vph"]
    d = d[d["main_served_vph"] >= 0.95 * d["q_tot_vph"] * (1 - d["ramp_share"])]
    d["q_m_bin_vph"] = (d["main_served_vph"] // bin_vph) * bin_vph + bin_vph / 2
    c = d.groupby("q_m_bin_vph").agg(n_runs=("seed", "size"), ramp_cap_vph=("ramp_served_vph", "mean"),
                                    ramp_cap_sd_vph=("ramp_served_vph", "std"),
                                    total_served_vph=("served_vph", "mean")).reset_index()
    return c[c["n_runs"] >= min_runs].reset_index(drop=True)


def demand_ratio(q_in_vph: float, lambda_r_vph: float, q_dis_vph: float) -> float:
    """X_m = (q_in + lambda_r) / q_dis, the demand ratio against the measured merge capacity."""
    return (q_in_vph + lambda_r_vph) / q_dis_vph


def main(argv: list[str]) -> None:
    args = [a for a in argv if not a.startswith("--")]
    path = Path(args[0]) if args else DEFAULT_SCENARIO
    base = ScenarioConfig.from_yaml(path)
    if base.ramp is None:
        raise SystemExit(f"{path} has no ramp; nothing to measure")
    out = ROOT / "results"
    runs_csv = out / f"{base.scenario_id}_merge_capacity_runs.csv"
    if "--from-csv" in argv:
        df = classify(pd.read_csv(runs_csv))
    else:
        if "--quick" in argv:                  # small example (CLAUDE.md rule 7)
            demands, shares, seeds = [1200, 1800], [0.28], [181]
        else:
            demands, shares, seeds = range(1000, 2001, 100), [0.1, 0.2, 0.28, 0.4], [181, 182, 183]
        df = sweep(path, demands, shares, seeds)
    est = estimate(df)
    curve = ramp_capacity_curve(df)
    q_c = capacity(load_params(base.param_set))["q_c_veh_per_h"]
    est["threshold_over_q_c"] = est["breakdown_threshold_vph"] / q_c
    pd.set_option("display.width", 160)
    print(f"scenario {base.scenario_id} | computed q_c = {q_c:.0f} veh/h | window {WARMUP_S:.0f}-{HORIZON_S:.0f} s")
    print(df.groupby(["ramp_share", "q_tot_vph"])[["served_vph", "ramp_served_vph", "queued_end_veh"]]
          .mean().round(0).to_string())
    print(est.round(3).to_string(index=False))
    print("\nramp capacity vs mainline flow served (saturated ramp, mainline fully served):")
    print(curve.round(0).to_string(index=False) if len(curve) else "  (not enough saturated runs)")
    if "--quick" not in argv:
        out.mkdir(exist_ok=True)
        if "--from-csv" not in argv:
            df.to_csv(runs_csv, index=False)
        est.to_csv(out / f"{base.scenario_id}_merge_capacity.csv", index=False)
        curve.to_csv(out / f"{base.scenario_id}_ramp_capacity_curve.csv", index=False)
        print(f"wrote results/{base.scenario_id}_merge_capacity.csv, ..._ramp_capacity_curve.csv")


if __name__ == "__main__":
    main(sys.argv[1:])
