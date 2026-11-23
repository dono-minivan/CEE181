"""E9 setting A: a braking disturbance travelling up a single-lane platoon (experiment log E9).

    python -m tmc.disturbance                # human baseline: T_r = 0 and 0.6 s, both disturbances
    python -m tmc.disturbance --quick        # one run, prints the metrics (small example, CLAUDE.md rule 7)

Setting (configs/scenario_e9_setting_a.yaml): one lane, a scripted lead (vehicle 0) and n followers in IDM
equilibrium at v_eq. At t_start the lead brakes at b_lead to v_low, holds for hold_s, then accelerates at the
IDM a back to v_eq. Nothing enters or exits (the road is long enough); the run lasts duration_s.

Metrics (definitions in design record section 7; simulator truth, allowed for evaluation):
  dv_i = v_eq - min_t v_i for vehicle index i = 0 (lead) .. n.
  g (main): slope of ln(dv_i) against i, fitted over vehicles with dv_i > DV_FLOOR and min v_i > V_STOP.
     At least MIN_POINTS vehicles must qualify, else g = NaN ("not estimable") with the reason.
  A = dv_n / dv_0 (last follower over lead), deficit at vehicle 10, stopped count (min v <= V_STOP),
  max deceleration of the followers, min net gap (all followers, all speeds), min time gap (net gap / v,
  only while v > V_TIMEGAP).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tmc.idm import IDMParams, equilibrium_gap, load_params
from tmc.network import FreewaySegment, ScenarioConfig

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "scenario_e9_setting_a.yaml"

DV_FLOOR = 0.01      # m/s, below this a deficit is numerical noise (design record section 7)
V_STOP = 0.5         # m/s, a vehicle whose min speed is <= this counts as stopped
MIN_POINTS = 5       # vehicles needed to fit g
V_TIMEGAP = 5.0      # m/s, time gaps are evaluated only above this speed


@dataclass(frozen=True)
class Disturbance:
    name: str
    v_low: float          # m/s, speed the lead brakes to
    b_lead: float         # m/s^2, lead braking rate (positive number)
    hold_s: float         # s at v_low before recovering
    t_start: float        # s, braking starts


class ScriptedLead:
    """Lead acceleration as a function of (t, v): cruise, brake at b_lead to v_low, hold, recover at a to v_eq.
    Exact: the last braking / recovery step is shortened so the lead lands on v_low / v_eq, never past it."""

    def __init__(self, d: Disturbance, v_eq: float, a_recover: float, dt: float):
        self.d, self.v_eq, self.a, self.dt = d, v_eq, a_recover, dt
        self.t_low: float | None = None     # time the lead reached v_low

    def __call__(self, t: float, v: float) -> float:
        d = self.d
        if t < d.t_start - 1e-9:
            return 0.0
        if self.t_low is None:
            if v <= d.v_low + 1e-9:
                self.t_low = t
            else:
                return -min(d.b_lead, (v - d.v_low) / self.dt)
        if t < self.t_low + d.hold_s - 1e-9:
            return 0.0
        return min(self.a, max(0.0, (self.v_eq - v) / self.dt))


def load_setting(path: Path = CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    d["disturbances"] = {k: Disturbance(name=k, **{kk: float(vv) for kk, vv in v.items()})
                         for k, v in d["disturbances"].items()}
    return d


def platoon_config(s: dict, p: IDMParams, reaction_time_s: float, dt: float | None = None) -> ScenarioConfig:
    v_eq, n = float(s["v_eq_mps"]), int(s["n_followers"])
    spacing = float(equilibrium_gap(v_eq, p)) + p.length
    x0 = (n + 1) * spacing + 10.0
    dt = float(s["dt_s"]) if dt is None else dt
    return ScenarioConfig(
        scenario_id=s["scenario_id"], demand_id="none", param_set=s["param_set"], seed=None,
        length_m=1e7, dt_s=dt, record_every_s=dt, max_time_s=float(s["duration_s"]),
        arrival_process="uniform", demand_profile=[],
        initial_vehicles=[[x0 - i * spacing, v_eq] for i in range(n + 1)],
        reaction_time_s=reaction_time_s)


def run_platoon(s: dict, dist: Disturbance, reaction_time_s: float, dt: float | None = None,
                p: IDMParams | None = None):
    p = p or load_params(s["param_set"])
    cfg = platoon_config(s, p, reaction_time_s, dt)
    lead = ScriptedLead(dist, float(s["v_eq_mps"]), p.a, cfg.dt_s)
    res = FreewaySegment(p, cfg, lead_accel=lead).run()
    return res, p


def growth_rate(dv: np.ndarray, v_min: np.ndarray) -> tuple[float, str, int]:
    """(g, reason, n_points). g = NaN with a reason when fewer than MIN_POINTS vehicles qualify."""
    i = np.arange(len(dv))
    ok = (dv > DV_FLOOR) & (v_min > V_STOP)
    if ok.sum() < MIN_POINTS:
        why = []
        if (dv <= DV_FLOOR).sum():
            why.append(f"{int((dv <= DV_FLOOR).sum())} below {DV_FLOOR} m/s")
        if (v_min <= V_STOP).sum():
            why.append(f"{int((v_min <= V_STOP).sum())} stopped")
        return float("nan"), "not estimable: " + ", ".join(why), int(ok.sum())
    return float(np.polyfit(i[ok], np.log(dv[ok]), 1)[0]), "", int(ok.sum())


def platoon_metrics(res, p: IDMParams, v_eq: float) -> tuple[dict, pd.DataFrame]:
    tr = res.trajectories
    x = tr.pivot(index="time_s", columns="vehicle_id", values="position_m")
    v = tr.pivot(index="time_s", columns="vehicle_id", values="speed_mps")
    a = tr.pivot(index="time_s", columns="vehicle_id", values="acceleration_mps2")
    ids = list(v.columns)                      # 0 = lead, then followers upstream
    X, V, A = x[ids].to_numpy(), v[ids].to_numpy(), a[ids].to_numpy()
    v_min = V.min(axis=0)
    dv = v_eq - v_min
    gap = X[:, :-1] - X[:, 1:] - p.length      # follower i+1 behind i
    vf = V[:, 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        tgap = np.where(vf > V_TIMEGAP, gap / vf, np.inf)
    g, why, npts = growth_rate(dv, v_min)
    n = len(ids) - 1
    m = {
        "g_per_veh": g, "g_note": why, "g_points": npts,
        "A": float(dv[-1] / dv[0]) if dv[0] > 0 else float("nan"),
        "dv_lead": float(dv[0]), "dv_veh10": float(dv[min(10, n)]), "dv_last": float(dv[-1]),
        "n_stopped": int((v_min[1:] <= V_STOP).sum()),
        "max_decel_mps2": float(-A[:, 1:].min()),
        "min_gap_m": float(gap.min()),
        "min_time_gap_s": float(tgap.min()),
        "end_speed_last_mps": float(V[-1, -1]),
    }
    per_vehicle = pd.DataFrame({"vehicle_index": np.arange(len(ids)), "min_speed_mps": v_min, "deficit_mps": dv})
    return m, per_vehicle


def human_baseline(s: dict, reaction_times=(0.0, 0.6)) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, per = [], []
    for name, dist in s["disturbances"].items():
        for tr_s in reaction_times:
            res, p = run_platoon(s, dist, tr_s)
            m, pv = platoon_metrics(res, p, float(s["v_eq_mps"]))
            rows.append({"scenario_id": s["scenario_id"], "disturbance": name, "case": "human",
                         "p_equipped": 0.0, "reaction_time_s": tr_s, **m})
            per.append(pv.assign(disturbance=name, reaction_time_s=tr_s))
    return pd.DataFrame(rows), pd.concat(per, ignore_index=True)


def plot_deficits(per: pd.DataFrame, scenario_id: str, path: Path, threshold: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, (name, grp) in zip(axes, per.groupby("disturbance", sort=False)):
        for tr_s, g in grp.groupby("reaction_time_s"):
            ax.semilogy(g["vehicle_index"], g["deficit_mps"], marker="o", ms=3,
                        label=f"human, T_r = {tr_s:g} s")
        ax.set_title(f"{scenario_id}: disturbance '{name}'")
        ax.set_xlabel("vehicle index i (0 = lead)")
        ax.set_ylabel("speed deficit v_eq - min v_i (m/s, log)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"E9 setting A, human baseline (slope of ln deficit = g; CAV test threshold {threshold}/veh)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str]) -> None:
    s = load_setting()
    if "--quick" in argv:
        dist = s["disturbances"]["nominal"]
        res, p = run_platoon(s, dist, 0.6)
        m, _ = platoon_metrics(res, p, float(s["v_eq_mps"]))
        print(f"{s['scenario_id']} nominal, human T_r = 0.6 s:")
        for k, v in m.items():
            print(f"  {k}: {v}")
        return
    df, per = human_baseline(s)
    out = ROOT / "results"
    df.to_csv(out / "E9A_human_baseline.csv", index=False)
    per.to_csv(out / "E9A_human_baseline_per_vehicle.csv", index=False)
    plot_deficits(per, s["scenario_id"], out / "E9A_human_baseline_deficits.png", s["g_threshold_per_veh"])
    pd.set_option("display.width", 200)
    print(df.drop(columns=["scenario_id", "case", "p_equipped"]).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
