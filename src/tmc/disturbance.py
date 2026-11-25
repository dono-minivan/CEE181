"""E9 setting A: a braking disturbance travelling up a single-lane platoon (experiment log E9).

    python -m tmc.disturbance                # human baseline: T_r = 0 and 0.6 s, both disturbances
    python -m tmc.disturbance --quick        # one run, prints the metrics (small example, CLAUDE.md rule 7)
    python -m tmc.disturbance --matrix       # E9 CAV matrix, seeds 181-184 -> results/E9A_matrix_*.csv, figures
    python -m tmc.disturbance --matrix --holdout   # confirmation runs (nominal group) on seeds 185-188
    python -m tmc.disturbance --matrix --set e9d   # E9d Ploeg extension -> results/E9D_matrix_*.csv, E9D_*.png
    options: --workers N (default: CPUs - 2), --groups nominal,degraded,sweep,tau

Setting (configs/scenario_e9_setting_a.yaml): one lane, a scripted lead (vehicle 0) and n followers in IDM
equilibrium at v_eq. At t_start the lead brakes at b_lead to v_low, holds for hold_s, then accelerates at the
IDM a back to v_eq. Nothing enters or exits (the road is long enough); the run lasts duration_s.

Metrics (definitions in design record section 7; simulator truth, allowed for evaluation):
  dv_i = v_eq - min_t v_i for vehicle index i = 0 (lead) .. n.
  g (main): slope of ln(dv_i) against i, fitted over vehicles with dv_i > DV_FLOOR and min v_i > V_STOP.
     At least MIN_POINTS vehicles must qualify, else g = NaN ("not estimable") with the reason.
  A = dv_n / dv_0 (last follower over lead), deficit at vehicle 10, stopped count (min v <= V_STOP),
  max deceleration of the followers, min net gap (all followers, all speeds), min time gap (net gap / v,
  only while v > V_TIMEGAP). With CAVs, the same gap/deceleration metrics split by follower type (equipped,
  human, human directly behind an equipped vehicle) and CACC fallback counts by reason.

Judgement (judge_matrix) applies the design record section 7 criteria as pre-registered: "disturbance reduced"
(nominal disturbance only), "gaps acceptable" (both disturbances), "robust communication".
"""
from __future__ import annotations

import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tmc.cav import delay_ticks
from tmc.idm import IDMParams, equilibrium_gap, load_params
from tmc.network import FreewaySegment, ScenarioConfig

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "scenario_e9_setting_a.yaml"

DV_FLOOR = 0.01      # m/s, below this a deficit is numerical noise (design record section 7)
V_STOP = 0.5         # m/s, a vehicle whose min speed is <= this counts as stopped
MIN_POINTS = 5       # vehicles needed to fit g
V_TIMEGAP = 5.0      # m/s, time gaps are evaluated only above this speed
GAP_MIN_M = 1.0      # m, gap criterion (design record section 7)
TIME_GAP_MIN_S = 0.5 # s, time-gap criterion, evaluated while v > V_TIMEGAP
DECEL_MAX = 4.0      # m/s^2, max deceleration of equipped vehicles (= b_safe)
SEEDS_TO_PASS = 3    # of 4 seeds


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


def equipped_mask(cav: dict | None, n_vehicles: int) -> np.ndarray:
    """The network's equipped draw (FreewaySegment._init_cav), reproduced to place each vehicle at its own
    controller's equilibrium gap. run_platoon checks it against the simulated assignment."""
    if not cav or float(cav.get("p_equipped", 0.0)) <= 0.0:
        return np.zeros(n_vehicles, dtype=bool)
    return np.random.default_rng(cav.get("placement_seed")).random(n_vehicles) < float(cav["p_equipped"])


def follower_gap(v: float, p: IDMParams, equipped: bool, cav: dict | None) -> float:
    """Equilibrium net gap at speed v: IDM s_e for humans and IDM-gap controllers; r + h v for a Ploeg
    controller with the constant time-headway policy (E9d PL-short)."""
    if equipped and cav and cav.get("controller") == "ploeg" and cav.get("spacing", "ctg") == "ctg":
        return float(cav.get("r_m", 2.0)) + float(cav.get("h_s", 1.0)) * v
    return float(equilibrium_gap(v, p))


def platoon_config(s: dict, p: IDMParams, reaction_time_s: float, dt: float | None = None,
                   cav: dict | None = None) -> ScenarioConfig:
    v_eq, n = float(s["v_eq_mps"]), int(s["n_followers"])
    eq = equipped_mask(cav, n + 1)
    x = [0.0]
    for i in range(1, n + 1):          # follower i sits at its own equilibrium gap behind vehicle i - 1
        x.append(x[-1] - follower_gap(v_eq, p, bool(eq[i]), cav) - p.length)
    x0 = -x[-1] + 10.0
    dt = float(s["dt_s"]) if dt is None else dt
    return ScenarioConfig(
        scenario_id=s["scenario_id"], demand_id="none", param_set=s["param_set"], seed=None,
        length_m=1e7, dt_s=dt, record_every_s=dt, max_time_s=float(s["duration_s"]),
        arrival_process="uniform", demand_profile=[],
        initial_vehicles=[[x0 + xi, v_eq] for xi in x],
        reaction_time_s=reaction_time_s, cav=cav)


def run_platoon(s: dict, dist: Disturbance, reaction_time_s: float, dt: float | None = None,
                p: IDMParams | None = None, cav: dict | None = None):
    """One setting A run. reaction_time_s applies to human vehicles; cav (network cav config) adds
    equipped vehicles, which have no reaction delay."""
    p = p or load_params(s["param_set"])
    cfg = platoon_config(s, p, reaction_time_s, dt, cav)
    lead = ScriptedLead(dist, float(s["v_eq_mps"]), p.a, cfg.dt_s)
    res = FreewaySegment(p, cfg, lead_accel=lead).run()
    if cav and "equipped" in res.vehicles:
        sim = res.vehicles.sort_values("vehicle_id")["equipped"].to_numpy(dtype=bool)
        if not np.array_equal(sim, equipped_mask(cav, len(sim))):
            raise RuntimeError("equipped assignment differs from the one used to place the platoon")
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


def platoon_metrics(res, p: IDMParams, v_eq: float, by_type: bool = False) -> tuple[dict, pd.DataFrame]:
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
    veq = res.vehicles.set_index("vehicle_id")["equipped"] if "equipped" in res.vehicles else None
    eq = np.array([bool(veq[i]) for i in ids]) if veq is not None else np.zeros(len(ids), dtype=bool)
    if by_type:
        m.update(_type_metrics(eq, A, gap, tgap))
    per_vehicle = pd.DataFrame({"vehicle_index": np.arange(len(ids)), "min_speed_mps": v_min, "deficit_mps": dv,
                                "equipped": eq})
    return m, per_vehicle


def _type_metrics(eq: np.ndarray, A: np.ndarray, gap: np.ndarray, tgap: np.ndarray) -> dict:
    """Gap and deceleration metrics split by follower type. Column j of gap/tgap is follower j+1 behind j."""
    fol_eq, lead_eq = eq[1:], eq[:-1]
    groups = {"equipped": fol_eq, "human": ~fol_eq, "human_behind_cav": (~fol_eq) & lead_eq}
    out = {"n_equipped": int(eq.sum()), "lead_equipped": bool(eq[0])}
    nan = float("nan")
    for k, sel in groups.items():
        out[f"n_fol_{k}"] = int(sel.sum())
        out[f"max_decel_{k}"] = float(-A[:, 1:][:, sel].min()) if sel.any() else nan
        out[f"min_gap_{k}"] = float(gap[:, sel].min()) if sel.any() else nan
        out[f"min_time_gap_{k}"] = float(tgap[:, sel].min()) if sel.any() else nan
    return out


def fallback_metrics(res, outages=()) -> dict:
    """CACC fallback counts (controller ticks) by reason. With outages: 'stale' ticks inside the outage
    window (plus the max-age allowance after it) and the last 'stale' tick after the outage ended."""
    ev = res.cav_events
    out = {f"fb_{r}": int((ev["reason"] == r).sum()) if len(ev) else 0
           for r in ("leader_unequipped", "no_message", "stale", "comms_off")}
    out["fb_vehicles"] = int(ev["vehicle_id"].nunique()) if len(ev) else 0
    if outages:
        start, end = min(a for a, _ in outages), max(b for _, b in outages)
        st = ev.loc[ev["reason"] == "stale", "time_s"] if len(ev) else pd.Series(dtype=float)
        out["fb_stale_in_outage"] = int(((st >= start - 1e-9) & (st <= end + 1e-9)).sum())
        after = st[st > end + 1e-9]
        out["last_stale_after_outage_s"] = float(after.max()) if len(after) else float("nan")
    return out


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


# --------------------------------------------------------------------------- E9 CAV matrix

def matrix_cases(s: dict, groups=None) -> list[dict]:
    """Expand the config's matrix block into cases: case_id, group, mode, p and the network cav keys."""
    mx = s["matrix"]
    nom = mx["nominal_cacc"]
    g = mx["groups"]
    want = set(groups) if groups else set(g)
    unknown = want - set(g)
    if unknown:
        raise ValueError(f"unknown matrix groups {sorted(unknown)}; known: {sorted(g)}")
    cases = []

    def add(group, mode, p, variant="", **kw):
        base = {"tau_s": float(nom["tau_s"]), "delay_s": float(nom["delay_s"]), "max_age_ticks": 5,
                "loss_prob": 0.0, "outages": []}
        base.update(kw)
        cid = f"{group}_{mode}_p{p:g}" + (f"_{variant}" if variant else "")
        cases.append({"case_id": cid, "group": group, "mode": mode, "p_equipped": float(p),
                      "variant": variant or "nominal", **base})

    if "nominal" in want:
        for mode in g["nominal"]["modes"]:
            for p in g["nominal"]["p"]:
                add("nominal", mode, p)
    if "degraded" in want:
        for p in g["degraded"]["p"]:
            for name, kw in g["degraded"]["variants"].items():
                kw = dict(kw)
                if "outages" in kw:
                    kw["outages"] = [tuple(map(float, o)) for o in kw["outages"]]
                for k in ("delay_s", "loss_prob"):
                    if k in kw:
                        kw[k] = float(kw[k])
                add("degraded", "cacc", p, name, **kw)
    if "sweep" in want:
        sw = g["sweep"]
        for p in sw["p"]:
            for d in sw["delays_s"]:
                add("sweep", "cacc", p, f"delay_{float(d):g}_nofallback", delay_s=float(d),
                    max_age_ticks=sw["max_age_ticks"])
    if "tau" in want:
        ta = g["tau"]
        for mode in ta["modes"]:
            for p in ta["p"]:
                add("tau", mode, p, f"tau_{float(ta['tau_s']):g}", tau_s=float(ta["tau_s"]))
    return cases


def e9d_cases(s: dict, groups=None) -> list[dict]:
    """E9d cases (Ploeg controller) from the config's matrix_e9d block. mode = feed-forward source."""
    mx = s["matrix_e9d"]
    nom = mx["nominal_cacc"]
    g = mx["groups"]
    want = set(groups) if groups else set(g)
    unknown = want - set(g)
    if unknown:
        raise ValueError(f"unknown matrix_e9d groups {sorted(unknown)}; known: {sorted(g)}")
    cases = []

    def add(group, ff, p, gcfg, variant="", **kw):
        c = {"case_id": f"{group}_{ff}_p{p:g}" + (f"_{variant}" if variant else ""), "group": group, "mode": ff,
             "p_equipped": float(p), "variant": variant or "nominal", "tau_s": float(nom["tau_s"]),
             "delay_s": float(nom["delay_s"]), "max_age_ticks": 5, "loss_prob": 0.0, "outages": [],
             "controller": "ploeg", "ff": ff, "spacing": gcfg["spacing"],
             "h_s": float(gcfg.get("h_s", float("nan"))), "r_m": float(gcfg.get("r_m", float("nan")))}
        c.update(kw)
        cases.append(c)

    for name in ("pl_idm", "pl_short"):
        if name in want:
            for ff in g[name]["ff"]:
                for p in g[name]["p"]:
                    add(name, ff, p, g[name])
    if "pl_sweep" in want:
        sw = g["pl_sweep"]
        for p in sw["p"]:
            for d in sw["delays_s"]:
                add("pl_sweep", "v2v", p, sw, f"delay_{float(d):g}_nofallback", delay_s=float(d),
                    max_age_ticks=sw["max_age_ticks"])
    return cases


def case_cav_config(case: dict, seed: int) -> dict:
    """Network cav config for one case and seed (placement seed = loss seed = run seed)."""
    c = {"p_equipped": case["p_equipped"], "mode": case["mode"], "placement_seed": int(seed),
         "tau_s": case["tau_s"], "delay_s": case["delay_s"], "max_age_ticks": case["max_age_ticks"],
         "loss_prob": case["loss_prob"], "loss_seed": int(seed),
         "outages": [list(o) for o in case["outages"]]}
    if case.get("controller") == "ploeg":
        c.update({"controller": "ploeg", "ff": case["ff"], "spacing": case["spacing"]})
        if case["spacing"] == "ctg":
            c.update({"h_s": case["h_s"], "r_m": case["r_m"]})
    return c


HUMAN_CASE = {"case_id": "human", "group": "human", "mode": "human", "p_equipped": 0.0, "variant": "nominal",
              "tau_s": float("nan"), "delay_s": float("nan"), "max_age_ticks": None, "loss_prob": 0.0,
              "outages": []}


def _run_one(job: tuple) -> tuple[dict, pd.DataFrame]:
    """Worker: one (case, disturbance, seed) run. case None = human baseline (deterministic)."""
    case, dist_name, seed = job
    s = load_setting()
    dist = s["disturbances"][dist_name]
    tr_s = float(s["matrix"]["human_reaction_time_s"])
    cav = None if case is None else case_cav_config(case, seed)
    info = dict(HUMAN_CASE if case is None else case)
    outages = tuple(info["outages"])
    info["outages"] = str([list(o) for o in outages])
    info["outage_end_s"] = max(b for _, b in outages) if outages else float("nan")
    base = {"scenario_id": s["scenario_id"], "disturbance": dist_name, "seed": int(seed),
            "reaction_time_human_s": tr_s, **info}
    try:
        res, p = run_platoon(s, dist, tr_s, cav=cav)
    except RuntimeError as err:          # the simulator stops at an overlap: record it as the run's outcome
        hit = re.search(r"collision/overlap at t=([\d.]+)s \(min gap (-?[\d.]+) m\)", str(err))
        if hit is None:
            raise
        nan = float("nan")
        row = {**base, "collided": True, "collision_t_s": float(hit.group(1)), "g_per_veh": nan,
               "g_note": f"not estimable: collision at t = {hit.group(1)} s", "g_points": 0, "A": nan,
               "n_stopped": -1, "max_decel_mps2": nan, "min_gap_m": float(hit.group(2)), "min_time_gap_s": nan,
               "max_decel_equipped": nan}
        return row, pd.DataFrame(columns=["vehicle_index", "min_speed_mps", "deficit_mps", "equipped",
                                          "case_id", "disturbance", "seed"])
    m, pv = platoon_metrics(res, p, float(s["v_eq_mps"]), by_type=True)
    row = {**base, "collided": False, "collision_t_s": float("nan"), **m, **fallback_metrics(res, outages)}
    pv = pv.assign(case_id=row["case_id"], disturbance=dist_name, seed=int(seed))
    return row, pv


def run_matrix(s: dict, seeds, groups=None, workers: int | None = None,
               which: str = "e9c") -> tuple[pd.DataFrame, pd.DataFrame]:
    """All cases x both disturbances x seeds, plus the human baseline once per disturbance (deterministic;
    its row is repeated for every seed so seed-paired comparisons can join on seed). which: 'e9c' (Eq. 11.26
    matrix) or 'e9d' (Ploeg extension)."""
    cases = e9d_cases(s, groups) if which == "e9d" else matrix_cases(s, groups)
    dists = list(s["disturbances"])
    jobs = [(None, d, seeds[0]) for d in dists] + [(c, d, sd) for c in cases for d in dists for sd in seeds]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    if workers == 1:
        out = [_run_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(_run_one, jobs, chunksize=1))
    rows = [r for r, _ in out]
    for h in [r for r in rows if r["case_id"] == "human"]:
        rows += [dict(h, seed=int(sd)) for sd in seeds[1:]]
    df = pd.DataFrame(rows).sort_values(["group", "case_id", "disturbance", "seed"], ignore_index=True)
    return df, pd.concat([pv for _, pv in out], ignore_index=True)


def _reduced(cand: pd.Series, ref: pd.Series, thr: float) -> bool:
    """One seed: g lower by >= thr; if either g is not estimable, A at least 10% lower instead."""
    if np.isfinite(cand["g_per_veh"]) and np.isfinite(ref["g_per_veh"]):
        return bool(ref["g_per_veh"] - cand["g_per_veh"] >= thr - 1e-12)
    return bool(cand["A"] <= 0.9 * ref["A"])


def _gaps_ok(c: pd.DataFrame) -> pd.Series:
    return ((c["min_gap_m"] >= GAP_MIN_M) & (c["min_time_gap_s"] >= TIME_GAP_MIN_S)
            & ~(c["max_decel_equipped"] > DECEL_MAX))


def judge_matrix(df: pd.DataFrame, thr: float, tick_s: float = 0.1) -> pd.DataFrame:
    """Design record section 7 criteria, applied mechanically. One row per (criterion, case, disturbance).
    pass = None where no pass/fail was registered (reported only)."""
    out = []
    key = ["disturbance", "seed"]
    by_case = {cid: c.set_index(key) for cid, c in df.groupby("case_id")}
    human = by_case["human"]

    def nominal_of(cid):
        c = by_case.get(cid)
        return None if c is None else c.xs("nominal", level="disturbance")

    # disturbance reduced (nominal disturbance): CACC vs human AND vs ACC, same p and seed; ACC vs human reported
    for p in sorted(df.loc[df["group"] == "nominal", "p_equipped"].unique()):
        acc = nominal_of(f"nominal_acc_p{p:g}")
        for mode in ("acc", "cacc"):
            sub = nominal_of(f"nominal_{mode}_p{p:g}")
            if sub is None:
                continue
            vs_h = [_reduced(sub.loc[sd], human.loc[("nominal", sd)], thr) for sd in sub.index]
            vs_a = [_reduced(sub.loc[sd], acc.loc[sd], thr) for sd in sub.index] if mode == "cacc" else None
            both = [a and b for a, b in zip(vs_h, vs_a)] if vs_a else vs_h
            out.append({"criterion": "disturbance_reduced" if mode == "cacc" else "acc_vs_human (reported)",
                        "case_id": f"nominal_{mode}_p{p:g}", "disturbance": "nominal", "seeds": len(sub),
                        "seeds_vs_human": int(sum(vs_h)), "seeds_vs_acc": int(sum(vs_a)) if vs_a else None,
                        "seeds_pass": int(sum(both)), "g_mean": float(sub["g_per_veh"].mean()),
                        "pass": bool(sum(both) >= min(SEEDS_TO_PASS, len(sub))) if mode == "cacc" else None})
    # gaps acceptable: every CAV case, both disturbances, every seed
    for (cid, dist), c in df[df["case_id"] != "human"].groupby(["case_id", "disturbance"]):
        ok = _gaps_ok(c)
        out.append({"criterion": "gaps_acceptable", "case_id": cid, "disturbance": dist, "seeds": len(c),
                    "seeds_pass": int(ok.sum()), "min_gap_m": float(c["min_gap_m"].min()),
                    "min_time_gap_s": float(c["min_time_gap_s"].min()),
                    "max_decel_equipped": float(c["max_decel_equipped"].max()),
                    "min_gap_human_behind_cav": float(c["min_gap_human_behind_cav"].min()),
                    "pass": bool(ok.all())})
    # robust communication (nominal disturbance): degraded CACC vs ACC at the same p and seed
    for p in sorted(df.loc[df["group"] == "degraded", "p_equipped"].unique()):
        acc = nominal_of(f"nominal_acc_p{p:g}")
        if acc is None:
            continue
        for cid in sorted(df.loc[(df["group"] == "degraded") & (df["p_equipped"] == p), "case_id"].unique()):
            c = nominal_of(cid)
            var = cid.split(f"_p{p:g}_", 1)[1]
            dg = c["g_per_veh"] - acc.loc[c.index, "g_per_veh"]
            row = {"criterion": f"robust_comm_{var}", "case_id": cid, "disturbance": "nominal", "seeds": len(c),
                   "g_mean": float(c["g_per_veh"].mean()), "g_minus_acc_min": float(dg.min()),
                   "g_minus_acc_max": float(dg.max()), "fb_stale_total": int(c["fb_stale"].sum()),
                   "gaps_ok": bool(_gaps_ok(c).all())}
            if var.startswith("delay_") and float(var.split("_")[1]) < 0.5:     # 0.3 s: no worse than ACC
                row["pass"] = bool((dg <= thr + 1e-12).all() and row["gaps_ok"])
            elif var.startswith("delay_"):                                      # 1.0 s: falls back, = ACC
                row["pass"] = bool((c["fb_stale"] > 0).all() and (dg.abs() <= thr + 1e-12).all())
            elif var.startswith("outage"):                                      # falls back, recovers
                # after the outage, the first new message is usable delay_ticks later; until then the
                # newest usable one is stale, so the last stale tick may be end + (delay_ticks - 1) ticks
                end = float(c["outage_end_s"].iloc[0])
                allow = end + (delay_ticks(float(c["delay_s"].iloc[0]), tick_s) - 1) * tick_s + 1e-6
                last = c["last_stale_after_outage_s"]
                row["last_stale_after_outage_s"] = float(last.max())
                row["pass"] = bool((c["fb_stale_in_outage"] > 0).all() and (last.isna() | (last <= allow)).all()
                                   and (dg.abs() <= thr + 1e-12).all() and row["gaps_ok"])
            else:
                row["pass"] = None          # 30% loss: no pass/fail registered, reported against prediction 6
            out.append(row)
    return pd.DataFrame(out)


def plot_matrix(df: pd.DataFrame, per: pd.DataFrame, out: Path, thr: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sid = df["scenario_id"].iloc[0]
    col = {"human": "0.3", "acc": "tab:blue", "cacc": "tab:orange"}
    hum = {d: df[(df["case_id"] == "human") & (df["disturbance"] == d)]["g_per_veh"].iloc[0]
           for d in df["disturbance"].unique()}
    # 1. g vs p by controller
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, dist in zip(axes, ("nominal", "strong")):
        h = hum[dist]
        ax.axhline(h, color=col["human"], lw=1, label=f"human (T_r = 0.6 s), g = {h:.4f}")
        if dist == "nominal":
            ax.axhline(h - thr, color=col["human"], ls=":", lw=1, label=f"human - {thr} (criterion)")
        ax.axhline(0, color="k", lw=0.5)
        for k, mode in enumerate(("acc", "cacc")):
            c = df[(df["group"] == "nominal") & (df["mode"] == mode) & (df["disturbance"] == dist)]
            xo = (k - 0.5) * 0.03
            ax.scatter(c["p_equipped"] + xo, c["g_per_veh"], s=12, color=col[mode], alpha=0.5)
            m = c.groupby("p_equipped")["g_per_veh"].mean()
            ax.plot(np.r_[0, m.index] + xo, np.r_[h, m.values], marker="o", color=col[mode],
                    label=f"{mode.upper()} (line = seed mean, dots = seeds)")
        ax.set_xlabel("penetration p (share of equipped vehicles)")
        ax.set_ylabel("growth rate g (1/veh)")
        ax.set_title(f"{sid}, disturbance '{dist}'" + (" (stress case)" if dist == "strong" else ""), fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "E9A_matrix_g_vs_p.png", dpi=150)
    plt.close(fig)
    # 2. per-vehicle deficits, nominal disturbance, first seed
    sd = int(df["seed"].min())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, p in zip(axes, (0.25, 0.5, 1.0)):
        for cid, lab, c in (("human", "human", col["human"]), (f"nominal_acc_p{p:g}", "ACC", col["acc"]),
                            (f"nominal_cacc_p{p:g}", "CACC", col["cacc"])):
            g = per[(per["case_id"] == cid) & (per["disturbance"] == "nominal") & (per["seed"] == sd)]
            if g.empty:
                continue
            ax.semilogy(g["vehicle_index"], g["deficit_mps"], marker=".", color=c, label=lab)
            e = g[g["equipped"]]
            ax.semilogy(e["vehicle_index"], e["deficit_mps"], ls="none", marker="s", mfc="none", ms=6, color=c)
        ax.set_title(f"p = {p:g}, seed {sd} (squares = equipped)", fontsize=9)
        ax.set_xlabel("vehicle index i (0 = lead)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("speed deficit v_eq - min v_i (m/s, log)")
    fig.suptitle(f"{sid}, nominal disturbance: per-vehicle deficits (slope of the log = g)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "E9A_matrix_deficits.png", dpi=150)
    plt.close(fig)
    # 3. delay sweep (p = 1, fallback disabled)
    sw = df[(df["group"] == "sweep") & (df["disturbance"] == "nominal")]
    if sw.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    m = sw.groupby("delay_s")["g_per_veh"].mean()
    ax.plot(m.index, m.values, marker="o", color=col["cacc"], label="CACC, p = 1, fallback disabled")
    nc = df[(df["case_id"] == "nominal_cacc_p1") & (df["disturbance"] == "nominal")]["g_per_veh"].mean()
    a = df[(df["case_id"] == "nominal_acc_p1") & (df["disturbance"] == "nominal")]["g_per_veh"].mean()
    ax.scatter([0.2], [nc], marker="*", s=80, color=col["cacc"], zorder=3,
               label=f"nominal CACC (0.15 s -> 2 ticks, fallback on), g = {nc:.4f}")
    ax.axhline(a, color=col["acc"], ls="--", label=f"ACC, p = 1, g = {a:.4f}")
    ax.axhline(hum["nominal"], color=col["human"], lw=1, label=f"human, g = {hum['nominal']:.4f}")
    ax.set_xlabel("effective message delay (s; multiples of the 0.1 s tick)")
    ax.set_ylabel("growth rate g (1/veh)")
    ax.set_title(f"{sid}, nominal disturbance: delay sweep", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "E9A_matrix_delay_sweep.png", dpi=150)
    plt.close(fig)


def judge_e9d(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    """E9d criteria as registered: 'disturbance reduced' (v2v vs human AND vs sensor at the same spacing
    policy and p, nominal disturbance, >= 3 of 4 seeds), the positive control (prediction 1: none above v2v
    by >= thr, every seed, nominal), and 'gaps acceptable' (every case, both disturbances)."""
    out = []
    key = ["disturbance", "seed"]
    by_case = {cid: c.set_index(key) for cid, c in df.groupby("case_id")}
    human = by_case["human"]

    def nom(cid):
        c = by_case.get(cid)
        return None if c is None else c.xs("nominal", level="disturbance")

    for grp in ("pl_idm", "pl_short"):
        for p in sorted(df.loc[df["group"] == grp, "p_equipped"].unique()):
            v2v, sen, none = (nom(f"{grp}_{ff}_p{p:g}") for ff in ("v2v", "sensor", "none"))
            if v2v is None:
                continue
            seeds = list(v2v.index)
            if sen is not None:
                vs_h = [_reduced(v2v.loc[sd], human.loc[("nominal", sd)], thr) for sd in seeds]
                vs_s = [_reduced(v2v.loc[sd], sen.loc[sd], thr) for sd in seeds]
                both = [a and b for a, b in zip(vs_h, vs_s)]
                out.append({"criterion": "disturbance_reduced", "case_id": f"{grp}_v2v_p{p:g}",
                            "disturbance": "nominal", "seeds": len(seeds), "seeds_vs_human": int(sum(vs_h)),
                            "seeds_vs_sensor": int(sum(vs_s)), "seeds_pass": int(sum(both)),
                            "g_v2v": float(v2v["g_per_veh"].mean()), "g_sensor": float(sen["g_per_veh"].mean()),
                            "pass": bool(sum(both) >= min(SEEDS_TO_PASS, len(seeds)))})
            if none is not None and p == 1.0:
                d = none["g_per_veh"] - v2v.loc[none.index, "g_per_veh"]
                n_coll = int(none["collided"].sum()) if "collided" in none else 0
                row = {"criterion": "positive_control (prediction 1)", "case_id": f"{grp}_none_p{p:g}",
                       "disturbance": "nominal", "seeds": len(d), "g_none": float(none["g_per_veh"].mean()),
                       "g_v2v": float(v2v["g_per_veh"].mean()), "none_minus_v2v_min": float(d.min()),
                       "none_collided_seeds": n_coll}
                if d.notna().all():
                    row["pass"] = bool((d >= thr - 1e-12).all())
                else:   # a collision ends the run, so g (and A) cannot be computed: the g test is not evaluable
                    row["pass"] = None
                    row["note"] = f"not evaluable by the g test: no feed-forward collided in {n_coll} of {len(d)} seeds"
                out.append(row)
    for (cid, dist), c in df[df["case_id"] != "human"].groupby(["case_id", "disturbance"]):
        ok = _gaps_ok(c)
        out.append({"criterion": "gaps_acceptable", "case_id": cid, "disturbance": dist, "seeds": len(c),
                    "seeds_pass": int(ok.sum()), "min_gap_m": float(c["min_gap_m"].min()),
                    "min_time_gap_s": float(c["min_time_gap_s"].min()),
                    "max_decel_equipped": float(c["max_decel_equipped"].max()), "pass": bool(ok.all())})
    return pd.DataFrame(out)


def plot_e9d(df: pd.DataFrame, per: pd.DataFrame, out: Path, thr: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sid = df["scenario_id"].iloc[0]
    col = {"none": "tab:red", "sensor": "tab:blue", "v2v": "tab:orange", "human": "0.3"}
    lab = {"none": "no feed-forward (Ploeg ACC)", "sensor": "sensor feed-forward (ACC)", "v2v": "V2V feed-forward (CACC)"}
    hum = {d: df[(df["case_id"] == "human") & (df["disturbance"] == d)]["g_per_veh"].iloc[0]
           for d in df["disturbance"].unique()}
    # 1. g by feed-forward source, per spacing policy and p
    panels = [("pl_idm", 0.5, "IDM gap, p = 0.5"), ("pl_idm", 1.0, "IDM gap, p = 1"),
              ("pl_short", 1.0, "short gap h = 1.0 s, p = 1")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, dist in zip(axes, ("nominal", "strong")):
        for k, ff in enumerate(("none", "sensor", "v2v")):
            xs, ys = [], []
            for j, (grp, p, _) in enumerate(panels):
                c = df[(df["case_id"] == f"{grp}_{ff}_p{p:g}") & (df["disturbance"] == dist)]
                if c.empty:
                    continue
                x = j + (k - 1) * 0.22
                n_coll = int(c["collided"].sum()) if "collided" in c else 0
                if n_coll:                      # a collision ends the run: no g, marked at the top edge
                    ax.annotate(f"collided {n_coll}/{len(c)}", (x, 1.0), xycoords=("data", "axes fraction"),
                                ha="center", va="top", fontsize=7, color=col[ff])
                    c = c[~c["collided"].astype(bool)]
                    if c.empty:
                        continue
                ax.scatter(np.full(len(c), x), c["g_per_veh"], s=10, color=col[ff], alpha=0.5)
                xs.append(x)
                ys.append(c["g_per_veh"].mean())
            ax.scatter(xs, ys, marker="D", s=40, color=col[ff], label=lab[ff])
        ax.axhline(hum[dist], color=col["human"], lw=1, label=f"human (T_r = 0.6 s), g = {hum[dist]:.4f}")
        if dist == "nominal":
            ax.axhline(hum[dist] - thr, color=col["human"], ls=":", lw=1, label=f"human - {thr}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(len(panels)), [t for _, _, t in panels], fontsize=8)
        ax.set_xlim(-0.6, len(panels) - 0.4)
        ax.set_ylabel("growth rate g (1/veh)")
        ax.set_title(f"{sid}, Ploeg controller, disturbance '{dist}'" + (" (stress)" if dist == "strong" else ""),
                     fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "E9D_g_by_feedforward.png", dpi=150)
    plt.close(fig)
    # 2. per-vehicle deficits at p = 1, nominal, first seed
    sd = int(df["seed"].min())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, grp, title in zip(axes, ("pl_idm", "pl_short"), ("IDM gap", "short gap h = 1.0 s")):
        g = per[(per["case_id"] == "human") & (per["disturbance"] == "nominal") & (per["seed"] == sd)]
        ax.semilogy(g["vehicle_index"], g["deficit_mps"], marker=".", color=col["human"], label="human")
        for ff in ("none", "sensor", "v2v"):
            g = per[(per["case_id"] == f"{grp}_{ff}_p1") & (per["disturbance"] == "nominal") & (per["seed"] == sd)]
            if g.empty:                         # collided: no completed run to plot
                ax.plot([], [], color=col[ff], label=f"{lab[ff]}: collided, not plotted")
                continue
            ax.semilogy(g["vehicle_index"], g["deficit_mps"], marker=".", color=col[ff], label=lab[ff])
        ax.set_title(f"{title}, p = 1, nominal disturbance", fontsize=9)
        ax.set_xlabel("vehicle index i (0 = lead)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("speed deficit v_eq - min v_i (m/s, log)")
    fig.suptitle(f"{sid}, E9d: per-vehicle deficits (slope of the log = g)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "E9D_deficits.png", dpi=150)
    plt.close(fig)
    # 3. delay sweep (PL-short, p = 1, fallback disabled)
    sw = df[(df["group"] == "pl_sweep") & (df["disturbance"] == "nominal")]
    if sw.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    m = sw.groupby("delay_s")["g_per_veh"].mean()
    ax.plot(m.index, m.values, marker="o", color=col["v2v"], label="CACC (V2V), fallback disabled")
    for ff in ("sensor", "none"):
        c = df[(df["case_id"] == f"pl_short_{ff}_p1") & (df["disturbance"] == "nominal")]
        if c.empty:
            continue
        if c["collided"].astype(bool).all():
            ax.plot([], [], color=col[ff], ls="--", label=f"{lab[ff]}: collided in every run")
            continue
        ax.axhline(c["g_per_veh"].mean(), color=col[ff], ls="--", label=f"{lab[ff]}, g = {c['g_per_veh'].mean():.4f}")
    ax.axhline(hum["nominal"], color=col["human"], lw=1, label=f"human, g = {hum['nominal']:.4f}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("effective message delay (s; multiples of the 0.1 s tick)")
    ax.set_ylabel("growth rate g (1/veh)")
    ax.set_title(f"{sid}, Ploeg controller, short gap h = 1.0 s, p = 1: delay sweep", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "E9D_delay_sweep.png", dpi=150)
    plt.close(fig)


def _arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def matrix_main(s: dict, argv: list[str]) -> None:
    holdout = "--holdout" in argv
    which = _arg(argv, "--set", "e9c")
    if which not in ("e9c", "e9d"):
        raise ValueError(f"--set must be e9c or e9d, got {which!r}")
    seeds = [int(x) for x in s["seeds_holdout" if holdout else "seeds"]]
    groups = _arg(argv, "--groups")
    default_holdout = ["pl_idm"] if which == "e9d" else ["nominal"]
    groups = groups.split(",") if groups else (default_holdout if holdout else None)
    workers = _arg(argv, "--workers")
    df, per = run_matrix(s, seeds, groups, int(workers) if workers else None, which)
    out = ROOT / "results"
    base = "E9D_matrix" if which == "e9d" else "E9A_matrix"
    tag = f"{base}_holdout" if holdout else base
    thr = float(s["g_threshold_per_veh"])
    df.to_csv(out / f"{tag}_runs.csv", index=False)
    per.to_csv(out / f"{tag}_per_vehicle.csv", index=False)
    jd = judge_e9d(df, thr) if which == "e9d" else judge_matrix(df, thr)
    jd.to_csv(out / f"{tag}_judgement.csv", index=False)
    if not holdout:
        (plot_e9d if which == "e9d" else plot_matrix)(df, per, out, thr)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 30)
    cols = ["g_per_veh", "A", "n_stopped", "max_decel_equipped", "max_decel_human", "min_gap_m", "min_time_gap_s",
            "min_gap_human_behind_cav", "fb_stale"]
    print(df.groupby(["case_id", "disturbance"])[cols].agg(["mean", "min", "max"]).round(4).to_string())
    print(jd.to_string(index=False))


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
    if "--matrix" in argv:
        matrix_main(s, argv)
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
