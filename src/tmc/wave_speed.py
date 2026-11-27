"""Wave-speed check (experiment log E10): simulated braking-wave speed vs the LWR shock speed.

    python -m tmc.wave_speed            # writes results/E10_wave_speed.csv and results/E10_wave_speed.png

Setting A platoon (tmc.disturbance: 30 followers at 15 m/s, scripted lead brakes to v_low, holds, recovers).
The LWR reference uses the IDM equilibrium as the fundamental diagram:
  shock (Rankine-Hugoniot):  omega = (q_B - q_A) / (k_B - k_A)      Lighthill & Whitham 1955; Lecture 03
  characteristic speed:      c(k) = dq/dk along the equilibrium curve
with k = 1 / (s_e(v) + l), q = v k (tmc.idm). Measured: the time each follower first falls below
v_mid = (v_eq + v_low) / 2 and its position then; the wave speed is the least-squares slope of position vs
time (fixed road frame, m/s, negative = moving upstream). Recovery: first climb back above v_mid after the
follower's minimum, fitted the same way.

Fixed-frame wave speed and per-vehicle delay tau (time the front takes to pass one vehicle) are linked by
omega = v_A - (s_A + l) / tau in the upstream state A. At 15 m/s, v_A and (s_A + l) / tau are both ~15-17 m/s,
so a small change in tau is a large change in omega; tau is reported next to omega for that reason.
Linear check: the IDM platoon transfer function's phase delay per vehicle (linear_delay) tends to the LWR tau
as omega -> 0 (LWR is the long-wave limit of the car-following model) and is larger at finite wavelengths.

Diagnostics added after the registered E10 run (disclosed in the log): a 150-follower platoon with a long hold
(`long_platoon`), where the wave grows into stop-and-go, compared block by block with the LWR shock to the
speed each block actually reaches.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tmc.disturbance import load_setting, run_platoon
from tmc.idm import IDMParams, equilibrium_flow, equilibrium_gap, equilibrium_partials, load_params

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results"
TOL_LWR = 0.10      # E10 pass criterion: braking front within 10% of omega (T_r = 0)
TOL_DT = 0.02       # and within 2% between dt 0.1 and 0.05


def eq_state(v: float, p: IDMParams) -> tuple[float, float]:
    """Equilibrium (density veh/m, flow veh/s) at speed v."""
    return float(1.0 / (equilibrium_gap(v, p) + p.length)), float(equilibrium_flow(v, p))


def lwr_shock_speed(v_a: float, v_b: float, p: IDMParams) -> float:
    """omega = (q_B - q_A) / (k_B - k_A) between the equilibrium states at speeds v_a and v_b (m/s)."""
    k_a, q_a = eq_state(v_a, p)
    k_b, q_b = eq_state(v_b, p)
    return (q_b - q_a) / (k_b - k_a)


def characteristic_speed(v: float, p: IDMParams, h: float = 1e-4) -> float:
    """c = dq/dk on the equilibrium curve at speed v (central difference in v)."""
    return lwr_shock_speed(max(v - h, 0.0), v + h, p)


def per_vehicle_delay(omega: float, v_a: float, p: IDMParams) -> float:
    """tau = 1 / (q_A - omega k_A): seconds for a front moving at omega (fixed frame) to pass one vehicle
    of the upstream equilibrium state at speed v_a."""
    k_a, q_a = eq_state(v_a, p)
    return 1.0 / (q_a - omega * k_a)


def linear_delay(v: float, omega_rad: float, p: IDMParams) -> float:
    """Phase delay per vehicle (s) of the linearized IDM platoon at speed v, angular frequency omega_rad:
    -arg G(i w) / w with G = (f_s - i w f_dv) / (-w^2 - i w (f_v + f_dv) + f_s) (tmc.idm.platoon_gain, T_r = 0)."""
    d = equilibrium_partials(v, p)
    iw = 1j * omega_rad
    G = (d["f_s"] - iw * d["f_dv"]) / (iw ** 2 - iw * (d["f_v"] + d["f_dv"]) + d["f_s"])
    return float(-np.angle(G) / omega_rad)


def crossings(tr: pd.DataFrame, v_mid: float, followers) -> pd.DataFrame:
    """Per follower: first downward crossing of v_mid (braking front) and first upward crossing after the
    follower's minimum (recovery), each with the position at that time. Linear interpolation between records.
    NaN where a follower never crosses (stays explicitly missing)."""
    rows = []
    for vid in followers:
        g = tr[tr["vehicle_id"] == vid].sort_values("time_s")
        t, x, v = (g[c].to_numpy() for c in ("time_s", "position_m", "speed_mps"))
        row = {"vehicle_index": int(vid), "v_min_mps": float(v.min()) if len(v) else np.nan}
        for name, idx in (("brake", _first_cross(v, v_mid, down=True)),
                          ("recover", _first_cross(v, v_mid, down=False, start=int(np.argmin(v)) if len(v) else 0))):
            if idx is None:
                row[f"t_{name}_s"] = row[f"x_{name}_m"] = np.nan
            else:
                f = (v_mid - v[idx]) / (v[idx + 1] - v[idx])
                row[f"t_{name}_s"] = float(t[idx] + f * (t[idx + 1] - t[idx]))
                row[f"x_{name}_m"] = float(x[idx] + f * (x[idx + 1] - x[idx]))
        rows.append(row)
    return pd.DataFrame(rows)


def _first_cross(v: np.ndarray, level: float, down: bool, start: int = 0) -> int | None:
    """Index i >= start with v[i] on one side of level and v[i+1] on the other (first such), else None."""
    a, b = v[start:-1], v[start + 1:]
    hit = (a >= level) & (b < level) if down else (a <= level) & (b > level)
    i = np.flatnonzero(hit)
    return int(i[0]) + start if len(i) else None


def fit_speed(t: np.ndarray, x: np.ndarray) -> tuple[float, float, int]:
    """Least-squares slope dx/dt (m/s), R^2, and number of points used (NaNs dropped)."""
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    if len(t) < 3:
        return float("nan"), float("nan"), int(len(t))
    slope, icpt = np.polyfit(t, x, 1)
    resid = x - (slope * t + icpt)
    r2 = 1.0 - float(resid @ resid) / float(((x - x.mean()) ** 2).sum())
    return float(slope), r2, int(len(t))


def measure(dist_name: str, reaction_time_s: float = 0.0, dt: float | None = None,
            s: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """One platoon run; returns (summary row, per-follower crossings)."""
    s = s or load_setting()
    dist = s["disturbances"][dist_name]
    v_eq = float(s["v_eq_mps"])
    res, p = run_platoon(s, dist, reaction_time_s, dt)
    v_mid = 0.5 * (v_eq + dist.v_low)
    per = crossings(res.trajectories, v_mid, range(1, int(s["n_followers"]) + 1))
    w_b, r2_b, n_b = fit_speed(per["t_brake_s"].to_numpy(), per["x_brake_m"].to_numpy())
    w_r, r2_r, n_r = fit_speed(per["t_recover_s"].to_numpy(), per["x_recover_m"].to_numpy())
    omega = lwr_shock_speed(v_eq, dist.v_low, p)
    row = {
        "disturbance": dist_name, "T_r_s": reaction_time_s, "dt_s": res.config.dt_s, "v_eq_mps": v_eq,
        "v_low_mps": dist.v_low, "v_mid_mps": v_mid,
        "omega_lwr_mps": omega, "c_at_v_eq_mps": characteristic_speed(v_eq, p),
        "c_at_v_low_mps": characteristic_speed(dist.v_low, p),
        "c_at_v_mid_mps": characteristic_speed(v_mid, p),
        "brake_front_mps": w_b, "brake_r2": r2_b, "brake_n": n_b,
        "brake_rel_err": (w_b - omega) / abs(omega),
        "tau_lwr_s": per_vehicle_delay(omega, v_eq, p), "tau_brake_s": per_vehicle_delay(w_b, v_eq, p),
        "recover_mps": w_r, "recover_r2": r2_r, "recover_n": n_r,
        "min_follower_speed_mps": float(per["v_min_mps"].min()),
    }
    per.insert(0, "T_r_s", reaction_time_s)
    per.insert(0, "dt_s", res.config.dt_s)
    per.insert(0, "disturbance", dist_name)
    return row, per


def long_platoon(n_followers: int = 150, hold_s: float = 600.0, duration_s: float = 900.0,
                 block: int = 25, s: dict | None = None) -> pd.DataFrame:
    """Diagnostic (after the registered run): long platoon, long hold, T_r = 0, dt 0.1. Per block of followers:
    measured braking-front speed vs the LWR shock from v_eq to the block's median minimum speed."""
    import copy
    import dataclasses
    s = copy.deepcopy(s or load_setting())
    s["n_followers"], s["duration_s"] = n_followers, duration_s
    p = load_params(s["param_set"])
    v_eq = float(s["v_eq_mps"])
    rows = []
    for name in ("nominal", "strong"):
        s["disturbances"][name] = dataclasses.replace(s["disturbances"][name], hold_s=hold_s)
        _, per = measure(name, 0.0, None, s)
        for a in range(0, n_followers, block):
            g = per.iloc[a:a + block]
            w_b, r2, n = fit_speed(g["t_brake_s"].to_numpy(), g["x_brake_m"].to_numpy())
            v_loc = max(float(g["v_min_mps"].median()), 0.0)
            om = lwr_shock_speed(v_eq, v_loc, p)
            rows.append({"disturbance": name, "followers": f"{a + 1}-{a + block}", "first": a + 1,
                         "median_v_min_mps": v_loc, "brake_front_mps": w_b, "brake_r2": r2,
                         "omega_lwr_local_mps": om, "rel_err": (w_b - om) / abs(om),
                         "tau_brake_s": per_vehicle_delay(w_b, v_eq, p), "tau_lwr_s": per_vehicle_delay(om, v_eq, p)})
    return pd.DataFrame(rows)


def plot(per: pd.DataFrame, df: pd.DataFrame, lp: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cases = df[(df["dt_s"] == 0.1)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=False)
    for ax, dist in zip(axes[:2], ["nominal", "strong"]):
        for tr_s, mk in ((0.0, "o"), (0.6, "^")):
            c = cases[(cases["disturbance"] == dist) & (cases["T_r_s"] == tr_s)]
            g = per[(per["disturbance"] == dist) & (per["T_r_s"] == tr_s) & (per["dt_s"] == 0.1)]
            if c.empty:
                continue
            r = c.iloc[0]
            ax.plot(g["t_brake_s"], g["x_brake_m"], mk, ms=4, color="#2f6fde" if tr_s == 0 else "#d6453a",
                    label=f"braking front, T_r = {tr_s} s: {r.brake_front_mps:.2f} m/s")
            ax.plot(g["t_recover_s"], g["x_recover_m"], mk, ms=4, mfc="none",
                    color="#2f6fde" if tr_s == 0 else "#d6453a",
                    label=f"recovery, T_r = {tr_s} s: {r.recover_mps:.2f} m/s")
        r0 = cases[(cases["disturbance"] == dist) & (cases["T_r_s"] == 0.0)].iloc[0]
        g0 = per[(per["disturbance"] == dist) & (per["T_r_s"] == 0.0) & (per["dt_s"] == 0.1)]
        t0, x0 = g0["t_brake_s"].iloc[0], g0["x_brake_m"].iloc[0]
        tt = np.array([g0["t_brake_s"].min(), g0["t_brake_s"].max()])
        ax.plot(tt, x0 + r0.omega_lwr_mps * (tt - t0), "k--", lw=1,
                label=f"LWR shock omega = {r0.omega_lwr_mps:.2f} m/s")
        ax.set_title(f"E10_{dist}: lead 15 -> {r0.v_low_mps:.0f} m/s, v_mid = {r0.v_mid_mps:.1f} m/s")
        ax.set_xlabel("time when the follower crosses v_mid (s)")
        ax.set_ylabel("position then (m, fixed road frame)")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=.3)
    ax = axes[2]
    for dist, col in (("nominal", "#2f6fde"), ("strong", "#d6453a")):
        g = lp[lp["disturbance"] == dist]
        x = g["first"] + 12
        ax.plot(x, g["brake_front_mps"], "o-", color=col, label=f"{dist}: measured front speed")
        ax.plot(x, g["omega_lwr_local_mps"], "s--", mfc="none", color=col,
                label=f"{dist}: LWR shock 15 m/s -> block's min speed")
    ax.axhline(-(load_params("baseline").s0 + load_params("baseline").length) / load_params("baseline").T,
               color="k", lw=.8, ls=":", label="-(s0 + l)/T (jam limit)")
    ax.set_title("E10 diagnostic: 150 followers, 600 s hold (T_r = 0)")
    ax.set_xlabel("follower index (block centre, blocks of 25)")
    ax.set_ylabel("braking-front speed (m/s, negative = upstream)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=.3)
    fig.suptitle("E10 wave-speed check (setting A platoon, T_r = 0 unless noted, dt 0.1 s): "
                 "left two panels, each point is one follower")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str]) -> None:
    s = load_setting()
    rows, pers = [], []
    for dist in ("nominal", "strong"):
        for tr_s, dt in ((0.0, 0.1), (0.0, 0.05), (0.6, 0.1)):
            r, per = measure(dist, tr_s, dt, s)
            rows.append(r)
            pers.append(per)
    df, per = pd.DataFrame(rows), pd.concat(pers, ignore_index=True)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "E10_wave_speed.csv", index=False)
    per.to_csv(OUT / "E10_wave_speed_per_vehicle.csv", index=False)
    lp = long_platoon(s=s)
    lp.to_csv(OUT / "E10_wave_speed_long_platoon.csv", index=False)
    plot(per, df, lp, OUT / "E10_wave_speed.png")
    pd.set_option("display.width", 200)
    print(df[["disturbance", "T_r_s", "dt_s", "omega_lwr_mps", "brake_front_mps", "brake_rel_err", "brake_r2",
              "tau_lwr_s", "tau_brake_s", "recover_mps", "c_at_v_eq_mps", "c_at_v_low_mps", "c_at_v_mid_mps",
              "min_follower_speed_mps"]].round(4).to_string(index=False))
    for dist in ("nominal", "strong"):
        a = df[(df.disturbance == dist) & (df.T_r_s == 0.0)].set_index("dt_s")
        lwr_ok = abs(a.loc[0.1, "brake_rel_err"]) <= TOL_LWR
        dt_ok = abs(a.loc[0.05, "brake_front_mps"] / a.loc[0.1, "brake_front_mps"] - 1) <= TOL_DT
        print(f"{dist}: within {TOL_LWR:.0%} of LWR: {'PASS' if lwr_ok else 'FAIL'}; "
              f"dt 0.1 vs 0.05 within {TOL_DT:.0%}: {'PASS' if dt_ok else 'FAIL'}")
    print()
    print("Diagnostic, long platoon (LWR shock to the speed each block reaches):")
    print(lp.round(3).to_string(index=False))
    p = load_params(s["param_set"])
    print()
    print("Linear IDM platoon delay per vehicle at 15 m/s (LWR long-wave tau = "
          f"{per_vehicle_delay(characteristic_speed(15.0, p), 15.0, p):.3f} s):")
    for period in (1000.0, 120.0, 60.0, 30.0, 20.0, 12.0):
        print(f"  period {period:6.0f} s: {linear_delay(15.0, 2 * np.pi / period, p):.3f} s")
    print(f"wrote {OUT / 'E10_wave_speed.csv'}, ..._per_vehicle.csv, ..._long_platoon.csv, E10_wave_speed.png")


if __name__ == "__main__":
    main(sys.argv[1:])
