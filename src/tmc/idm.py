"""Intelligent Driver Model (IDM) dynamics, ballistic update, and derived quantities.

Units: SI throughout (m, s, m/s, m/s^2). Convert to mph / veh/h only at outputs.

Sources (see docs/source_to_design.md):
  - IDM acceleration: Treiber, Hennecke & Helbing (2000), Eqs. 6-7;
    Treiber & Kesting (2013), Eqs. 11.14-11.15.
  - Ballistic update incl. stop-within-step correction: Treiber & Kesting (2013), Eqs. 10.9-10.10
    (and Treiber & Kanagaraj 2015 for the stopping case).
  - Equilibrium gap s_e(v) and flow q_e(v) follow from setting dv/dt = 0, dv = 0.

Everything derived (capacity, critical speed, jam density) is computed here, never copied.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import yaml

DEFAULT_PARAM_FILE = Path(__file__).resolve().parents[2] / "configs" / "idm_baseline.yaml"


@dataclass(frozen=True)
class IDMParams:
    v0: float      # desired speed, m/s
    T: float       # safe time gap, s
    s0: float      # minimum (jam) gap, m
    a: float       # maximum acceleration, m/s^2
    b: float       # comfortable deceleration, m/s^2
    delta: float   # acceleration exponent, -
    length: float  # vehicle length, m

    def to_dict(self) -> dict:
        return asdict(self)


def load_params(set_name: str = "baseline", path: str | Path = DEFAULT_PARAM_FILE) -> IDMParams:
    """Load a named parameter set ("baseline" or "sensitivity") from the YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return IDMParams(**{k: float(v) for k, v in data[set_name].items()})


# --------------------------------------------------------------------------- dynamics

def desired_gap(v, dv, p: IDMParams):
    """s*(v, dv) = s0 + max(0, v*T + v*dv / (2*sqrt(a*b))), with dv = v - v_leader (approach rate)."""
    v = np.asarray(v, dtype=float)
    dv = np.asarray(dv, dtype=float)
    return p.s0 + np.maximum(0.0, v * p.T + v * dv / (2.0 * np.sqrt(p.a * p.b)))


def acceleration(v, s, dv, p: IDMParams):
    """IDM acceleration a[1 - (v/v0)^delta - (s*/s)^2].

    s = bumper-to-bumper gap to the leader (m); use np.inf for a free road (no leader).
    dv = v - v_leader (m/s); ignored when s is inf.
    """
    v = np.asarray(v, dtype=float)
    s = np.asarray(s, dtype=float)
    free = p.a * (1.0 - (v / p.v0) ** p.delta)
    with np.errstate(divide="ignore", invalid="ignore"):
        interaction = np.where(np.isinf(s), 0.0, (desired_gap(v, dv, p) / s) ** 2)
    return free - p.a * interaction


def ballistic_update(x, v, acc, dt: float):
    """One ballistic step. If a vehicle would reverse within the step, it stops exactly where v hits 0.

    Returns (x_new, v_new). Inputs are not modified.
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    acc = np.asarray(acc, dtype=float)
    v_new = v + acc * dt
    stops = v_new < 0.0
    x_new = x + v * dt + 0.5 * acc * dt**2
    if np.any(stops):
        # distance to stop: v^2 / (2|a|) = -v^2 / (2 a)  (acc < 0 whenever v_new < 0 and v >= 0)
        x_new = np.where(stops, x - 0.5 * v**2 / np.where(stops, acc, 1.0), x_new)
        v_new = np.where(stops, 0.0, v_new)
    return x_new, v_new


# --------------------------------------------------------------------------- equilibrium / FD

def equilibrium_gap(v, p: IDMParams):
    """Steady-state gap s_e(v) = (s0 + v T) / sqrt(1 - (v/v0)^delta); inf at v >= v0."""
    v = np.asarray(v, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        root = 1.0 - (v / p.v0) ** p.delta
        return np.where(root > 0, (p.s0 + v * p.T) / np.sqrt(np.clip(root, 1e-300, None)), np.inf)


def equilibrium_flow(v, p: IDMParams):
    """Steady-state flow q_e(v) = v / (s_e(v) + l), veh/s."""
    return np.asarray(v, dtype=float) / (equilibrium_gap(v, p) + p.length)


def fundamental_diagram(p: IDMParams, n: int = 4001):
    """Return (density veh/m, flow veh/s, speed m/s) along the equilibrium curve."""
    v = np.linspace(0.0, p.v0 * (1 - 1e-9), n)
    k = 1.0 / (equilibrium_gap(v, p) + p.length)
    return k, v * k, v


def capacity(p: IDMParams, n: int = 200001) -> dict:
    """Maximum equilibrium flow and where it occurs (computed numerically on a fine speed grid)."""
    v = np.linspace(1e-6, p.v0 * (1 - 1e-9), n)
    q = equilibrium_flow(v, p)
    i = int(np.argmax(q))
    k_c = q[i] / v[i]
    return {
        "q_c_veh_per_s": float(q[i]),
        "q_c_veh_per_h": float(q[i] * 3600.0),
        "v_c_mps": float(v[i]),
        "k_c_veh_per_m": float(k_c),
        "k_jam_veh_per_m": float(1.0 / (p.s0 + p.length)),
    }


# --------------------------------------------------------------------------- IIDM (comparison only)
# Improved IDM, Treiber & Kesting (2013) Sec. 11.3.7, Eqs. 11.22-11.24. NOT used for any vehicle in
# the simulator (design record section 7: all vehicle types use the IDM base). It is here to compute
# what an IIDM base would change: its steady-state gap is exactly s0 + v T, which raises capacity.

def iidm_acceleration(v, s, dv, p: IDMParams):
    """IIDM acceleration with z = s*(v, dv) / s (same s* as the IDM); s = inf means a free road."""
    v = np.asarray(v, dtype=float)
    s = np.asarray(s, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z = np.where(np.isinf(s), 0.0, desired_gap(v, dv, p) / s)
        below = v <= p.v0
        a_free = np.where(below, p.a * (1.0 - (v / p.v0) ** p.delta),
                          -p.b * (1.0 - (p.v0 / np.maximum(v, 1e-12)) ** (p.a * p.delta / p.b)))
        a_free_safe = np.where(np.abs(a_free) > 1e-12, a_free, 1e-12)
        below_close = p.a * (1.0 - z**2)                                     # v <= v0, z >= 1
        below_far = a_free * (1.0 - z ** (2.0 * p.a / a_free_safe))          # v <= v0, z < 1
        above_close = a_free + p.a * (1.0 - z**2)                            # v > v0, z >= 1
        acc = np.where(below, np.where(z >= 1.0, below_close, below_far),
                       np.where(z >= 1.0, above_close, a_free))
    return acc


def iidm_capacity(p: IDMParams, n: int = 200001) -> dict:
    """Maximum IIDM equilibrium flow q = v / (s0 + v T + l), on a speed grid over (0, v0).

    The flow rises with v, so the maximum is the supremum as v -> v0: v0 / (s0 + v0 T + l).
    Computed numerically (grid), not from the closed form; the test checks both agree.
    """
    v = np.linspace(1e-6, p.v0 * (1 - 1e-9), n)
    q = v / (p.s0 + v * p.T + p.length)
    i = int(np.argmax(q))
    return {"q_c_veh_per_s": float(q[i]), "q_c_veh_per_h": float(q[i] * 3600.0), "v_c_mps": float(v[i])}


# --------------------------------------------------------------------------- string stability

def equilibrium_partials(v, p: IDMParams) -> dict:
    """Analytic partial derivatives of the IDM acceleration f(s, v, dv) at equilibrium (s = s_e(v), dv = 0).

    Code convention: dv = v - v_leader (approach rate), so f_dv < 0 (approaching -> brake).
    f_v is taken with dv held fixed. Valid for 0 < v < v0.
    """
    v = np.asarray(v, dtype=float)
    s = equilibrium_gap(v, p)
    s_star = p.s0 + v * p.T                                   # s*(v, 0)
    f_s = 2.0 * p.a * s_star**2 / s**3
    f_v = -p.a * (p.delta * v ** (p.delta - 1.0) / p.v0**p.delta + 2.0 * s_star * p.T / s**2)
    f_dv = -p.a * s_star * v / (s**2 * np.sqrt(p.a * p.b))
    return {"s_e": s, "f_s": f_s, "f_v": f_v, "f_dv": f_dv}


def string_stability_margin(v, p: IDMParams):
    """Linear string-stability margin of an IDM platoon at equilibrium speed v (m/s).

    Criterion: Treiber & Kesting (2013) Eq. 15.77 (Wilson's condition for f(s, v, v_l) models).
    String-stable iff  f_v^2 / 2 + f_v f_dv - f_s >= 0.
    The f_v f_dv term is written for this module's dv = v - v_leader; it is positive (both partials < 0),
    i.e. responding to the approach rate damps waves. With the book's leader-speed form it is
    -f_v f_{v_l}, the same quantity. Margin < 0 means small disturbances grow along the platoon.
    """
    d = equilibrium_partials(v, p)
    return 0.5 * d["f_v"] ** 2 + d["f_v"] * d["f_dv"] - d["f_s"]


def platoon_gain(v, omega, p: IDMParams, reaction_time_s: float = 0.0):
    """Linear per-vehicle speed gain |G(i omega)| = |V_follower / V_leader| at equilibrium speed v.

    Linearizing ds/dt = v_l - v, dv/dt(t) = [f_s ds + f_v dv + f_dv (dv - dv_l)](t - T_r) gives, with
    E = exp(-i w T_r) (reaction delay on all inputs, design record section 7):
        G(i w) = E (f_s - i w f_dv) / (-w^2 - i w E (f_v + f_dv) + E f_s).
    T_r = 0 gives the undelayed IDM; then |G| > 1 at some omega  <=>  string_stability_margin(v) < 0.
    A wave of angular frequency omega grows by a factor |G| per vehicle (ln|G| = per-vehicle growth rate).
    """
    d = equilibrium_partials(v, p)
    iw = 1j * np.asarray(omega, dtype=float)
    E = np.exp(-iw * reaction_time_s)
    G = E * (d["f_s"] - iw * d["f_dv"]) / (iw**2 - iw * E * (d["f_v"] + d["f_dv"]) + E * d["f_s"])
    return np.abs(G)


def max_platoon_gain(v: float, p: IDMParams, omega_max: float = 3.0, n: int = 30001,
                     reaction_time_s: float = 0.0) -> dict:
    """Largest per-vehicle gain over omega in (0, omega_max] rad/s, and the wave period where it occurs."""
    w = np.linspace(omega_max / n, omega_max, n)
    g = platoon_gain(v, w, p, reaction_time_s)
    i = int(np.argmax(g))
    return {"gain": float(g[i]), "growth_per_veh": float(np.log(g[i])), "period_s": float(2 * np.pi / w[i])}


def unstable_speeds_by_gain(p: IDMParams, reaction_time_s: float = 0.0, n: int = 400) -> np.ndarray:
    """Equilibrium speeds (grid of n over (0, v0)) where some wave grows: |G(omega)| > 1 for some omega.

    Works with a reaction delay, where the T&K Eq. 15.77 margin does not apply. Near the stability
    boundary only long waves grow and |G| - 1 ~ omega^2, so omega is log-spaced down to 1e-4 rad/s and
    the tolerance is tight. Note: with the delay on ALL inputs, exp(-i w T_r) cancels from the O(omega^2)
    terms of |G|^2, so the delay does not move the long-wave boundary; it raises growth of finite waves.
    With T_r = 0 this must agree with string_unstable_ranges (tested).
    """
    v = np.linspace(p.v0 * 0.01, p.v0 * 0.99, n)
    w = np.geomspace(1e-4, 3.0, 4000)
    return np.array([vi for vi in v if (platoon_gain(vi, w, p, reaction_time_s) - 1.0).max() > 1e-12])


def string_unstable_ranges(p: IDMParams, n: int = 20001) -> list[tuple[float, float]]:
    """Equilibrium-speed intervals (m/s) where the IDM platoon is string-unstable, on a grid over (0, v0)."""
    v = np.linspace(p.v0 * 1e-4, p.v0 * (1 - 1e-6), n)
    unstable = string_stability_margin(v, p) < 0
    ranges, start = [], None
    for vi, u in zip(v, unstable):
        if u and start is None:
            start = vi
        if not u and start is not None:
            ranges.append((float(start), float(prev)))
            start = None
        prev = vi
    if start is not None:
        ranges.append((float(start), float(prev)))
    return ranges


if __name__ == "__main__":  # small example (CLAUDE.md rule 7)
    for name in ("baseline", "sensitivity"):
        prm = load_params(name)
        c = capacity(prm)
        print(f"[{name}] q_c = {c['q_c_veh_per_h']:.0f} veh/h at v_c = {c['v_c_mps']:.1f} m/s "
              f"({c['v_c_mps'] * 2.23694:.1f} mph), k_c = {c['k_c_veh_per_m'] * 1609.344:.1f} veh/mi, "
              f"k_jam = {c['k_jam_veh_per_m'] * 1609.344:.0f} veh/mi")
        ci = iidm_capacity(prm)
        print(f"[{name}] IIDM comparison: q_c = {ci['q_c_veh_per_h']:.0f} veh/h "
              f"(+{100 * (ci['q_c_veh_per_h'] / c['q_c_veh_per_h'] - 1):.1f}% vs IDM)")
        for lo, hi in string_unstable_ranges(prm):
            print(f"[{name}] string-unstable for {lo:.1f} < v_e < {hi:.1f} m/s ({hi * 2.23694:.0f} mph)")
        u = unstable_speeds_by_gain(prm, reaction_time_s=0.6)
        g15 = max_platoon_gain(15.0, prm, reaction_time_s=0.6)
        print(f"[{name}] with T_r = 0.6 s: unstable up to {u.max():.1f} m/s; at 15 m/s max growth "
              f"{g15['growth_per_veh']:.4f}/veh (period {g15['period_s']:.0f} s)")
