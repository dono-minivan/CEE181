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


if __name__ == "__main__":  # small example (CLAUDE.md rule 7)
    for name in ("baseline", "sensitivity"):
        prm = load_params(name)
        c = capacity(prm)
        print(f"[{name}] q_c = {c['q_c_veh_per_h']:.0f} veh/h at v_c = {c['v_c_mps']:.1f} m/s "
              f"({c['v_c_mps'] * 2.23694:.1f} mph), k_c = {c['k_c_veh_per_m'] * 1609.344:.1f} veh/mi, "
              f"k_jam = {c['k_jam_veh_per_m'] * 1609.344:.0f} veh/mi")
