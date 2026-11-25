"""Ploeg et al. 2011 CACC controller (experiment log E9d, setting A only) and its string-stability theory.

    python -m tmc.ploeg        # theory table (Eq. 17) + one CACC and one ACC follower behind a braking leader

Source: Ploeg, Scheepers, van Nunen, van de Wouw & Nijmeijer 2011, "Design and experimental evaluation of
cooperative adaptive cruise control", IEEE ITSC 2011, pp. 260-265 (TU/e accepted manuscript).
  Spacing policy (Eq. 1): d_r = r + h v;  spacing error (Eq. 2): e = d - d_r.
  Controller (Eq. 8):     h du/dt = -u + k_p e + k_d de/dt + k_dd d2e/dt2 + u_{i-1}
      u = commanded acceleration, u_{i-1} = the preceding vehicle's commanded acceleration (V2V feed-forward).
      k_p = 0.2, k_d = 0.7, k_dd = 0 (Sec. V-A). The paper's ACC omits u_{i-1} (Sec. V-B).
  Vehicle (Eq. 3): first-order actuator lag tau. The identified input delay phi = 0.2 s (Eq. 18) is omitted,
      as everywhere in E9.
  Theory (Eq. 17): Gamma(s) = (D(s) + G(s) K(s)) / (H(s) (1 + G(s) K(s))),
      G = 1 / (s^2 (tau s + 1)), K = k_p + k_d s, H = h s + 1, D = feed-forward path from u_{i-1}.

Implementation (E9d design, D1-D4 confirmed 2026-11-24):
  Spacing policy 'ctg' (Ploeg Eq. 1, r and h given) or 'idm' (e = s - s_e(v), the IDM equilibrium gap, with
      local h = ds_e/dv, so the steady state equals the human one).
  de/dt = (v_l - v) - h a_own: the sensed approach rate and the vehicle's own realized acceleration (an
      on-board measurement; Eq. 2 differentiated).
  Feed-forward u_ff: 'none' (Ploeg's ACC), 'sensor' (the E9 ACC estimate of the leader's REALIZED
      acceleration, tmc.cav.LeaderAccelEstimator), 'v2v' (the leader's COMMANDED acceleration via
      tmc.cav.V2VChannel; falls back to the sensor estimate, logged, under the E9 rules). A missing estimate
      (fewer than two samples since a leader change) means the feed-forward term is omitted for that tick
      (the controller then acts as Ploeg's ACC); it is never replaced by a number.
  Discretization: Eq. 8 is a first-order filter in u with time constant h; updated once per TICK_S with an
      exact zero-order-hold step u += (1 - exp(-TICK_S / h)) (k_p e + k_d de/dt + u_ff - u). The command and
      the filter state are clipped to [A_MIN, a] (anti-windup).
  No leader: the IDM free-road acceleration (not used in setting A, where the lead is scripted).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from tmc.cav import A_MIN, MAX_AGE_TICKS, T_FILTER_S, TICK_S, LeaderAccelEstimator, V2VChannel, lag_step
from tmc.idm import IDMParams, acceleration, equilibrium_gap, load_params

KP, KD = 0.2, 0.7          # Ploeg et al. 2011, Sec. V-A (k_dd = 0)
FF_SOURCES = ("none", "sensor", "v2v")
SPACINGS = ("ctg", "idm")


# --------------------------------------------------------------------------- theory (Eq. 17)

def feedforward_path(ff: str, s: np.ndarray, tau: float, delay_s: float = 0.0,
                     sample_lag_s: float = 0.05, t_filter_s: float = T_FILTER_S) -> np.ndarray:
    """D(s), the path from the leader's command u_{i-1} to the follower's feed-forward input.
    v2v: pure delay. sensor: the leader's actuator lag (realized acceleration), the average sampling lag of
    the backward difference (~half a tick) and the low-pass filter. none: 0."""
    if ff == "v2v":
        return np.exp(-delay_s * s)
    if ff == "sensor":
        return np.exp(-sample_lag_s * s) / ((tau * s + 1.0) * (t_filter_s * s + 1.0))
    if ff == "none":
        return np.zeros_like(s)
    raise ValueError(f"ff must be one of {FF_SOURCES}, got {ff!r}")


def gamma_peak(h: float, ff: str, tau: float = 0.1, delay_s: float = 0.0, kp: float = KP, kd: float = KD,
               w: np.ndarray | None = None) -> tuple[float, float]:
    """(||Gamma||_inf, frequency of the peak in rad/s), on a log grid 1e-3..1e2 rad/s."""
    w = np.logspace(-3, 2, 40000) if w is None else w
    s = 1j * w
    G = 1.0 / (s ** 2 * (tau * s + 1.0))
    K = kp + kd * s
    g = np.abs((feedforward_path(ff, s, tau, delay_s) + G * K) / ((h * s + 1.0) * (1.0 + G * K)))
    i = int(g.argmax())
    return float(g[i]), float(w[i])


def min_stable_h(ff: str, tau: float = 0.1, delay_s: float = 0.0, lo: float = 0.01, hi: float = 20.0) -> float:
    """Smallest time headway h with ||Gamma||_inf <= 1 (bisection; ||Gamma|| decreases with h)."""
    def unstable(h):
        return gamma_peak(h, ff, tau, delay_s)[0] > 1.0 + 1e-6
    if unstable(hi):
        return math.inf
    if not unstable(lo):
        return lo
    for _ in range(50):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if unstable(m) else (lo, m)
    return hi


def idm_local_headway(v: float, p: IDMParams, dv: float = 1e-4) -> float:
    """ds_e/dv of the IDM equilibrium gap (central difference)."""
    return float((equilibrium_gap(v + dv, p) - equilibrium_gap(max(v - dv, 0.0), p)) / (v + dv - max(v - dv, 0.0)))


# --------------------------------------------------------------------------- controller

@dataclass
class PloegController:
    """One equipped vehicle with the Ploeg controller. Same call pattern as tmc.cav.CAVController:
    command() once per tick (leader-first), actuate() every simulation step."""
    vid: int
    p: IDMParams
    ff: str = "v2v"
    spacing: str = "ctg"
    h_s: float = 1.0
    r_m: float = 2.0
    tau_s: float = 0.1
    max_age_ticks: int | None = MAX_AGE_TICKS
    kp: float = KP
    kd: float = KD
    estimator: LeaderAccelEstimator = field(default_factory=LeaderAccelEstimator)
    u: float = 0.0
    a_realized: float = 0.0
    leader: int | None = None
    leader_since_tick: int = 0
    log: list = field(default_factory=list)

    def __post_init__(self):
        if self.ff not in FF_SOURCES:
            raise ValueError(f"ff must be one of {FF_SOURCES}, got {self.ff!r}")
        if self.spacing not in SPACINGS:
            raise ValueError(f"spacing must be one of {SPACINGS}, got {self.spacing!r}")
        self.mode = "cacc" if self.ff == "v2v" else "acc"     # for the network: who needs a V2V channel

    def reference(self, v: float) -> tuple[float, float]:
        """(desired net gap, local time headway h) at speed v."""
        if self.spacing == "ctg":
            return self.r_m + self.h_s * v, self.h_s
        return float(equilibrium_gap(v, self.p)), idm_local_headway(v, self.p)

    def command(self, tick: int, v: float, s: float, approach_rate: float, leader: int | None,
                channel: V2VChannel | None = None, leader_equipped: bool = False) -> dict:
        if leader != self.leader:
            self.leader, self.leader_since_tick = leader, tick
            self.estimator.reset()
        if leader is None or not math.isfinite(s):
            self.u = float(min(max(acceleration(v, np.inf, 0.0, self.p), A_MIN), self.p.a))
            return {"u": self.u, "u_ff": None, "source": "free", "fallback": ""}
        a_sensor = self.estimator.update(v - approach_rate)
        u_ff, source, reason = None, "none", ""
        if self.ff == "sensor":
            u_ff, source = a_sensor, ("sensor" if a_sensor is not None else "missing")
        elif self.ff == "v2v":
            msg = channel.latest(leader, tick, self.leader_since_tick) if channel is not None else None
            if msg is not None and (self.max_age_ticks is None or tick - msg[1] <= self.max_age_ticks):
                u_ff, source = msg[0], "v2v"
            else:
                reason = ("comms_off" if channel is None else
                          "leader_unequipped" if not leader_equipped else
                          "no_message" if msg is None else "stale")
                self.log.append((tick, reason))
                u_ff, source = a_sensor, ("sensor" if a_sensor is not None else "missing")
        s_ref, h = self.reference(v)
        e = s - s_ref
        e_dot = -approach_rate - h * self.a_realized
        target = self.kp * e + self.kd * e_dot + (u_ff if u_ff is not None else 0.0)
        self.u += (1.0 - math.exp(-TICK_S / h)) * (target - self.u)
        self.u = min(max(self.u, A_MIN), self.p.a)
        return {"u": self.u, "u_ff": u_ff, "source": source, "fallback": reason}

    def actuate(self, dt: float) -> float:
        self.a_realized = lag_step(self.a_realized, self.u, dt, self.tau_s)
        return self.a_realized


# --------------------------------------------------------------------------- small example

def _demo() -> None:
    p = load_params("baseline")
    print("Ploeg et al. 2011 Eq. 17, k_p = 0.2, k_d = 0.7, tau = 0.1 s")
    print(f"  check vs paper: h_min CACC (0.15 s delay) = {min_stable_h('v2v', delay_s=0.15):.3f} s (paper 0.67); "
          f"ACC without feed-forward = {min_stable_h('none'):.3f} s (paper 3.16)")
    h_idm = idm_local_headway(15.0, p)
    print(f"  IDM equilibrium at 15 m/s: s_e = {float(equilibrium_gap(15.0, p)):.2f} m, local h = {h_idm:.3f} s")
    for h in (h_idm, 1.0):
        row = ", ".join(f"{name} {gamma_peak(h, ff, delay_s=d)[0]:.4f}" for name, ff, d in
                        (("none", "none", 0), ("sensor", "sensor", 0), ("v2v 0.2 s", "v2v", 0.2)))
        print(f"  h = {h:.3f} s: ||Gamma||_inf {row}")
    # one follower behind a leader that brakes at 2 m/s^2 from 15 to 10 m/s (CTG, h = 1 s)
    for ff in ("v2v", "none"):
        ch = V2VChannel(delay_s=0.15)
        c = PloegController(1, p, ff=ff, spacing="ctg", h_s=1.0)
        xl, vl, x, v, dt = 100.0, 15.0, 100.0 - 5.0 - (2.0 + 15.0), 15.0, TICK_S
        vmin, smin = v, 1e9
        for k in range(400):
            ul = -2.0 if (1.0 <= k * dt < 3.5) else 0.0
            ch.send(0, k, ul)
            c.command(k, v, xl - x - p.length, v - vl, 0, ch, True)
            a = c.actuate(dt)
            xl, vl = xl + vl * dt + 0.5 * ul * dt * dt, vl + ul * dt
            x, v = x + v * dt + 0.5 * a * dt * dt, v + a * dt
            vmin, smin = min(vmin, v), min(smin, xl - x - p.length)
        print(f"  follower ff={ff:5s}: min speed {vmin:.2f} m/s (leader 10.00), min gap {smin:.2f} m")


if __name__ == "__main__":
    _demo()
