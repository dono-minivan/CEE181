"""CAV controller core (design record section 7, experiment log E9): ACC and CACC on an IDM base.

Pure pieces, unit-tested in tests/test_cav.py; wiring into the simulator is separate (network.py).

    python -m tmc.cav          # small example: one ACC and one CACC follower behind a braking leader

Control law (Treiber & Kesting 2013, Sec. 11.3.8):
  CAH (Eq. 11.25), with the effective leader acceleration a~_l = min(a_l, a):
      a_CAH = v^2 a~_l / (v_l^2 - 2 s a~_l)                 if v_l (v - v_l) <= -2 s a~_l
            = a~_l - (v - v_l)^2 Theta(v - v_l) / (2 s)      otherwise
    Guard (not in the book): the first branch is used only when its denominator is > 0. At v_l = 0 and
    a~_l = 0 the book's first branch is 0/0; the second branch then gives -v^2/(2s), stopping behind
    the leader, which is the physical limit.
  ACC blend (Eq. 11.26), base = IDM (team decision, section 7), coolness c = 0.99:
      a_ACC = a_IDM                                                   if a_IDM >= a_CAH
            = (1 - c) a_IDM + c [a_CAH + b tanh((a_IDM - a_CAH) / b)]  otherwise
  No leader (free road): a_IDM free-road term only.

Leader-acceleration input a_l (the only difference between ACC and CACC):
  ACC: leader speed = own speed - sensed approach rate, sampled every TICK_S, backward difference,
       first-order low-pass T_f (exact discretization y += (1 - exp(-TICK_S/T_f)) (x - y), first sample
       initializes y). Before two samples exist since the last leader change, a_l is MISSING (None) and the
       command is a_IDM alone (CAH needs a_l; a missing value is never replaced by 0, CLAUDE.md rule 1).
  CACC: the leader's COMMANDED acceleration from V2V (Ploeg et al. 2011). Messages on the TICK_S grid,
       usable at the first tick >= send tick + delay ticks, delay ticks = ceil(round(delay / TICK_S, 9)).
       Falls back to the ACC estimate (logged) when: the leader is not equipped, no message from the current
       leader since the leader change (the link restarts), or the newest usable message is > MAX_AGE_TICKS old.
Command to motion: command clipped to [A_MIN, a], held for one tick (zero-order hold); realized acceleration
follows a first-order lag tau, exact per simulation step: a_r += (1 - exp(-dt/tau)) (u - a_r).
Controller inputs (permitted): own speed, sensed gap and approach rate, the tracked leader's identity (to
address V2V and detect a leader change), and received messages. leader_equipped is used ONLY to label a
fallback reason in the log, never in the control logic.
"""
from __future__ import annotations

import math
import random
import sys
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from tmc.idm import IDMParams, acceleration, load_params

TICK_S = 0.1          # sensing / message / command interval (fixed controller property)
MAX_AGE_TICKS = 5     # message older than this (0.5 s) -> fallback
A_MIN = -8.0          # m/s^2 command floor (T&K Fig. 11.9)
COOLNESS = 0.99
T_FILTER_S = 0.3      # ACC leader-acceleration low-pass time constant


def delay_ticks(delay_s: float, tick_s: float = TICK_S) -> int:
    """Message delay in whole ticks: ceil(round(delay / tick, 9)), so 0.15 s -> 2 and 0.3 s -> 3 exactly."""
    if delay_s < 0:
        raise ValueError("delay must be >= 0")
    return int(math.ceil(round(delay_s / tick_s, 9)))


# --------------------------------------------------------------------------- control law

def cah_acceleration(s: float, v: float, v_l: float, a_l: float, p: IDMParams) -> float:
    """Constant-acceleration heuristic (T&K Eq. 11.25) with a~_l = min(a_l, a)."""
    a_eff = min(a_l, p.a)
    s = max(s, 1e-6)
    denom = v_l * v_l - 2.0 * s * a_eff
    if v_l * (v - v_l) <= -2.0 * s * a_eff and denom > 1e-12:
        return v * v * a_eff / denom
    dv = v - v_l
    return a_eff - (dv * dv / (2.0 * s) if dv > 0 else 0.0)


def acc_blend(a_idm: float, a_cah: float, p: IDMParams, c: float = COOLNESS) -> float:
    """ACC model (T&K Eq. 11.26) blending the base (IDM) and CAH accelerations."""
    if a_idm >= a_cah:
        return a_idm
    return (1.0 - c) * a_idm + c * (a_cah + p.b * math.tanh((a_idm - a_cah) / p.b))


def acc_command(s: float, v: float, approach_rate: float, a_l: float | None, p: IDMParams) -> float:
    """Command from sensed gap s, own speed v, approach rate dv = v - v_l and leader acceleration a_l.
    s = inf: free road. a_l = None (missing): IDM only. Clipped to [A_MIN, a]."""
    a_idm = float(acceleration(v, s, approach_rate, p))
    if math.isfinite(s) and a_l is not None:
        a_idm = acc_blend(a_idm, cah_acceleration(s, v, v - approach_rate, a_l, p), p)
    return min(max(a_idm, A_MIN), p.a)


def lag_step(a_realized: float, u: float, dt: float, tau: float) -> float:
    """First-order actuator lag over one simulation step (exact for a command held over the step)."""
    return a_realized + (1.0 - math.exp(-dt / tau)) * (u - a_realized)


# --------------------------------------------------------------------------- ACC sensor estimate

@dataclass
class LeaderAccelEstimator:
    """a_l from sensed leader speed: backward difference over TICK_S, then a first-order low-pass."""
    t_filter_s: float = T_FILTER_S
    tick_s: float = TICK_S
    _v_prev: float | None = None
    _y: float | None = None

    def reset(self) -> None:
        self._v_prev, self._y = None, None

    def update(self, v_leader: float) -> float | None:
        if self._v_prev is None:
            self._v_prev = v_leader
            return None
        x = (v_leader - self._v_prev) / self.tick_s
        self._v_prev = v_leader
        if self._y is None:
            self._y = x
        else:
            self._y += (1.0 - math.exp(-self.tick_s / self.t_filter_s)) * (x - self._y)
        return self._y


# --------------------------------------------------------------------------- V2V channel

@dataclass
class V2VChannel:
    """Broadcast channel on the tick grid. Each message is lost with probability loss_prob (seeded) or
    during an outage [start_s, end_s); delivered messages become usable delay ticks after sending."""
    delay_s: float = 0.15
    loss_prob: float = 0.0
    outages: list = field(default_factory=list)     # [(start_s, end_s), ...]
    seed: int | None = None
    tick_s: float = TICK_S

    def __post_init__(self):
        self.d = delay_ticks(self.delay_s, self.tick_s)
        self._rng = random.Random(self.seed)
        self._q: dict[int, deque] = {}              # sender -> deque of (send_tick, usable_tick, value)
        self.sent = 0
        self.lost = 0

    def send(self, sender: int, tick: int, value: float) -> None:
        self.sent += 1
        t = tick * self.tick_s
        lost = any(a - 1e-9 <= t < b - 1e-9 for a, b in self.outages)
        if self.loss_prob > 0:                       # draw for every message so loss is reproducible
            lost = (self._rng.random() < self.loss_prob) or lost
        if lost:
            self.lost += 1
            return
        q = self._q.setdefault(sender, deque())
        q.append((tick, tick + self.d, value))
        while len(q) > 64:
            q.popleft()

    def latest(self, sender: int, now_tick: int, since_tick: int) -> tuple[float, int] | None:
        """Newest message from `sender` usable at now_tick and sent at or after since_tick: (value, send_tick)."""
        for send_tick, usable, value in reversed(self._q.get(sender, ())):
            if usable <= now_tick:
                return (value, send_tick) if send_tick >= since_tick else None
        return None


# --------------------------------------------------------------------------- per-vehicle controller

@dataclass
class CAVController:
    """One equipped vehicle. mode 'acc' or 'cacc'. Call command() once per tick (leader-first order),
    then actuate() every simulation step."""
    vid: int
    mode: str
    p: IDMParams
    tau_s: float = 0.1
    estimator: LeaderAccelEstimator = field(default_factory=LeaderAccelEstimator)
    u: float = 0.0                  # current command (held over the tick)
    a_realized: float = 0.0
    leader: int | None = None
    leader_since_tick: int = 0
    log: list = field(default_factory=list)   # (tick, reason) for every tick spent in fallback

    def __post_init__(self):
        if self.mode not in ("acc", "cacc"):
            raise ValueError(f"mode must be 'acc' or 'cacc', got {self.mode!r}")

    def command(self, tick: int, v: float, s: float, approach_rate: float, leader: int | None,
                channel: V2VChannel | None = None, leader_equipped: bool = False) -> dict:
        if leader != self.leader:                   # cut-in, leader exit, or merge: link and estimate restart
            self.leader, self.leader_since_tick = leader, tick
            self.estimator.reset()
        a_sensor = None if leader is None else self.estimator.update(v - approach_rate)
        a_l, source, reason = a_sensor, ("sensor" if a_sensor is not None else "none"), ""
        if self.mode == "cacc" and leader is not None:
            msg = channel.latest(leader, tick, self.leader_since_tick) if channel is not None else None
            if msg is not None and tick - msg[1] <= MAX_AGE_TICKS:
                a_l, source = msg[0], "v2v"
            else:
                reason = ("comms_off" if channel is None else
                          "leader_unequipped" if not leader_equipped else
                          "no_message" if msg is None else "stale")
                self.log.append((tick, reason))
        self.u = acc_command(s, v, approach_rate, a_l, self.p)
        return {"u": self.u, "a_l": a_l, "source": source, "fallback": reason}

    def actuate(self, dt: float) -> float:
        self.a_realized = lag_step(self.a_realized, self.u, dt, self.tau_s)
        return self.a_realized


# --------------------------------------------------------------------------- small example

def _demo() -> None:
    """Leader brakes at 2 m/s^2 from 15 to 10 m/s; ACC and CACC followers start in IDM equilibrium."""
    from tmc.idm import ballistic_update, equilibrium_gap
    p = load_params("baseline")
    dt, s_e = 0.1, float(equilibrium_gap(15.0, p))
    for mode in ("acc", "cacc"):
        ch = V2VChannel(delay_s=0.15)
        ctl = CAVController(vid=1, mode=mode, p=p)
        xl, vl, xf, vf = s_e + p.length, 15.0, 0.0, 15.0
        vmin, min_gap = vf, s_e
        for k in range(600):
            t = k * dt
            al = -2.0 if 5.0 <= t and vl > 10.0 + 1e-9 else 0.0
            al = max(al, (10.0 - vl) / dt) if al < 0 else al
            ch.send(0, k, al)                        # leader broadcasts its commanded acceleration
            ctl.command(k, vf, xl - xf - p.length, vf - vl, 0, ch, True)
            af = ctl.actuate(dt)
            (xl,), (vl,) = ballistic_update([xl], [vl], [al], dt)
            (xf,), (vf,) = ballistic_update([xf], [vf], [af], dt)
            vmin, min_gap = min(vmin, vf), min(min_gap, xl - xf - p.length)
        print(f"{mode.upper():4s}: follower min speed {vmin:.2f} m/s (leader 10.00), min gap {min_gap:.1f} m, "
              f"fallback ticks {len(ctl.log)}")


if __name__ == "__main__":
    _demo()
