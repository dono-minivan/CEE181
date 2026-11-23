"""Unit checks of the CAV controller core (tmc.cav): hand-computable cases for every piece."""
import math

import numpy as np
import pytest

from tmc.cav import (A_MIN, MAX_AGE_TICKS, TICK_S, CAVController, LeaderAccelEstimator, V2VChannel,
                     acc_blend, acc_command, cah_acceleration, delay_ticks, lag_step)
from tmc.idm import acceleration, equilibrium_gap, load_params

P = load_params("baseline")


# --------------------------------------------------------------------------- CAH and blend (T&K 11.25-11.26)

def test_cah_hand_cases():
    # second branch, closing in on a constant-speed leader: a~_l - dv^2 / (2 s)
    assert cah_acceleration(30.0, 20.0, 10.0, -1.0, P) == pytest.approx(-1.0 - 100.0 / 60.0)
    # first branch (leader faster but braking): v^2 a~_l / (v_l^2 - 2 s a~_l)
    assert cah_acceleration(20.0, 10.0, 20.0, -2.0, P) == pytest.approx(100.0 * -2.0 / (400.0 + 80.0))
    # stopped leader, a_l = 0: the guarded case, -v^2 / (2 s)
    assert cah_acceleration(50.0, 10.0, 0.0, 0.0, P) == pytest.approx(-1.0)
    # effective leader acceleration is capped at a
    assert cah_acceleration(30.0, 15.0, 15.0, 2.0, P) == pytest.approx(P.a)
    # steady following: zero
    assert cah_acceleration(30.0, 15.0, 15.0, 0.0, P) == 0.0


def test_blend_rules_and_bounds():
    assert acc_blend(-0.5, -1.0, P) == -0.5                      # a_IDM >= a_CAH: IDM unchanged
    a_idm, a_cah = -3.0, -0.5
    expect = 0.01 * a_idm + 0.99 * (a_cah + P.b * math.tanh((a_idm - a_cah) / P.b))
    assert acc_blend(a_idm, a_cah, P) == pytest.approx(expect)
    rng = np.random.default_rng(1)
    for a_i, a_c in rng.uniform(-8, 1, size=(500, 2)):
        r = acc_blend(a_i, a_c, P)
        assert min(a_i, a_c) - 1e-12 <= r <= max(a_i, a_c) + 1e-12   # never harsher than IDM, never above CAH
    assert acc_blend(-1.0, -1.0 + 1e-12, P) == pytest.approx(-1.0, abs=1e-9)   # continuous at the switch


def test_acc_command_equals_idm_in_equilibrium_and_on_free_road():
    for v in (5.0, 15.0, 25.0):
        s = float(equilibrium_gap(v, P))
        assert acc_command(s, v, 0.0, 0.0, P) == pytest.approx(0.0, abs=1e-12)   # same steady state as IDM
    assert acc_command(math.inf, 20.0, 0.0, -3.0, P) == pytest.approx(float(acceleration(20.0, np.inf, 0.0, P)))
    # missing a_l: IDM alone (no CAH), not a_l = 0
    s, v, dv = 20.0, 20.0, 5.0
    assert acc_command(s, v, dv, None, P) == pytest.approx(max(float(acceleration(v, s, dv, P)), A_MIN))
    assert acc_command(s, v, dv, None, P) != acc_command(s, v, dv, 0.0, P)


def test_command_is_clipped():
    assert acc_command(0.5, 30.0, 20.0, -5.0, P) == A_MIN
    assert acc_command(math.inf, 0.0, 0.0, None, P) == pytest.approx(P.a)


# --------------------------------------------------------------------------- estimator and lag

def test_estimator_missing_then_step_response():
    e = LeaderAccelEstimator()
    assert e.update(15.0) is None                               # one sample: no estimate yet
    assert e.update(15.0) == 0.0                                # first difference initializes the filter
    v, out = 15.0, []
    for _ in range(10):                                         # leader starts braking at exactly 1 m/s^2
        v -= 1.0 * TICK_S
        out.append(e.update(v))
    n = np.arange(1, 11)
    assert np.allclose(out, -(1 - np.exp(-n * TICK_S / 0.3)), atol=1e-12)
    e.reset()
    assert e.update(10.0) is None


def test_lag_exact_step_response():
    a, dt, tau = 0.0, 0.05, 0.1
    for k in range(1, 21):
        a = lag_step(a, 1.0, dt, tau)
        assert a == pytest.approx(1 - math.exp(-k * dt / tau), abs=1e-12)


# --------------------------------------------------------------------------- ticks and channel

def test_delay_ticks():
    assert [delay_ticks(d) for d in (0.0, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.1)] == [0, 1, 2, 2, 3, 5, 7, 10, 11]
    with pytest.raises(ValueError):
        delay_ticks(-0.1)


def test_channel_delay_since_and_age():
    ch = V2VChannel(delay_s=0.15)                               # 2 ticks
    for k in range(10):
        ch.send(7, k, float(k))
    assert ch.latest(7, 1, 0) is None                           # nothing usable before tick 2
    assert ch.latest(7, 2, 0) == (0.0, 0)
    assert ch.latest(7, 9, 0) == (7.0, 7)                       # newest usable is 2 ticks old
    assert ch.latest(7, 9, 8) is None                           # nothing sent since the link restart yet
    assert ch.latest(3, 9, 0) is None                           # nobody else sent


def test_channel_loss_is_seeded_and_outage_drops_everything():
    a, b = V2VChannel(loss_prob=0.3, seed=181), V2VChannel(loss_prob=0.3, seed=181)
    for k in range(2000):
        a.send(1, k, 0.0)
        b.send(1, k, 0.0)
    assert a.lost == b.lost and 0.27 < a.lost / a.sent < 0.33
    o = V2VChannel(delay_s=0.1, outages=[(1.0, 1.5)])
    for k in range(30):
        o.send(1, k, float(k))
    assert o.lost == 5                                          # ticks 10-14
    assert o.latest(1, 15, 0) == (9.0, 9)                       # last message before the outage


# --------------------------------------------------------------------------- controller and fallback

def _drive(ctl, ch, ticks, leader=0, equipped=True, send=True):
    out = []
    for k in ticks:
        if send and ch is not None:
            ch.send(leader, k, -1.0)
        out.append(ctl.command(k, 15.0, 30.0, 0.0, leader, ch, equipped))
    return out


def test_cacc_uses_v2v_when_fresh_and_logs_each_fallback_reason():
    ch = V2VChannel(delay_s=0.15)
    c = CAVController(1, "cacc", P)
    out = _drive(c, ch, range(5))
    assert [o["fallback"] for o in out[:2]] == ["no_message", "no_message"]   # link starts: 2-tick delay
    assert out[2]["source"] == "v2v" and out[2]["a_l"] == -1.0
    # leader unequipped (never sends)
    c2 = CAVController(2, "cacc", P)
    assert _drive(c2, ch, range(3), leader=5, equipped=False, send=False)[-1]["fallback"] == "leader_unequipped"
    # stale: 1.0 s delay = 10 ticks > MAX_AGE_TICKS
    slow = V2VChannel(delay_s=1.0)
    c3 = CAVController(3, "cacc", P)
    o3 = _drive(c3, slow, range(15))
    assert o3[-1]["fallback"] == "stale" and o3[-1]["source"] == "sensor"
    assert MAX_AGE_TICKS == 5
    # comms off
    c4 = CAVController(4, "cacc", P)
    assert _drive(c4, None, range(3))[-1]["fallback"] == "comms_off"
    assert len(c.log) == 2 and all(r == "no_message" for _, r in c.log)


def test_leader_change_restarts_link_and_estimator():
    ch = V2VChannel(delay_s=0.15)
    c = CAVController(1, "cacc", P)
    _drive(c, ch, range(10), leader=0)
    for k in range(10):
        ch.send(9, k, -2.0)                                     # new leader has been broadcasting all along
    o = c.command(10, 15.0, 30.0, 0.0, 9, ch, True)
    assert o["fallback"] == "no_message" and o["a_l"] is None   # only messages sent since the change count
    ch.send(9, 10, -2.0); ch.send(9, 11, -2.0); ch.send(9, 12, -2.0)
    assert c.command(11, 15.0, 30.0, 0.0, 9, ch, True)["fallback"] == "no_message"
    o = c.command(12, 15.0, 30.0, 0.0, 9, ch, True)
    assert o["source"] == "v2v" and o["a_l"] == -2.0


def test_cacc_with_comms_off_is_identical_to_acc():
    """E9 gate at unit level: same inputs -> same commands, tick by tick."""
    rng = np.random.default_rng(3)
    acc, cacc = CAVController(1, "acc", P), CAVController(1, "cacc", P)
    v_l = 15.0
    for k in range(300):
        v_l = max(0.0, v_l + rng.normal(0, 0.1))
        v, s = 15.0 + rng.normal(0, 0.5), 25.0 + rng.normal(0, 3)
        a = acc.command(k, v, s, v - v_l, 0)
        b = cacc.command(k, v, s, v - v_l, 0, None, True)
        assert a["u"] == b["u"] and a["a_l"] == b["a_l"]
