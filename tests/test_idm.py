"""Verification of IDM dynamics in isolation: free road, equilibrium, stopping, derived quantities."""
import numpy as np
import pytest

from tmc.idm import (acceleration, ballistic_update, capacity, equilibrium_flow, equilibrium_gap,
                     fundamental_diagram, load_params)

P = load_params("baseline")


def test_params_load_from_yaml():
    assert P.v0 == 33.3 and P.T == 1.6 and P.s0 == 2.0 and P.a == 0.73 and P.b == 1.67
    s = load_params("sensitivity")
    assert s.T == 1.0 and s.a == 1.0 and s.b == 1.5


def test_free_road_acceleration_bounds():
    v = np.linspace(0, P.v0, 50)
    acc = acceleration(v, np.full_like(v, np.inf), np.zeros_like(v), P)
    assert acc[0] == pytest.approx(P.a)            # standing start: a
    assert acc[-1] == pytest.approx(0.0, abs=1e-12)  # at v0: 0
    assert np.all(np.diff(acc) < 0)               # decreasing in v


def test_equilibrium_gap_gives_zero_acceleration():
    v = np.linspace(0.5, 0.99 * P.v0, 40)
    acc = acceleration(v, equilibrium_gap(v, P), np.zeros_like(v), P)
    assert np.allclose(acc, 0.0, atol=1e-10)


def test_ballistic_stops_inside_step_without_reversing():
    x, v = ballistic_update(np.array([10.0]), np.array([1.0]), np.array([-5.0]), dt=1.0)
    assert v[0] == 0.0
    assert x[0] == pytest.approx(10.0 + 1.0**2 / (2 * 5.0))


def _follow(lead_x, lead_v, fx, fv, dt, steps, lead_speed_fn=None):
    """Integrate one follower behind a prescribed leader. Returns arrays of gap and follower speed."""
    gaps, speeds = [], []
    for k in range(steps):
        s = lead_x - fx - P.length
        gaps.append(s)
        speeds.append(fv)
        acc = float(acceleration(fv, s, fv - lead_v, P))
        (fx,), (fv,) = ballistic_update([fx], [fv], [acc], dt)
        lead_x += lead_v * dt
    return np.array(gaps), np.array(speeds)


def test_equilibrium_following_is_stationary():
    v_e = 20.0
    s_e = float(equilibrium_gap(v_e, P))
    gaps, speeds = _follow(lead_x=s_e + P.length, lead_v=v_e, fx=0.0, fv=v_e, dt=0.1, steps=3000)
    assert np.max(np.abs(gaps - s_e)) < 1e-6
    assert np.max(np.abs(speeds - v_e)) < 1e-6


@pytest.mark.parametrize("dt", [0.05, 0.1, 0.2, 0.5])
def test_approach_to_stopped_vehicle_never_collides(dt):
    gaps, speeds = _follow(lead_x=300.0, lead_v=0.0, fx=0.0, fv=P.v0, dt=dt, steps=int(200 / dt))
    assert gaps.min() > 0
    assert speeds.min() >= 0
    # IDM braking from v0 overshoots s0 slightly (converges to ~1.89 m as dt -> 0 for baseline params);
    # this is a model property, not a numerical error. Require a safe stop close to s0.
    assert 0.9 * P.s0 < gaps[-1] <= P.s0 + 0.05
    assert speeds[-1] == pytest.approx(0.0, abs=1e-3)


def test_capacity_is_recomputed_and_consistent():
    c = capacity(P)
    k, q, v = fundamental_diagram(P)
    assert c["q_c_veh_per_s"] == pytest.approx(q.max(), rel=1e-4)
    assert c["q_c_veh_per_s"] < 1.0 / P.T             # s_e > s0 + vT  =>  q < 1/T
    assert 0 < c["v_c_mps"] < P.v0
    assert c["k_jam_veh_per_m"] == pytest.approx(1 / (P.s0 + P.length))
    assert equilibrium_flow(c["v_c_mps"], P) == pytest.approx(c["q_c_veh_per_s"])
    # shorter T / larger a should raise capacity (sensitivity direction check)
    assert capacity(load_params("sensitivity"))["q_c_veh_per_s"] > c["q_c_veh_per_s"]


# --------------------------------------------------------------------------- string stability

from dataclasses import replace

from tmc.idm import (equilibrium_partials, max_platoon_gain, platoon_gain, string_stability_margin,
                     string_unstable_ranges)


@pytest.mark.parametrize("v_e", [5.0, 15.0, 22.0, 28.0])
def test_analytic_partials_match_finite_differences(v_e):
    d = equilibrium_partials(v_e, P)
    s, h = float(d["s_e"]), 1e-5
    f_s = (acceleration(v_e, s + h, 0.0, P) - acceleration(v_e, s - h, 0.0, P)) / (2 * h)
    f_v = (acceleration(v_e + h, s, 0.0, P) - acceleration(v_e - h, s, 0.0, P)) / (2 * h)
    f_dv = (acceleration(v_e, s, h, P) - acceleration(v_e, s, -h, P)) / (2 * h)
    assert float(d["f_s"]) == pytest.approx(float(f_s), rel=1e-5)
    assert float(d["f_v"]) == pytest.approx(float(f_v), rel=1e-5)
    assert float(d["f_dv"]) == pytest.approx(float(f_dv), rel=1e-5)


def test_unstable_ranges_reproduce_values_quoted_in_E9():
    """Experiment log E9 quotes: baseline unstable below ~22.9 m/s; T = 0.6 s below ~24.5 m/s."""
    (lo, hi), = string_unstable_ranges(P)
    assert lo < 0.1 and hi == pytest.approx(22.9, abs=0.05)
    (lo6, hi6), = string_unstable_ranges(replace(P, T=0.6))
    assert hi6 == pytest.approx(24.5, abs=0.05)
    assert hi6 > hi  # shorter time gap alone destabilizes (the E9 argument)
    # E9 setting A runs at 15 m/s, which must sit inside the unstable range
    assert string_stability_margin(15.0, P) < 0


def test_stronger_acceleration_stabilizes():
    """Direction check: a more responsive driver (larger a) narrows the unstable range."""
    hi = string_unstable_ranges(P)[0][1]
    hi_fast = string_unstable_ranges(replace(P, a=1.5))
    assert not hi_fast or hi_fast[0][1] < hi


@pytest.mark.parametrize("v_e", [5.0, 10.0, 15.0, 20.0, 22.5, 23.5, 26.0, 30.0])
def test_transfer_function_agrees_with_margin(v_e):
    """Two analytic routes agree: peak gain > 1 exactly where the T&K 15.77 margin is negative."""
    unstable = string_stability_margin(v_e, P) < 0
    assert (max_platoon_gain(v_e, P)["gain"] > 1.0) == unstable


def _sinusoid_gain(v_e, period, n=20, dt=0.1, t_end=1200.0, amp=0.05):
    """Simulate an open platoon behind a leader oscillating at `period`; return the measured gain per vehicle."""
    w = 2 * np.pi / period
    s_e = float(equilibrium_gap(v_e, P))
    x = -np.arange(n + 1) * (s_e + P.length)          # index 0 = leader
    v = np.full(n + 1, v_e)
    steps = int(t_end / dt)
    hist = np.empty((steps, n + 1))
    for k in range(steps):
        t = k * dt
        ramp = min(1.0, t / 120.0)                    # fade the forcing in to limit transients
        a_lead = amp * w * (ramp * np.cos(w * t) + (1 / (120.0 * w) if t < 120 else 0.0) * np.sin(w * t))
        gap = x[:-1] - x[1:] - P.length
        acc = np.concatenate([[a_lead], acceleration(v[1:], gap, v[1:] - v[:-1], P)])
        x, v = ballistic_update(x, v, acc, dt)
        assert np.all(x[:-1] - x[1:] - P.length > 0)
        hist[k] = v
    tail = hist[int(0.6 * steps):]                    # periodic steady state
    amp_v = 0.5 * (tail.max(axis=0) - tail.min(axis=0))
    return (amp_v[-1] / amp_v[0]) ** (1.0 / n)


@pytest.mark.parametrize("v_e, period", [(15.0, None), (28.0, 60.0)])
def test_simulated_platoon_gain_matches_linear_theory(v_e, period):
    """The simulator reproduces the analytic per-vehicle gain. At 15 m/s (unstable range) the most
    amplified wave grows; at 28 m/s (stable) a 60 s wave decays. (In the stable range the peak sits at
    omega -> 0 with |G| -> 1, so a fixed finite period is used there.)"""
    if period is None:
        period = max_platoon_gain(v_e, P)["period_s"]
    g_theory = float(platoon_gain(v_e, 2 * np.pi / period, P))
    g_sim = _sinusoid_gain(v_e, period)
    assert g_sim == pytest.approx(g_theory, abs=2e-3)
    assert (g_sim > 1.0) == (string_stability_margin(v_e, P) < 0)
