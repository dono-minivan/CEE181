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
