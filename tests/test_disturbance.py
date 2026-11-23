"""Checks of the E9 setting A harness (tmc.disturbance): scripted lead, g fit, metric bookkeeping."""
import numpy as np
import pytest

from tmc.disturbance import (DV_FLOOR, MIN_POINTS, V_STOP, Disturbance, growth_rate, load_setting,
                             platoon_metrics, run_platoon)
from tmc.idm import equilibrium_gap, load_params

P = load_params("baseline")
S = load_setting()


def test_setting_matches_E9_spec():
    assert S["v_eq_mps"] == 15.0 and S["n_followers"] == 30
    nom, strong = S["disturbances"]["nominal"], S["disturbances"]["strong"]
    assert (nom.v_low, nom.b_lead, nom.hold_s) == (10.0, 2.0, 10.0)
    assert (strong.v_low, strong.b_lead, strong.hold_s) == (5.0, 2.0, 10.0)
    assert S["g_threshold_per_veh"] == pytest.approx(np.log(1 / 0.9) / 30, abs=1e-4)


def test_scripted_lead_profile():
    """Lead alone: cruises, brakes at exactly b_lead, lands on v_low, holds hold_s, recovers at a to v_eq."""
    s = dict(S, n_followers=0, duration_s=60.0)
    d = Disturbance("t", v_low=10.0, b_lead=2.0, hold_s=10.0, t_start=10.0)
    res, _ = run_platoon(s, d, 0.0)
    tr = res.trajectories
    t, v, a = tr["time_s"].to_numpy(), tr["speed_mps"].to_numpy(), tr["acceleration_mps2"].to_numpy()
    assert np.all(v[t < 10.0] == 15.0)
    assert a.min() == pytest.approx(-2.0)
    assert v.min() == pytest.approx(10.0, abs=1e-9)
    t_low = t[np.argmax(v <= 10.0 + 1e-9)]
    assert t_low == pytest.approx(12.5, abs=0.11)                   # (15 - 10) / 2 s after the start
    hold = (t >= t_low) & (t < t_low + 10.0 - 1e-9)
    assert np.all(np.abs(v[hold] - 10.0) < 1e-9)
    assert a.max() == pytest.approx(P.a)
    assert v[-1] == pytest.approx(15.0, abs=1e-9) and v.max() <= 15.0 + 1e-9


def test_growth_rate_recovers_known_slope_and_applies_floor():
    i = np.arange(31)
    dv = 2.0 * np.exp(0.02 * i)
    g, why, n = growth_rate(dv, np.full(31, 10.0))
    assert g == pytest.approx(0.02, abs=1e-12) and why == "" and n == 31
    # decayed wave: deficits below the floor are excluded, too few left -> not estimable
    dv2 = 0.5 * np.exp(-1.5 * i)                    # above the floor only for i = 0, 1, 2
    g2, why2, n2 = growth_rate(dv2, np.full(31, 10.0))
    assert n2 == int((dv2 > DV_FLOOR).sum()) < MIN_POINTS and np.isnan(g2) and "below" in why2
    # stopped vehicles are excluded and counted in the reason
    vmin = np.where(i > 3, 0.0, 10.0)
    g3, why3, _ = growth_rate(dv, vmin)
    assert np.isnan(g3) and "27 stopped" in why3
    assert V_STOP == 0.5 and DV_FLOOR == 0.01


def test_metrics_on_undisturbed_platoon():
    """No disturbance (start after the run ends): deficits 0, g not estimable, gaps = equilibrium gap."""
    s = dict(S, n_followers=5, duration_s=30.0)
    d = Disturbance("none", v_low=10.0, b_lead=2.0, hold_s=10.0, t_start=1e9)
    res, p = run_platoon(s, d, 0.6)
    m, per = platoon_metrics(res, p, 15.0)
    s_e = float(equilibrium_gap(15.0, P))
    assert np.allclose(per["deficit_mps"], 0.0, atol=1e-9)
    assert np.isnan(m["g_per_veh"]) and m["n_stopped"] == 0
    assert m["min_gap_m"] == pytest.approx(s_e, abs=1e-6)
    assert m["min_time_gap_s"] == pytest.approx(s_e / 15.0, abs=1e-6)
    assert m["max_decel_mps2"] == pytest.approx(0.0, abs=1e-9)
