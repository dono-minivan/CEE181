"""Wave-speed check (tmc.wave_speed, experiment log E10)."""
import numpy as np
import pandas as pd
import pytest

from tmc import wave_speed as w
from tmc.idm import load_params

P = load_params("baseline")


def test_characteristic_speed_tends_to_jam_limit():
    # near v = 0, s_e ~ s0 + vT, so q = (1 - k (s0 + l)) / T and dq/dk = -(s0 + l) / T
    assert w.characteristic_speed(0.01, P) == pytest.approx(-(P.s0 + P.length) / P.T, rel=1e-3)


def test_shock_between_close_states_is_the_characteristic():
    assert w.lwr_shock_speed(15.0, 14.999, P) == pytest.approx(w.characteristic_speed(15.0, P), rel=1e-3)
    assert w.lwr_shock_speed(15.0, 10.0, P) == pytest.approx(w.lwr_shock_speed(10.0, 15.0, P))


def test_shock_speed_and_per_vehicle_delay_are_consistent():
    # omega = v_A - (s_A + l) / tau, the same relation read the other way
    om = w.lwr_shock_speed(15.0, 10.0, P)
    tau = w.per_vehicle_delay(om, 15.0, P)
    k_a, _ = w.eq_state(15.0, P)
    assert 15.0 - (1.0 / k_a) / tau == pytest.approx(om)


def test_linear_platoon_delay_tends_to_lwr_in_the_long_wave_limit():
    tau_lwr = w.per_vehicle_delay(w.characteristic_speed(15.0, P), 15.0, P)
    assert w.linear_delay(15.0, 2 * np.pi / 5000.0, P) == pytest.approx(tau_lwr, rel=2e-3)
    assert w.linear_delay(15.0, 2 * np.pi / 60.0, P) > 1.1 * tau_lwr      # dispersive at finite wavelength


def test_crossings_and_fit_recover_a_known_front():
    t = np.arange(0.0, 120.0, 0.1)
    rows = []
    for i in range(1, 21):
        t_i = 10.0 + 1.8 * i                                  # front passes one vehicle every 1.8 s
        v = 15.0 - 5.0 / (1.0 + np.exp(-(t - t_i) * 3.0))      # crosses 12.5 m/s exactly at t_i
        x = -31.55 * i + 15.0 * t
        rows.append(pd.DataFrame({"vehicle_id": i, "time_s": t, "position_m": x, "speed_mps": v}))
    per = w.crossings(pd.concat(rows), 12.5, range(1, 21))
    assert np.allclose(per["t_brake_s"], 10.0 + 1.8 * per["vehicle_index"], atol=1e-3)
    assert per["t_recover_s"].isna().all()                    # never climbs back: stays missing, not zero
    slope, r2, n = w.fit_speed(per["t_brake_s"].to_numpy(), per["x_brake_m"].to_numpy())
    assert n == 20 and r2 > 0.9999
    assert slope == pytest.approx(15.0 - 31.55 / 1.8, abs=1e-3)


def test_e10_strong_front_reproduces_and_is_dt_insensitive():
    a, _ = w.measure("strong", 0.0, 0.1)
    b, _ = w.measure("strong", 0.0, 0.05)
    assert a["brake_n"] == 30 and a["brake_r2"] > 0.99
    assert a["brake_front_mps"] == pytest.approx(-3.448, abs=0.01)  # E10 result
    assert abs(b["brake_front_mps"] / a["brake_front_mps"] - 1) < w.TOL_DT
    assert abs(a["tau_brake_s"] / a["tau_lwr_s"] - 1) < 0.05       # per vehicle within 5% of LWR
