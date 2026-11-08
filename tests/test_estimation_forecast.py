"""Estimation and forecast units (detector data only)."""
import numpy as np
import pandas as pd
import pytest

from tmc.estimation import RampQueueEstimator, density_vpm, flow_vph
from tmc.evaluate import forecast_skill
from tmc.forecast import ExpSmoothing, queue_forecast
from tmc.interfaces import Observation


class View:
    def __init__(self):
        self.o = {}

    def latest(self, sid, n=1):
        return [self.o[sid]] if sid in self.o else []


def obs(sid, t, count, occ=None, flag="ok"):
    return Observation(sid, t - 30.0, t, count, None, flag, occ)


def test_flow_and_density():
    assert flow_vph(obs("a", 30.0, 10)) == pytest.approx(1200.0)
    assert flow_vph(None) is None
    assert density_vpm(obs("a", 30.0, 10, occ=0.14), 5.0, 2.0) == pytest.approx(0.02)
    assert density_vpm(obs("a", 30.0, 10), 5.0, 2.0) is None          # no occupancy -> missing, not 0


def test_queue_counting_accumulates_once_per_interval():
    e, v = RampQueueEstimator(), View()
    v.o = {"R_dem": obs("R_dem", 30.0, 5), "R_pass": obs("R_pass", 30.0, 2)}
    assert e.update(v) == 3
    assert e.update(v) == 3                       # same interval again: not double counted
    v.o = {"R_dem": obs("R_dem", 60.0, 1), "R_pass": obs("R_pass", 60.0, 4)}
    assert e.update(v) == 0                       # 6 in, 6 out


def test_queue_estimate_missing_stays_missing():
    e, v = RampQueueEstimator(), View()
    assert e.update(v) is None
    v.o = {"R_dem": obs("R_dem", 30.0, 5, flag="missing"), "R_pass": obs("R_pass", 30.0, 2)}
    assert e.update(v) is None
    v.o = {"R_dem": obs("R_dem", 60.0, 5), "R_pass": obs("R_pass", 60.0, 2)}
    assert e.update(v) is None                    # a lost interval invalidates the count, never guessed


def test_exp_smoothing():
    es = ExpSmoothing(alpha=0.5)
    assert es.update(None) is None
    assert es.update(100.0) == 100.0
    assert es.update(200.0) == pytest.approx(150.0)
    assert es.update(None) == pytest.approx(150.0)       # missing keeps the level
    assert es.rmse is None                                # one error so far
    es.update(150.0)
    assert es.rmse == pytest.approx(np.sqrt((100.0 ** 2 + 0.0) / 2))


def test_queue_forecast_mass_balance():
    assert queue_forecast(20, 500, 300, 60) == pytest.approx(20 + 200 / 60)
    assert queue_forecast(1, 0, 900, 60) == 0.0
    assert queue_forecast(None, 500, 300, 60) is None


def test_forecast_skill_perfect_forecast():
    t = np.arange(0, 300, 30.0)
    q = np.arange(len(t), dtype=float)                      # queue grows 1 per interval
    tr = pd.DataFrame({"time_s": t, "queue_est_veh": q, "queue_fc_veh": q + 2,  # 60 s = 2 intervals ahead
                       "ramp_demand_smoothed_vph": 300.0, "ramp_demand_meas_vph": 300.0})
    sk = forecast_skill(tr)
    assert sk["queue_fc_mae_veh"] == pytest.approx(0.0)
    assert sk["queue_persist_mae_veh"] == pytest.approx(2.0)
    assert sk["demand_fc_mae_vph"] == pytest.approx(0.0)
