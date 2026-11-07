"""Detector checks: count audit against simulator truth, speed/occupancy on a known case, passivity."""
import copy
from pathlib import Path

import numpy as np
import pytest

from tmc.detectors import DetectorBank, DetectorSpec
from tmc.idm import load_params
from tmc.network import FreewaySegment, ScenarioConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
S1 = CONFIGS / "scenario_s1_onramp.yaml"
S2 = CONFIGS / "scenario_s2_twolane_onramp.yaml"
I = 30.0


def short_s1(detectors, main=1300.0, ramp=400.0, horizon=600.0):
    cfg = ScenarioConfig.from_yaml(S1)
    cfg.demand_profile = [[0.0, horizon, main]]
    cfg.ramp.demand_profile = [[0.0, horizon, ramp]]
    cfg.record_every_s = 5.0
    cfg.detectors = detectors
    return cfg


AUDIT = [
    DetectorSpec("main_arr", kind="arrivals", boundary="main", interval_s=I),
    DetectorSpec("ramp_arr", kind="arrivals", boundary="ramp", interval_s=I),
    DetectorSpec("up", position_m=1000.0, lane_id=0, interval_s=I),
    DetectorSpec("ramp_stop", position_m=1260.0, lane_id=-1, interval_s=I),   # entrance is at 1250 m
    DetectorSpec("down", position_m=1900.0, lane_id=0, interval_s=I),
    DetectorSpec("exit", position_m=3000.0, lane_id=0, interval_s=I),
]


@pytest.fixture(scope="module")
def audit_run():
    cfg = short_s1(AUDIT)
    return FreewaySegment(load_params(cfg.param_set), cfg).run()


def binned(times, end):
    edges = np.arange(0.0, end + I, I)
    return np.histogram(times[np.isfinite(times)], bins=edges)[0]


def counts_of(obs, sid):
    return obs.loc[obs["sensor_id"] == sid].sort_values("interval_start_s")["count"].to_numpy()


def test_count_totals_match_truth(audit_run):
    r = audit_run
    assert r.cleared
    v, obs = r.vehicles, r.observations
    tot = obs.groupby("sensor_id")["count"].sum()
    n_main, n_ramp = int((v["origin"] == "main").sum()), int((v["origin"] == "ramp").sum())
    assert tot["main_arr"] == n_main
    assert tot["ramp_arr"] == n_ramp
    assert tot["up"] == n_main                 # upstream of the merge: mainline vehicles only
    assert tot["ramp_stop"] == n_ramp
    assert tot["down"] == n_main + n_ramp      # downstream of the merge zone (1750 m): everyone
    assert tot["exit"] == len(v)


def test_interval_counts_match_truth(audit_run):
    """Per-interval audit: the exit loop sits on the boundary, so its counts must equal the binned
    simulator exit times exactly; the arrival counters must equal the binned arrival times."""
    r = audit_run
    v, obs, end = r.vehicles, r.observations, r.end_time_s
    ex = counts_of(obs, "exit")
    truth = binned(v["exit_s"].to_numpy(), end)
    assert np.array_equal(ex, truth[:len(ex)]) and truth[len(ex):].sum() == 0
    arr = counts_of(obs, "ramp_arr")
    truth = binned(v.loc[v["origin"] == "ramp", "arrival_s"].to_numpy(), end)
    assert np.array_equal(arr, truth[:len(arr)])


def test_intervals_are_contiguous_and_missing_is_explicit(audit_run):
    obs = audit_run.observations
    for sid, g in obs.groupby("sensor_id"):
        g = g.sort_values("interval_start_s")
        assert g["interval_start_s"].iloc[0] == 0.0
        assert np.allclose(g["interval_start_s"].to_numpy()[1:], g["interval_end_s"].to_numpy()[:-1])
        assert set(g["quality_flag"].iloc[:-1]) == {"ok"}
    loops = obs[obs["sensor_id"].isin(["up", "ramp_stop", "down", "exit"])]
    # speed is NaN exactly when nothing crossed (never a silent 0); otherwise physical
    assert (loops["mean_speed_mps"].isna() == (loops["count"] == 0)).all()
    sp = loops["mean_speed_mps"].dropna()
    assert (sp > 0).all() and (sp <= load_params("baseline").v0 + 1e-6).all()
    assert loops["occupancy"].between(0.0, 1.0).all()
    arr = obs[obs["sensor_id"] == "main_arr"]
    assert arr["mean_speed_mps"].isna().all() and arr["occupancy"].isna().all()


def test_single_vehicle_speed_and_occupancy():
    """One vehicle at v0 on an empty road (IDM acceleration 0): spot speed = v0, and the loop is
    occupied for (l + field_length) / v0 seconds, to within one step."""
    p = load_params("baseline")
    cfg = ScenarioConfig.from_yaml(CONFIGS / "scenario_s0_single_lane.yaml")
    cfg.demand_profile = []
    cfg.initial_vehicles = [[100.0, p.v0]]
    cfg.detectors = [DetectorSpec("d", position_m=1000.0, lane_id=0, interval_s=60.0, field_length_m=2.0)]
    r = FreewaySegment(p, cfg).run()
    hit = r.observations[r.observations["count"] > 0]
    assert len(hit) == 1 and hit["count"].iloc[0] == 1
    assert hit["mean_speed_mps"].iloc[0] == pytest.approx(p.v0, rel=1e-9)
    occ_time = hit["occupancy"].iloc[0] * 60.0
    assert occ_time == pytest.approx((p.length + 2.0) / p.v0, abs=cfg.dt_s + 1e-9)


def test_station_equals_sum_of_lane_loops():
    cfg = ScenarioConfig.from_yaml(S2)
    cfg.demand_profile = [[0.0, 400.0, 2400.0]]
    cfg.ramp.demand_profile = [[0.0, 400.0, 0.0]]
    cfg.record_every_s = 5.0
    cfg.detectors = [DetectorSpec("st", position_m=1200.0, interval_s=I),
                     DetectorSpec("l0", position_m=1200.0, lane_id=0, interval_s=I),
                     DetectorSpec("l1", position_m=1200.0, lane_id=1, interval_s=I)]
    obs = FreewaySegment(load_params(cfg.param_set), cfg).run().observations
    st, l0, l1 = (obs[obs["sensor_id"] == s].sort_values("interval_start_s") for s in ("st", "l0", "l1"))
    assert np.array_equal(st["count"].to_numpy(), l0["count"].to_numpy() + l1["count"].to_numpy())
    assert np.allclose(st["occupancy"].to_numpy(),
                       (l0["occupancy"].to_numpy() + l1["occupancy"].to_numpy()) / 2)
    assert st["count"].sum() > 0


def test_detectors_do_not_change_the_simulation():
    base = short_s1([])
    with_det = copy.deepcopy(base)
    with_det.detectors = AUDIT
    a = FreewaySegment(load_params(base.param_set), base).run()
    b = FreewaySegment(load_params(base.param_set), with_det).run()
    assert a.summary() == b.summary()
    assert a.trajectories.equals(b.trajectories)
    assert a.observations.empty


def test_bad_specs_rejected():
    with pytest.raises(ValueError):
        DetectorBank([DetectorSpec("a", position_m=10.0, interval_s=30.05)], 1, 5.0, 0.1)
    with pytest.raises(ValueError):
        DetectorBank([DetectorSpec("a", kind="arrivals", boundary="side")], 1, 5.0, 0.1)
    with pytest.raises(ValueError):
        DetectorBank([DetectorSpec("a", position_m=10.0), DetectorSpec("a", position_m=20.0)], 1, 5.0, 0.1)
    with pytest.raises(ValueError):
        DetectorBank([DetectorSpec("a", position_m=10.0, lane_id=2)], 2, 5.0, 0.1)
