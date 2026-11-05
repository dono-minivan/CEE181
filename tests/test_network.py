"""Verification of the single-lane segment: conservation, free motion, gaps, boundaries, time step."""
import numpy as np
import pytest

from tmc.idm import capacity, load_params
from tmc.network import ScenarioConfig, SingleLaneSegment, generate_arrivals

P = load_params("baseline")


def cfg(**kw):
    base = dict(scenario_id="test", demand_id="test", param_set="baseline", seed=1, length_m=1000.0,
                dt_s=0.1, record_every_s=0.1, max_time_s=3600.0, arrival_process="uniform",
                demand_profile=[[0, 300, 1200]])
    base.update(kw)
    return ScenarioConfig(**base)


def run(**kw):
    c = cfg(**kw)
    return SingleLaneSegment(P, c).run()


def test_free_motion_single_vehicle():
    res = run(length_m=5000.0, demand_profile=[], initial_vehicles=[[0.0, 0.0]])
    tr = res.trajectories
    assert np.all(np.diff(tr["speed_mps"].to_numpy()) >= -1e-12)  # never slows on an empty road
    assert tr["speed_mps"].max() < P.v0
    assert tr["acceleration_mps2"].max() <= P.a + 1e-12
    assert tr["speed_mps"].iloc[-1] > 0.95 * P.v0
    assert res.cleared


def test_conservation_every_step_and_nothing_dropped():
    res = run(demand_profile=[[0, 300, 2400], [300, 600, 600]])  # over capacity, then under
    c = res.counts
    assert np.all(c["n_arrived"] == c["n_queued"] + c["n_on_road"] + c["n_exited"])
    s = res.summary()
    assert s["n_arrived"] == len(generate_arrivals([[0, 300, 2400], [300, 600, 600]], "uniform", 1))
    assert s["n_exited"] == s["n_arrived"] and s["unmet_demand_veh"] == 0 and res.cleared
    assert s["peak_entry_queue_veh"] > 0             # over-capacity demand must queue, not vanish
    assert res.min_gap_m > 0


def test_time_limit_reports_leftovers_instead_of_dropping():
    res = run(demand_profile=[[0, 600, 2400]], max_time_s=300.0)
    s = res.summary()
    assert not res.cleared
    assert s["n_arrived"] == s["n_exited"] + s["unmet_demand_veh"] + s["on_road_at_end_veh"]
    assert s["unmet_demand_veh"] > 0 and s["demand_not_yet_arrived_veh"] > 0
    assert s["mean_travel_time_s"] > 0


def test_saturated_entry_discharges_near_capacity():
    res = run(demand_profile=[[0, 900, 2400]], record_every_s=1.0)
    e = np.sort(res.vehicles["entry_s"].to_numpy())
    e = e[(e > 300) & (e < 850)]                    # queue present throughout this window
    q_entry = 3600 * (len(e) - 1) / (e[-1] - e[0])
    assert q_entry == pytest.approx(capacity(P)["q_c_veh_per_h"], rel=0.05)


def test_no_overlap_under_heavy_poisson_demand():
    res = run(arrival_process="poisson", seed=7, demand_profile=[[0, 600, 3000]], record_every_s=1.0)
    assert res.min_gap_m > P.s0                      # no stop-and-go here, gaps stay above jam gap
    assert res.cleared


def test_time_step_sensitivity():
    """Halving/doubling dt must not change aggregate results meaningfully."""
    out = {dt: run(dt_s=dt, demand_profile=[[0, 300, 2400], [300, 600, 800]], record_every_s=1.0).summary()
           for dt in (0.05, 0.1, 0.2)}
    ref = out[0.05]
    for dt in (0.1, 0.2):
        assert out[dt]["mean_travel_time_s"] == pytest.approx(ref["mean_travel_time_s"], rel=0.01)
        assert out[dt]["total_travel_time_veh_h"] == pytest.approx(ref["total_travel_time_veh_h"], rel=0.01)
