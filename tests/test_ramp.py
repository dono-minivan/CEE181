"""Verification of the on-ramp merge: conservation, gaps, lane end, merging, no-ramp equivalence.

Every check runs under each merge-behavior variant, because the merge model is still provisional
(docs/experiment_log.md E1) and whichever variant is chosen must stay verified.
"""
import numpy as np
import pytest

from tmc.idm import load_params
from tmc.network import MAIN, RAMP, STOP_MARGIN_M, FreewaySegment, RampConfig, ScenarioConfig

P = load_params("baseline")
VARIANTS = {
    "plain": {},
    "yield": {"cooperative_yield": True},
    "yield_ratio": {"cooperative_yield": True, "yield_min_speed_ratio": 0.5},
    "adapt": {"speed_adaptation": True},
}


def run(variant="plain", qm=1200, qr=400, horizon=600, **kw):
    ramp = RampConfig(merge_start_m=600.0, merge_end_m=800.0, approach_length_m=200.0,
                      demand_profile=[[0, horizon, qr]], **VARIANTS[variant])
    cfg = ScenarioConfig("test_ramp", "test", "baseline", 3, 1500.0, 0.1, 0.5, 7200.0, "poisson",
                         [[0, horizon, qm]], ramp=ramp)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return FreewaySegment(P, cfg).run()


@pytest.fixture(scope="module", params=list(VARIANTS))
def res(request):
    return run(request.param, qm=1300, qr=500)  # X ~ 1.03: congested merge exercises every branch


def test_conservation_every_step(res):
    c = res.counts
    assert np.all(c["n_arrived"] == c["n_queued"] + c["n_on_road"] + c["n_exited"])
    assert np.all(c["n_queued"] == c["n_queued_main"] + c["n_queued_ramp"])
    assert np.all(c["n_on_road"] == c["n_on_main"] + c["n_on_ramp"])


def test_everyone_merges_and_exits(res):
    v, s = res.vehicles, res.summary()
    assert res.cleared and s["unmet_demand_veh"] == 0
    ramp = v[v["origin"] == "ramp"]
    assert ramp["merge_s"].notna().all()
    assert (ramp["entry_s"] <= ramp["merge_s"]).all() and (ramp["merge_s"] <= ramp["exit_s"]).all()
    assert v.loc[v["origin"] == "main", "merge_s"].isna().all()


def test_positive_gaps_and_lane_end(res):
    assert res.min_gap_m > 0
    tr = res.trajectories
    on_ramp = tr[tr["lane_id"] == RAMP]
    # constant-deceleration stop lands on the stop point up to float rounding (< 1 mm; margin is 1 m)
    assert on_ramp["position_m"].max() <= res.config.ramp.merge_end_m - STOP_MARGIN_M + 1e-3
    assert tr["speed_mps"].min() >= 0


def test_merges_only_inside_merge_zone(res):
    tr, v = res.trajectories, res.vehicles
    first_main = tr[tr["lane_id"] == MAIN].groupby("vehicle_id")["position_m"].first()
    ramp_ids = v.loc[v["origin"] == "ramp", "vehicle_id"]
    x_at_merge = first_main.loc[ramp_ids]
    # first mainline record comes up to one record interval after the merge, so allow that much travel
    assert (x_at_merge >= res.config.ramp.merge_start_m).all()
    assert (x_at_merge <= res.config.ramp.merge_end_m + P.v0 * res.config.record_every_s).all()


def test_zero_ramp_demand_matches_single_lane():
    """A configured ramp with no demand must not change mainline results."""
    with_ramp = run(qr=0, qm=1500).summary()
    cfg = ScenarioConfig("test_ramp", "test", "baseline", 3, 1500.0, 0.1, 0.5, 7200.0, "poisson", [[0, 600, 1500]])
    without = FreewaySegment(P, cfg).run().summary()
    for k in ("n_arrived", "total_travel_time_veh_h", "end_time_s", "min_gap_m"):
        assert with_ramp[k] == pytest.approx(without[k])
