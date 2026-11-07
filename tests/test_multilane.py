"""Verification of multi-lane mainline + MOBIL lane changing (network.FreewaySegment, n_lanes >= 2).

These check the lane-change MECHANICS (conservation, safety, cooldown, incentive, entry capacity). They do
not claim the two-lane on-ramp merge is credible yet: see docs/experiment_log.md E2.
"""
import numpy as np
import pytest

from tmc.idm import capacity, load_params
from tmc.network import MAIN, RAMP, FreewaySegment, LaneChangeConfig, RampConfig, ScenarioConfig

P = load_params("baseline")


def cfg(n_lanes=2, length=1500.0, profile=((0, 300, 2400),), ramp=None, **kw):
    c = ScenarioConfig("test_multilane", "test", "baseline", 5, length, 0.1, 0.1, 3600.0, "poisson",
                       [list(r) for r in profile], ramp=ramp, n_lanes=n_lanes)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def run(**kw):
    return FreewaySegment(P, cfg(**kw)).run()


def slow_leader_case(**lc_kw):
    """Fast vehicle (lane 0, x=200, 30 m/s) closing on a slow one (lane 0, x=400, 5 m/s); lane 1 empty."""
    return run(length=3000.0, profile=(), initial_vehicles=[[400.0, 5.0, 0], [200.0, 30.0, 0]],
               lane_change=LaneChangeConfig(**lc_kw))


@pytest.fixture(scope="module")
def busy():
    """Two lanes + ramp with cooperative yielding: produces many lane changes and merges."""
    ramp = RampConfig(600.0, 800.0, 200.0, [[0, 300, 600]], cooperative_yield=True)
    return run(profile=((0, 300, 2600),), ramp=ramp)


def test_overtakes_slow_leader():
    res = slow_leader_case()
    lc = res.lane_changes
    fast = 1  # vehicle ids: initial vehicles sorted downstream-first -> slow = 0, fast = 1
    assert ((lc["vehicle_id"] == fast) & (lc["from_lane"] == 0) & (lc["to_lane"] == 1)).any()
    assert res.min_gap_m > P.s0
    assert res.cleared


def test_no_change_without_incentive():
    """A threshold no gain can reach means nobody changes; the follower just brakes behind the leader."""
    res = slow_leader_case(threshold_mps2=100.0)
    assert len(res.lane_changes) == 0
    assert res.min_gap_m > 0 and res.cleared


def test_lane_changes_can_be_disabled():
    res = slow_leader_case(enabled=False)
    assert len(res.lane_changes) == 0


def test_conservation_two_lanes_with_ramp(busy):
    c = busy.counts
    assert np.all(c["n_arrived"] == c["n_queued"] + c["n_on_road"] + c["n_exited"])
    assert np.all(c["n_on_main"] == c["n_on_lane_0"] + c["n_on_lane_1"])
    assert np.all(c["n_on_road"] == c["n_on_main"] + c["n_on_ramp"])
    s = busy.summary()
    assert busy.cleared and s["unmet_demand_veh"] == 0 and s["n_exited"] == s["n_arrived"]
    assert busy.min_gap_m > 0
    assert busy.trajectories["speed_mps"].min() >= 0


def test_lane_ids_and_merge_target(busy):
    assert set(busy.trajectories["lane_id"].unique()) <= {RAMP, 0, 1}
    lc = busy.lane_changes
    merges = lc[lc["kind"] == "merge"]
    assert len(merges) == busy.summary()["n_merged"] > 0
    assert (merges["from_lane"] == RAMP).all() and (merges["to_lane"] == MAIN).all()
    d = lc[lc["kind"] == "discretionary"]
    assert len(d) > 0
    assert ((d["to_lane"] - d["from_lane"]).abs() == 1).all() and (d["to_lane"] >= 0).all()


def test_every_change_satisfies_mobil_safety(busy):
    """In the recorded state right after a step's lane changes (record_every = dt), the changer and its
    new follower decelerate no harder than b_safe. Holds even when several vehicles change in one step,
    because each change was re-checked against the lanes as already updated."""
    b_safe = busy.config.lane_change.b_safe_mps2
    tr = busy.trajectories.set_index(["time_s", "lane_id"]).sort_index()
    d = busy.lane_changes[busy.lane_changes["kind"] == "discretionary"]
    assert len(d) > 20
    for _, e in d.iterrows():
        snap = tr.loc[(e["time_s"], e["to_lane"])].reset_index(drop=True)
        me = snap[snap["vehicle_id"] == e["vehicle_id"]]
        assert len(me) == 1 and me["acceleration_mps2"].iloc[0] >= -b_safe - 1e-9
        behind = snap[snap["position_m"] < me["position_m"].iloc[0]]
        if len(behind):
            follower = behind.loc[behind["position_m"].idxmax()]
            assert follower["acceleration_mps2"] >= -b_safe - 1e-9


def test_cooldown_between_changes(busy):
    lc = busy.lane_changes.sort_values(["vehicle_id", "time_s"])
    gaps = lc.groupby("vehicle_id")["time_s"].diff().dropna()
    assert (gaps >= busy.config.lane_change.cooldown_s - 1e-9).all()


def test_vehicle_table_counts_changes(busy):
    d = busy.lane_changes[busy.lane_changes["kind"] == "discretionary"]
    assert busy.vehicles["n_lane_changes"].sum() == len(d)
    assert busy.vehicles.loc[busy.vehicles["origin"] == "main", "entry_lane"].isin([0, 1]).all()


def test_saturated_two_lane_entry_discharges_near_twice_capacity():
    res = run(length=1000.0, profile=((0, 900, 4800),), arrival_process="uniform", record_every_s=1.0)
    e = np.sort(res.vehicles["entry_s"].to_numpy())
    e = e[(e > 300) & (e < 850)]
    q_entry = 3600 * (len(e) - 1) / (e[-1] - e[0])
    assert q_entry == pytest.approx(2 * capacity(P)["q_c_veh_per_h"], rel=0.05)


def test_lanes_balanced_without_ramp():
    res = run(length=2000.0, profile=((0, 600, 2400),), record_every_s=1.0)
    share = res.trajectories["lane_id"].value_counts(normalize=True)
    assert 0.4 <= share.get(0, 0) <= 0.6 and 0.4 <= share.get(1, 0) <= 0.6


def test_two_lane_time_step_sensitivity():
    out = {dt: run(dt_s=dt, length=2000.0, profile=((0, 300, 4200), (300, 600, 1600)),
                   record_every_s=1.0).summary() for dt in (0.05, 0.1, 0.2)}
    ref = out[0.05]
    for dt in (0.1, 0.2):
        assert out[dt]["total_travel_time_veh_h"] == pytest.approx(ref["total_travel_time_veh_h"], rel=0.01)
