"""Zipper courtesy merge (ramp.courtesy = "zipper", experiment log E5)."""
from pathlib import Path

import numpy as np
import pytest

from tmc.idm import load_params
from tmc.network import FreewaySegment, ScenarioConfig

S1Z = Path(__file__).resolve().parents[1] / "configs" / "scenario_s1z_onramp_zipper.yaml"


@pytest.fixture(scope="module")
def zipper_run():
    cfg = ScenarioConfig.from_yaml(S1Z)
    cfg.demand_profile = [[0.0, 600.0, 1400.0]]
    cfg.ramp.demand_profile = [[0.0, 600.0, 500.0]]   # X ~ 1.09: the merge saturates
    cfg.record_every_s = cfg.dt_s                      # every step, so followers at merge time are exact
    cfg.detectors = []
    return FreewaySegment(load_params(cfg.param_set), cfg).run()


def test_config_parsed():
    assert ScenarioConfig.from_yaml(S1Z).ramp.courtesy == "zipper"


def test_zipper_safe_conserving_and_merges_in_zone(zipper_run):
    r = zipper_run                                   # run() raises on any gap <= 0
    assert r.cleared and r.min_gap_m > 0
    v = r.vehicles
    assert v["merge_s"].notna().sum() == (v["origin"] == "ramp").sum()
    mg = r.lane_changes[r.lane_changes["kind"] == "merge"]
    rc = r.config.ramp
    assert mg["position_m"].between(rc.merge_start_m, rc.merge_end_m).all()


def two_vehicle_state(x_main: float):
    """Ramp vehicle (id 0) at 1600 m, 20 m/s in the merge zone; lane-0 vehicle (id 1) at x_main, 25 m/s."""
    cfg = ScenarioConfig.from_yaml(S1Z)
    cfg.demand_profile, cfg.ramp.demand_profile, cfg.detectors = [], [], []
    cfg.initial_vehicles = [[1700.0, 20.0], [x_main, 25.0]]
    seg = FreewaySegment(load_params(cfg.param_set), cfg)
    seg.main.remove_at(0)
    seg.rlane.insert(0, 0, 1600.0, 20.0)
    return seg


def free_acc(p, v):
    return p.a * (1 - (v / p.v0) ** p.delta)


def test_courtesy_yields_once_with_bounded_braking():
    seg = two_vehicle_state(1500.0)             # 95 m gap: yielding needs only mild braking
    p = seg.p
    a = seg._accelerations()[0][0][0]
    assert -p.b <= a < 0                         # follower eases off for the ramp vehicle
    seg.let_in[1] = True                         # ...but not if it has already let one in
    assert seg._accelerations()[0][0][0] == pytest.approx(free_acc(p, 25.0))


def test_courtesy_refused_when_braking_would_exceed_b():
    seg = two_vehicle_state(1550.0)             # 45 m gap at +5 m/s closing: IDM wants ~ -3 m/s^2 > b
    assert seg._accelerations()[0][0][0] == pytest.approx(free_acc(seg.p, 25.0))


def test_saturated_ramp_is_not_starved(zipper_run):
    """X ~ 1.09 for 600 s: the ramp keeps merging throughout (E4 default merge starves it at this load)."""
    mg = zipper_run.vehicles["merge_s"].dropna()
    assert ((mg > 300) & (mg <= 600)).sum() >= 0.5 * 500 * 300 / 3600   # >= half the ramp demand, 300-600 s


S3 = S1Z.parent / "scenario_s3_onramp_zipgap.yaml"


def test_s3_zipper_with_gap_seeking_safe_and_conserving():
    """S3 merge (zipper + gap seeking), saturated for 600 s: no overlap (run() raises on gap <= 0),
    conservation at every recorded step, every ramp vehicle merges inside the zone, all clear."""
    cfg = ScenarioConfig.from_yaml(S3)
    assert cfg.ramp.courtesy == "zipper" and cfg.ramp.gap_seeking
    cfg.demand_profile = [[0.0, 600.0, 1100.0]]
    cfg.ramp.demand_profile = [[0.0, 600.0, 450.0]]
    cfg.record_every_s = 1.0
    cfg.detectors = []
    r = FreewaySegment(load_params(cfg.param_set), cfg).run()
    assert r.cleared and r.min_gap_m > 0
    c = r.counts
    assert (c["n_arrived"] == c["n_queued"] + c["n_on_road"] + c["n_exited"]).all()
    mg = r.lane_changes[r.lane_changes["kind"] == "merge"]
    assert len(mg) == (r.vehicles["origin"] == "ramp").sum()
    assert mg["position_m"].between(cfg.ramp.merge_start_m, cfg.ramp.merge_end_m).all()


def test_bad_courtesy_value_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(S1Z.read_text(encoding="utf-8").replace("courtesy: zipper", "courtesy: always"),
                   encoding="utf-8")
    with pytest.raises(ValueError):
        ScenarioConfig.from_yaml(bad)
