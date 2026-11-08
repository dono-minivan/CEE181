"""Ramp meter + controllers: control law, override, missing data, actuator, estimation vs truth."""
import copy
from pathlib import Path

import numpy as np
import pytest

from tmc.control import Controller, MeterLimits
from tmc.idm import load_params
from tmc.interfaces import Observation
from tmc.network import FreewaySegment, ScenarioConfig

S3 = Path(__file__).resolve().parents[1] / "configs" / "scenario_s3_onramp_zipgap.yaml"


class FakeView:
    """Stands in for DetectorView: returns whatever observations the test sets."""

    def __init__(self):
        self.obs = {}

    def set(self, sid, t, count=0, occ=None):
        self.obs[sid] = Observation(sid, t - 30.0, t, count, None, "ok", occ)

    def latest(self, sid, n=1):
        return [self.obs[sid]] if sid in self.obs else []


def test_alinea_law_and_dark_at_max():
    c = Controller("alinea", o_set=0.10, limits=MeterLimits(r_min_vph=240, r_max_vph=900))
    v = FakeView()
    for k, occ in enumerate([0.30, 0.30]):
        t = 30.0 * (k + 1)
        v.set("R_dem", t, 2); v.set("R_pass", t, 2); v.set("M_mrg", t, 5, occ)
        c.update(t, v)
    # r starts at 900; 900 + 70*100*(0.10-0.30) = -500 -> clipped 240; again -> 240
    assert c.trace[-1]["r_law_vph"] == pytest.approx(240.0)
    assert c.trace[-1]["action"] == "set_rate" and c.trace[-1]["rate_cmd_vph"] == pytest.approx(240.0)
    v.set("M_mrg", 90.0, 5, 0.05); v.set("R_dem", 90.0, 2); v.set("R_pass", 90.0, 2)
    c.update(90.0, v)                                  # 240 + 70*100*0.05 = 590
    assert c.trace[-1]["r_law_vph"] == pytest.approx(590.0)
    for t in (120.0, 150.0):
        v.set("M_mrg", t, 5, 0.0); v.set("R_dem", t, 2); v.set("R_pass", t, 2)
        c.update(t, v)                                 # reaches r_max -> meter off
    assert c.trace[-1]["r_law_vph"] == pytest.approx(900.0) and c.trace[-1]["action"] == "dark"


def test_queue_override_goes_dark():
    c = Controller("pretimed", rate_vph=300.0, limits=MeterLimits(n_max_veh=35))
    v = FakeView()
    v.set("R_dem", 30.0, 40); v.set("R_pass", 30.0, 2)     # 38 waiting > n_max
    cmd = c.update(30.0, v)
    assert cmd.action == "dark" and c.trace[-1]["override"] and c.trace[-1]["queue_est_veh"] == 38


def test_forecast_triggers_override_before_limit():
    c = Controller("pretimed", rate_vph=240.0, limits=MeterLimits(n_max_veh=35, horizon_s=60))
    v = FakeView()
    v.set("R_dem", 30.0, 38); v.set("R_pass", 30.0, 6)     # N = 32 < 35, lambda = 4560 veh/h -> N_fc >> 35
    assert c.update(30.0, v).action == "dark"
    assert c.trace[-1]["queue_est_veh"] == 32 and c.trace[-1]["queue_fc_veh"] > 35


def test_missing_measurement_holds_last_command():
    c = Controller("alinea", o_set=0.10)
    v = FakeView()
    v.set("R_dem", 30.0, 2); v.set("R_pass", 30.0, 2); v.set("M_mrg", 30.0, 5, 0.40)
    first = c.update(30.0, v)
    del v.obs["M_mrg"]
    v.set("R_dem", 60.0, 2); v.set("R_pass", 60.0, 2)
    assert c.update(60.0, v) is first
    assert c.trace[-1]["reason"].startswith("missing:M_mrg")


def short_s3(control, horizon=900.0, q=1300.0, r=0.28):
    cfg = ScenarioConfig.from_yaml(S3)
    cfg.demand_profile = [[0.0, horizon, q * (1 - r)]]
    cfg.ramp.demand_profile = [[0.0, horizon, q * r]]
    cfg.record_every_s = 1.0
    cfg.control = control
    return cfg


@pytest.fixture(scope="module")
def metered():
    cfg = short_s3({"mode": "pretimed", "rate_vph": 300.0})
    seg = FreewaySegment(load_params(cfg.param_set), cfg)
    return seg, seg.run()


def test_metered_run_safe_conserving_and_cleared(metered):
    _, r = metered
    assert r.cleared and r.min_gap_m > 0
    c = r.counts
    assert (c["n_arrived"] == c["n_queued"] + c["n_on_road"] + c["n_exited"]).all()


def test_release_headway_and_standing_start(metered):
    """While the meter is on, consecutive releases are >= 3600/rate apart, and released vehicles start
    at v = 0 at the stop line."""
    seg, r = metered
    tr = r.control_trace
    on = tr[tr["action"] == "set_rate"]["time_s"]
    v = r.vehicles[r.vehicles["origin"] == "ramp"].sort_values("entry_s")
    e = v["entry_s"].to_numpy()
    both_on = np.array([((on <= a) & (on > a - 30)).any() and ((on <= b) & (on > b - 30)).any()
                        for a, b in zip(e[:-1], e[1:])])
    assert both_on.sum() > 20
    assert (np.diff(e)[both_on] >= 3600.0 / 300.0 - 1e-6).all()
    # Standing start: at its first recorded state, a vehicle released at entry_s can be no faster than
    # a * (t - entry_s) (it started from 0 and IDM acceleration never exceeds a).
    first = r.trajectories[(r.trajectories["lane_id"] == -1)].groupby("vehicle_id").first()
    ent = v.set_index("vehicle_id")["entry_s"]
    rel = [vid for vid in ent.index if vid in first.index and
           any((on <= ent[vid]) & (on > ent[vid] - 30))]
    f = first.loc[rel]
    tau = f["time_s"] - ent.loc[rel]
    assert len(rel) > 20
    assert (f["speed_mps"] <= seg.p.a * tau + 1e-6).all()


def test_queue_estimate_matches_truth(metered):
    """Count-in/count-out estimate vs the simulator's ramp queue at control times (evaluation only).
    Off by at most the vehicles between the stop line and the R_pass loop (10 m: <= 2 vehicles)."""
    _, r = metered
    tr = r.control_trace.dropna(subset=["queue_est_veh"])
    truth = r.counts.set_index("time_s")["n_queued_ramp"]
    err = [abs(row.queue_est_veh - truth.loc[truth.index[truth.index <= row.time_s].max()])
           for row in tr.itertuples()]
    assert max(err) <= 2 and tr["queue_est_veh"].max() >= 5


def test_dark_controller_changes_nothing():
    a = FreewaySegment(load_params("baseline"), short_s3(None)).run()
    b = FreewaySegment(load_params("baseline"), short_s3({"mode": "none"})).run()
    assert a.summary() == b.summary()
    assert len(b.control_trace) > 0 and (b.control_trace["action"] == "dark").all()


def test_controller_requires_detectors():
    cfg = short_s3({"mode": "pretimed", "rate_vph": 300.0})
    cfg.detectors = []
    with pytest.raises(ValueError):
        FreewaySegment(load_params("baseline"), cfg)


def test_bad_modes_rejected():
    with pytest.raises(ValueError):
        Controller("fuzzy")
    with pytest.raises(ValueError):
        Controller("pretimed")
    with pytest.raises(ValueError):
        Controller("alinea")
