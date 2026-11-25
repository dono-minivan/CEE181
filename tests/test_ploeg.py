"""E9d: Ploeg et al. 2011 controller (tmc.ploeg) and its gates in setting A.

Theory reproduces the paper; filter discretization exact; equilibrium held for both spacing policies;
V2V with comms off = sensor feed-forward; missing feed-forward never replaced by a number; fallback logged;
only permitted inputs; collision-free; time-step sensitivity; case list = E9d registration.
"""
import math

import numpy as np
import pytest

from tmc import ploeg as pl
from tmc.cav import TICK_S, V2VChannel
from tmc.disturbance import (Disturbance, e9d_cases, equipped_mask, load_setting, platoon_metrics,
                             run_platoon)
from tmc.idm import equilibrium_gap, load_params

P = load_params("baseline")
S = load_setting()
NOM = S["disturbances"]["nominal"]
T_R = 0.6
NONE = Disturbance("none", v_low=10.0, b_lead=2.0, hold_s=10.0, t_start=1e9)


def pcav(p, ff="v2v", spacing="ctg", seed=181, **kw):
    c = {"p_equipped": p, "controller": "ploeg", "ff": ff, "spacing": spacing, "placement_seed": seed}
    if spacing == "ctg":
        c.update({"h_s": 1.0, "r_m": 2.0})
    c.update(kw)
    return c


# --------------------------------------------------------------------------- theory

def test_theory_reproduces_paper_thresholds():
    """Ploeg et al. 2011: h_min = 0.67 s at theta = 0.15 s (Sec. V-B); ACC string-stable for h >= 3.16 s."""
    assert pl.min_stable_h("v2v", delay_s=0.15) == pytest.approx(0.67, abs=0.005)
    assert pl.min_stable_h("none") == pytest.approx(3.16, abs=0.005)
    # zero delay: ||Gamma|| = 1 exactly at any h (paper Sec. III-B)
    for h in (0.3, 1.0, 2.0):
        assert pl.gamma_peak(h, "v2v", delay_s=0.0)[0] == pytest.approx(1.0, abs=1e-5)   # sup reached as w -> 0


def test_theory_values_quoted_in_E9d():
    h_idm = pl.idm_local_headway(15.0, P)
    assert h_idm == pytest.approx(1.786, abs=0.002)
    assert pl.gamma_peak(h_idm, "none")[0] == pytest.approx(1.097, abs=0.001)
    assert pl.gamma_peak(h_idm, "sensor")[0] == pytest.approx(1.000, abs=0.001)
    assert pl.gamma_peak(1.0, "none")[0] == pytest.approx(1.185, abs=0.001)
    assert pl.gamma_peak(1.0, "sensor")[0] == pytest.approx(1.036, abs=0.001)
    assert pl.gamma_peak(1.0, "v2v", delay_s=0.2)[0] == pytest.approx(1.000, abs=0.001)
    assert pl.min_stable_h("v2v", delay_s=0.2) == pytest.approx(0.78, abs=0.01)
    assert pl.min_stable_h("sensor") == pytest.approx(1.18, abs=0.01)


# --------------------------------------------------------------------------- controller unit checks

def test_filter_step_is_exact_first_order():
    """Constant error, no feed-forward, h = 1: u(t) = k_p e (1 - exp(-t/h)) on the tick grid."""
    c = pl.PloegController(1, P, ff="none", spacing="ctg", h_s=1.0, r_m=2.0)
    v, e0 = 10.0, 1.0
    s = 2.0 + 1.0 * v + e0
    for k in range(1, 6):
        c.command(k, v, s, 0.0, 0)
        assert c.u == pytest.approx(pl.KP * e0 * (1 - math.exp(-k * TICK_S / 1.0)), rel=1e-12)


def test_missing_feedforward_is_omitted_not_replaced():
    c = pl.PloegController(1, P, ff="sensor", spacing="ctg")
    out = c.command(0, 15.0, 17.0, 0.0, 0)
    assert out["u_ff"] is None and out["source"] == "missing"
    out = c.command(1, 15.0, 17.0, 0.0, 0)
    assert out["u_ff"] == 0.0 and out["source"] == "sensor"


def test_v2v_fallback_reasons_and_sensor_backup():
    ch = V2VChannel(delay_s=0.15)
    c = pl.PloegController(1, P, ff="v2v", spacing="ctg")
    out = c.command(0, 15.0, 17.0, 0.0, 0, ch, True)              # nothing usable yet
    assert out["fallback"] == "no_message" and out["source"] == "missing"
    for k in range(1, 3):
        ch.send(0, k, -1.0)
    c.command(1, 15.0, 17.0, 0.0, 0, ch, True)
    c.command(2, 15.0, 17.0, 0.0, 0, ch, True)
    out = c.command(3, 15.0, 17.0, 0.0, 0, ch, True)              # message from tick 1 usable at tick 3
    assert out["source"] == "v2v" and out["u_ff"] == -1.0 and out["fallback"] == ""
    out = pl.PloegController(2, P, ff="v2v").command(0, 15.0, 17.0, 0.0, 0, ch, False)
    assert out["fallback"] == "leader_unequipped"


def test_bad_options_rejected():
    with pytest.raises(ValueError):
        pl.PloegController(1, P, ff="radar")
    with pytest.raises(ValueError):
        pl.PloegController(1, P, spacing="cth")


# --------------------------------------------------------------------------- simulator gates

@pytest.mark.parametrize("spacing,ff", [("ctg", "v2v"), ("ctg", "none"), ("idm", "sensor"), ("idm", "v2v")])
def test_equipped_platoon_holds_its_equilibrium(spacing, ff):
    res, _ = run_platoon(dict(S, duration_s=60.0), NONE, T_R, cav=pcav(1.0, ff, spacing))
    tr = res.trajectories
    assert np.abs(tr["speed_mps"] - 15.0).max() < 1e-9
    assert np.abs(tr["acceleration_mps2"]).max() < 1e-9


def test_initial_gaps_follow_each_controller():
    s = dict(S, duration_s=0.2)
    cav = pcav(0.5, "v2v", "ctg")
    res, _ = run_platoon(s, NONE, T_R, cav=cav)
    eq = equipped_mask(cav, 31)
    x0 = res.trajectories[res.trajectories["time_s"] == 0.0].sort_values("vehicle_id")["position_m"].to_numpy()
    gaps = x0[:-1] - x0[1:] - P.length
    expect = np.where(eq[1:], 2.0 + 1.0 * 15.0, float(equilibrium_gap(15.0, P)))
    assert np.allclose(gaps, expect, atol=1e-9)


def test_v2v_with_comms_off_equals_sensor_feedforward():
    a, _ = run_platoon(dict(S, duration_s=80.0), NOM, T_R, cav=pcav(1.0, "v2v", "ctg", comms=False))
    b, _ = run_platoon(dict(S, duration_s=80.0), NOM, T_R, cav=pcav(1.0, "sensor", "ctg"))
    ka = a.trajectories.sort_values(["time_s", "vehicle_id"])[["position_m", "speed_mps"]].to_numpy()
    kb = b.trajectories.sort_values(["time_s", "vehicle_id"])[["position_m", "speed_mps"]].to_numpy()
    assert np.array_equal(ka, kb)
    assert set(a.cav_events["reason"]) == {"comms_off"}


def test_controller_inputs_are_permitted(monkeypatch):
    seen = []
    orig = pl.PloegController.command

    def spy(self, tick, v, s, approach_rate, leader, channel=None, leader_equipped=False):
        seen.append((tick, v, s, approach_rate, leader, channel, leader_equipped))
        return orig(self, tick, v, s, approach_rate, leader, channel, leader_equipped)

    monkeypatch.setattr(pl.PloegController, "command", spy)
    run_platoon(dict(S, duration_s=20.0), NOM, T_R, cav=pcav(0.5, "v2v", "idm"))
    assert seen
    for tick, v, s, dv, leader, ch, eq in seen:
        assert type(tick) is int and type(v) is float and type(s) is float and type(dv) is float
        assert leader is None or type(leader) is int
        assert ch is None or isinstance(ch, V2VChannel)
    c = pl.PloegController(1, P)
    assert not any(hasattr(v, "lanes") or hasattr(v, "trajectories") for v in vars(c).values())


def test_ploeg_rejected_outside_setting_A():
    from tmc.network import FreewaySegment, ScenarioConfig
    cfg = ScenarioConfig.from_yaml("configs/scenario_s3r_onramp_zipgap_tr06.yaml") \
        if hasattr(ScenarioConfig, "from_yaml") else None
    if cfg is None:
        pytest.skip("no YAML loader on ScenarioConfig")
    cfg.cav = pcav(0.5)
    with pytest.raises(ValueError):
        FreewaySegment(P, cfg)


@pytest.mark.parametrize("spacing,ff,p_eq,dist", [("ctg", "v2v", 1.0, "strong"), ("ctg", "v2v", 1.0, "nominal"),
                                                  ("idm", "sensor", 0.5, "strong"), ("idm", "v2v", 0.5, "nominal")])
def test_runs_are_collision_free(spacing, ff, p_eq, dist):
    res, p = run_platoon(S, S["disturbances"][dist], T_R, cav=pcav(p_eq, ff, spacing))   # raises on overlap
    m, _ = platoon_metrics(res, p, 15.0)
    assert m["min_gap_m"] > 0


@pytest.mark.parametrize("spacing,ff,p_eq", [("ctg", "v2v", 1.0), ("idm", "sensor", 0.5)])
def test_time_step_sensitivity(spacing, ff, p_eq):
    """E9 gate: metrics change < 2% across dt = 0.025 / 0.05 / 0.1 s; g within 0.1 x the test margin."""
    out = {}
    for dt in (0.025, 0.05, 0.1):
        res, p = run_platoon(S, NOM, T_R, dt=dt, cav=pcav(p_eq, ff, spacing))
        out[dt], _ = platoon_metrics(res, p, 15.0)
    for key in ("A", "dv_last", "dv_veh10", "min_gap_m"):
        for dt in (0.025, 0.05):
            assert out[dt][key] == pytest.approx(out[0.1][key], rel=0.02), (key, dt)
    for dt in (0.025, 0.05):
        assert abs(out[dt]["g_per_veh"] - out[0.1]["g_per_veh"]) < 0.1 * S["g_threshold_per_veh"]


def test_e9d_cases_match_registration():
    cases = e9d_cases(S)
    ids = {c["case_id"] for c in cases}
    assert len(cases) == len(ids) == 6 + 3 + 5
    assert {(c["ff"], c["p_equipped"]) for c in cases if c["group"] == "pl_idm"} == \
        {(f, p) for f in ("none", "sensor", "v2v") for p in (0.5, 1.0)}
    assert all(c["spacing"] == "idm" for c in cases if c["group"] == "pl_idm")
    short = [c for c in cases if c["group"] in ("pl_short", "pl_sweep")]
    assert all(c["spacing"] == "ctg" and c["h_s"] == 1.0 and c["r_m"] == 2.0 and c["p_equipped"] == 1.0
               for c in short)
    sw = [c for c in cases if c["group"] == "pl_sweep"]
    assert sorted(c["delay_s"] for c in sw) == [0.0, 0.2, 0.3, 0.5, 1.0]
    assert all(c["ff"] == "v2v" and c["max_age_ticks"] is None for c in sw)
    assert all(c["tau_s"] == 0.1 and c["controller"] == "ploeg" for c in cases)
    nominal = [c for c in cases if c["group"] != "pl_sweep"]
    assert all(c["delay_s"] == 0.15 and c["max_age_ticks"] == 5 for c in nominal)


def test_matrix_records_a_collision_instead_of_crashing():
    """PL-short without feed-forward collides in the strong disturbance (seen in the gate tests before the
    E9d matrix; E9d prediction 5: reported, not tuned away). The matrix row must record it and fail the gap
    criterion; the simulator itself still refuses to continue past an overlap."""
    from tmc.disturbance import _gaps_ok, _run_one
    case = next(c for c in e9d_cases(S, ["pl_short"]) if c["ff"] == "none")
    row, per = _run_one((case, "strong", 181))
    assert row["collided"] and row["collision_t_s"] > 10.0 and row["min_gap_m"] <= 0
    assert np.isnan(row["g_per_veh"]) and "collision" in row["g_note"]
    import pandas as pd
    assert not bool(_gaps_ok(pd.DataFrame([row])).iloc[0])


def test_judge_e9d_rules():
    import pandas as pd
    from tmc.disturbance import judge_e9d
    thr = float(S["g_threshold_per_veh"])
    rows = []

    def r(cid, grp, g, sd, **kw):
        base = dict(case_id=cid, group=grp, p_equipped=1.0, g_per_veh=g, A=np.exp(30 * g), seed=sd,
                    disturbance="nominal", min_gap_m=10.0, min_time_gap_s=1.2, max_decel_equipped=2.0,
                    collided=False)
        base.update(kw)
        return base
    for sd in (181, 182, 183, 184):
        rows += [r("human", "human", 0.0119, sd, p_equipped=0.0),
                 r("pl_short_v2v_p1", "pl_short", -0.006, sd), r("pl_short_sensor_p1", "pl_short", -0.002, sd),
                 r("pl_short_none_p1", "pl_short", np.nan, sd, A=np.nan, collided=True, min_gap_m=-0.1,
                   min_time_gap_s=np.nan, max_decel_equipped=np.nan),
                 r("pl_idm_v2v_p1", "pl_idm", -0.020, sd), r("pl_idm_sensor_p1", "pl_idm", -0.019, sd),
                 r("pl_idm_none_p1", "pl_idm", 0.03, sd)]
    jd = judge_e9d(pd.DataFrame(rows), thr).set_index(["criterion", "case_id", "disturbance"])
    assert jd.loc[("disturbance_reduced", "pl_short_v2v_p1", "nominal"), "pass"]          # 0.004 below sensor
    assert not jd.loc[("disturbance_reduced", "pl_idm_v2v_p1", "nominal"), "pass"]        # only 0.001 below
    assert jd.loc[("positive_control (prediction 1)", "pl_idm_none_p1", "nominal"), "pass"]
    pc = jd.loc[("positive_control (prediction 1)", "pl_short_none_p1", "nominal")]
    assert pc["pass"] is None and pc["none_collided_seeds"] == 4 and "not evaluable" in pc["note"]
    assert not jd.loc[("gaps_acceptable", "pl_short_none_p1", "nominal"), "pass"]
