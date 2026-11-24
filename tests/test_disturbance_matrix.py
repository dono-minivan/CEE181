"""Checks of the E9 setting A CAV matrix (tmc.disturbance): case list matches the E9 pre-registration,
config translation, metrics by vehicle type, outage recovery bookkeeping, and the judgement rules."""
import numpy as np
import pandas as pd
import pytest

from tmc.disturbance import (_type_metrics, case_cav_config, fallback_metrics, judge_matrix, load_setting,
                             matrix_cases, platoon_metrics, run_platoon)

S = load_setting()
NOM = S["disturbances"]["nominal"]
THR = float(S["g_threshold_per_veh"])


def test_matrix_cases_match_E9_preregistration():
    cases = matrix_cases(S)
    ids = {c["case_id"] for c in cases}
    assert len(cases) == len(ids) == 6 + 8 + 5 + 4
    # nominal: ACC and CACC at p = 0.25 / 0.5 / 1.0, tau 0.1 s, delay 0.15 s, fallback at age > 5 ticks
    for mode in ("acc", "cacc"):
        for p in (0.25, 0.5, 1.0):
            c = next(c for c in cases if c["case_id"] == f"nominal_{mode}_p{p:g}")
            assert (c["tau_s"], c["delay_s"], c["max_age_ticks"], c["loss_prob"], c["outages"]) == \
                (0.1, 0.15, 5, 0.0, [])
    # degraded CACC at p = 0.5 and 1.0
    for p in (0.5, 1.0):
        d = {c["variant"]: c for c in cases if c["group"] == "degraded" and c["p_equipped"] == p}
        assert set(d) == {"delay_0.3", "delay_1.0", "loss_0.3", "outage_5s"}
        assert d["delay_0.3"]["delay_s"] == 0.3 and d["delay_1.0"]["delay_s"] == 1.0
        assert d["loss_0.3"]["loss_prob"] == 0.3 and d["loss_0.3"]["delay_s"] == 0.15
        (a, b), = d["outage_5s"]["outages"]
        assert b - a == pytest.approx(5.0)
        assert all(c["mode"] == "cacc" and c["max_age_ticks"] == 5 for c in d.values())
    # delay sweep: p = 1, fallback disabled, 0 / 0.2 / 0.3 / 0.5 / 1.0 s
    sw = [c for c in cases if c["group"] == "sweep"]
    assert sorted(c["delay_s"] for c in sw) == [0.0, 0.2, 0.3, 0.5, 1.0]
    assert all(c["p_equipped"] == 1.0 and c["max_age_ticks"] is None and c["mode"] == "cacc" for c in sw)
    # tau sensitivity: ACC and CACC at p = 0.5 and 1.0
    ta = [c for c in cases if c["group"] == "tau"]
    assert {(c["mode"], c["p_equipped"]) for c in ta} == {(m, p) for m in ("acc", "cacc") for p in (0.5, 1.0)}
    assert all(c["tau_s"] == 0.5 for c in ta)
    assert S["matrix"]["human_reaction_time_s"] == 0.6
    with pytest.raises(ValueError):
        matrix_cases(S, ["nope"])


def test_case_config_uses_run_seed_for_placement_and_loss():
    c = next(c for c in matrix_cases(S, ["degraded"]) if c["variant"] == "loss_0.3")
    cfg = case_cav_config(c, 183)
    assert cfg["placement_seed"] == cfg["loss_seed"] == 183
    assert cfg["loss_prob"] == 0.3 and cfg["mode"] == "cacc" and cfg["max_age_ticks"] == 5


def test_type_metrics_split():
    eq = np.array([True, False, True, False])          # lead CAV, human, CAV, human
    A = np.array([[0.0, -1.0, -2.0, -3.0], [0.0, -0.5, -4.5, -1.0]])
    gap = np.array([[10.0, 8.0, 6.0], [9.0, 7.0, 5.0]])    # follower 1, 2, 3
    tg = gap / 10.0
    m = _type_metrics(eq, A, gap, tg)
    assert m["n_equipped"] == 2 and m["lead_equipped"]
    assert m["n_fol_equipped"] == 1 and m["max_decel_equipped"] == 4.5 and m["min_gap_equipped"] == 7.0
    assert m["n_fol_human"] == 2 and m["max_decel_human"] == 3.0 and m["min_gap_human"] == 5.0
    # humans directly behind a CAV: followers 1 (behind lead) and 3 (behind vehicle 2)
    assert m["n_fol_human_behind_cav"] == 2 and m["min_gap_human_behind_cav"] == 5.0


def test_platoon_metrics_by_type_on_mixed_run():
    s = dict(S, duration_s=60.0)
    res, p = run_platoon(s, NOM, 0.6, cav={"p_equipped": 0.5, "mode": "acc", "placement_seed": 181})
    m, pv = platoon_metrics(res, p, 15.0, by_type=True)
    eq = res.vehicles.set_index("vehicle_id")["equipped"]
    assert pv["equipped"].tolist() == [bool(eq[i]) for i in range(31)]
    n_eq_fol = int(eq.loc[1:30].sum())
    assert m["n_equipped"] == int(eq.sum()) and m["n_fol_human"] == 30 - n_eq_fol and 0 < n_eq_fol < 30
    assert min(m["min_gap_equipped"], m["min_gap_human"]) == pytest.approx(m["min_gap_m"])


def test_outage_recovery_bookkeeping():
    """Nominal delay (2 ticks): stale from age > 5 into the outage, last stale tick 1 tick after it ends."""
    res, _ = run_platoon(dict(S, duration_s=40.0), NOM, 0.6,
                         cav={"p_equipped": 1.0, "mode": "cacc", "placement_seed": 181, "outages": [[10.0, 15.0]]})
    f = fallback_metrics(res, ((10.0, 15.0),))
    assert f["fb_stale_in_outage"] > 0 and f["fb_leader_unequipped"] == 0
    assert f["last_stale_after_outage_s"] == pytest.approx(15.1)


def _row(case_id, group, mode, p, g, seed, dist="nominal", **kw):
    base = dict(case_id=case_id, group=group, mode=mode, p_equipped=p, g_per_veh=g, A=np.exp(30 * g), seed=seed,
                disturbance=dist, min_gap_m=10.0, min_time_gap_s=1.5, max_decel_equipped=2.0,
                min_gap_human_behind_cav=10.0, fb_stale=0, delay_s=0.15, outage_end_s=np.nan,
                last_stale_after_outage_s=np.nan, fb_stale_in_outage=0)
    base.update(kw)
    return base


def test_judgement_rules():
    rows = []
    for sd in (181, 182, 183, 184):
        rows.append(_row("human", "human", "human", 0.0, 0.0119, sd, max_decel_equipped=np.nan))
        rows.append(_row("nominal_acc_p1", "nominal", "acc", 1.0, 0.0100, sd))
        # CACC: 3 of 4 seeds beat both human and ACC by >= THR; seed 184 beats human (by 0.0039) but not ACC
        rows.append(_row("nominal_cacc_p1", "nominal", "cacc", 1.0, 0.0100 - THR - 1e-4 if sd < 184 else 0.0080, sd))
        rows.append(_row("degraded_cacc_p1_delay_0.3", "degraded", "cacc", 1.0, 0.0100 + THR + 1e-4 * (sd == 182),
                         sd, delay_s=0.3))
        rows.append(_row("degraded_cacc_p1_delay_1.0", "degraded", "cacc", 1.0, 0.0100, sd, delay_s=1.0, fb_stale=50))
        rows.append(_row("degraded_cacc_p1_outage_5s", "degraded", "cacc", 1.0, 0.0101, sd, outage_end_s=15.0,
                         fb_stale=40, fb_stale_in_outage=40, last_stale_after_outage_s=15.1))
        rows.append(_row("nominal_cacc_p1", "nominal", "cacc", 1.0, 0.02, sd, dist="strong",
                         min_time_gap_s=0.4 if sd == 181 else 1.0))
    jd = judge_matrix(pd.DataFrame(rows), THR).set_index(["criterion", "case_id", "disturbance"])
    r = jd.loc[("disturbance_reduced", "nominal_cacc_p1", "nominal")]
    assert (r["seeds_vs_human"], r["seeds_vs_acc"], r["seeds_pass"], r["pass"]) == (4, 3, 3, True)
    assert jd.loc[("acc_vs_human (reported)", "nominal_acc_p1", "nominal"), "pass"] is None
    assert not jd.loc[("robust_comm_delay_0.3", "degraded_cacc_p1_delay_0.3", "nominal"), "pass"]   # 1 seed > THR
    assert jd.loc[("robust_comm_delay_1.0", "degraded_cacc_p1_delay_1.0", "nominal"), "pass"]
    assert jd.loc[("robust_comm_outage_5s", "degraded_cacc_p1_outage_5s", "nominal"), "pass"]
    # gap criterion applies to the strong (stress) disturbance in full: one seed with a 0.4 s time gap fails it
    assert not jd.loc[("gaps_acceptable", "nominal_cacc_p1", "strong"), "pass"]
    assert jd.loc[("gaps_acceptable", "nominal_cacc_p1", "nominal"), "pass"]


def test_not_estimable_g_falls_back_to_A():
    rows = []
    for sd in (181, 182, 183, 184):
        rows.append(_row("human", "human", "human", 0.0, 0.0119, sd, A=1.40))
        rows.append(_row("nominal_acc_p1", "nominal", "acc", 1.0, 0.0100, sd, A=1.35))
        rows.append(_row("nominal_cacc_p1", "nominal", "cacc", 1.0, np.nan, sd, A=1.20))   # 1.20 <= 0.9 * 1.35
    jd = judge_matrix(pd.DataFrame(rows), THR).set_index(["criterion", "case_id"])
    assert jd.loc[("disturbance_reduced", "nominal_cacc_p1"), "pass"]
