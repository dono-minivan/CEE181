"""E9 setting B harness (experiment log E9e): every-step gap tracker, scenario build, judgement rules.

Written before any setting B run. The judgement tests use synthetic run tables, so they fix the rules
(E7c "better", E9e D1 congestion reduction, D2 gaps with the human-relative time-gap bound) independently
of any result.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tmc import setting_b as sb
from tmc.idm import load_params
from tmc.network import GAP_TYPES, FreewaySegment, ScenarioConfig

ROOT = Path(__file__).resolve().parents[1]
P = load_params("baseline")


# --------------------------------------------------------------------------- gap tracker (D2)

@pytest.fixture(scope="module")
def short_merge_run():
    """S3r merge, half equipped, 250 s, every step recorded, so the tracker can be audited exactly."""
    cfg = ScenarioConfig.from_yaml(ROOT / "configs" / "scenario_s3r_onramp_zipgap_tr06.yaml")
    cfg.max_time_s = 250.0
    cfg.record_every_s = cfg.dt_s
    cfg.cav = {"p_equipped": 0.5, "mode": "cacc", "placement_seed": 181}
    return FreewaySegment(P, cfg).run()


def _audit_from_trajectories(res) -> dict:
    eq = res.vehicles.set_index("vehicle_id")["equipped"]
    tr = res.trajectories.sort_values(["time_s", "lane_id", "position_m"], ascending=[True, True, False])
    out = {f"{m}_{k}": (np.inf if m.startswith("min") else 0.0)
           for k in GAP_TYPES for m in ("min_gap_m", "min_time_gap_s", "max_decel_mps2")}
    for _, g in tr.groupby(["time_s", "lane_id"]):
        if len(g) < 2:
            continue
        x, v, a = g["position_m"].to_numpy(), g["speed_mps"].to_numpy(), g["acceleration_mps2"].to_numpy()
        ids = g["vehicle_id"].to_numpy()
        s = x[:-1] - x[1:] - P.length
        fe, le = eq.loc[ids[1:]].to_numpy(), eq.loc[ids[:-1]].to_numpy()
        for key, sel in (("eq", fe), ("hh", ~fe & ~le), ("he", ~fe & le)):
            if not sel.any():
                continue
            out[f"min_gap_m_{key}"] = min(out[f"min_gap_m_{key}"], s[sel].min())
            out[f"max_decel_mps2_{key}"] = max(out[f"max_decel_mps2_{key}"], -a[1:][sel].min())
            fast = sel & (v[1:] > 5.0)
            if fast.any():
                out[f"min_time_gap_s_{key}"] = min(out[f"min_time_gap_s_{key}"], (s[fast] / v[1:][fast]).min())
    return out


def test_gap_tracker_matches_an_audit_of_every_recorded_step(short_merge_run):
    res = short_merge_run
    audit = _audit_from_trajectories(res)
    assert set(res.gap_stats) == set(audit)
    for k, v in audit.items():
        assert res.gap_stats[k] == pytest.approx(v, abs=1e-9), k


def test_gap_tracker_overall_minimum_equals_simulator_min_gap(short_merge_run):
    res = short_merge_run
    assert min(res.gap_stats[f"min_gap_m_{k}"] for k in GAP_TYPES) == pytest.approx(res.min_gap_m, abs=1e-12)
    assert res.vehicles["equipped"].any() and (~res.vehicles["equipped"]).any()
    for k in GAP_TYPES:                                  # every follower type occurs at p = 0.5
        assert np.isfinite(res.gap_stats[f"min_gap_m_{k}"])


def test_gap_tracker_human_only_has_no_equipped_types():
    cfg = ScenarioConfig.from_yaml(ROOT / "configs" / "scenario_s3r_onramp_zipgap_tr06.yaml")
    cfg.max_time_s = 120.0
    st = FreewaySegment(P, cfg).run().gap_stats
    assert np.isfinite(st["min_gap_m_hh"])
    assert st["min_gap_m_eq"] == np.inf and st["min_gap_m_he"] == np.inf


# --------------------------------------------------------------------------- scenario build

def test_capacity_is_read_from_the_e9b2_result():
    est = pd.read_csv(sb.CAP_CSV)
    assert "S3r2" in sb.CAP_CSV.name and "s3r2" in sb.S3R.name
    row = est.loc[np.isclose(est["ramp_share"], 0.28), "sat_discharge_vph"]
    assert sb.merge_capacity_s3r() == pytest.approx(float(row.iloc[0]))


@pytest.mark.parametrize("x", [1.20, 1.35])
def test_build_config_demand_and_cav(x):
    c_m = sb.merge_capacity_s3r()
    h = sb.build_config(x, 182, c_m)
    assert h.cav is None and h.control is None and h.seed == 182 and h.dt_s == 0.05
    assert h.leader_change_delay == "restart"
    peak = h.demand_profile[1][2] + h.ramp.demand_profile[1][2]
    assert peak == pytest.approx(x * c_m)
    assert h.ramp.demand_profile[1][2] / peak == pytest.approx(0.28)
    assert [r[:2] for r in h.demand_profile] == [[0.0, 600.0], [600.0, 1500.0], [1500.0, 2400.0]]
    a = sb.build_config(x, 182, c_m, "cacc", 0.5, dt=0.1)
    assert a.cav == {"p_equipped": 0.5, "mode": "cacc", "placement_seed": 182, "tau_s": 0.1, "delay_s": 0.15}
    assert a.dt_s == 0.1
    assert h.reaction_time_s == a.reaction_time_s == 0.6      # humans keep T_r in every setting B run


def test_cases_match_the_registration():
    assert sb.CASES == [("human", 0.0), ("acc", 0.25), ("cacc", 0.25), ("acc", 0.5), ("cacc", 0.5),
                        ("acc", 1.0), ("cacc", 1.0)]
    assert sb.X_PEAKS == [1.35, 1.20] and sb.SEEDS == [181, 182, 183, 184]


# --------------------------------------------------------------------------- congestion and discharge

def _obs(speeds_mrg, flows_down):
    rows = []
    for i, (v, q) in enumerate(zip(speeds_mrg, flows_down)):
        t0 = 30.0 * i
        rows.append({"sensor_id": "M_mrg", "interval_start_s": t0, "interval_end_s": t0 + 30, "count": 5,
                     "mean_speed_mps": v, "occupancy": 0.05})
        rows.append({"sensor_id": "M_down", "interval_start_s": t0, "interval_end_s": t0 + 30, "count": q / 120.0,
                     "mean_speed_mps": 25.0, "occupancy": 0.05})
    return pd.DataFrame(rows)


def test_merge_discharge_uses_congested_intervals_only_and_is_nan_without_congestion():
    o = _obs([30, 10, 10, 10, 30, 10], [1500, 900, 1000, 1100, 1500, 2000])   # last 10 is a lone interval
    mins, cong = sb.congestion_minutes(o, "M_mrg", 18.55, 0.183)
    assert mins == 1.5
    assert sb.merge_discharge(o, cong) == pytest.approx(1000.0)
    o2 = _obs([30, 30, 30], [1500, 1500, 1500])
    _, cong2 = sb.congestion_minutes(o2, "M_mrg", 18.55, 0.183)
    assert np.isnan(sb.merge_discharge(o2, cong2))


# --------------------------------------------------------------------------- judgement rules

def _table(rows):
    base = {"x_peak": 1.35, "dt_s": 0.1, "served_600_2100_vph": 1000.0, "merge_discharge_vph": 950.0,
            "congested_min_M_mrg": 10.0}
    base.update({f"min_gap_m_{k}": 10.0 for k in GAP_TYPES})
    base.update({f"min_time_gap_s_{k}": 1.0 for k in GAP_TYPES})
    base.update({"max_decel_mps2_eq": 2.0})
    return pd.DataFrame([{**base, **r} for r in rows])


def _runs(case, ttts, **cols):
    mode, p = case
    return [{"mode": mode, "p": p, "seed": s, "ttt_veh_h": t, **{k: v[i] for k, v in cols.items()}}
            for i, (s, t) in enumerate(zip([181, 182, 183, 184], ttts))]


def test_better_rule_needs_2pct_and_3_of_4_seeds():
    df = _table(_runs(("human", 0.0), [10, 10, 10, 10]) + _runs(("acc", 1.0), [9.0, 9.0, 9.0, 10.5]))
    r = sb.better(df, ("acc", 1.0), ("human", 0.0))
    assert r["n_seeds_lower"] == 3 and r["ttt_pct"] == pytest.approx(-6.25) and r["better"]
    df = _table(_runs(("human", 0.0), [10, 10, 10, 10]) + _runs(("acc", 1.0), [9.9, 9.9, 9.9, 9.9]))
    assert not sb.better(df, ("acc", 1.0), ("human", 0.0))["better"]          # 1% only
    df = _table(_runs(("human", 0.0), [10, 10, 10, 10]) + _runs(("acc", 1.0), [5, 5, 10.1, 10.1]))
    assert not sb.better(df, ("acc", 1.0), ("human", 0.0))["better"]          # 2 of 4 seeds only


def test_d1_needs_10pct_and_1min_below_both_human_and_acc():
    cm = "congested_min_M_mrg"
    human = _runs(("human", 0.0), [10] * 4, **{cm: [10, 10, 10, 10]})
    acc = _runs(("acc", 0.5), [10] * 4, **{cm: [8, 8, 8, 8]})
    cacc = _runs(("cacc", 0.5), [10] * 4, **{cm: [7, 7, 7.5, 8]})       # vs ACC: -1, -1, -0.5, 0
    r = sb.d1_reduced(_table(human + acc + cacc), 0.5)
    assert r["d1_seeds_pass"] == 2 and not r["d1_reduced"]
    cacc = _runs(("cacc", 0.5), [10] * 4, **{cm: [7, 7, 7, 8]})
    assert sb.d1_reduced(_table(human + acc + cacc), 0.5)["d1_seeds_pass"] == 3
    # 10% but under the 1-minute floor
    small = _runs(("human", 0.0), [10] * 4, **{cm: [2, 2, 2, 2]}) + _runs(("acc", 0.5), [10] * 4, **{cm: [2] * 4})
    small += _runs(("cacc", 0.5), [10] * 4, **{cm: [1.5] * 4})
    assert sb.d1_reduced(_table(small), 0.5)["d1_seeds_pass"] == 0
    # nothing to reduce
    zero = _runs(("human", 0.0), [10] * 4, **{cm: [0] * 4}) + _runs(("acc", 0.5), [10] * 4, **{cm: [0] * 4})
    zero += _runs(("cacc", 0.5), [10] * 4, **{cm: [0] * 4})
    assert sb.d1_reduced(_table(zero), 0.5)["d1_seeds_pass"] == 0


def test_gaps_time_bound_relaxes_to_the_same_seed_human_minimum():
    tg = "min_time_gap_s_hh"
    human = _runs(("human", 0.0), [10] * 4, **{tg: [0.30, 0.8, 0.8, 0.8]})
    acc = _runs(("acc", 1.0), [10] * 4, **{"min_time_gap_s_eq": [0.35, 0.6, 0.6, 0.6]})
    r = sb.gaps_ok(_table(human + acc), "acc", 1.0)
    assert r["gaps_ok"] and r["time_gap_bound_relaxed_seeds"] == 1
    acc = _runs(("acc", 1.0), [10] * 4, **{"min_time_gap_s_eq": [0.25, 0.6, 0.6, 0.6]})   # worse than human
    assert not sb.gaps_ok(_table(human + acc), "acc", 1.0)["gaps_ok"]
    acc = _runs(("acc", 1.0), [10] * 4, **{"min_time_gap_s_eq": [0.6, 0.45, 0.6, 0.6]})    # 0.5 s bound holds
    assert not sb.gaps_ok(_table(human + acc), "acc", 1.0)["gaps_ok"]


def test_gaps_net_gap_bound_relaxes_to_human_and_decel_stays_absolute():
    human = _runs(("human", 0.0), [10] * 4, **{"min_gap_m_hh": [0.5, 0.8, 5, 5]})
    acc = _runs(("acc", 1.0), [10] * 4, **{"min_gap_m_eq": [0.6, 0.8, 5, 5]})
    r = sb.gaps_ok(_table(human + acc), "acc", 1.0)
    assert r["gaps_ok"] and r["net_gap_bound_relaxed_seeds"] == 2
    acc = _runs(("acc", 1.0), [10] * 4, **{"min_gap_m_he": [0.4, 5, 5, 5]})          # worse than human
    assert not sb.gaps_ok(_table(human + acc), "acc", 1.0)["gaps_ok"]
    acc = _runs(("acc", 1.0), [10] * 4, **{"min_gap_m_eq": [5, 5, 0.9, 5]})           # human seed fine: 1 m
    assert not sb.gaps_ok(_table(human + acc), "acc", 1.0)["gaps_ok"]
    acc = _runs(("acc", 1.0), [10] * 4, **{"max_decel_mps2_eq": [4.5, 2, 2, 2]})
    assert not sb.gaps_ok(_table(human + acc), "acc", 1.0)["gaps_ok"]


def test_judge_produces_one_row_per_cav_case():
    rows = _runs(("human", 0.0), [10] * 4)
    for m, p in sb.CASES[1:]:
        rows += _runs((m, p), [9] * 4)
    jd = sb.judge(_table(rows))
    assert len(jd) == 6 and jd["better_than_human"].all() and jd["throughput_preserved"].all()
    assert not jd.loc[jd["mode"] == "cacc", "cacc_better_than_acc"].any()
