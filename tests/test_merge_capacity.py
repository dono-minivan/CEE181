"""Checks for the merge-capacity measurement (tmc.merge_capacity)."""
from pathlib import Path

import pandas as pd
import pytest

from tmc.merge_capacity import classify, demand_ratio, estimate, ramp_capacity_curve, steady_run
from tmc.network import ScenarioConfig

S1 = Path(__file__).resolve().parents[1] / "configs" / "scenario_s1_onramp.yaml"


def test_estimate_on_synthetic_sweep():
    rows = []
    for q, served, ok in [(1000, 1000, True), (1100, 1100, True), (1200, 1150, False),
                          (1300, 1290, True), (1400, 1150, False), (1500, 1150, False)]:
        rows.append({"q_tot_vph": q, "ramp_share": 0.2, "seed": 1, "served_vph": served,
                     "ramp_served_vph": 0, "queued_end_veh": 0, "demand_limited": ok})
    est = estimate(pd.DataFrame(rows)).iloc[0]
    assert est["served_at_max_vph"] == pytest.approx(1150)
    assert est["n_saturated_runs"] == 3
    assert est["breakdown_threshold_vph"] == 1100   # 1300 passed, but 1200 already failed


def test_demand_ratio():
    assert demand_ratio(1300, 500, 1500) == pytest.approx(1.2)


def test_low_demand_is_fully_served_and_inputs_untouched():
    base = ScenarioConfig.from_yaml(S1)
    before = (base.demand_profile, base.ramp.demand_profile, base.seed)
    r = steady_run(base, 800.0, 0.25, seed=181, horizon=900.0, warmup=300.0)
    assert r["served_vph"] == pytest.approx(800.0, rel=0.15)   # Poisson count over a 600 s window
    assert r["queued_end_veh"] < 10
    assert (base.demand_profile, base.ramp.demand_profile, base.seed) == before


def test_queue_criterion_ignores_poisson_shortfall():
    """A run that served less than nominal demand only because fewer vehicles arrived is not saturated."""
    df = classify(pd.DataFrame([{"q_tot_vph": 1000, "ramp_share": 0.1, "seed": 182, "served_vph": 955,
                                 "ramp_served_vph": 90, "queued_end_veh": 0}]))
    assert bool(df["demand_limited"].iloc[0])


def test_ramp_capacity_curve_uses_ramp_limited_runs_only():
    rows = [  # saturated, mainline fully served (q_m = 1000): ramp leftover 300 / 320 / 310
        {"q_tot_vph": 1500, "ramp_share": 1 / 3, "seed": s, "served_vph": 1000 + c, "ramp_served_vph": c,
         "queued_end_veh": 80} for s, c in [(1, 300), (2, 320), (3, 310)]]
    rows.append({"q_tot_vph": 2000, "ramp_share": 0.1, "seed": 1, "served_vph": 1500,  # mainline queued too
                 "ramp_served_vph": 0, "queued_end_veh": 200})
    rows.append({"q_tot_vph": 900, "ramp_share": 0.1, "seed": 1, "served_vph": 900,   # not saturated
                 "ramp_served_vph": 90, "queued_end_veh": 0})
    c = ramp_capacity_curve(classify(pd.DataFrame(rows)), min_runs=1)
    assert len(c) == 1 and c["q_m_bin_vph"].iloc[0] == 1050
    assert c["ramp_cap_vph"].iloc[0] == pytest.approx(310)
