"""Verification of the human reaction-time switch (design record section 7, experiment log E9).

a(t) = a_IDM(s(t - T_r), v(t - T_r), v_l(t - T_r)) for car-following; decisions (merge, courtesy, MOBIL,
lane end) stay undelayed. Checks: T_r = 0 reproduces committed results exactly; the delayed inputs are
exactly the stored state T_r ago; steady state unchanged; simulated platoon gain matches the delayed
linear theory; collision-free and conserving on S0 and S3; time-step sensitivity.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tmc.idm import (acceleration, equilibrium_gap, load_params, max_platoon_gain, platoon_gain,
                     string_unstable_ranges, unstable_speeds_by_gain)
from tmc.network import FreewaySegment, ScenarioConfig

ROOT = Path(__file__).resolve().parents[1]
P = load_params("baseline")
T_R = 0.6


def cfg(**kw):
    base = dict(scenario_id="test", demand_id="test", param_set="baseline", seed=1, length_m=1000.0,
                dt_s=0.1, record_every_s=0.1, max_time_s=3600.0, arrival_process="uniform",
                demand_profile=[[0, 300, 1200]])
    base.update(kw)
    return ScenarioConfig(**base)


def platoon(v_e, n):
    """Initial vehicles [x, v] in IDM equilibrium at v_e, leader (vehicle 0) most downstream."""
    s_e = float(equilibrium_gap(v_e, P))
    x0 = (n + 1) * (s_e + P.length) + 10.0
    return [[x0 - i * (s_e + P.length), v_e] for i in range(n + 1)]


# --------------------------------------------------------------------------- switch off = old model

def test_off_reproduces_committed_S0_results_exactly():
    """T_r = 0 (the default) must reproduce results/S0_single_lane_vehicles.csv (committed before the switch)."""
    c = ScenarioConfig.from_yaml(ROOT / "configs" / "scenario_s0_single_lane.yaml")
    assert c.reaction_time_s == 0.0
    res = FreewaySegment(load_params(c.param_set), c).run()
    ref = pd.read_csv(ROOT / "results" / "S0_single_lane_vehicles.csv")
    got = res.vehicles.set_index("vehicle_id").loc[ref["vehicle_id"]]
    assert np.allclose(got["exit_s"].to_numpy(), ref["exit_s"].to_numpy(), rtol=0, atol=1e-9)
    assert np.allclose(got["entry_s"].to_numpy(), ref["entry_s"].to_numpy(), rtol=0, atol=1e-9)


def test_time_step_must_divide_reaction_time():
    with pytest.raises(ValueError):
        FreewaySegment(P, cfg(reaction_time_s=0.25, dt_s=0.1))
    with pytest.raises(ValueError):
        FreewaySegment(P, cfg(reaction_time_s=-0.1))
    assert FreewaySegment(P, cfg(reaction_time_s=0.6, dt_s=0.2)).D == 3


# --------------------------------------------------------------------------- the delayed inputs

def test_applied_acceleration_uses_state_exactly_T_r_ago():
    """Recompute each follower's acceleration by hand from the recorded trajectory T_r earlier."""
    def brake(t, v):
        return -1.5 if 5.0 <= t < 9.0 else (0.5 if t >= 9.0 and v < 20.0 else 0.0)
    c = cfg(length_m=3000.0, demand_profile=[], initial_vehicles=platoon(20.0, 3), reaction_time_s=T_R,
            max_time_s=60.0)
    res = FreewaySegment(P, c, lead_accel=brake).run()
    tr = res.trajectories
    D = int(round(T_R / c.dt_s))
    x = tr.pivot(index="time_s", columns="vehicle_id", values="position_m").to_numpy()
    v = tr.pivot(index="time_s", columns="vehicle_id", values="speed_mps").to_numpy()
    a = tr.pivot(index="time_s", columns="vehicle_id", values="acceleration_mps2").to_numpy()
    for i in (1, 2, 3):
        s_d = x[:-D, i - 1] - x[:-D, i] - P.length
        expect = acceleration(v[:-D, i], s_d, v[:-D, i] - v[:-D, i - 1], P)
        assert np.allclose(a[D:, i], expect, atol=1e-12)
        # and it is NOT the undelayed response once the disturbance has started
        s_now = x[D:, i - 1] - x[D:, i] - P.length
        now = acceleration(v[D:, i], s_now, v[D:, i] - v[D:, i - 1], P)
        assert np.abs(a[D:, i] - now).max() > 1e-3
    # first T_r after the start: no history yet, so undelayed
    s0_ = x[:D, 0] - x[:D, 1] - P.length
    assert np.allclose(a[:D, 1], acceleration(v[:D, 1], s0_, v[:D, 1] - v[:D, 0], P), atol=1e-12)


def test_steady_state_unchanged():
    """A platoon in IDM equilibrium stays there with the delay on (the delay does not change q_c)."""
    c = cfg(length_m=20000.0, demand_profile=[], initial_vehicles=platoon(15.0, 10), reaction_time_s=T_R,
            max_time_s=300.0, record_every_s=1.0)
    res = FreewaySegment(P, c, lead_accel=lambda t, v: 0.0).run()
    tr = res.trajectories
    assert np.abs(tr["speed_mps"] - 15.0).max() < 1e-9
    assert np.abs(tr["acceleration_mps2"]).max() < 1e-9


# --------------------------------------------------------------------------- against the delayed theory

def test_theory_delay_does_not_move_long_wave_boundary_but_raises_growth():
    """Analytic: with the delay on all inputs, the unstable speed range is unchanged (long-wave limit)
    while the most amplified wave at 15 m/s grows faster and is shorter."""
    u0 = unstable_speeds_by_gain(P, 0.0)
    u6 = unstable_speeds_by_gain(P, T_R)
    assert u0.max() == pytest.approx(string_unstable_ranges(P)[0][1], abs=0.1)   # gain route = Eq. 15.77
    assert u6.max() == pytest.approx(u0.max(), abs=0.1)
    g0, g6 = max_platoon_gain(15.0, P), max_platoon_gain(15.0, P, reaction_time_s=T_R)
    assert g6["growth_per_veh"] > 1.3 * g0["growth_per_veh"]
    assert g6["period_s"] < g0["period_s"]
    # T_r = 0 reduces to the undelayed transfer function
    w = np.linspace(0.01, 2, 50)
    assert np.allclose(platoon_gain(15.0, w, P, 0.0), platoon_gain(15.0, w, P))


def _sim_gain(v_e, period, reaction_time_s, n=20, amp=0.05, t_end=1200.0):
    w = 2 * np.pi / period

    def sinus(t, v):  # leader speed v_e + amp * ramp(t) * sin(w t), faded in over 120 s
        r, dr = (t / 120.0, 1 / 120.0) if t < 120.0 else (1.0, 0.0)
        return amp * (r * w * np.cos(w * t) + dr * np.sin(w * t))

    c = cfg(length_m=1e6, demand_profile=[], initial_vehicles=platoon(v_e, n), reaction_time_s=reaction_time_s,
            max_time_s=t_end, record_every_s=0.5)
    tr = FreewaySegment(P, c, lead_accel=sinus).run().trajectories
    tail = tr[tr["time_s"] >= 0.6 * t_end]
    amp_v = tail.groupby("vehicle_id")["speed_mps"].agg(lambda s: 0.5 * (s.max() - s.min()))
    return (amp_v[n] / amp_v[0]) ** (1.0 / n)


def test_simulator_matches_delayed_theory():
    """The real simulator (network.py with the delay) reproduces the delayed per-vehicle gain."""
    period = max_platoon_gain(15.0, P, reaction_time_s=T_R)["period_s"]
    g_theory = float(platoon_gain(15.0, 2 * np.pi / period, P, T_R))
    g_sim = _sim_gain(15.0, period, T_R)
    assert g_sim == pytest.approx(g_theory, abs=2e-3)
    # the same wave without the delay grows less, in theory and in the simulator
    assert _sim_gain(15.0, period, 0.0) < g_sim - 5e-3


# --------------------------------------------------------------------------- safety, conservation, dt

def test_S0_with_reaction_time_is_safe_and_conserving():
    c = ScenarioConfig.from_yaml(ROOT / "configs" / "scenario_s0_single_lane.yaml")
    c.reaction_time_s = T_R
    res = FreewaySegment(load_params(c.param_set), c).run()   # raises on any overlap
    cnt = res.counts
    assert np.all(cnt["n_arrived"] == cnt["n_queued"] + cnt["n_on_road"] + cnt["n_exited"])
    assert res.cleared and res.min_gap_m > 0


def test_S3_merge_with_reaction_time_is_safe_and_conserving():
    """E9 gate: humans with T_r = 0.6 s must be collision-free at the merge (S3 baseline, X_m ~ 1.03)."""
    c = ScenarioConfig.from_yaml(ROOT / "configs" / "scenario_s3_onramp_zipgap.yaml")
    c.reaction_time_s = T_R
    res = FreewaySegment(load_params(c.param_set), c).run()
    cnt = res.counts
    assert np.all(cnt["n_arrived"] == cnt["n_queued"] + cnt["n_on_road"] + cnt["n_exited"])
    assert res.cleared and res.min_gap_m > 0


def _dt_spread(reaction_time_s, key):
    out = {dt: FreewaySegment(P, cfg(dt_s=dt, reaction_time_s=reaction_time_s, record_every_s=1.0,
                                     demand_profile=[[0, 300, 2400], [300, 600, 800]])).run().summary()[key]
           for dt in (0.05, 0.1, 0.2)}
    return max(abs(out[dt] / out[0.05] - 1) for dt in (0.1, 0.2))


@pytest.mark.parametrize("key", ["total_travel_time_veh_h", "mean_travel_time_s"])
def test_time_step_sensitivity_with_reaction_time(key):
    """Human-only dt check with T_r on (dt = 0.05 / 0.1 / 0.2 s). Threshold = the E9 verification gate
    (< 2%, pre-registered). The undelayed model on the same case already varies ~0.6% (and ~1.2% from
    dt = 0.025 to 0.1, non-monotonic: entry-time quantization at the point-queue boundary), so we also
    require that the delay does not make the dt sensitivity materially worse."""
    spread_tr = _dt_spread(T_R, key)
    assert spread_tr < 0.02
    assert spread_tr < _dt_spread(0.0, key) + 0.005


# --------------------------------------------------------------------------- leader-change rule (E9e, 2026-11-25)

S3R = ROOT / "configs"


def _s3r(rule: str, t_end: float = 900.0) -> ScenarioConfig:
    c = ScenarioConfig.from_yaml(S3R / "scenario_s3r_onramp_zipgap_tr06.yaml")
    c.leader_change_delay = rule
    c.max_time_s = t_end
    c.dt_s = 0.05
    return c


def test_restart_rule_is_identical_without_leader_changes():
    """Setting A (one lane, no entries, no merges): 'restart' must not change a single number."""
    from tmc.disturbance import ScriptedLead, load_setting, platoon_config
    s = load_setting()
    dist = s["disturbances"]["nominal"]
    out = []
    for rule in ("delayed_gap", "restart"):
        cfg_ = platoon_config(s, P, 0.6, None, None)
        cfg_.leader_change_delay = rule
        res = FreewaySegment(P, cfg_, lead_accel=ScriptedLead(dist, float(s["v_eq_mps"]), P.a, cfg_.dt_s)).run()
        out.append(res.trajectories[["position_m", "speed_mps"]].to_numpy())
    assert np.array_equal(out[0], out[1])


def test_restart_rule_removes_stale_gap_braking_at_the_merge():
    """S3r merge at dt 0.05: the old rule produces decelerations far beyond any physical limit right after
    merges (a stale, tiny delayed gap); with 'restart' the human maximum stays within the IDM's range
    for real gaps and the run stays collision-free and conserving."""
    old = FreewaySegment(P, _s3r("delayed_gap")).run()
    c = _s3r("restart")
    new = FreewaySegment(P, c).run()
    assert old.gap_stats["max_decel_mps2_hh"] > 100.0          # the artifact (documents why the rule exists)
    assert new.gap_stats["max_decel_mps2_hh"] < 20.0
    assert new.min_gap_m > 0
    s = new.summary()
    assert s["n_exited"] + s["on_road_at_end_veh"] + s["unmet_demand_veh"] == s["n_arrived"]


def test_leader_change_rule_rejects_unknown_values():
    c = _s3r("sometimes")
    with pytest.raises(ValueError):
        FreewaySegment(P, c)
