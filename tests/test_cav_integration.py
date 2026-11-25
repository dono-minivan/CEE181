"""E9 verification gates for equipped vehicles in the simulator (design record section 7).

p = 0 identical to the baseline; CACC with comms off identical to ACC; equipped vehicles hold the IDM
equilibrium; controllers get only permitted scalar inputs; leader-first order (a 0-tick delay is usable the
same tick); gap > 0 and conservation, incl. the merge; every fallback logged; time-step sensitivity.
"""
from pathlib import Path

import numpy as np
import pytest

from tmc import cav as cavmod
from tmc.disturbance import Disturbance, load_setting, platoon_metrics, run_platoon
from tmc.idm import acceleration, load_params
from tmc.network import FreewaySegment, ScenarioConfig

ROOT = Path(__file__).resolve().parents[1]
P = load_params("baseline")
S = load_setting()
NOM = S["disturbances"]["nominal"]
T_R = 0.6


def cav(p, mode="cacc", seed=181, **kw):
    return {"p_equipped": p, "mode": mode, "placement_seed": seed, **kw}


def traj(res):
    return res.trajectories.sort_values(["time_s", "vehicle_id"])[["position_m", "speed_mps"]].to_numpy()


# --------------------------------------------------------------------------- identities

def test_p0_is_identical_to_no_cav_config():
    a, _ = run_platoon(S, NOM, T_R)
    b, _ = run_platoon(S, NOM, T_R, cav=cav(0.0))
    assert np.array_equal(traj(a), traj(b))
    c = ScenarioConfig.from_yaml(ROOT / "configs" / "scenario_s3r_onramp_zipgap_tr06.yaml")
    r1 = FreewaySegment(P, c).run()
    c.cav = cav(0.0)
    r2 = FreewaySegment(P, c).run()
    assert np.array_equal(r1.vehicles["exit_s"].to_numpy(), r2.vehicles["exit_s"].to_numpy())
    assert not r2.vehicles["equipped"].any()


@pytest.mark.parametrize("p_eq", [0.5, 1.0])
def test_cacc_with_comms_off_is_identical_to_acc(p_eq):
    a, _ = run_platoon(S, NOM, T_R, cav=cav(p_eq, "acc"))
    b, _ = run_platoon(S, NOM, T_R, cav=cav(p_eq, "cacc", comms=False))
    assert np.array_equal(traj(a), traj(b))
    assert set(b.cav_events["reason"]) == {"comms_off"}


@pytest.mark.parametrize("mode", ["acc", "cacc"])
def test_equipped_platoon_holds_idm_equilibrium(mode):
    """Same steady state as humans: a fully equipped platoon behind a constant-speed lead stays put."""
    none = Disturbance("none", v_low=10.0, b_lead=2.0, hold_s=10.0, t_start=1e9)
    res, _ = run_platoon(dict(S, duration_s=120.0), none, T_R, cav=cav(1.0, mode))
    tr = res.trajectories
    assert np.abs(tr["speed_mps"] - 15.0).max() < 1e-9
    assert np.abs(tr["acceleration_mps2"]).max() < 1e-9


# --------------------------------------------------------------------------- inputs and order

def test_controller_receives_only_permitted_scalar_inputs(monkeypatch):
    seen = []
    orig = cavmod.CAVController.command

    def spy(self, tick, v, s, approach_rate, leader, channel=None, leader_equipped=False):
        seen.append((tick, v, s, approach_rate, leader, channel, leader_equipped))
        return orig(self, tick, v, s, approach_rate, leader, channel, leader_equipped)

    monkeypatch.setattr(cavmod.CAVController, "command", spy)
    run_platoon(dict(S, duration_s=20.0), NOM, T_R, cav=cav(0.5))
    assert seen
    for tick, v, s, dv, leader, ch, eq in seen:
        assert type(tick) is int and type(v) is float and type(s) is float and type(dv) is float
        assert leader is None or type(leader) is int
        assert ch is None or isinstance(ch, cavmod.V2VChannel)
        assert type(eq) is bool
    # a controller holds no reference to the simulator, only its own state
    c = cavmod.CAVController(1, "cacc", P)
    assert not any(hasattr(v, "lanes") or hasattr(v, "trajectories") for v in vars(c).values())


def test_leader_first_order_makes_zero_delay_usable_same_tick():
    """delay 0, p = 1: every follower's leader is equipped and has already broadcast this tick, so no
    fallback ever happens (with follower-first order, tick 0 would show 'no_message')."""
    res, _ = run_platoon(dict(S, duration_s=30.0), NOM, T_R, cav=cav(1.0, delay_s=0.0))
    assert len(res.cav_events) == 0


# --------------------------------------------------------------------------- safety, conservation, logging

@pytest.mark.parametrize("mode,p_eq,dist", [("acc", 0.5, "nominal"), ("acc", 1.0, "strong"),
                                            ("cacc", 0.25, "nominal"), ("cacc", 1.0, "strong")])
def test_platoon_runs_are_collision_free(mode, p_eq, dist):
    res, p = run_platoon(S, S["disturbances"][dist], T_R, cav=cav(p_eq, mode))   # raises on overlap
    m, _ = platoon_metrics(res, p, 15.0)
    assert m["min_gap_m"] > 0
    assert res.vehicles["equipped"].sum() == np.sum(np.random.default_rng(181).random(31) < p_eq)


def test_every_fallback_is_logged_with_a_reason():
    res, _ = run_platoon(dict(S, duration_s=30.0), NOM, T_R, cav=cav(0.5))
    eq = res.vehicles.set_index("vehicle_id")["equipped"]
    ev = res.cav_events
    assert set(ev["reason"]) <= {"no_message", "leader_unequipped", "stale", "comms_off"}
    # every equipped follower (not the lead) behind a human is in fallback on every tick
    n_ticks = int(round(30.0 / cavmod.TICK_S)) + 1
    for vid in range(1, 31):
        if eq[vid] and not eq[vid - 1]:
            assert (ev[ev["vehicle_id"] == vid]["reason"] == "leader_unequipped").sum() == n_ticks
    # a 5 s outage: fallback 'stale' appears during it and stops after it
    res2, _ = run_platoon(dict(S, duration_s=40.0), NOM, T_R, cav=cav(1.0, outages=[[10.0, 15.0]]))
    st = res2.cav_events[res2.cav_events["reason"] == "stale"]
    assert len(st) and st["time_s"].min() > 10.0 and st["time_s"].max() < 15.0 + 0.3


def test_merge_with_equipped_vehicles_is_safe_and_conserving():
    c = ScenarioConfig.from_yaml(ROOT / "configs" / "scenario_s3r_onramp_zipgap_tr06.yaml")
    c.cav = cav(0.5)
    res = FreewaySegment(P, c).run()
    cnt = res.counts
    assert np.all(cnt["n_arrived"] == cnt["n_queued"] + cnt["n_on_road"] + cnt["n_exited"])
    assert res.cleared and res.min_gap_m > 0
    v = res.vehicles
    assert v.loc[v["origin"] == "ramp", "equipped"].any() and v["merge_s"].notna().sum() > 0


def test_decision_acceleration_of_equipped_equals_idm_in_checks():
    """Merge/courtesy checks: equipped use ACC with a_l missing = IDM (clipped), as confirmed."""
    seg = FreewaySegment(P, ScenarioConfig(scenario_id="t", demand_id="t", param_set="baseline", seed=1,
                                           length_m=1000.0, dt_s=0.1, record_every_s=1.0, max_time_s=1.0,
                                           arrival_process="uniform", demand_profile=[[0, 10, 1000]],
                                           cav=cav(1.0)))
    for v, s, dv in [(15.0, 20.0, 3.0), (25.0, 8.0, 5.0), (5.0, 50.0, -2.0)]:
        assert seg._decision_acc(0, v, s, dv) == pytest.approx(max(float(acceleration(v, s, dv, P)), -8.0))


# --------------------------------------------------------------------------- time step

@pytest.mark.parametrize("mode", ["acc", "cacc"])
def test_time_step_sensitivity_with_cavs(mode):
    """E9 gate: metrics change < 2% across dt = 0.025 / 0.05 / 0.1 s (CAV dt set; humans T_r = 0.6 s)."""
    out = {}
    for dt in (0.025, 0.05, 0.1):
        res, p = run_platoon(S, NOM, T_R, dt=dt, cav=cav(0.5, mode))
        out[dt], _ = platoon_metrics(res, p, 15.0)
    for key in ("A", "dv_last", "dv_veh10", "min_gap_m"):
        ref = out[0.1][key]
        for dt in (0.025, 0.05):
            assert out[dt][key] == pytest.approx(ref, rel=0.02), (key, dt, out[dt][key], ref)
    # g itself is small (~0.01/veh): require agreement well inside the 0.0035 test margin
    for dt in (0.025, 0.05):
        assert abs(out[dt]["g_per_veh"] - out[0.1]["g_per_veh"]) < 0.1 * S["g_threshold_per_veh"]


def test_dt_must_divide_tick():
    with pytest.raises(ValueError):
        run_platoon(dict(S, duration_s=1.0), NOM, 0.0, dt=0.2, cav=cav(1.0))


# --------------------------------------------------------------------------- broadcast = 'effective' (E9e, 2026-11-25)

def test_effective_broadcast_is_identical_in_setting_a():
    """No lane-end / courtesy / gap-seeking demand in setting A, so nothing is ever amended."""
    for p_eq in (0.5, 1.0):
        a, _ = run_platoon(S, NOM, T_R, cav=cav(p_eq, "cacc"))
        b, _ = run_platoon(S, NOM, T_R, cav=cav(p_eq, "cacc", broadcast="effective"))
        assert np.array_equal(traj(a), traj(b))


def test_channel_amend_replaces_only_this_ticks_message():
    ch = cavmod.V2VChannel(delay_s=0.15)
    ch.send(7, 10, 1.0)
    assert not ch.amend(7, 9, -3.0)                    # not the newest tick
    assert ch.amend(7, 10, -2.0)
    assert ch.latest(7, 12, 0) == (-2.0, 10)
    lossy = cavmod.V2VChannel(delay_s=0.15, outages=[(0.0, 10.0)])
    lossy.send(7, 10, 1.0)                              # lost in the outage
    assert not lossy.amend(7, 10, -2.0) and lossy.latest(7, 12, 0) is None


def test_effective_broadcast_sends_min_of_command_and_applied_at_the_merge():
    """S3r2 merge, all equipped: after each tick, every equipped vehicle's message equals its command u,
    or min(u, applied acceleration) when another demand (lane end, gap seeking, courtesy) bound it."""
    checked = {"bound": 0, "free": 0}

    class Spy(FreewaySegment):
        def _amend_broadcasts(self, tick, accs, acc_r):
            super()._amend_broadcasts(tick, accs, acc_r)
            for ln, acc in [*zip(self.lanes, accs), (self.rlane, acc_r)]:
                for i, vid in enumerate(ln.ids):
                    ctl, a_r = self.ctl.get(int(vid)), self._a_real.get(int(vid))
                    q = self.channel._q.get(int(vid))
                    if ctl is None or a_r is None or not q or q[-1][0] != tick:
                        continue
                    bound = acc[i] < a_r - 1e-12
                    want = min(ctl.u, float(acc[i])) if bound else ctl.u
                    assert q[-1][2] == pytest.approx(want, abs=1e-12)
                    checked["bound" if bound and want < ctl.u else "free"] += 1

    c = ScenarioConfig.from_yaml(ROOT / "configs" / "scenario_s3r2_onramp_zipgap_tr06_restart.yaml")
    c.max_time_s = 400.0
    c.cav = cav(1.0, "cacc", broadcast="effective")
    res = Spy(P, c).run()
    assert checked["bound"] > 0 and checked["free"] > 0
    assert res.min_gap_m > 0


def test_broadcast_rejects_unknown_values():
    c = ScenarioConfig.from_yaml(ROOT / "configs" / "scenario_s3r2_onramp_zipgap_tr06_restart.yaml")
    c.cav = cav(1.0, "cacc", broadcast="sometimes")
    with pytest.raises(ValueError):
        FreewaySegment(P, c)
