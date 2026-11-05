# Design Record (one evolving document)

## 1. Question and prediction
Is actuated freeway on-ramp metering better than pre-timed or no signal at all volumes? Prediction: actuated wins only in a middle band near capacity (X about 0.9 to 1.05). See Intermediate 1 for the full reasoning and "what would change our conclusion".

## 2. Model and assumptions
IDM, ballistic update, parameters in `configs/idm_baseline.yaml`. Limitations: no reaction time (ACC-like), deterministic driving (stochastic only in arrivals).

Baseline simulator (`src/tmc/network.py`, single lane, no merge yet):
- Arrivals: piecewise-constant demand, Poisson (seeded) or uniform.
- Upstream boundary: point queue. A vehicle enters at x = 0 when the gap to the last vehicle is at least s0 + v_last*T, at speed v_last (v0 if the road is empty). Waiting vehicles are never dropped, and their wait counts in travel time.
- Downstream boundary: free exit. Exit time is interpolated within the step.
- On-ramp (`configs/scenario_s1_onramp.yaml`): ramp entrance (stop line, future meter site) -> 250 m approach -> 250 m acceleration lane (1500-1750 m). Ramp vehicles merge when the MOBIL safety criterion holds (gaps > 0, merging vehicle and new follower both decelerate no harder than b_safe = 4 m/s^2). If they cannot merge, they stop 1 m before the lane end. Optional switches: cooperative yielding, speed adaptation.
- **Merge model is PROVISIONAL.** With one mainline lane, the merge discharges only ~1430-1490 veh/h and breaks down even at X = 0.92 (experiment log E1 correction). Options: (a) two mainline lanes with MOBIL lane changing; (b) recalibrate the single-lane merge against a target merge capacity; (c) keep it and define X against the measured merge capacity instead of q_c. Team decision pending.
- Derived values are computed in `src/tmc/idm.py`. Baseline q_c = 1742 veh/h at v_c = 18.6 m/s (41.5 mph); k_jam = 1/(s0+l) = 230 veh/mi.

## 3. Interfaces
See `src/tmc/interfaces.py`.

## 4. Verification log
(conservation, free motion, equilibrium, gaps, boundaries, time step, detector audit: record date, result)

| Date | Check | Test | Result |
|---|---|---|---|
| 2026-11-05 | Free motion: accel <= a, v < v0, monotone speed-up | test_network::test_free_motion_single_vehicle, test_idm::test_free_road_acceleration_bounds | pass |
| 2026-11-05 | Equilibrium: accel = 0 at s_e(v); follower at s_e stays put 300 s | test_idm::test_equilibrium_* | pass (err < 1e-6) |
| 2026-11-05 | Gap > 0: stop behind a standing vehicle from v0, dt 0.05 to 0.5 s | test_idm::test_approach_to_stopped_vehicle_never_collides | pass. Final gap ~1.89 m (< s0 = 2 m) is an IDM property: it converges as dt -> 0 |
| 2026-11-05 | Conservation: arrived = queued + on road + exited every step | test_network::test_conservation_every_step_and_nothing_dropped | pass |
| 2026-11-05 | Boundary: truncated run reports leftovers, never drops | test_network::test_time_limit_reports_leftovers_instead_of_dropping | pass |
| 2026-11-05 | Boundary: saturated entrance discharges at q_c (within 5%) | test_network::test_saturated_entry_discharges_near_capacity | pass (1740 vs 1742 veh/h in E0) |
| 2026-11-05 | Time step: TTT and mean TT within 1% for dt = 0.05/0.1/0.2 s | test_network::test_time_step_sensitivity | pass |
| 2026-11-05 | Ramp: conservation, all ramp vehicles merge inside the merge zone and exit, gap > 0, no ramp vehicle past the lane end. Run for all 4 merge variants | tests/test_ramp.py | pass (34/34 total) |
| 2026-11-05 | Ramp configured with zero demand gives identical mainline results | test_ramp::test_zero_ramp_demand_matches_single_lane | pass |
| 2026-11-05 | Visual check in the animated viewer (`python -m tmc.viz`) | manual | found the lane-end artifact (E1 correction) |
| | Detector count audit | todo with detectors.py | |

## 5. Results

## 6. Design changes made in response to feedback

## 7. Final pathway decision (CAV or signal)
Open.

## 8. AI-use and verification disclosure
AI used to gather papers, format deliverables, summarize topics, and draft verification/fallback plans (per Intermediate 1). Record any further use here and how output was checked.

- 2026-11-05: Claude Code (AI assistant) drafted `idm.py`, `network.py`, `example.py`, and the tests. Checked by: the verification tests above, comparing simulated entry discharge against independently computed q_c, and reviewing the time-space and count plots. The team must review and be able to explain each module.
- 2026-11-05: Claude Code drafted the on-ramp merge, `tests/test_ramp.py`, and the animated viewer (`src/tmc/viz.py`, `viewer_template.html`). Checked with the tests and by watching the viewer, which exposed a merge-model artifact (E1 correction). Merge calibration is still open.
