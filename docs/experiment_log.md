# Experiment Log
Template per entry: ID | date | prediction (written BEFORE running) | change made | result | interpretation | config/commit

## E0 | 2026-11-05 | Single-lane IDM baseline with entry queue (S0_single_lane)
- **Prediction (written before running):** Computed IDM capacity is about 1742 veh/h (baseline params). Period 1 (1200 veh/h) is free flow with no persistent entry queue. Period 2 (2000 veh/h) exceeds what the entrance can admit, so an entry queue grows. The entry rule admits roughly capacity, so we expect about (2000 - ~1750)/3600 * 600 = ~40 veh at peak. In period 3 (800 veh/h) the queue clears within a few minutes. With no downstream bottleneck, congestion stays at the entrance, the mainline never stops, and min gap stays > s0-ish. All vehicles exit (cleared = True), and conservation holds at every recorded step.
- **Change:** First simulator (`src/tmc/idm.py`, `src/tmc/network.py`), config `configs/scenario_s0_single_lane.yaml`, seed 181, dt 0.1 s.
- **Result:** 822 arrived, 822 exited, cleared at t = 2502 s, unmet 0. Peak entry queue 64 veh (predicted ~40). Measured saturated entry discharge 1740 veh/h, matching computed q_c = 1742 veh/h. Min speed 38.6 mph (the capacity state, v_c = 41.5 mph). The mainline never stopped, and min gap was 29.8 m. TTT 36.6 veh*h, including 8.3 veh*h of entry waiting. Conservation held at every step (pytest).
- **Interpretation:** The prediction held in shape and mechanism. The queue-size miss came from demand realization, not the model: seed 181 drew 353 arrivals in period 2 vs 333 expected, and 353 - 1740/6 = 63, which matches the peak of 64. Lesson: state queue predictions per realized arrivals, or average over seeds. With no downstream bottleneck, the entrance itself is the bottleneck and discharges at q_c, so later merge experiments need mainline demand < q_c to be meaningful.

## E1 | 2026-11-05 | Unmetered on-ramp merge (S1_onramp_nocontrol)
- **Prediction (written before running):** Before 600 s, X = 1600/1742 = 0.92: the merge works, short ramp waits, no persistent queue. From 600 to 1500 s, X ~ 1.03 is over capacity. The merge becomes the active bottleneck and congestion forms just upstream of merge_start (1500 m), spreading back toward the entrance (a backward-moving shock). Downstream of the merge, flow is at or below q_c ~ 1740 veh/h. Mainline min speed falls well below v_c (41.5 mph), possibly stop-and-go. Ramp vehicles occasionally stop at the end of the acceleration lane waiting for gaps. After 1500 s, X = 1100/1742 = 0.63, so the congestion dissolves and everything clears (cleared = True). No collisions (gap > 0 everywhere), and conservation holds.
- **Change:** Added on-ramp + acceleration lane + MOBIL-safety merge to `src/tmc/network.py`. Config `configs/scenario_s1_onramp.yaml`, seed 181, dt 0.1 s.
- **Result:** 974 arrived (216 ramp), all merged and exited, cleared at 2665 s, conservation held. During 900-1500 s, mean mainline speed was 29 mph at 1000-1500 m (min 0, so stop-and-go upstream of the merge), 24 mph in the merge zone, and 49 mph downstream. Discharge 900-1600 s was 1543 veh/h, about 11% below q_c = 1742. Peak ramp queue was 40 veh (point queue, on top of vehicles physically queued on the ramp). TTT 71.3 veh*h: 32.8 mainline-origin, 38.5 ramp-origin. Ramp-queue wait 4.8 veh*h. Min gap 0.79 m (> 0, < s0).
- **Interpretation:** The bottleneck mechanism matched the prediction. Two things were not predicted. (1) Discharge fell below q_c (a capacity drop): stop-and-go plus low-speed merges cut outflow. This is relevant to metering, because a meter that prevents the breakdown could recover that ~200 veh/h. (2) Ramp vehicles carry most of the delay (22% of vehicles, 54% of TTT) because mainline drivers never yield, so the merge itself acts like an implicit meter. Before comparing control modes, decide whether to add mainline courtesy or MOBIL politeness, since that choice changes the no-signal baseline. Min gap < s0 comes from merges accepted at low speed under the b_safe criterion. It is safe (> 0) but should be watched.

### E1 correction (2026-11-05, same day): the E1 interpretation is withdrawn
The animated viewer (`python -m tmc.viz`) showed ramp vehicles stopped bumper-to-bumper while the mainline beside them moved at ~45 mph. Cause: the acceleration-lane end was modeled as an IDM standing obstacle. IDM's approach-rate term makes vehicles brake about v^2/(2 sqrt(ab)) ahead (~180 m at 20 m/s), so they slowed for the whole 250 m lane. A stopped vehicle then needs a ~85-140 m mainline gap to merge safely, which almost never occurs. So "the merge acts as an implicit meter" was a model artifact, not a traffic effect.

**Changes:** The lane end is now a kinematic stop (brake only once v^2/(2d) >= b). Added two optional behaviors, both off by default: speed adaptation to the adjacent mainline leader, and cooperative yielding by the mainline follower (after Hidas 2005).

**Follow-up scan** (steady demand 1800 s, seed 181, mainline 1300 veh/h, ramp 300 or 500 veh/h). "Out" is the exit flow over 600-1800 s, in veh/h.

| Variant | X = 0.92 out / upstream mph / ramp wait veh*h | X = 1.03 out / upstream mph / ramp wait veh*h |
|---|---|---|
| plain MOBIL, b_safe 4 (current default) | 1434 / 60.8 / 1.8 | 1464 / 36.7 / 17.0 |
| plain, b_safe 2 | 1389 / 61.4 / 5.2 | 1398 / 61.3 / 26.6 |
| + yielding, b_safe 4 | 1181 / 8.5 / 0.02 | 1112 / 3.9 / 0.04 |
| + speed adaptation, b_safe 4 | 1424 / 61.2 / 2.7 | 1447 / 39.7 / 23.7 |
| + adaptation + yielding, b_safe 4 | 1428 / 58.7 / 3.9 | 1447 / 39.7 / 23.7 |

**Interpretation:** No variant is credible yet. Even at X = 0.92 (below q_c), every variant reaches min speed 0 and discharges only ~1400-1490 veh/h, about 15-20% below q_c = 1742. Without yielding, the ramp starves once a vehicle stops at the lane end. With yielding, the mainline collapses instead. The likely structural cause: one mainline lane with T = 1.6 s leaves no room to absorb a merge, and mainline drivers cannot move over. **Decision needed before any control comparison:** see the design record, section 2 (merge model). S1 with current defaults: TTT 67.9 veh*h, ramp wait 5.5 veh*h, peak ramp queue 48, discharge 1483 veh/h, cleared at 2614 s.

## E2 | 2026-11-07 | Two mainline lanes + symmetric MOBIL lane changing (S2_twolane_onramp_nocontrol)
- **Prediction (stated before any two-lane run, in the planning discussion):** With a second lane, right-lane drivers can move left to absorb merging vehicles. So (1) at X = 0.92 the merge no longer breaks down, and (2) at X ~ 1.03 outflow gets much closer to 2 q_c = 3485 veh/h (computed) than the single lane's ~1460 out of 1742 did. With no ramp, the two lanes split traffic about 50/50, and a saturated entrance discharges about 2 q_c.
- **Change:** `n_lanes` mainline lanes (lane 0 = right, ramp lane id now -1) in `src/tmc/network.py`. Symmetric MOBIL discretionary lane changes (p = 0.2, threshold 0.1 m/s^2, b_safe 4 m/s^2, 3 s cooldown). Decisions use the state at t and are applied largest gain first, each re-checked for safety. One entry queue feeds both lanes. Lane-change log in `SimResult.lane_changes`. The ramp merge rule is unchanged, now into lane 0. Config `configs/scenario_s2_twolane_onramp.yaml` (S1 demand x2, so the X profile matches S1), seed 181, dt 0.1 s.
- **Regression:** S0 and S1 summaries and trajectories are bit-identical to the pre-change code.
- **Result, no ramp (prediction held):** At 2400 veh/h, the lanes split 50/50 with 2 lane changes in 1800 s. Saturated entrance discharge was 3435 veh/h at 4200 veh/h demand (98.6% of 2 q_c); the test checks it within 5%. Gaps stay > 29 m.
- **Result, with ramp (prediction failed):** Variant scan, steady demand 1800 s, seed 181, mainline 2600 veh/h, ramp 600 (X = 0.92) or 1000 (X = 1.03). Columns: exit flow over 600-1800 s / ramp merges per hour over 600-1800 s / mean mainline speed at 1000-1500 m.

| Variant | X = 0.92 out / ramp vph / upstream mph | X = 1.03 out / ramp vph / upstream mph |
|---|---|---|
| plain MOBIL merge (S2 default) | 2736 / 0 / 61.5 | 2736 / 0 / 61.5 |
| + cooperative yield | 2415 / 483 / 8.7 (9520 lane changes) | 2442 / 402 / 9.2 |
| + yield, min speed ratio 0.5 | 2742 / 0 / 61.5 | 3087 / 294 / 20.0 |
| + speed adaptation | 2736 / 0 / 61.5 | 2736 / 0 / 61.5 |
| plain, lane-end trigger 3 m/s^2 | 2769 / 0 / 61.5 | 2808 / 0 / 61.1 |
| adaptation, lane-end trigger 3 m/s^2 | 2934 / 123 / 54.0 | 2736 / 0 / 61.5 |

S2 itself: 1995 vehicles (435 ramp), all merged and exited, cleared at 2903 s. TTT 190.2 veh*h, of which ramp-queue wait 94.0. Peak ramp queue 271. 814 discretionary lane changes; median 0 per vehicle, max 12. The heavy changers (15 vehicles) hop lanes at ~2 mph in the queue beside the stalled ramp. Min gap 0.79 m.
- **Interpretation:** The lane-change mechanics verify (tests/test_multilane.py), but the second lane does not fix the merge. Mechanism: ramp merges stop entirely after ~600 s while both mainline lanes run at 61 mph. The first ramp vehicle that fails to find a gap brakes to the lane end. Stopped, it needs a ~165 m gap to keep a 28 m/s lane-0 follower within b_safe, and lane-0 spacing at 1300 veh/h is ~76 m. Lane-0 drivers have no reason to move left, because ramp vehicles do not enter their MOBIL accelerations. Unrestricted yielding gives them that reason, but then they yield to stopped ramp vehicles and the mainline collapses. Contributing causes: (a) the lane-end trigger b = 1.67 m/s^2 means braking starts ~230 m before the end at 28 m/s, i.e. almost the whole 250 m acceleration lane; (b) a ramp vehicle at mainline speed stays beside the same car (no relative speed to sweep gaps). Conclusion: the merge model, not the lane count, is the binding problem. The "no-signal" baseline is not credible until it is fixed. Decision pending, see design record section 2.

## E3 | 2026-11-07 | Merge fixes on two lanes: gap seeking + anticipatory lane changes
- **Prediction (written before running):** Gap seeking alone stops the lane-end deadlock at X = 0.92, but some ramp vehicles still stop in heavier traffic. Anticipation alone helps less: it opens gaps only for drivers already in the zone, and it can't fix ramp vehicles traveling at the same speed as the mainline. Both together should deliver nearly all ramp demand at X = 0.92 (outflow ~3200 veh/h, no stops). At X = 1.03, outflow should be about 3300-3400, with congestion upstream of the merge.
- **Change:** Two `RampConfig` switches, both off by default (S1 and S2 results unchanged: TTT 67.87 and 190.17 veh*h).
  - `gap_seeking`: in the merge zone, a ramp vehicle steers its front bumper toward the center of the larger adjacent lane-0 gap, with a = kx (x* - x) + kv (v* - v), kx = 0.1 s^-2, kv = 0.6 s^-1, clipped to [-b, a]. It is never above the IDM and lane-end limits.
  - `anticipation_m`: lane-0 drivers from merge_start - anticipation_m to merge_end count the nearest merge-zone ramp vehicle ahead as a leader in lane-change decisions only (not in driving). The same applies to lane-1 drivers judging a move into lane 0.
- **Result:** Same scan as E2 (steady 1800 s, seed 181, mainline 2600 veh/h, ramp 600 / 1000). Columns: exit flow over 600-1800 s / ramp merges per hour / mean mainline speed at 1000-1500 m / TTT veh*h.

| Variant (2 lanes) | X = 0.92 | X = 1.03 |
|---|---|---|
| plain (E2 baseline) | 2736 / 0 / 61.5 / 120.2 | 2736 / 0 / 61.5 / 210.0 |
| gap seeking | 2739 / 0 / 61.5 / 111.3 | 2757 / 0 / 61.5 / 178.4 |
| anticipation 200 m | 3072 / 492 / 33.7 / 71.5 | 2724 / 657 / 10.3 / 163.6 |
| gap + anticipation 200 m | 2913 / 354 / 22.9 / 85.9 | 2922 / 486 / 16.6 / 146.7 |
| gap + anticipation 400 m | 2907 / 489 / 19.0 / 85.6 | 2937 / 396 / 22.1 / 141.3 |
| gap + anticipation 200 m + lane-end trigger 3 m/s^2 | 3084 / 450 / 37.4 / 67.5 | 2883 / 384 / 16.1 / 150.9 |

Single lane (S1 demand) for comparison: gap seeking changes outflow by < 3% (1473 vs 1434 at X = 0.92, 1449 vs 1464 at X = 1.03), and anticipation has no effect (no lane to move to). Min mainline speed is 0 in every run.
- **Interpretation:** The prediction failed on mechanism and on size. Gap seeking alone does almost nothing: the deadlock is a stopped vehicle at the lane end, and no gap the PD law aims for is ever big enough for it. Anticipation is what breaks the deadlock, because right-lane drivers now move left, but it moves the breakdown onto the mainline: upstream speed falls to 20-37 mph already at X = 0.92. The best variant discharges 3084 veh/h (88% of 2 q_c). Every variant breaks down below capacity, so the two-lane merge is still not a credible no-control baseline. **Decision (team, 2026-11-07):** stop tuning the two-lane merge. Keep the code and switches (verified, off by default) and move on to the other merge-model options (design record section 2: (b) recalibrate the single-lane merge, (c) define X against the measured merge capacity).

## E4 | 2026-11-07 | Measured merge capacity of the single-lane merge (option (c))
- **Prediction (written before running):** Using the S1 geometry and the default (plain MOBIL) merge, the queue discharge rate once the merge is saturated is about 1450-1500 veh/h, about 15% below q_c = 1742. This matches the E1 scan, where outflow sat at 1434-1464 for both X = 0.92 and 1.03. It falls slightly as the ramp share grows, because each merge costs more mainline headway than a through vehicle. The breakdown threshold (highest total demand served with no growing queue) is lower than the discharge rate, about 1300-1400 veh/h. E1 broke down already at 1600 veh/h, but a 300 s warm-up before the demand step may have helped there. If the threshold comes out lower than q_dis, the merge has no capacity drop in the usual sense: it is simply slow from the start.
- **Change:** `src/tmc/merge_capacity.py` (new). Steady Poisson demand for 3600 s on the S1 geometry and default merge, served flow and end-of-window queue measured over 1200-3600 s. Sweep: q_tot = 1000-2000 veh/h in steps of 100, ramp share r = 0.10 / 0.20 / 0.28 / 0.40, seeds 181-183 (132 runs). Raw runs in `results/S1_onramp_nocontrol_merge_capacity_runs.csv`.
- **Method correction (before interpreting):** The first saturation test also required served >= 0.97 q_tot. Seed 182 drew ~5% fewer arrivals than nominal, so runs with no queue at all were labeled saturated (e.g. served 955 at q_tot 1000, queue 0). Saturation is now judged by the end-of-window queue only (>= 10 veh). Re-estimated from the saved runs with `python -m tmc.merge_capacity --from-csv`.
- **Result:**
  - Breakdown threshold (highest q_tot with no queue for every seed, and none below it): 1200 veh/h for r = 0.10-0.28 (0.69 q_c), 1000 veh/h for r = 0.40 (0.57 q_c). Predicted 1300-1400.
  - Saturated served flow does **not** level off. It keeps rising with demand while the ramp share of it collapses. At q_tot = 2000: 1706 served with 0 from the ramp (r = 0.10), 1434 with 228 from the ramp (r = 0.40). In 70 of 87 saturated runs the mainline got >= 95% of its demand, so the queue was entirely on the ramp.
  - Ramp capacity against the mainline flow actually served (saturated ramp, mainline fully served, `results/S1_onramp_nocontrol_ramp_capacity_curve.csv`): c_r = 518 / 430 / 326 / 284 / 180 / 124 / 44 / 11 / 1 veh/h at q_m = 850 / 950 / 1050 / 1150 / 1250 / 1350 / 1450 / 1550 / 1650 veh/h. Total served while the ramp still gets real flow: 1350-1490 veh/h (0.78-0.86 q_c).
- **Interpretation:** The prediction failed on mechanism. The single-lane merge is not a bottleneck with one discharge rate. It behaves like a **give-way junction**: the mainline keeps priority, and the ramp is a minor stream that gets the gaps left over, so c_r falls to ~0 at q_m ~ 1600 (a gap-acceptance capacity curve, as for a minor road). Consequences:
  1. Option (c) "X against one measured q_dis" is not well-defined: the capacity depends on the split. If we keep this merge, X has to be stated per split, or as ramp demand / c_r(q_m).
  2. For the project question, the mechanism that metering exploits (ramp inflow breaks down the mainline, causing a capacity drop that a meter can prevent) is mostly absent. The unmetered merge already meters the ramp, with no rate control and no queue limit. Under this model a meter can only lower ramp flow further. So the likely result is "metering never helps mainline throughput; it only moves and caps the ramp queue". That would answer the question, but it would be an artifact of mainline drivers never yielding, not a traffic finding.
  3. Real merges (Hidas 2005: forced and cooperative merges) do not starve the ramp. So option (b), recalibrating the merge so ramp vehicles cannot be shut out (e.g. a forced or cooperative merge after waiting at the lane end, with tightly bounded yielding), is needed before the control comparison is credible. Team decision pending (design record section 2).
- Open side note: summary `min_mainline_speed_mps` is ~0 even in some low-demand runs, but a focused 900 + 100 veh/h run had no mainline vehicle below 5 m/s. Check which vehicles and where before using min speed as a metric.

## E5 | 2026-11-07 | Zipper courtesy merge (option (b): the ramp cannot be shut out)
- **Change planned:** `RampConfig.courtesy = "zipper"` (default "none", so S1/S2 stay unchanged). For each acceleration-lane vehicle in the merge zone, its would-be lane-0 follower treats it as a leader (cooperative merge, after Hidas 2005) if (1) the gap to it is > 0, (2) the IDM response needs no harder braking than b, and (3) that follower has not already let a ramp vehicle in (each mainline driver lets in at most one: a zipper). The merge itself still uses the unchanged MOBIL safety test (b_safe = 4 m/s^2). Unlike the E1 yielding switch, no driver yields twice, so courtesy braking is bounded.
- **Success criteria (fixed before running; all must hold for the fix to count):**
  1. No-queue threshold >= 0.85 q_c (~1480 veh/h) at every ramp share 0.10-0.40 (E4: 0.57-0.69 q_c).
  2. The ramp is not starved: in saturated runs the ramp gets at least half of min(its demand, half the served flow). E4: ~0 at high mainline flow.
  3. A merge capacity exists: the saturated served flow is roughly flat in demand (spread <= 10% within a share) and lies in 0.85-1.0 q_c.
  4. Above capacity, congestion reaches the mainline upstream of the merge (M_up speed falls well below v_c = 41.5 mph), instead of the queue landing only on the ramp. Metering needs this mechanism.
  5. Verification holds: gap > 0, conservation, all existing tests pass, S1/S2 unchanged with courtesy off.
- **Prediction (written before running):** Below the threshold the zipper rarely acts (ramp vehicles find gaps), so behavior is close to E4 but without lane-end deadlocks, and the threshold rises to ~1400-1500 veh/h (0.80-0.86 q_c). Criterion 1 is the most likely to fail: each courtesy brake is a small disturbance, and at T = 1.6 s on one lane it may seed stop-and-go below 0.85 q_c, as yielding did in E1. When saturated, the ramp share approaches min(r, 0.5), and the total discharge is ~1450-1600 veh/h (0.83-0.92 q_c), below q_c because merges at low speed cost headway (a capacity drop). Congestion forms on the mainline upstream of the merge, so criterion 4 holds.
- **Result** (same sweep as E4, config `configs/scenario_s1z_onramp_zipper.yaml`, raw runs in `results/S1z_onramp_zipper_nocontrol_merge_capacity_runs.csv`):
  - Criterion 1 FAILED: the no-queue threshold is 1200 / 1200 / 1100 / 1000 veh/h for r = 0.10 / 0.20 / 0.28 / 0.40 (0.57-0.69 q_c), no better than E4.
  - Criterion 2 passed: when saturated, the ramp gets about half (r = 0.28, q_tot >= 1500: ~400 ramp + ~400 mainline).
  - Criterion 3 FAILED: the saturated discharge collapses to ~800 veh/h (0.46 q_c) at r = 0.28 / 0.40 and ~850-1200 at r = 0.20. A capacity drop of > 50% is far outside what is observed on real freeways.
  - Criterion 4 passed, too strongly: the mainline queue spills back to the entrance (upstream lane-0 speeds 2-17 mph).
  - Criterion 5 passed: tests/test_zipper.py (6 tests) plus the full suite; S1/S2 unchanged with courtesy off.
- **Diagnosis** (one run, 1150 + 450 veh/h for 1500 s): after 600 s, all 103 merges happened at x = 1749 m (the lane end) at 0 m/s. The single-file ramp lane queues behind its stopped head vehicle. The yielding lane-0 driver stops beside it, and each pair restarts from rest at one point, ~4.5 s per vehicle with a = 0.73 m/s^2. Ramp vehicles further back never merge earlier: they approach the crawling mainline (~2 mph) at higher speed, so the new-leader safety test fails.
- **Interpretation:** The prediction failed on criteria 1 and 3, and the failure is mechanistic, not a tuning issue. Bounded courtesy fixes ramp starvation but turns the merge into a stop-start point bottleneck. Real zipper merges roll at low speed along the acceleration lane. The missing ingredient is that ramp vehicles match the speed of the lane-0 traffic beside them. The zipper switch stays (verified, off by default). Next: E5b.

## E5b | 2026-11-07 | Zipper + ramp speed matching (targeted scan)
- **Change:** No new code. Zipper courtesy combined with the existing speed-adaptation (E1) or gap-seeking (E3) switch, so ramp vehicles sync with the adjacent lane-0 traffic and can merge while rolling.
- **Scan:** steady demand, r = 0.28, q_tot = 1300 / 1500 / 1800 veh/h (0.75 / 0.86 / 1.03 q_c), seed 181, horizon 2400 s, window 900-2400 s. Variants: zipper alone (reference), zipper + adaptation, zipper + gap seeking. A variant that looks viable then gets the full E4-style sweep against the E5 criteria.
- **Prediction (written before running):** Speed matching moves merges upstream of the lane end and above walking speed, so saturated discharge rises from ~800 to ~1300-1500 veh/h with ramp share ~0.3-0.5. Adaptation helps more than gap seeking (gap seeking aims at gaps that, in dense traffic, are too small). At 1300 veh/h both stay queue-free. At 1500 (0.86 q_c) it is uncertain: breakdown is likely, because courtesy braking still disturbs a single lane at T = 1.6 s.
- **Result:** served flow over 900-2400 s (veh/h), with the ramp part in brackets:

| Variant | q_tot 1300 (0.75 q_c) | 1500 (0.86 q_c) | 1800 (1.03 q_c) |
|---|---|---|---|
| zipper alone | 1010 (338) | 804 (398) | 811 (403) |
| zipper + speed adaptation | 1183 (341) | 1102 (386) | 811 (406) |
| zipper + gap seeking | 1255 (341) | 1169 (410) | 1157 (490) |

- **Interpretation:** Speed matching helps as predicted, and gap seeking helps more than adaptation, which contradicts the prediction. Gap seeking holds ~1160 veh/h when saturated, while adaptation falls back to the 800 veh/h stop-start state at 1800. But the best variant reaches only 0.66 q_c, so criterion 3 (>= 0.85 q_c) fails for every variant, and even 1300 veh/h is not fully served. No full sweep was run. Conclusion: no single-lane merge variant tried so far (E1, E4, E5, E5b) gets anywhere near q_c. The binding limit looks structural: one lane with T = 1.6 s and a = 0.73 m/s^2 cannot absorb merges without large headway losses. Team decision needed (design record section 2).
- **Method flaw found:** `merge_capacity.steady_run` counts as "queued" only the point queues and the ramp lane. Vehicles stored in a mainline jam are not counted (zipper at 1300: served 1010 with "queued" 6). This can overstate the no-queue threshold whenever the mainline congests. E4 is mostly unaffected (the mainline was fully served in 70 of 87 saturated runs). E5's failing verdict stands, since its thresholds can only be lower. Fix before the next sweep: also require served >= 0.97 x realized arrivals in the window.

## E6 | 2026-11-07 | Measured capacity of the zipper + gap-seeking merge (option 1, team decision)
- **Decision (team, 2026-11-07):** Option 1. Accept the zipper + gap-seeking merge as the model's bottleneck and state the demand ratio against its MEASURED capacity: X_m = (q_in + lambda_r) / C_m, instead of X = .../q_c. Config `configs/scenario_s3_onramp_zipgap.yaml` (S3).
- **Method change first:** a run counts as saturated if the end queue is >= 10 veh OR served < 0.97 x the realized arrival flow in the window (fixes the E5b flaw, where vehicles stored in a mainline jam were invisible). Unit test: test_merge_capacity::test_mainline_jam_counts_as_saturated.
- **Acceptance criteria for S3 as the no-control baseline (fixed before running).** These test bottleneck behavior, not closeness to q_c:
  1. A capacity exists: within each ramp share the saturated served flow varies little (CV <= 5%), so C_m(r) is well-defined. If C_m differs by more than 15% across shares, X_m uses C_m(r) for the scenario's share.
  2. The ramp is not starved: saturated ramp share >= 0.5 x min(r, 0.5).
  3. When saturated, the mainline is congested too (served mainline < its demand), so a meter has something to protect.
  4. Capacity drop (1 - discharge / highest queue-free flow) between 0 and 20%.
  5. Verification: zipper + gap safety and conservation tests pass, and time-step sensitivity on one saturated case (dt = 0.05 / 0.1 / 0.2 s) keeps served flow within 3%.
- **Prediction (written before running):** C_m ~ 1150-1250 veh/h for r >= 0.20, saturated CV ~3-5%, capacity drop ~3-10%, saturated ramp share 0.35-0.45 (zipper alternation, reduced by mainline vehicles that pass without yielding). At r = 0.10 there are fewer merges, so C_m is higher (~1350-1450) and the cross-share spread exceeds 15%, so X_m will need C_m(r). Breakdown threshold ~1100-1200 veh/h. Criterion 5 is the main risk: gap seeking is a feedback law, so its result may depend on dt.
- **Result** (132 runs, `results/S3_onramp_zipgap_nocontrol_merge_capacity_runs.csv`; estimates in `..._merge_capacity.csv`):

| r | queue-free threshold | highest queue-free flow | saturated discharge C_m (CV) | capacity drop | saturated ramp share |
|---|---|---|---|---|---|
| 0.10 | 1100 | 1538 | 1503 (9.8%) | 2.2% | 0.06 |
| 0.20 | 1200 | 1337 | 1257 (6.6%) | 6.0% | 0.27 |
| 0.28 | 1100 | 1230 | **1157 (4.4%)** | 5.9% | 0.41 |
| 0.40 | n/a (one seed at 1000 fails the served test) | 1209 | 1104 (3.4%) | 8.7% | 0.49 |

  - Criterion 1: passed for r = 0.28 and 0.40, FAILED for r = 0.20 and 0.10. At low shares the ramp's slice swings between starved and served, so the discharge is not stable. The spread across shares is 36%, so capacity is share-dependent: C_m(r).
  - Criterion 2 passed (ramp share 0.06 / 0.27 / 0.41 / 0.49 >= 0.05 / 0.10 / 0.14 / 0.20). Criterion 3 passed: in 81% of saturated runs the mainline got < 95% of its demand. Criterion 4 passed: drop 2-9%. Criterion 5 passed: tests/test_zipper.py (7 tests), and the dt sensitivity at q_tot 1500, r 0.28, seeds 181-182 gives mean served 1188 / 1159 / 1186 veh/h for dt = 0.05 / 0.1 / 0.2 s (max 2.5% from dt = 0.1, no trend).
- **Interpretation:** The prediction held on C_m (1157, predicted 1150-1250), on the drop, the ramp share and the share dependence. It missed on CV at r = 0.20 (6.6% vs <= 5%). **Verdict: S3 is accepted as the no-control baseline, restricted to a fixed ramp share r = 0.28**, where every criterion holds. Control experiments hold r = 0.28 and vary X_m = q_tot / 1157. Low-share capacity is not claimed. The merge discharges at 0.66 q_c, which is a property of this single-lane IDM merge and must be stated as a limitation.
- **S3 demand rescaled** to S1's intended X profile with r = 0.28: X_m = 0.92 / 1.03 / 0.63 gives q_tot = 1064 / 1192 / 729 veh/h, split 766+298 / 858+334 / 525+204 (rounded). Seed 181: no breakdown, queues <= 3, TTT 18.6 veh*h, cleared, min gap 0.81 m. This is consistent with E6: 1192 lies between C_m (1157, the discharge after breakdown) and the highest queue-free flow (~1230). The band 1.0 < X_m < ~1.06 is near capacity, where breakdown may or may not happen. That is where a meter is expected to matter, so the control sweep must resolve it with several seeds.

## E7 | 2026-11-08 | Control-mode comparison across demand levels (the project question)
- **Change:** Ramp meter at the S3 stop line (one vehicle per green, standing start), `src/tmc/estimation.py` (queue by vehicle counting), `src/tmc/forecast.py` (exponential smoothing, queue mass balance), `src/tmc/control.py` (none / pre-timed / ALINEA, a queue override at n_max = 35 using the 60 s queue forecast, a decision trace), `src/tmc/evaluate.py`. ALINEA measures occupancy at M_mrg, 50 m upstream of the merge, because congestion forms upstream of this merge (see tmc.control). Setpoint range 0.08-0.15 from no-control occupancy at M_mrg: 0.05-0.06 in free flow, 0.16-0.50 congested (S3, r = 0.28).
- **Design:** S3, r = 0.28, X_m profile 0.85 / X_peak / 0.60 (600 / 900 / 900 s), X_peak = 0.85 / 0.95 / 1.00 / 1.05 / 1.10 / 1.20 / 1.35, seeds 181-184, run until clear. Pre-timed rates 240-600 veh/h (best per level). ALINEA setpoints 0.08 / 0.10 / 0.12 / 0.15 (one global setpoint, plus the best per level for reference). 308 runs.
- **Disclosure:** before writing this prediction I ran the `--quick` example (X_peak 0.95 and 1.10, seed 181, pre-timed 300/420, ALINEA 0.10). There, pre-timed was 14% / 21% worse than no control, and ALINEA stayed dark and matched no control.
- **Prediction (Intermediate 1, restated in X_m and updated by the quick run):**
  1. X_peak <= 0.95: no control is best. ALINEA stays dark and ties it. Pre-timed is worse (every ramp vehicle stops, and standing-start merges at ~19 m/s disturb the mainline).
  2. Near capacity (1.0 <= X_peak <= 1.1): Intermediate 1 predicted that actuated wins here. Expected: ALINEA beats no control by a small margin (0-5% TTT) only in seeds where no control breaks down. Pre-timed stays worse than both. The standing-start merge cost may cancel ALINEA's benefit; if so, the "actuated wins in the middle band" hypothesis is not supported in this model.
  3. X_peak >= 1.2: the ramp queue reaches n_max and the override flushes it, so ALINEA converges to no control. Pre-timed is still worst.
  4. Forecast: the smoothed ramp-demand forecast beats persistence (lower MAE). The queue forecast is about as good as persistence, because queues at a 60 s horizon change little.
- **Sweep-range correction (before interpreting):** in the first pass, pre-timed's best rate (600) and ALINEA's best global setpoint (0.15) both sat at the top of their swept ranges. Extended with `python -m tmc.evaluate --extend` (pre-timed 700/800/900 = up to r_max, ALINEA 0.20/0.25). After that, ALINEA's global best is 0.20 (interior). Pre-timed's best is 600-900, where 900 is the meter's physical limit. All runs: `results/E7_runs.csv` (420 runs), summary `results/E7_summary.csv`, plot `results/E7_ttt_vs_X.png`.
- **Result** (mean TTT over seeds 181-184, veh*h; change vs no control):

| X_peak | No signal | Pre-timed, best rate per level | ALINEA o_set 0.20 (global) | ALINEA meter on | No-control min 30-s speed |
|---|---|---|---|---|---|
| 0.85 | 13.94 | 15.75 (+13%, 600) | 13.94 (0%) | 0% of time | 62 mph |
| 0.95 | 15.36 | 17.36 (+13%, 600) | 15.36 (0%) | 0% | 62 mph |
| 1.00 | 15.78 | 18.33 (+16%, 800) | 15.78 (0%) | 0% | 63 mph |
| 1.05 | 17.55 | 19.52 (+11%, 900) | 17.55 (0%) | 0% | 58 mph |
| 1.10 | 17.71 | 20.42 (+15%, 800) | 17.71 (0%) | 0% | 52 mph |
| 1.20 | 21.41 | 24.88 (+16%, 600) | 21.29 (-0.6%) | 6% | 5 mph |
| 1.35 | 31.37 | 35.15 (+12%, 900) | 28.82 (-8.1%) | 24% | 2 mph |

  - Paired by seed, ALINEA 0.20 vs no control: X = 1.35: -1.3 / -2.4 / -2.3 / -4.2 veh*h (4 of 4 better). X = 1.20: -0.3 / -0.1 / -0.1 / 0.0.
  - **Held-out check** (seeds 185-188, not used to choose the setpoint; `results/E7_holdout_runs.csv`): X = 1.35: ALINEA -2.7 / -3.5 / -2.7 / -5.4 veh*h (4 of 4 better, mean -8.3%). X = 1.20: +0.2 / +1.3 / -1.4 / -2.0 (mixed, mean -1.7%). Pre-timed 900 was worse than no control in 8 of 8.
  - No ramp queue exceeded n_max (peak 23, no overrides). Unmet demand 0 and all runs cleared. Throughput 600-2100 s is equal across modes within ~1% (demand-limited window).
  - Forecast skill (metered runs): ramp-demand smoothing MAE 163 veh/h vs persistence 200. Queue forecast MAE 0.72 veh vs persistence 0.81.
- **Interpretation (against the prediction):**
  1. Held: below X ~ 1.1, no control is best, ALINEA stays dark and ties it, and pre-timed costs 11-16%.
  2. FAILED: in 1.0-1.1 the no-control merge did not break down within the 900 s peak (min speed >= 52 mph), so there was nothing for a meter to prevent, and ALINEA correctly stayed off. The near-capacity range where breakdown happens only sometimes sits higher than X_m ~ 1.1. X_m is stated against the after-breakdown discharge C_m, and the highest queue-free flow is ~6% above it (E6).
  3. FAILED: at X = 1.35, ALINEA did not converge to no control. It cut TTT by ~8% in 8 of 8 seeds (in-sample and held-out), keeping the ramp queue <= 23 < n_max. At X = 1.2 the effect is small and inconsistent.
  4. Held: the demand forecast beats persistence by ~19%. The queue forecast is only slightly better (60 s is short relative to queue dynamics).
  - **Answer to the project question (this model, r = 0.28):** actuated metering is not better at all demand levels. It ties no control up to X_m ~ 1.1 (because it switches itself off) and is better only in heavy overload (X_m ~ 1.35), by ~8% TTT. The Intermediate 1 "middle band" prediction is not supported: the band where actuated wins is shifted to higher demand. Pre-timed metering is worse than no control at every level tested. The likely mechanism is the standing start: every metered vehicle stops at the stop line and reaches the merge at ~19 m/s, below mainline speed, which disturbs the mainline (no-control min speed 52 mph vs 29 mph pre-timed at X = 1.10).
  - **Limits and what would change the conclusion:** 4 + 4 seeds. The ALINEA setpoint was tuned on seeds 181-184 (the held-out check addresses this). Merge capacity is 0.66 q_c (single-lane IDM merge). The standing-start release and the 250 m approach are modeling choices: a longer approach or a rolling release would likely help pre-timed (a candidate follow-up, E8). The peak lasts 900 s; longer peaks would move breakdown to lower X_m.
