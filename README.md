# CEE 181 Design Project: Freeway Simulation and Ramp Control

Donovan Lee and Anthony Lee, UCLA CEE 181, Fall 2026.

Freeway segment simulated with the Intelligent Driver Model (IDM), virtual detectors, and three ramp-control modes (no signal, pre-timed, actuated). See `CLAUDE.md` for project rules and `docs/` for the design record.

## Setup (Windows)
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

On Python 3.15 PyYAML has no prebuilt wheel. If its build fails asking for a C++ compiler, install the pure-Python version first:
`$env:PYYAML_FORCE_LIBYAML="0"; pip install pyyaml`, then rerun the line above.

## Run
Reproducible run command (single-lane IDM baseline, scenario `configs/scenario_s0_single_lane.yaml`):

    python -m tmc.example

It prints a summary and writes `results/S0_single_lane_time_space.png`, `..._counts.png`, and `..._vehicles.csv`. Pass another scenario YAML path as an argument to run it. Module examples: `python -m tmc.idm` prints computed capacity; `python -m tmc.detectors` runs S1 and prints the virtual detector flows.

Animated viewer (mainline, on-ramp, queues; vehicles colored by speed; click a car to trace it):

    python -m tmc.viz --open                                  # on-ramp scenario S1
    python -m tmc.viz configs/scenario_s0_single_lane.yaml --open
    python -m tmc.viz configs/scenario_s2_twolane_onramp.yaml --open   # two lanes + lane changing

This writes `results/<scenario_id>_viewer.html`, a self-contained file (single-run viewers are not committed; `.gitignore` skips them).

Ramp control in the viewer (meter signal, detectors, the controller's decision at each moment, and a timeline of ramp queue and meter rate; E7 demand at a chosen peak X_m):

    python -m tmc.viz --e7 1.35 --control alinea:0.20 --open
    python -m tmc.viz --e7 1.35 --control none --open        # same demand and seed, no signal
    python -m tmc.viz --e7 1.10 --control pretimed:800 --open
    python -m tmc.viz configs/scenario_s3_onramp_zipgap.yaml --control alinea:0.20 --seed 182 --open

E9 viewers (equipped vehicles carry a roof marker, orange while a CACC vehicle is in fallback; click a car to see its driver type and where its leader-acceleration input comes from):

    python -m tmc.viz --e9b 1.35 --mode cacc_eff --p 0.5 --open   # setting B: S3r2 merge, modes human / acc / cacc / cacc_eff
    python -m tmc.viz --e9a nominal --mode cacc --p 0.5 --open     # setting A: platoon braking wave, follows the lead
    python -m tmc.viz --examples                                   # regenerate results/viewer.html (all example runs)

All example runs in one page: open `results/viewer.html` and pick a run from the dropdown (grouped: basic scenarios, E7 ramp metering, E9 setting B, E9 setting A). Switching keeps the time, close-up position and width, and selected car, so you see the same moment in the other run; `[` / `]` step through the runs in a group. Add `#run=<key>` to the URL to open a given run, e.g. `viewer.html#run=E9B_Xp1.35_acc_p1.00_s181&t=1500`.
 Add `#t=1200&zoom=300&x=1600` to the URL to open at a given time, width, and position.

Ramp control (S3 baseline: zipper + gap-seeking merge, detector-only controllers):

    python -m tmc.control configs/scenario_s3_onramp_zipgap.yaml alinea 0.10     # prints the decision trace
    python -m tmc.control configs/scenario_s3_onramp_zipgap.yaml pretimed 400
    python -m tmc.evaluate --quick       # small control comparison (~1 min)
    python -m tmc.evaluate               # full E7 sweep (writes results/E7_*.csv and E7_ttt_vs_X.png)
    python -m tmc.merge_capacity configs/scenario_s3_onramp_zipgap.yaml --from-csv   # E6 capacity estimate

CAV pathway, E9 setting A (braking wave in a 30-vehicle platoon, `configs/scenario_e9_setting_a.yaml`):

    python -m tmc.disturbance --quick    # one run, prints the metrics
    python -m tmc.cav                    # CAV controller core: one ACC and one CACC follower behind a braking leader
    python -m tmc.disturbance            # human baseline (T_r = 0 and 0.6 s), writes results/E9A_human_baseline*
    python -m tmc.disturbance --matrix   # E9 CAV matrix, seeds 181-184 (~3 min on 14 cores), writes results/E9A_matrix_*
    python -m tmc.disturbance --matrix --holdout   # nominal group on held-out seeds 185-188
    python -m tmc.ploeg                  # Ploeg et al. 2011 CACC: string-stability theory (Eq. 17) + a one-follower example
    python -m tmc.disturbance --matrix --set e9d   # E9d extension (Ploeg controller), writes results/E9D_*

CAV pathway, E9 setting B (S3r2 merge, humans with T_r = 0.6 s restarted at leader changes, dt 0.05 s; experiment log E9e):

    python -m tmc.setting_b --quick              # human vs ACC p = 1, X 1.35, seed 181 (~1 min)
    python -m tmc.setting_b --run 1.35 --with-eff   # every case, seeds 181-184, writes results/E9B_runs.csv, E9B_judgement.csv, figures
    python -m tmc.setting_b --holdout 1.35 all   # every case on held-out seeds 185-188
    python -m tmc.merge_capacity configs/scenario_s3r2_onramp_zipgap_tr06_restart.yaml --share=0.28 --demands=800:2000:100   # E9b2 C_m

## Tests
    pytest

## Status
Complete for the shared TMC foundation and the CAV final pathway. Verified by 199 tests (`pytest`; see `docs/design_record.md` section 4).
- Simulator: IDM mainline (1 or 2+ lanes, symmetric MOBIL lane changing) with entry queue, free exit and an on-ramp merge; zipper courtesy + gap-seeking merge (S3, the no-control baseline). Animated viewer.
- Observation and control: virtual detectors (loops, stations, ramp demand counter) with a count audit; queue estimation and short-horizon forecasts; ramp meter with no-signal / pre-timed / ALINEA control, a forecast-based queue override and decision traces; evaluation sweep (E7).
- Human drivers (E9): optional reaction delay T_r = 0.6 s on car-following (`drivers: reaction_time_s`). With `leader_change_delay: restart` (S3r2, used in setting B) the delay restarts after every leader change: for T_r after a merge, cut-in or leader exit the driver uses current inputs. The earlier rule (`delayed_gap`, S3r, E9b) let just-merged vehicles react to a stale, near-zero gap and brake unrealistically hard (experiment log E9e).
- CAV pathway: textbook ACC (Treiber & Kesting Eq. 11.26 on an IDM base) and CACC (leader's commanded acceleration by V2V, Ploeg et al. 2011) with actuator lag, tick-based V2V channel and logged fallbacks (`cav.py`); setting A platoon experiments (E9c, plus the Ploeg CACC extension E9d) and setting B merge experiments (E9e, `setting_b.py`). Merge capacities at r = 0.28: S3 (no delay) C_m = 1157 veh/h (E6); S3r2 (T_r = 0.6 s, restart rule, dt 0.05 s) C_m = 1069 veh/h (E9b2). Setting B states X_m against 1069.

## Known limitations
- Merge capacity, and therefore every X_m, holds only for ramp share r = 0.28: C_m = 1157 veh/h for S3 (0.66 of the computed IDM capacity q_c; E1-E7) and 1069 veh/h for S3r2 (E9 setting B). S3r's C_m = 978 (E9b) was depressed by the stale-gap artifact and is superseded.
- The two-lane merge option is parked: it deadlocks once a ramp vehicle stops at the lane end (E2), and the gap-seeking and anticipation fixes reach at most 88% of 2 q_c (E3). All control and CAV results use the single-lane S3 merge.
- Lane changes are instantaneous (no lateral motion). MOBIL parameters (p = 0.2, threshold 0.1 m/s^2, b_safe = 4 m/s^2) were checked against Kesting, Treiber & Helbing 2007 and Treiber & Kesting 2013 (`docs/source_to_design.md`); the two sources differ on b_safe (4 vs 2 m/s^2), and we use the original MOBIL value. The merge safety check limits braking, not the gap, so merging vehicles can land ~0.8 m behind a faster leader (E9e).
- Human drivers have a reaction delay on car-following only; merge, courtesy and lane-change decisions are undelayed, and with the restart rule humans react to a cut-in without delay (optimistic for humans). No estimation errors or noise; randomness comes only from arrivals and CAV placement.
- CAV merge decisions are identical to human ones (no cooperative merging), so the CAV experiments cannot show a merge-decision benefit.
- With T_r on, results depend on the time step: setting B runs at dt 0.05 s (dt 0.1 s biased human TTT by ~20% on S3r, E9e).
- The entrance discharges at most about q_c. Mainline demand above that queues at the entrance.

## AI-use disclosure
Claude Code (AI assistant) drafted code, tests, and documentation drafts in this repo. The team reviews each piece and must be able to explain it. What was drafted, and how it was checked, is logged in `docs/design_record.md` section 8:

- 2026-11-05: IDM simulator, entry/exit boundaries, on-ramp merge, example run, verification tests, animated viewer.
- 2026-11-07: multi-lane mainline with MOBIL lane changing, gap-seeking/anticipation switches, E2/E3 diagnostics; virtual detectors and count-audit tests; merge-capacity tool (E4); zipper courtesy merge (E5).
- 2026-11-08: merge-capacity revision and S3 baseline (E6); ramp meter, `estimation.py`, `forecast.py`, `control.py`, `evaluate.py` and tests (E7); viewer support for the meter signal, detectors and decision traces.
- 2026-11-15: `report_figures.py` (progress-report figures, `results/PR_*`, congestion-minutes definition).
- 2026-11-23/24: reaction-time switch, CAV controller core (`cav.py`), CAV integration in `network.py`, E9 setting A harness and matrix runner (`disturbance.py`), E9b capacity re-measurement.
- 2026-11-24: Ploeg et al. 2011 CACC controller and string-stability theory (`ploeg.py`), E9d extension runner and tests.
- 2026-11-25: viewer update for E9 (`viz.py`, `viewer_template.html`: equipped markers, fallback state, driver panel, `--e9b` / `--e9a` / `--examples`, `tests/test_viz.py`).
- 2026-11-26: combined viewer `results/viewer.html` (all example runs, dropdown run switcher keeping time and view; `viz.write_combined`, written by `--examples`; replaces the 11 separate committed viewers).
- 2026-11-25: E9 setting B runner (`setting_b.py`), every-step gap tracker, reaction-delay restart at leader changes (S3r2), effective-command V2V broadcast option, E9b2 capacity re-measurement.

Earlier (Intermediate 1), AI was used to gather papers, format deliverables, summarize topics, and draft verification/fallback plans. No AI output is accepted as a result until it passes the verification tests and the checks recorded in the design record.
