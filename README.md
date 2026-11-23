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

This writes `results/<scenario_id>_viewer.html`, a self-contained file.

Ramp control in the viewer (meter signal, detectors, the controller's decision at each moment, and a timeline of ramp queue and meter rate; E7 demand at a chosen peak X_m):

    python -m tmc.viz --e7 1.35 --control alinea:0.20 --open
    python -m tmc.viz --e7 1.35 --control none --open        # same demand and seed, no signal
    python -m tmc.viz --e7 1.10 --control pretimed:800 --open
    python -m tmc.viz configs/scenario_s3_onramp_zipgap.yaml --control alinea:0.20 --seed 182 --open
 Add `#t=1200&zoom=300&x=1600` to the URL to open at a given time, width, and position.

Ramp control (S3 baseline: zipper + gap-seeking merge, detector-only controllers):

    python -m tmc.control configs/scenario_s3_onramp_zipgap.yaml alinea 0.10     # prints the decision trace
    python -m tmc.control configs/scenario_s3_onramp_zipgap.yaml pretimed 400
    python -m tmc.evaluate --quick       # small control comparison (~1 min)
    python -m tmc.evaluate               # full E7 sweep (writes results/E7_*.csv and E7_ttt_vs_X.png)
    python -m tmc.merge_capacity configs/scenario_s3_onramp_zipgap.yaml --from-csv   # E6 capacity estimate

CAV pathway, E9 setting A (braking wave in a 30-vehicle platoon, `configs/scenario_e9_setting_a.yaml`):

    python -m tmc.disturbance --quick    # one run, prints the metrics
    python -m tmc.disturbance            # human baseline (T_r = 0 and 0.6 s), writes results/E9A_human_baseline*

## Tests
    pytest

## Status
IDM mainline (1 or 2+ lanes, symmetric MOBIL lane changing) with entry queue, free exit, and an on-ramp merge (provisional merge model, see experiment log E1 and E2). Verified by 50 tests (see `docs/design_record.md` section 4). Animated viewer available. Virtual detectors (loops, stations, ramp demand counter) with a count audit. Merge-capacity measurement (E4, E6); no-control baseline S3 (zipper + gap-seeking merge, X_m stated against its measured capacity C_m). Ramp meter with no-signal / pre-timed / ALINEA control, a forecast-based queue override and decision traces; evaluation sweep (E7). Not yet built: the final pathway (tentatively CAV, subject to change).

## Known limitations
- The S3 merge discharges C_m ~ 1157 veh/h at ramp share 0.28 (0.66 of the computed IDM capacity q_c). Control results are stated in X_m = demand / C_m and hold only for r = 0.28 (E6).
- The on-ramp merge model is provisional. Single lane: the merge discharges ~1430-1490 veh/h, below q_c (E1). Two lanes: the merge deadlocks once a ramp vehicle stops at the lane end (E2); the optional gap-seeking and anticipation fixes reach at most 88% of 2 q_c (E3), so the two-lane option is parked. Not a valid no-control baseline yet.
- Lane changes are instantaneous (no lateral motion). MOBIL parameters are still to be verified against the primary source.
- Deterministic IDM drivers (no reaction time or noise). Randomness comes only from arrivals.
- The entrance discharges at most about q_c. Mainline demand above that queues at the entrance.

## AI-use disclosure
Claude Code (AI assistant) drafted code, tests, and documentation drafts in this repo. The team reviews each piece and must be able to explain it. What was drafted, and how it was checked, is logged in `docs/design_record.md` section 8:

- 2026-11-05: IDM simulator, entry/exit boundaries, on-ramp merge, example run, verification tests, animated viewer.
- 2026-11-07: multi-lane mainline with MOBIL lane changing, gap-seeking/anticipation switches, E2/E3 diagnostics; virtual detectors and count-audit tests; merge-capacity tool (E4); zipper courtesy merge (E5).
- 2026-11-08: merge-capacity revision and S3 baseline (E6); ramp meter, `estimation.py`, `forecast.py`, `control.py`, `evaluate.py` and tests (E7); viewer support for the meter signal, detectors and decision traces.
- 2026-11-15: `report_figures.py` (progress-report figures, `results/PR_*`, congestion-minutes definition).

Earlier (Intermediate 1), AI was used to gather papers, format deliverables, summarize topics, and draft verification/fallback plans. No AI output is accepted as a result until it passes the verification tests and the checks recorded in the design record.
