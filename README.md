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

This writes `results/<scenario_id>_viewer.html`, a self-contained file. Add `#t=1200&zoom=300&x=1600` to the URL to open at a given time, width, and position.

Ramp control (S3 baseline: zipper + gap-seeking merge, detector-only controllers):

    python -m tmc.control configs/scenario_s3_onramp_zipgap.yaml alinea 0.10     # prints the decision trace
    python -m tmc.control configs/scenario_s3_onramp_zipgap.yaml pretimed 400
    python -m tmc.evaluate --quick       # small control comparison (~1 min)
    python -m tmc.evaluate               # full E7 sweep (writes results/E7_*.csv and E7_ttt_vs_X.png)
    python -m tmc.merge_capacity configs/scenario_s3_onramp_zipgap.yaml --from-csv   # E6 capacity estimate

## Tests
    pytest

## Status
IDM mainline (1 or 2+ lanes, symmetric MOBIL lane changing) with entry queue, free exit, and an on-ramp merge (provisional merge model, see experiment log E1 and E2). Verified by 50 tests (see `docs/design_record.md` section 4). Animated viewer available. Virtual detectors (loops, stations, ramp demand counter) with a count audit. Merge-capacity measurement (E4, E6); no-control baseline S3 (zipper + gap-seeking merge, X_m stated against its measured capacity C_m). Ramp meter with no-signal / pre-timed / ALINEA control, a forecast-based queue override and decision traces; evaluation sweep (E7). Not yet built: the final pathway (CAV or signal, undecided).

## Known limitations
- The S3 merge discharges C_m ~ 1157 veh/h at ramp share 0.28 (0.66 of the computed IDM capacity q_c). Control results are stated in X_m = demand / C_m and hold only for r = 0.28 (E6).
- The on-ramp merge model is provisional. Single lane: the merge discharges ~1430-1490 veh/h, below q_c (E1). Two lanes: the merge deadlocks once a ramp vehicle stops at the lane end (E2); the optional gap-seeking and anticipation fixes reach at most 88% of 2 q_c (E3), so the two-lane option is parked. Not a valid no-control baseline yet.
- Lane changes are instantaneous (no lateral motion). MOBIL parameters are still to be verified against the primary source.
- Deterministic IDM drivers (no reaction time or noise). Randomness comes only from arrivals.
- The entrance discharges at most about q_c. Mainline demand above that queues at the entrance.

## AI-use disclosure
Claude Code drafted the initial simulator and tests (2026-11-05) and the multi-lane / MOBIL extension, its tests, and the E2 diagnostics (2026-11-07). Verification and review are recorded in `docs/design_record.md` section 8.
