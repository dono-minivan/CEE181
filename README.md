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

It prints a summary and writes `results/S0_single_lane_time_space.png`, `..._counts.png`, and `..._vehicles.csv`. Pass another scenario YAML path as an argument to run it. Module examples: `python -m tmc.idm` prints computed capacity.

## Tests
    pytest

## Status
Single-lane IDM simulator with entry queue and free exit, verified by 17 tests (see `docs/design_record.md` section 4). Not yet built: on-ramp merge/bottleneck, detectors, estimation/forecast, controllers.

## Known limitations
- Single lane, no lane changing, no on-ramp yet.
- Deterministic IDM drivers (no reaction time or noise). Randomness comes only from arrivals.
- The entrance discharges at most about q_c. Mainline demand above that queues at the entrance.

## AI-use disclosure
Claude Code drafted the initial simulator and tests (2026-11-05). Verification and review are recorded in `docs/design_record.md` section 8.
