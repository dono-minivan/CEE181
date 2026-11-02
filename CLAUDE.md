# CEE 181 Design Project (Fall 2026): Freeway Simulation + Traffic Management

Team: Donovan Lee, Anthony Lee (2-person group, regular CEE 181 track; no CEE 281 extras required).
Authoritative requirements: `CEE181281_Fall2026_Design_Project_Guide_Revised.pdf` (in the Claude project knowledge). Our Intermediate 1 report is the flexible plan, not a contract. Update it when evidence says so.
Lecture notes arrive gradually; when new ones are added, fold relevant decisions into `docs/design_record.md`.

## The project in one paragraph
One continuing project: a bounded freeway segment (arrivals, exits, a bottleneck / controlled ramp-entry queue) with virtual detectors -> observation -> estimation + short-horizon forecast -> basic closed-loop TMC (ramp release rule). Then ONE final pathway: CAV control OR traffic-signal control. Doing both earns nothing extra.

Our Intermediate 1 question: is actuated ramp metering better than pre-timed or no signal at all demand levels? Prediction: actuated wins only in a middle band near capacity (demand ratio X = (q_in + lambda_r)/q_c about 0.9 to 1.05). Better = lower total travel time (veh*h), including ramp waiting, run until residual queues clear, ramp queue within n_max. Also report mainline throughput, peak ramp queue, unmet demand, min mainline speed.

## Open decisions (ask the user, don't guess)
- Final pathway: CAV or signal (to be chosen). Ramp metering is only the shared TMC foundation and does not by itself satisfy the final pathway.
- Role split. Guide examples: A = vehicle dynamics + verification, B = arrivals/bottleneck/detectors. Fill in `docs/contributions.md`.
- Lane changing (mentioned as next step in the report) is optional. Start single-lane, verify, then extend.

## Model baseline (from Intermediate 1)
IDM, ballistic update, SI units internally; convert to mph / veh/h / veh/mi only at outputs.
Baseline params (Treiber-Hennecke-Helbing 2000, Table I): v0=33.3 m/s, T=1.6 s, s0=2 m, a=0.73 m/s^2, b=1.67 m/s^2, delta=4, l=5 m.
Sensitivity set (Treiber-Kesting Table 11.2): T=1.0, a=1.0, b=1.5.
Values in `configs/idm_baseline.yaml`. Any derived number (capacity, wave speed) must be recomputed in code, not copied from the report.

## Repo layout
- `src/tmc/`: the package. Planned modules: `interfaces.py` (exchange formats), `idm.py` (dynamics), `network.py` (segment, arrivals, merge/bottleneck), `detectors.py`, `estimation.py`, `forecast.py`, `control.py` (no-signal / pre-timed / actuated), `evaluate.py`, plus the final pathway.
- `tests/`: pytest checks (conservation, free motion, equilibrium, gaps, time-step sensitivity, detector counts).
- `configs/`: scenario and parameter YAMLs (one file per scenario; seeds and demand IDs live here).
- `docs/`: single evolving design record + experiment log + source-to-design table + contribution record.
- `results/`: generated plots/tables (each labeled with scenario_id, axes, units).

## Rules for working in this repo
1. Interfaces (guide's "minimum exchange format"): seconds and meters internally; explicit sampling intervals; consistent vehicle IDs. Unavailable quantities stay explicitly missing (NaN/None), never silently zero.
2. Controllers see ONLY permitted detector measurements. Simulator truth and future trajectories are for evaluation only.
3. Verify before interpreting: vehicle conservation, free-road and equilibrium motion, gap > 0, boundary behavior, time-step sensitivity, detector count audit. A "traffic effect" is not accepted until these pass.
4. Fair comparisons: same demand, initial conditions, horizon, and measurement definitions across modes. Pre-timed baseline = best rate from a sweep. Run until residual queues clear; report unmet demand, never silently drop vehicles at a full boundary.
5. Every experiment gets a record in `docs/experiment_log.md`: prediction, change, result, interpretation. Predict before running. Negative results are fine if supported.
6. Every equation or parameter traces to a source in `docs/source_to_design.md`.
7. Each module runs with a small example; one reproducible run command lives in the README.
8. Keep AI-use disclosure current (README + design record). Claims about real-world crash reduction are out of scope.
9. Commit small and often; message format `component: what changed`. Don't commit `.venv`.

## Commands
- Windows setup: `python -m venv .venv`, then `.venv\Scripts\activate`, then `pip install -r requirements.txt`
- Tests: `pytest`
- Run example: `python -m tmc.example` (to be added with the first simulator)

## Grading reminders
Technical score = 60% shared evidence + 40% individual work. Each partner must be able to trace a vehicle through the code, explain a decision trace, and predict a changed-case response. Keep `docs/contributions.md` honest and specific (component, code location, tests, conclusions).
Design record length: about 3-4 pages for intermediate reviews, 6 for the final, plus references/plots/code.
