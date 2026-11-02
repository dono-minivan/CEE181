# Design Record (one evolving document)

## 1. Question and prediction
Is actuated freeway on-ramp metering better than pre-timed or no signal at all volumes? Prediction: actuated wins only in a middle band near capacity (X about 0.9 to 1.05). See Intermediate 1 for the full reasoning and "what would change our conclusion".

## 2. Model and assumptions
IDM, ballistic update, parameters in `configs/idm_baseline.yaml`. Limitations: no reaction time (ACC-like), deterministic.

## 3. Interfaces
See `src/tmc/interfaces.py`.

## 4. Verification log
(conservation, free motion, equilibrium, gaps, boundaries, time step, detector audit: record date, result)

## 5. Results

## 6. Design changes made in response to feedback

## 7. Final pathway decision (CAV or signal)
Open.

## 8. AI-use and verification disclosure
AI used to gather papers, format deliverables, summarize topics, and draft verification/fallback plans (per Intermediate 1). Record any further use here and how output was checked.
