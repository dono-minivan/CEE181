# Individual Contribution Record

> **Tentative split (team, 2026-11-19). Subject to change.** Roles follow the CAV-pathway split suggested in the project guide: one partner owns control and communication logic, the other owns integration and comparative experiments. Code locations and tests marked *planned* do not exist yet. Update each row as work is finished, so it states what each person actually built, tested and concluded.

| Student | Role | Component owned | Code location | Tests | Conclusions owned |
|---|---|---|---|---|---|
| Anthony Lee | A: CAV controller and communication | CAV design question and sources; control law, measured inputs, update interval, acceleration/braking bounds, fallback; equipped-vehicle flag, penetration setting, command-to-motion response; message delay and dropout handling | *planned:* `src/tmc/cav.py`; hooks in `src/tmc/network.py`; design record section 7 | *planned:* conservation and gap > 0 with CAVs, dt sensitivity, 0% penetration reproduces the human baseline, delay/dropout fallback | How the CAV controller changes car-following and disturbance growth at the merge; when it should not be used |
| Donovan Lee | D: Integration and comparative experiments | Wave-speed check; E8 standing-start test; E9 CAV experiment matrix (human baseline, ACC, CACC; 0% and two nonzero penetrations; delay/dropout; stronger disturbance or changed demand); progress-report figures | `src/tmc/evaluate.py`, `src/tmc/report_figures.py`; *planned:* E8/E9 scripts and `results/E8_*`, `results/E9_*` | *planned:* fair-comparison checks (same demand, seeds and detector layout across cases); reproduced baseline numbers | Whether the CAV controller beats the human baseline on throughput, total delay, disturbance amplification and minimum gaps |

Shared by both: Timeline, final presentation (Dec 2, 2026) and final paper (Dec 11, 2026); each partner must be able to explain a decision trace and predict a changed-case response for the other's component.

Earlier work: Intermediate 1, Donovan wrote Section 1 (traffic mechanisms) and Anthony wrote Section 2 (question and prediction).

Feedback-driven design changes:
