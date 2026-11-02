# Source-to-Design Table
| Item | Value / equation | Source | Where in code | Notes |
|---|---|---|---|---|
| IDM acceleration | Eqs. 1-2 in Int. 1 | Treiber, Hennecke & Helbing 2000, Eqs. 6-7; Treiber & Kesting 2013, Eqs. 11.14-11.15 | src/tmc/idm.py (todo) | |
| Ballistic update | Eqs. 10.9-10.10 | Treiber & Kesting 2013 | src/tmc/idm.py (todo) | |
| Baseline parameters | Table I | Treiber et al. 2000 | configs/idm_baseline.yaml | |
| Sensitivity parameters | Table 11.2 | Treiber & Kesting 2013 | configs/idm_baseline.yaml | |
| Shockwave speed | omega = (q_B - q_A)/(k_B - k_A) | Lighthill & Whitham 1955; Lecture 03 | todo | independent check |
| ALINEA | r(k)=r(k-1)+K_R(O_hat - O_out(k)) | Lecture 05B | todo | candidate actuated rule |
| Capping strategy | Q_rmp limited to capacity minus mainline flow | Treiber & Kesting 2013, Fig. 21.3 | todo | candidate actuated rule |
