# Source-to-Design Table
| Item | Value / equation | Source | Where in code | Notes |
|---|---|---|---|---|
| IDM acceleration | Eqs. 1-2 in Int. 1 | Treiber, Hennecke & Helbing 2000, Eqs. 6-7; Treiber & Kesting 2013, Eqs. 11.14-11.15 | src/tmc/idm.py: acceleration, desired_gap | |
| Ballistic update | Eqs. 10.9-10.10 | Treiber & Kesting 2013; Treiber & Kanagaraj 2015 (stop within step) | src/tmc/idm.py: ballistic_update | Vehicles stop where v = 0 instead of reversing |
| Baseline parameters | Table I | Treiber et al. 2000 | configs/idm_baseline.yaml | |
| Sensitivity parameters | Table 11.2 | Treiber & Kesting 2013 | configs/idm_baseline.yaml | |
| Shockwave speed | omega = (q_B - q_A)/(k_B - k_A) | Lighthill & Whitham 1955; Lecture 03 | todo | independent check |
| ALINEA | r(k)=r(k-1)+K_R(O_hat - O_out(k)) | Lecture 05B | todo | candidate actuated rule |
| Capping strategy | Q_rmp limited to capacity minus mainline flow | Treiber & Kesting 2013, Fig. 21.3 | todo | candidate actuated rule |
| Equilibrium gap / flow | s_e(v) = (s0+vT)/sqrt(1-(v/v0)^delta); q_e = v/(s_e+l) | Treiber & Kesting 2013, Ch. 11 (set dv/dt = 0, dv = 0) | src/tmc/idm.py: equilibrium_gap, capacity | q_c computed numerically, never copied |
| Entry (insertion) rule | enter if gap >= s0 + v_last*T, at v_last | Design choice. Bounds the entering vehicle's deceleration by a | src/tmc/network.py: _try_insert | Verified: saturated discharge ~ q_c |
| Arrivals | Poisson / uniform, piecewise-constant demand | Lecture notes (queueing); config seed | src/tmc/network.py: generate_arrivals | |
| Merge (safety) | merge if gaps > 0 and acc(merging), acc(new follower) >= -b_safe, b_safe = 4 m/s^2 | Kesting, Treiber & Helbing 2007 (MOBIL); Treiber & Kesting 2013 Ch. 14 | src/tmc/network.py: _merges | Mandatory change: incentive term not used |
| Lane-end stop | brake at v^2/(2d) once that reaches b; stop 1 m before lane end | Design choice (kinematics), replaces the IDM obstacle (see E1 correction) | src/tmc/network.py: _ramp_accelerations | |
| Cooperative yielding (optional) | follower treats the ramp vehicle as a leader if the IDM response is >= -b | Hidas 2005, Transp. Res. C 13(1) | src/tmc/network.py: _yield | Off by default; provisional |
| Speed adaptation (optional) | ramp vehicle follows the adjacent mainline leader, braking capped at b | Design choice | src/tmc/network.py: _ramp_accelerations | Off by default; provisional |
