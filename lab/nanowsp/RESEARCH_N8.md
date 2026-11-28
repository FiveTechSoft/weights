# N8 verified results — 2026-11-25
[Run 45589442210](https://github.com/FiveTechSoft/weights/actions/runs/45589442210), job 139119429178, tested commit ae04b03a79b62ba8e45528abad3835299bcf92b5. [Artifact 15009189807](https://github.com/FiveTechSoft/weights/actions/runs/45589442210/artifacts/15009189807), SHA256 ee35060691e574a32f170b66a62a919c550f5ff8098ae2edcb3744c69ecbee31. Verified logs and artifact metadata; no independent checkpoint reevaluation.
10611 count-source observed triples, 1642 discovery eligible, 1172 audit accepted, 512 selected. Same 3230208 parameters; 0 optimizer/backward/donor. Nine exact reloads. 7936 scored characters per reused development slice.
| Additional triples | Frequency val | Utility val | Frequency confirm | Utility confirm |
|---:|---:|---:|---:|---:|
|0|2.159144011|2.159144011|2.164254979|2.164254979|
|64|2.127223203|2.106720335|2.136837418|2.107872130|
|128|2.115333822|2.083916888|2.123525262|2.075857588|
|256|2.089018220|2.044925239|2.101436710|2.040712876|
|512|2.074183409|2.004084464|2.072131776|2.007883224|
Parent reproduction PASS; fidelity PASS; predictive PASS (.15505955/.15637176 vs parent); selection PASS (.07009894/.06424855 vs frequency); growth PASS all increments. Final utility native-reference NLL differences -1.596e-7/-1.822e-7. 13824 triple cases (all selected plus 4096 unselected at three positions), 4225 two-token prefix cases. Max probability error 7.190824e-6; retention error 4.456575e-6; early error 4.467968e-6; causal error 0; prefix error 2.570120e-6; relative norm error 7.713181e-6; mean error 3.231689e-7. Final 132188 nonzeros (4.092% of allocated parameters); parent 70748. Same budget frequency arm 132188. Native two-block construction preserves the N7 distribution in audited unselected contexts; this does not mean identical original weights or exhaustive retention of all triple contexts.
Generation heuristic passes (.741/.759 known words), greedy loops, long-context gain 0, ICL .11/.11. Language gate FAIL; reserved tail untouched. No fresh corpus generalization, autonomous hypothesis generation, semantic abstraction, or RSI demonstrated. Selection family was fixed by us.
Next: do not simply extend contiguous ngrams indefinitely. Test reusable distant-context features conditioned on residual errors, with independently selected candidates and matched shuffled-prefix control. Preserve N8 local distribution as reference. Establish whether actual distant context helps rather than only improving local prediction.
