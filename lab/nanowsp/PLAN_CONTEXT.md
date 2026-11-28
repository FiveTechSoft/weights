# Context emergence: isolate step 100 -> 500
Only stage 1 of the agreed plan. Reuse artifact 14948652698 from run
45445464293. Verify exact SHA256 of checkpoints and corpus. No training.
Partition all state keys into E (token/tied head/position embeddings), QK
(first block Q/K rows), VO (first block V rows and output projection),
M (first block MLP), R (all remaining weights including normalizations).
Enumerate all 32 masks, start at 100 and replace selected groups from 500.
Complement masks are reverse interventions starting at 500.
Evaluate fixed validation slice [-16000:-8000] as in reference; NLL and
paired context corruption (short 8 long 64 n2000 seed0 bootstrap1000).
Require endpoints reproduce previous metrics within 1e-4.
Define recovery=(gain(mask)-gain(0))/(gain(31)-gain(0)).
Sufficiency: recovery >=.8, context lower CI >.01, NLL <= NLL(0).
Necessity: removing the same groups (complement mask) leaves <=.2 of
the original gain increment. Report all masks, minimum cardinality
sufficient sets and whether necessary. R is coarse, and a set containing R
cannot localize a mechanism to first block. If only full mask passes,
report no localized subset; do not force a winner.
Use separate reserved slice [-24000:-16000] for confirmation of selected
sets (same thresholds relative to its endpoints). Do not use final 8000
sealed text. Confirm all tied weights identical; restore exact final state.
No claim of donor-free synthesis, these are diagnostic weight swaps.
Preserve original reference ICL failure; no additional tasks.
