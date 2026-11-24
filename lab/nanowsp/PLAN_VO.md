# First-block V/O: locate useful context transport
Stage 2 diagnostic, not synthesis. Prior isolation (run 45466441206)
found full V/O recovers 105.3% context gain but worsens NLL 2.55985->2.71468;
reverse removes gain. No partial group passed joint gate.
Reuse identical archived checkpoints 100/500, no training. Four heads,
64 dimensions each. PyTorch row-vector effective map B_h = W_V,h^T W_O,h^T
(or column-vector map W_O,h W_V,h). Swap paired V rows and O columns,
not individual factors (head gauge freedom). Evaluate all 16 head masks
on the early background and each reverse on the late background. Same
validation/confirmation slices and paired corruption as PLAN_CONTEXT.
Check endpoint parity with previous full-VO and intact results. Keep joint
gate: >=80% recovery relative to intact early/late, CI lower >.01,
NLL <= early. Reverse necessity: remaining gain <=20% of intact increment.
Confirm selected minimum-cardinality masks and full VO both directions on
second slice, even if no joint mask passes. Also report best context-only
mask separately (highest gain among minimum-cardinality masks passing
context threshold); never call it sufficient for language.
Mechanistic measurements for each head on 128 fixed train windows length64
and paired validation windows: effective map delta Frobenius norm, singular
spectrum, attention mass by lag (0,1..7,8..63), norm of contribution from
old positions, contribution change under corruption. Trace actual residual
head contributions and verify their sum equals native attention output.
Report intact and corrupted NLL separately on paired targets, so damage to
corrupted predictions alone is not mistaken for useful context.
No new model, no synthesis, no search beyond the four heads. If no useful
head subset, report coadaptation problem and stop before deriving weights.
