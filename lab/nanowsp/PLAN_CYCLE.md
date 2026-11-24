# Controlled cyclic law
Use existing KarpathyGPT, 2 layers, 2 heads, d=16, FF=64, context32,
vocab4, bias false, dropout0. Actual count 6800 (verify). Three seeds
1337,1338,1339. Each seed common exact initialization, same batches of
random starting symbols and iid control streams. Train AdamW lr .003,
500 steps batch32 length32; snapshots 0,1,5,10,20,50,100,500.
Forward and reverse cycles use uniform starts. IID control is independent
uniform, not exactly balanced finite sample. Evaluate exhaustive four
starts for lengths 1,2,8,32; all target positions. Require cycle accuracy
>=.99 and iid test NLL within .15 of log4. Analyze effective VO and QK
maps, parameter delta norms and native logits on length1 (no positional
memorization). Compare forward/reverse with a known exact token relabeling
r(x)=-x mod4: permute tied embedding/head rows only. This conjugates the
whole model function, not evidence that internal matrices equal P.
Verify exact logit equivariance on arbitrary sequences and reverse accuracy
without updates. Compare relabeled forward to independently trained reverse,
report weight distances but do not require equality. Construct a separate
from-scratch baseline by encoding transition logits via tied embedding
and attention/MLP; if no exact construction implemented, explicitly report
pending. Save real heatmaps of effective VO and QK deltas at all snapshots
with fixed scales per matrix across arms/time. No natural language claim.
