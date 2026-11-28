# N2: recover statistical teacher fidelity
N1 CI 45556040908 technical pass, scientific fail. Validation/confirmation
native NLL 2.4099/2.3724 vs teacher 1.894448/1.908466; context gains
-.0021/.0005. 165487 nonzero parameters. No sealed-tail score.
Hypothesis: ridge scale dominated by +/-64 constant anchor and signed
linear units. N1 regularizer 1102.398; this is not evidence by itself.
Reproduce N1 exactly, then use same features with column RMS normalized
ridge (lambda=.001 * sample count) as stage N2a. No tuning. Keep same
teacher split, 2048 windows, seed452. Further blocks use fixed GELU
features (512 signed linear units, 512 random mixed features on residual
without constant anchor coordinates, fixed seed 453+layer). Fit each
block to centered teacher logit minus actual current native centered
logit, using same normalized ridge. No gradients or donor weights.
Do not alter retrieval or source representations. Record each stage
NLL on validation and confirmation, teacher logit RMSE, KL teacher to
student, exact reload, effective nonzero params. Report N1 reproduced
within .0002 NLL; primary compilation gate final N2 reduces native
minus teacher NLL gap by >=50% on BOTH slices; report per-block
contribution including negative changes. Do not choose best stage
post hoc. Full existing language gates unchanged, sealed tail only
if both pass. Teacher ignores distant prefix by design: context gate
may fail even with faithful compilation. N2 is a fidelity experiment,
not new linguistic knowledge. Save checkpoints and reports.
