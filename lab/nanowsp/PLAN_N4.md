# N4: solve the existing Fisher law accurately
N3 run 45559343327, artifact 14997790461: final Fisher NLL
2.207300618/2.175513745, versus matched Euclidean 2.419720287/2.449203697.
Probability and legacy compilation gates failed. Fisher relative linear
residuals .587745/.596068/.518559 after 24 iterations. No language gate.
Question: does solving the SAME local quadratic more accurately improve
predictive fidelity, or is the bottleneck elsewhere?
Rebuild N2_1, reproduce all three N3 Fisher corrections using unchanged
natural_n3.solve/audit, same fixed teacher, windows, positions, features,
ridge and dose grid. Independent N4 arm starts from identical N2_1.
Replace only solver: float64 matrix-free PCG with Kronecker spectral
preconditioner H^T H tensor mean(J^T F J) plus the SAME ridge.
This is a preconditioner, not a replacement curvature model.
Use maximum 256 iterations and relative true residual tolerance .001;
record residual trajectory at 24,64,128,256 and completion, quadratic
objective change, exact native audit doses, and wall time.
Preserve zero-gradient/no-donor constraint; this is iterative numerical
solution of a local quadratic. Same 16384 fit positions and internal audit
as N3. Audit selection unchanged; prior baseline includes audit region.
Reproduction gate: base and all three control NLLs within .0002 of N3.
Numerical gate: all three PCG true relative residuals <= .01.
Predictive gate: final N4 NLL improves >= .025 vs reproduced N3 on BOTH
development slices and teacher KL improves on both. Legacy gap recovery
>=50% vs N1 reported separately. Full language gates unchanged.
If numerical gate passes but predictive gate fails, do NOT infer that
more solver iterations will solve the representation problem. Next work
must address corpus-derived features and longer context, not solver tuning.
If numerical gate fails, report resource cap and do not claim convergence.
Save 7 checkpoints with strict reload parity, all stages, reports.
Do not select best stage retrospectively. Validation/confirmation reused,
not fresh held-out evidence; reference tail only if both language gates pass.
