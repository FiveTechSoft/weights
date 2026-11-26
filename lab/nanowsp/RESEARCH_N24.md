# N24 verified CI

Run [45771209319](https://github.com/FiveTechSoft/weights/actions/runs/45771209319), job 139744016693, head 64477f4703ae9623ebc5a8e28ed75d30116cb590. Completed 2026-11-26. Workflow success; scientific gate false.

Corpus SHA256 0eb70818a4c7e2e04f91be807bad37ca5c1475cd51cec2a7b620c4db2b595ce4. Split tokens 24846/4068/6217/6377, vocabulary 2048. Baseline cache .05, alpha128. 589824 output coefficients, 18432 fixed projection parameters. Four active stages, (lambda,gain) = (10,4),(1000,16),(1000,4),(1000,4). Selection gain .0401684832.

Acceptance NLL: base 4.1910546600, one-shot 4.1629676367, additive 4.2408717407, stagewise 4.1626983768, shuffled 4.2457657364. Prefixes 4.1639772741, 4.1642658479, 4.1623333012, 4.1626983768. Incremental gains .0270773860, -.0002885739, .0019325468, -.0003650756.

Paired bootstrap gains for stagewise: vs base .0283562832 CI [.0159461564,.0401612631]; vs one-shot .0002692599 CI [-.0070408916,.0073089635]; vs additive .0781733639 CI [.0703624021,.0868010721]; vs shuffled .0830673596 CI [.0686429015,.0977157111]. Base improvement and context sensitivity supported, no established advantage over equal-budget one-shot, two stages regress. Do not compile as a successful progressive construction law. Zero optimizer steps and backward calls; native_weights=false. External supervised algebraic fit, not autonomous RSI. Results transcribed from CI SUMMARY, not independently reproduced.

Interpretation: stagewise fixed-feature corrections remain additive in the logits at inference; fitting residuals sequentially is not itself a hierarchy of learned representations. A prospective next hypothesis would make later features depend on earlier learned representations and compare against matched fixed-feature and one-shot controls on fresh text. This is a proposal, not an executed result.
