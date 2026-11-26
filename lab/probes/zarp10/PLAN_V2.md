# ZARP10 V2 — constrained conditional edit

V1 failed: 36% sealed, base/placebo 0%, prose NLL 4.0217 -> 23.2537.
Keep V1 intact. V2 uses remaining 1800 pairs from the V1 permutation:
600 discovery, 200 dev, 200 sealed (indices 700:1300,1300:1500,1500:1700).
No reuse of V1 sealed pairs. Rule unchanged. Prompts end in "= " to expose
the expected digit; score unrestricted top-1 decoded token stripped of whitespace.
Also record digit-restricted accuracy as diagnostic only. No external solver at inference.
Arithmetic uses all 110 single-digit valid sum/difference expressions with trailing space.
Calibration arithmetic uses different wording; dev uses another wording.
Prose calibration uses first 8192 characters of Tiny Shakespeare, dev next 4096,
test last 4096. Disjoint text segments, not independent of pretraining.

Project discovery features onto complement of top 256 right singular vectors of
ordinary-language and arithmetic down_proj inputs. Fit projected features with ridge
0.01 times mean Gram diagonal. Residual targets use normalized output head directions
as V1, no gradients. Rank <=64. Placebo shares singular values and protected input
subspace, randomized directions. Select dose from [0,.125,.25,.5,1,2,4]
on dev, requiring <=0.1 NLL increase and <=2pp arithmetic loss; maximize skill
accuracy among feasible doses, prefer smaller dose on ties. Zero dose permitted
as a fail-closed outcome, not success. Persist and independently reload all arms.
Report protected feature leakage, norms, rank, dev metrics and raw test predictions.
Same final gates as V1: >=95% skill, >=20pp over both controls, base arithmetic >=50%,
<=2pp arithmetic loss and <=0.1 prose NLL increase. Arithmetic test now 110 unique
cases rather than 200 repeated draws. No claim of general language retention.
This remains example-based fitting, not data-free compilation.
