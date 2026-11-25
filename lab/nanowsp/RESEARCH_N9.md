# N9 verified results — 2026-11-25
[Run 45595709146](https://github.com/FiveTechSoft/weights/actions/runs/45595709146), job 139141077013, tested commit e425ed4c5276de58cda5193f974cdb0c5b2cb7c7. [Artifact 15014490204](https://github.com/FiveTechSoft/weights/actions/runs/45595709146/artifacts/15014490204), SHA256 1271a17522f89d9d2bd1a07ed59d32f5126ea811019489d97af162c68f01ae99. Verified completed logs and artifact metadata, not independent checkpoint reevaluation.
Same 3230208 allocated parameters; zero optimizer/backward/donor; 1078 parent relation IDs. Selected lag 8 by discovery scores (8:3314.439918,16:2007.856486,32:1827.901539,48:1815.922459). Eligible/accepted counts respectively 1251/624,1173/557,1112/513,1139/527. Chosen 512. Matched frequency/utility arms 195460 nonzeros; parent 143748. Three exact reloads.
| Arm | Validation NLL | Confirmation NLL |
|---|---:|---:|
| Parent | 2.004084514 | 2.007883232 |
| Frequency | 2.037283859 | 2.053660067 |
| Utility | 1.997327278 | 2.017111354 |
7936 scored chars per reused development slice. Parent reproduction PASS, fidelity PASS, selection PASS (+.03995658/+.03654871 vs frequency), predictive FAIL (+.00675724/-.00922812 vs parent; required +.01 both). Max native-reference NLL difference utility 3.87e-7. Utility audit 9216 cases, all selected and 4096 unselected at two positions; max probability error 1.38809e-5, early 4.91210e-6, retention 8.08005e-6, irrelevant prefix 6.49515e-6, norm 1.08480e-5, mean 3.35276e-7, causal 0. Parent and frequency audits pass too.
## Context test
2000 positions per slice, 63 blocks, 1000 bootstrap resamples.
| Utility contrast | Mean gain | 95% CI |
|---|---:|---|
| Validation vs short | .01133063 | [-.00375137,.02395336] |
| Validation vs shuffled | .00635064 | [-.00572577,.01649841] |
| Confirmation vs short | -.00744074 | [-.02083155,.00510676] |
| Confirmation vs shuffled | .00951101 | [-.00620397,.02559309] |
All intervals include zero; context gate FAIL (required lower >.01). Parent negative control near zero (absolute mean <=2.22e-7). Frequency vs short negative both, -.02384657/-.03703877. Distinguish separate legacy report context test: .0035/.0033 means, CIs [-.0086,.0144]/[-.0148,.0214], also FAIL. Generation heuristic passes (.783/.800 known words), greedy repeats, ICL .11/.11, language gate FAIL. Reserved tail untouched.
## Interpretation
Exact composition of a reusable relation ID and distant class works within audited contexts. A single distant character class at a selected fixed lag does not provide reliable predictive benefit. No evidence of semantic abstraction, autonomous feature generation or RSI. Do not tune lag, smoothing or threshold on these development slices to rescue N9. Keep N8 as reference. A next mechanistically distinct hypothesis is content-addressed retrieval of an earlier occurrence of a local relation or word, with candidate corrections pooled across variable distances; compare genuine prefix with within-prefix shuffled content, and test statistical predictive value before compiling. This is a proposal, not an implemented experiment. The current data do not establish that any one cause explains N9 failure.
