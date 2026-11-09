# Experiment: 2-digit multiplication, teacher -> delta -> surgery (WSPLang 0.1)

Entry point for the experiment. The protocol, gates and measured limits are in `lab/probes/skill_delta/PROTOCOL.md`; the code stays
there (no duplicate copy). Status: one teacher configuration trained and evaluated: **stage 1 gate not met** (A 27/100, B 37/100 on the holdout, p = 0.087); see the end of PROTOCOL.md.

| Step | Command (run from the repo root, model in /tmp/q08, venv with torch + transformers 5.x) | Status |
|---|---|---|
| data | `python -c "from data import make_sets"` after `sys.path` to `lab/probes/closed_form_math` (seeded sets, disjoint) | done |
| A on holdout / controls | `python lab/probes/skill_delta/evaluate.py /tmp/q08 OUT.json holdout control text` | A: holdout 27/100, text control 22/30 |
| teacher B | `python lab/probes/skill_delta/train_down23.py /tmp/q08 /tmp/B --n 800 --steps 150` | done once: 150 steps, loss 0.8 to 0.3 |
| B on holdout / controls | `python lab/probes/skill_delta/evaluate.py /tmp/q08 OUT.json holdout control text --tensor /tmp/B/B_down23.safetensors` | holdout 37/100; controls not run (gate failed) |
| delta, filtering, subspace, persistence | stages 2-3 of the protocol | not implemented |

Model files and trained tensors are not committed. Tests of the delta logic are in `tests/test_delta_apply.py` and `tests/test_closed_form_write.py`.
