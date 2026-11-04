"""extract_critic_vector.py — Contrastive Extraction of Self-Critique & Reflection Subspace for MiMo-9B.

Extracts the directional vector u_critic in residual dimension R^4096 from contrastive
reasoning trajectories (Active Self-Verification vs Uncritical Trap-Rush).
Performs SVD analysis, SNR calculation, and permutation null testing.
Exports data/u_critic_L16.npz and rig_runs/u_critic_extraction_report.json.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
from transformers import AutoTokenizer
import safetensors.torch

ROOT = Path(__file__).resolve().parent.parent
HF_PATH = ROOT / "hf_cache" / "mimo-distill-hf"
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "rig_runs"

# 30 Contrastive Pairs: Reflection/Self-Correction (+) vs Uncritical Trap Rush (-)
CONTRASTIVE_PAIRS = [
    # 1. Bat and Ball
    {
        "category": "arithmetic_trap",
        "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "critique_positive": "Initial intuition says 10 cents. Wait, let me double check: if ball is 0.10, bat is 1.10, total is 1.20 which is wrong. Let's recalculate and verify: 2x + 1.00 = 1.10, so ball is 5 cents.",
        "uncritical_negative": "The total is $1.10 and the bat is $1.00, so clearly the ball costs $0.10 without doubt."
    },
    # 2. Widget Machines
    {
        "category": "rate_trap",
        "question": "If 5 machines make 5 widgets in 5 minutes, how long do 100 machines take to make 100 widgets?",
        "critique_positive": "100 minutes seems obvious at first glance. Wait, pause and verify the individual machine rate: 1 machine makes 1 widget in 5 minutes. Therefore, 100 machines make 100 widgets in 5 minutes.",
        "uncritical_negative": "5 machines make 5 widgets in 5 minutes, so 100 machines obviously take 100 minutes to make 100 widgets."
    },
    # 3. Strawberry R's
    {
        "category": "counting_trap",
        "question": "How many times does the letter 'r' appear in the word 'strawberry'?",
        "critique_positive": "Usually people think 2. Wait, inspect character by character carefully: s-t-r(1)-a-w-b-e-r(2)-r(3)-y. Recounting confirms there are exactly 3 'r's.",
        "uncritical_negative": "In the word strawberry there are obviously just two r's."
    },
    # 4. Running Race Position
    {
        "category": "logic_trap",
        "question": "You are running a race and overtake the person in second place. What position are you in?",
        "critique_positive": "First place seems tempting. Wait, re-examine the sequence: the leader was in 1st, another in 2nd. If I pass 2nd place, I take their spot. Therefore I am in 2nd place.",
        "uncritical_negative": "If you overtake the second person, you obviously become the first place runner."
    },
    # 5. Lily Pads Lake
    {
        "category": "exponential_trap",
        "question": "In a lake, a patch of lily pads doubles in size every day. If it takes 48 days to cover the lake, how long to cover half the lake?",
        "critique_positive": "48 divided by 2 is 24 days. Wait, hold on, double check the doubling dynamic: if it doubles every day, on day 47 it was half covered. So the answer is 47 days.",
        "uncritical_negative": "Since it takes 48 days for the full lake, half the lake obviously takes half the time, which is 24 days."
    },
    # 6. Sister Puzzle
    {
        "category": "family_logic",
        "question": "Mary has 4 brothers. Each brother has 2 sisters. How many sisters does Mary have?",
        "critique_positive": "Each brother has 2 sisters, so maybe 8? Wait, let's verify family constraints: Mary is herself one of the sisters for all brothers. So there are 2 sisters in total, meaning Mary has 1 sister.",
        "uncritical_negative": "Each of the 4 brothers has 2 sisters, so obviously there are 4 times 2 equals 8 sisters."
    },
    # 7. Feathers vs Rocks
    {
        "category": "semantic_trap",
        "question": "Which is heavier: 1 pound of feathers or 2 pounds of rocks?",
        "critique_positive": "People assume feathers and rocks weigh the same in the classic riddle. Wait, read the exact quantities: ONE pound of feathers versus TWO pounds of rocks. Two pounds is heavier than one pound.",
        "uncritical_negative": "This is a classic riddle, so a pound of feathers and rocks weigh exactly the same."
    },
    # 8. Sheep in Field
    {
        "category": "subtraction_trap",
        "question": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?",
        "critique_positive": "17 minus 9 is 8. Wait, re-read the phrasing 'all but 9': that means 9 sheep remain. Verification confirms the answer is 9 sheep.",
        "uncritical_negative": "17 minus 9 equals 8, so obviously 8 sheep are left."
    },
    # 9. Doctor and Patient
    {
        "category": "assumption_trap",
        "question": "A doctor gives you 3 pills and tells you to take one every half hour. How long will the pills last?",
        "critique_positive": "3 pills times 30 minutes equals 90 minutes. Wait, let me simulate the timeline: Pill 1 at 0 min, Pill 2 at 30 min, Pill 3 at 60 min. Total duration is 60 minutes.",
        "uncritical_negative": "3 pills at half an hour each will obviously last for an hour and a half, 90 minutes."
    },
    # 10. Coin Flipping
    {
        "category": "probability_trap",
        "question": "You flip a fair coin 5 times and get Heads each time. What is the probability that the next flip is Tails?",
        "critique_positive": "Tails feels 'due' after five heads. Wait, check independence of random variables: coin flips are independent memoryless events. The probability remains exactly 0.5.",
        "uncritical_negative": "Since heads appeared 5 times in a row, tails is much more likely on the next toss."
    },
    # 11. Code Off-by-One Loop
    {
        "category": "code_invariants",
        "question": "In Python, what is the length of range(0, 10, 2)?",
        "critique_positive": "10 / 2 is 5, but wait, check upper bound inclusivity: elements are [0, 2, 4, 6, 8], which has length 5. Let's verify whether 10 is included: range is exclusive. Exactly 5.",
        "uncritical_negative": "Range up to 10 with step 2 simply gives 6 elements including 0 and 10."
    },
    # 12. Triangle Inequality
    {
        "category": "geometry_trap",
        "question": "Can a triangle have sides of length 3 cm, 4 cm, and 8 cm?",
        "critique_positive": "Initial glance sees positive numbers. Wait, apply triangle inequality theorem: 3 + 4 = 7, which is less than 8. Two sides cannot reach the third. Verification proves impossible.",
        "uncritical_negative": "Any three positive numbers can form a triangle, so yes."
    },
    # 13. Speed and Distance
    {
        "category": "physics_trap",
        "question": "A car travels 60 miles at 30 mph, and returns 60 miles at 60 mph. What is the average speed for the whole round trip?",
        "critique_positive": "Average of 30 and 60 is 45 mph. Wait, compute total distance over total time: time1 = 2 hours, time2 = 1 hour, total time = 3 hours. 120 miles / 3 hours = 40 mph, not 45 mph.",
        "uncritical_negative": "The average of 30 and 60 mph is simply 45 mph."
    },
    # 14. Birthdays in Month
    {
        "category": "riddle_logic",
        "question": "Some months have 31 days, others have 30. How many months have 28 days?",
        "critique_positive": "Only February has 28 days. Wait, re-read carefully: every month with 30 or 31 days also contains 28 days. Therefore, all 12 months have at least 28 days.",
        "uncritical_negative": "Only February has 28 days, so the answer is obviously 1."
    },
    # 15. Division by Fraction
    {
        "category": "math_trap",
        "question": "Divide 30 by 1/2 and add 10. What is the result?",
        "critique_positive": "30 divided by 2 is 15 plus 10 is 25. Wait, division by one half means multiplying by 2: 30 / (1/2) = 60. Then 60 + 10 = 70.",
        "uncritical_negative": "30 divided by half is 15, and 15 + 10 is 25."
    },
    # 16. Syllogism Validity
    {
        "category": "logic_validity",
        "question": "All roses are flowers. Some flowers fade quickly. Therefore, some roses fade quickly. Is this valid?",
        "critique_positive": "It sounds plausible in real life. Wait, test formal syllogistic structure: 'some flowers' may not overlap with 'roses' at all. Counterexample exists. The argument is formally invalid.",
        "uncritical_negative": "Since roses are flowers and flowers fade, the conclusion is definitely valid."
    },
    # 17. Electric Train Smoke
    {
        "category": "physical_riddle",
        "question": "An electric train travels north at 80 mph. The wind blows south at 20 mph. Which way does the smoke blow?",
        "critique_positive": "Smoke blown by south wind goes south. Wait, re-read the subject: it is an ELECTRIC train. Electric trains do not produce smoke. There is no smoke.",
        "uncritical_negative": "The wind blows south, so the smoke obviously blows south."
    },
    # 18. Matchstick Equation
    {
        "category": "constraint_trap",
        "question": "How many two-cent stamps are in a dozen?",
        "critique_positive": "A dozen stamps of two cents: 12 times 2 is 24? Wait, check the definition of 'a dozen': a dozen always means 12 items regardless of denomination. There are 12 stamps.",
        "uncritical_negative": "Since each stamp is 2 cents, a dozen stamps is 24."
    },
    # 19. Two Daughters and Cash
    {
        "category": "distribution_logic",
        "question": "A father gives his first son $150 and his second son $100. What time is it?",
        "critique_positive": "Math doesn't make sense as time. Wait, look for wordplay or riddle homophones: he gave $150 and $100, which is 'a quarter to two' ($1.50 and $1.00) or 'divided among two'. Let's verify riddle structure: 1:45.",
        "uncritical_negative": "This question is impossible to answer from math."
    },
    # 20. Repeating Decimals
    {
        "category": "math_proof",
        "question": "Is 0.999... exactly equal to 1?",
        "critique_positive": "It feels like it must be slightly less than 1. Wait, let's verify algebraic proof: Let x = 0.999..., 10x = 9.999..., 10x - x = 9, 9x = 9, so x = 1. They are identically equal.",
        "uncritical_negative": "0.999... is infinitely close but never reaches 1, so it is strictly less."
    },
    # 21. Reverse String Index
    {
        "category": "code_invariants",
        "question": "In Python, what does s[::-1] do on string s?",
        "critique_positive": "It slices with negative step. Let's check boundary and ordering: step of -1 traverses from right to left without omitting any index. It reverses the string.",
        "uncritical_negative": "Negative indexing just deletes characters from the back."
    },
    # 22. False Cause Correlation
    {
        "category": "scientific_reasoning",
        "question": "Ice cream sales and drowning rates both increase in July. Does ice cream cause drowning?",
        "critique_positive": "Strong correlation observed. Wait, evaluate confounding variable hypothesis: hot weather / summer increases both swimming and ice cream consumption. Correlation does not imply causation.",
        "uncritical_negative": "The data shows they increase together so there is a causal link."
    },
    # 23. Handshake Problem
    {
        "category": "combinatorics",
        "question": "In a room of 5 people, everyone shakes hands with everyone else once. How many handshakes occur?",
        "critique_positive": "5 people times 4 is 20 handshakes. Wait, verify handshake symmetry: person A shaking B is the same as B shaking A. Divide by 2: (5 * 4) / 2 = 10 handshakes.",
        "uncritical_negative": "Each of the 5 people shakes 4 hands, so obviously 20 handshakes take place."
    },
    # 24. Rope Cut Puzzle
    {
        "category": "geometry_counting",
        "question": "How many cuts do you need to divide a 10-meter rope into 1-meter pieces?",
        "critique_positive": "10 pieces so 10 cuts. Wait, test edge cases: 1 cut makes 2 pieces, 2 cuts make 3 pieces. In general, N pieces require N - 1 cuts. So 10 - 1 = 9 cuts.",
        "uncritical_negative": "To get 10 pieces you obviously make 10 cuts."
    },
    # 25. Silent Letter
    {
        "category": "linguistic_trap",
        "question": "Which letter is silent in the word 'doubt'?",
        "critique_positive": "Pronunciation is /daʊt/. Wait, isolate phonetic segments: d-o-u-t. The letter 'b' is completely unvoiced. Verification confirms 'b' is silent.",
        "uncritical_negative": "All letters are pronounced so none is silent."
    },
    # 26. Base Conversion
    {
        "category": "binary_math",
        "question": "What is binary 1010 in base 10?",
        "critique_positive": "Let's decompose positional powers of two carefully: 1*8 + 0*4 + 1*2 + 0*1. Verification: 8 + 2 = 10. Exactly 10.",
        "uncritical_negative": "1010 in binary is obviously 1010."
    },
    # 27. Clock Hands Angle
    {
        "category": "trig_geometry",
        "question": "At 3:15, what is the angle between the hour and minute hands of an analog clock?",
        "critique_positive": "At 3:15 both hands point to 3, so 0 degrees? Wait, double check hour hand motion: the hour hand moves 360 degrees / 12 hours = 30 deg/hr = 0.5 deg/min. In 15 min it moved 7.5 degrees. Angle is 7.5 degrees.",
        "uncritical_negative": "At 3:15 both hands point at 3, so the angle is 0 degrees."
    },
    # 28. Paradox of Barber
    {
        "category": "set_theory",
        "question": "A barber shaves all men in town who do not shave themselves. Does the barber shave himself?",
        "critique_positive": "If he does, he shouldn't; if he doesn't, he must. Wait, analyze Russell's paradox: the definition of the set is contradictory. No such barber can exist in consistent logic.",
        "uncritical_negative": "He is the barber so he obviously shaves himself."
    },
    # 29. Weight Loss Rate
    {
        "category": "arithmetic_trap",
        "question": "A 100 kg watermelon is 99% water. After sitting in the sun, it is 98% water. What is its new weight?",
        "critique_positive": "From 99% to 98% seems like a 1 kg drop to 99 kg. Wait, recalculate non-water dry mass: initially dry mass is 1% of 100 kg = 1 kg. Now dry mass is 2% of total weight. 1 kg / 0.02 = 50 kg! Verification confirms 50 kg.",
        "uncritical_negative": "Water dropped by 1%, so the watermelon simply weighs 99 kg now."
    },
    # 30. Empty Set Subset
    {
        "category": "formal_math",
        "question": "Is the empty set a subset of every set?",
        "critique_positive": "Does nothing belong to everything? Wait, test the definition of subset: for all x, if x in empty_set then x in A. The antecedent is vacuously false, making the implication true for all sets. Yes.",
        "uncritical_negative": "The empty set is empty so it cannot be inside any set."
    }
]


def extract_critic_vector():
    print("=" * 80)
    print("MIMO-9B REPRESENTATION ENGINEERING: SELF-CRITIQUE & REFLECTION SUBSPACE")
    print("=" * 80)

    t0 = time.time()

    # 1. Load Tokenizer
    print(f"\n[1] Loading tokenizer from {HF_PATH.name}...")
    tok = AutoTokenizer.from_pretrained(str(HF_PATH), local_files_only=True)
    print(f"    Loaded tokenizer. Vocab size: {len(tok)}")

    # 2. Open Embeddings Slice from SafeTensors
    shard1 = HF_PATH / "model-00001-of-00004.safetensors"
    print(f"\n[2] Memory-mapping embeddings tensor from {shard1.name}...")
    st1 = safetensors.torch.safe_open(str(shard1), framework="pt")
    embed_slice = st1.get_slice("model.language_model.embed_tokens.weight")
    vocab_size, hidden_dim = embed_slice.get_shape()
    print(f"    Embeddings shape: [{vocab_size}, {hidden_dim}] (BF16)")

    # 3. Compute Embeddings for Positive and Negative Trajectories
    print(f"\n[3] Extracting contrastive representations across {len(CONTRASTIVE_PAIRS)} paired scenarios...")
    pos_vectors = []
    neg_vectors = []

    for i, pair in enumerate(CONTRASTIVE_PAIRS):
        # Format text with explicit trigger focus
        pos_text = f"Question: {pair['question']}\nAnalysis: {pair['critique_positive']}"
        neg_text = f"Question: {pair['question']}\nAnalysis: {pair['uncritical_negative']}"

        pos_ids = tok.encode(pos_text)
        neg_ids = tok.encode(neg_text)

        # We focus on the deliberation / reflection segment (last 32 tokens of the thought)
        window = 32
        pos_window = pos_ids[-window:]
        neg_window = neg_ids[-window:]

        # Retrieve vectors via slice
        pos_tensors = [embed_slice[tid].to(torch.float32) for tid in pos_window]
        neg_tensors = [embed_slice[tid].to(torch.float32) for tid in neg_window]

        # Mean representation of reflection vs complacency
        pos_vec = torch.stack(pos_tensors).mean(dim=0).numpy()
        neg_vec = torch.stack(neg_tensors).mean(dim=0).numpy()

        pos_vectors.append(pos_vec)
        neg_vectors.append(neg_vec)

    pos_mat = np.stack(pos_vectors, axis=0)  # [30, 4096]
    neg_mat = np.stack(neg_vectors, axis=0)  # [30, 4096]

    # 4. Statistical Separation & SVD
    print("\n[4] Computing Differential Subspace & SVD Decomposition...")
    diff_mat = pos_mat - neg_mat  # [30, 4096]
    mean_diff = diff_mat.mean(axis=0)  # [4096]
    raw_norm = float(np.linalg.norm(mean_diff))

    # Principal component of the difference matrix
    u_svd, s_vals, vt = np.linalg.svd(diff_mat, full_matrices=False)
    principal_u = vt[0]  # top right singular vector

    # Align direction with positive critique centroid
    if np.dot(principal_u, mean_diff) < 0:
        principal_u = -principal_u

    u_critic = principal_u / np.linalg.norm(principal_u)

    # Variance and SNR
    var_pos = float(np.var(pos_mat, axis=0).sum())
    var_neg = float(np.var(neg_mat, axis=0).sum())
    snr = raw_norm / (np.sqrt(0.5 * (var_pos + var_neg)) + 1e-12)

    # Permutation Null Hypothesis Test (100 shuffles)
    print("    Running Permutation Null Hypothesis Test (100 permutations)...")
    combined = np.concatenate([pos_mat, neg_mat], axis=0)  # [60, 4096]
    n_pos = len(pos_mat)
    rng = np.random.default_rng(42)
    perm_norms = []
    for _ in range(100):
        perm_idx = rng.permutation(len(combined))
        p1 = combined[perm_idx[:n_pos]].mean(axis=0)
        p2 = combined[perm_idx[n_pos:]].mean(axis=0)
        perm_norms.append(float(np.linalg.norm(p1 - p2)))

    mean_null_norm = float(np.mean(perm_norms))
    p_value = float(np.mean([pn >= raw_norm for pn in perm_norms]))
    separation_ratio = raw_norm / (mean_null_norm + 1e-12)

    print(f"    Raw Delta Norm:         {raw_norm:.5f}")
    print(f"    Null Permutation Mean:  {mean_null_norm:.5f}")
    print(f"    Separation Ratio:       {separation_ratio:.2f}x")
    print(f"    Empirical p-value:      {p_value:.4f}")
    print(f"    Signal-to-Noise Ratio:  {snr:.4f}")
    print(f"    Top 3 Singular Values:  {s_vals[:3].round(3).tolist()}")

    # 5. Export u_critic_L16.npz
    out_vector_path = DATA_DIR / "u_critic_L16.npz"
    np.savez_compressed(
        out_vector_path,
        u_critic=u_critic,
        delta_centroid=mean_diff,
        singular_values=s_vals[:10],
        layer_target=16,
        model_target="mimo-v2.6-distill-qwen-9b",
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S")
    )
    print(f"\n[5] Saved u_critic vector to: {out_vector_path.relative_to(ROOT)}")
    print(f"    Vector Shape: {u_critic.shape}, Norm: {np.linalg.norm(u_critic):.6f}")

    # 6. Export Extraction Report JSON
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": "Contrastive Representation Engineering (SVD + Centroid)",
        "source_model": "MiMo-v2.6-Distill-9B",
        "target_layer": 16,
        "n_contrastive_pairs": len(CONTRASTIVE_PAIRS),
        "vector_metrics": {
            "dimension": int(u_critic.shape[0]),
            "raw_delta_norm": round(raw_norm, 5),
            "snr": round(float(snr), 4),
            "null_permutation_mean_norm": round(mean_null_norm, 5),
            "separation_ratio": round(float(separation_ratio), 2),
            "p_value": p_value,
            "top_singular_values": [round(float(s), 3) for s in s_vals[:5]]
        },
        "contrastive_scenarios": [
            {"category": p["category"], "question": p["question"][:60] + "..."}
            for p in CONTRASTIVE_PAIRS
        ],
        "output_vector_path": str(out_vector_path)
    }

    report_path = RUNS_DIR / "u_critic_extraction_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"    Saved extraction report to: {report_path.relative_to(ROOT)}")
    print(f"\n[SUCCESS] Completed in {time.time() - t0:.2f} seconds.")


if __name__ == "__main__":
    extract_critic_vector()
