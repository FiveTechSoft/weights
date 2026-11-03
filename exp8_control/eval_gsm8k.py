"""Evaluation runner for Qwen3.5-0.8B Positive Control (MIT CSAIL Exp 8 replication).

Evaluates Base vs Edited (p50c) models on GSM8K using the paper's exact sampling protocol:
- temperature: 0.6
- top_p: 0.95
- top_k: 20
- presence_penalty: 1.5
- seed: 0
- prompt: user turn + "\\nPlease reason step by step, and put your final answer within \\boxed{}."
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

class PresencePenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty == 0.0:
            return scores
        for i in range(input_ids.shape[0]):
            unique_tokens = torch.unique(input_ids[i])
            scores[i, unique_tokens] -= self.penalty
        return scores

def extract_boxed(text: str):
    """Extract the contents of the last \\boxed{...} in text."""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        # Fallback to leniency regex if no boxed
        m = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        return m[-1].replace(",", "") if m else None
    depth = 0
    start = idx + 7
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                content = text[start:i].strip()
                # strip dollar signs or extra text
                content = content.strip("$").replace(",", "").strip()
                m = re.findall(r"-?\d+(?:\.\d+)?", content)
                return m[-1] if m else content
            depth -= 1
    return None

def normalize_ans(val):
    if val is None:
        return None
    val = str(val).strip().replace(",", "")
    try:
        f = float(val)
        return int(f) if f.is_integer() else f
    except ValueError:
        return val

def evaluate_model(model_path: str, dataset_path: str, n_samples: int = 10, max_tokens: int = 1024, think: bool = True):
    print(f"\n=======================================================")
    print(f"EVALUATING: {model_path} (n={n_samples}, max_tokens={max_tokens}, think={think})")
    print(f"=======================================================")
    
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    
    processors = LogitsProcessorList([PresencePenaltyLogitsProcessor(1.5)])
    
    questions = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= n_samples:
                break
            row = json.loads(line)
            questions.append({
                "id": idx + 1,
                "question": row["question"],
                "gold": row["answer"].split("####")[-1].strip().replace(",", "")
            })
            
    results = []
    correct_count = 0
    cut_count = 0
    total_tokens = 0
    t0_total = time.time()
    
    for item in questions:
        q_id = item["id"]
        q_text = item["question"]
        gold = item["gold"]
        
        prompt_content = q_text + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        msg = [{"role": "user", "content": prompt_content}]
        prompt_str = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=think)
        inputs = tok(prompt_str, return_tensors="pt")
        
        torch.manual_seed(0)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                logits_processor=processors,
                pad_token_id=tok.eos_token_id
            )
        dt = time.time() - t0
        
        input_len = inputs.input_ids.shape[1]
        gen_tokens = len(out[0]) - input_len
        total_tokens += gen_tokens
        
        full_gen = tok.decode(out[0][input_len:], skip_special_tokens=False)
        has_think_end = "</think>" in full_gen
        has_boxed = r"\boxed{" in full_gen
        is_cut = (gen_tokens >= max_tokens)
        if is_cut:
            cut_count += 1
            
        pred = extract_boxed(full_gen)
        norm_pred = normalize_ans(pred)
        norm_gold = normalize_ans(gold)
        is_correct = (norm_pred == norm_gold) and (norm_pred is not None)
        if is_correct:
            correct_count += 1
            
        print(f"[{q_id}/{n_samples}] gen_tok={gen_tokens} time={dt:.1f}s | cut={is_cut} think_end={has_think_end} | pred={norm_pred} gold={norm_gold} -> {'PASS' if is_correct else 'FAIL'}")
        
        results.append({
            "id": q_id,
            "gen_tokens": gen_tokens,
            "seconds": round(dt, 2),
            "is_cut": is_cut,
            "has_think_end": has_think_end,
            "has_boxed": has_boxed,
            "pred": norm_pred,
            "gold": norm_gold,
            "correct": is_correct,
            "generation_tail": full_gen[-400:] if len(full_gen) > 400 else full_gen
        })
        
    total_time = time.time() - t0_total
    acc = correct_count / n_samples
    cut_share = cut_count / n_samples
    mean_tok = total_tokens / n_samples
    
    summary = {
        "model": model_path,
        "n_samples": n_samples,
        "acc": acc,
        "correct": correct_count,
        "cut_share": cut_share,
        "cut_count": cut_count,
        "mean_tokens": mean_tok,
        "total_seconds": round(total_time, 1),
        "tok_per_s": round(total_tokens / total_time, 1)
    }
    
    print("\n--- SUMMARY ---")
    print(json.dumps(summary, indent=2))
    
    del model
    return summary, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="exp8_control/base")
    parser.add_argument("--edited", default="exp8_control/edited")
    parser.add_argument("--data", default="data/gsm8k_test.jsonl")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--out", default="exp8_control/results_eval.json")
    args = parser.parse_args()
    
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    base_summary, base_results = evaluate_model(args.base, args.data, n_samples=args.n, max_tokens=args.max_tokens)
    edited_summary, edited_results = evaluate_model(args.edited, args.data, n_samples=args.n, max_tokens=args.max_tokens)
    
    full_report = {
        "base": {"summary": base_summary, "results": base_results},
        "edited": {"summary": edited_summary, "results": edited_results},
        "delta": {
            "acc_delta": edited_summary["acc"] - base_summary["acc"],
            "cut_share_delta": edited_summary["cut_share"] - base_summary["cut_share"],
            "mean_tokens_delta": edited_summary["mean_tokens"] - base_summary["mean_tokens"]
        }
    }
    
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
        
    print(f"\nEvaluation saved to {args.out}")
    print(f"Accuracy: Base {base_summary['acc']*100:.1f}% -> Edited {edited_summary['acc']*100:.1f}% (Delta: +{(edited_summary['acc'] - base_summary['acc'])*100:.1f}%)")
    print(f"Cut Share (Loops): Base {base_summary['cut_share']*100:.1f}% -> Edited {edited_summary['cut_share']*100:.1f}% (Delta: {(edited_summary['cut_share'] - base_summary['cut_share'])*100:.1f}%)")

if __name__ == "__main__":
    main()
