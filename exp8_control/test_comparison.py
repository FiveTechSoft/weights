import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

class PresencePenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty == 0.0:
            return scores
        # scores: (batch_size, vocab_size)
        # input_ids: (batch_size, cur_seq_len)
        for i in range(input_ids.shape[0]):
            unique_tokens = torch.unique(input_ids[i])
            scores[i, unique_tokens] -= self.penalty
        return scores

def run_test():
    prompt_text = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nPlease reason step by step, and put your final answer within \\boxed{}."
    
    tok = AutoTokenizer.from_pretrained("exp8_control/base")
    msg = [{"role": "user", "content": prompt_text}]
    prompt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    
    print("=== PROMPT ===")
    print(prompt)
    
    inputs = tok(prompt, return_tensors="pt")
    processors = LogitsProcessorList([PresencePenaltyLogitsProcessor(1.5)])
    
    # Test BASE
    print("\n--- RUNNING BASE MODEL ---")
    base_model = AutoModelForCausalLM.from_pretrained("exp8_control/base", dtype=torch.bfloat16)
    torch.manual_seed(0)
    t0 = time.time()
    with torch.no_grad():
        out_base = base_model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            logits_processor=processors,
            pad_token_id=tok.eos_token_id
        )
    dt_base = time.time() - t0
    base_text = tok.decode(out_base[0][inputs.input_ids.shape[1]:])
    print(f"Base Output ({dt_base:.2f}s, {len(out_base[0])-inputs.input_ids.shape[1]} tokens):")
    print(base_text)
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Test EDITED
    print("\n--- RUNNING EDITED MODEL (p50c) ---")
    edited_model = AutoModelForCausalLM.from_pretrained("exp8_control/edited", dtype=torch.bfloat16)
    torch.manual_seed(0)
    t0 = time.time()
    with torch.no_grad():
        out_edited = edited_model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            logits_processor=processors,
            pad_token_id=tok.eos_token_id
        )
    dt_edited = time.time() - t0
    edited_text = tok.decode(out_edited[0][inputs.input_ids.shape[1]:])
    print(f"Edited Output ({dt_edited:.2f}s, {len(out_edited[0])-inputs.input_ids.shape[1]} tokens):")
    print(edited_text)

if __name__ == "__main__":
    run_test()
