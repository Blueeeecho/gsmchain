"""Prepare SFT v2 training data with CoT responses (matches 8-shot style).

Output: sft_train_v2.jsonl with fields: {id, messages, prompt, response, category}
  - prompt: user-side chat (Q: ...\nA:)
  - response: assistant CoT (Let's think step by step. ... The final answer is N.)
  - matches sft_train.jsonl field schema so train_sft_trl.py works
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path("/home/wwq416/snap/wwq/math-chain")
RAW_TRAIN = ROOT / "chaingsm_data/data/raw/train-00000-of-00001.jsonl"
AUGMENTED = ROOT / "chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/source_augmented_with_traces.jsonl"
OUT = ROOT / "chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/sft_train_v2.jsonl"

SYSTEM_PROMPT = "You are an expert math problem solver. Solve step by step."


def clean_gsm8k_solution(sol: str) -> str:
    s = re.sub(r"<<[^>]+>>", "", sol)
    s = re.sub(r"\n*####.*$", "", s, flags=re.DOTALL)
    return s.strip()


def build_cot_from_raw_train(rec: dict) -> str:
    body = clean_gsm8k_solution(rec["answer"])
    final = rec["final_answer"]
    return f"Let's think step by step. {body} The final answer is {final}."


def build_cot_from_gold_trace(trace: list, final_answer: str) -> str:
    parts = []
    for step in trace:
        desc = step["description"].rstrip(".")
        expr = step["expression"]
        val = step["value"]
        parts.append(f"{desc}, {expr} = {val}.")
    body = " ".join(parts)
    return f"Let's think step by step. {body} The final answer is {final_answer}."


def main() -> None:
    samples = []
    n_orig = 0
    n_var = 0

    # 1) Original: raw GSM8K train
    with RAW_TRAIN.open() as f:
        for line in f:
            r = json.loads(line)
            user = f"Q: {r['question']}\nA:"
            assistant = build_cot_from_raw_train(r)
            samples.append({
                "id": f"{r['base_id']}_original",
                "base_id": r["base_id"],
                "category": "original",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                "prompt": user,
                "response": assistant,
            })
            n_orig += 1

    # 2) Variant: source_augmented_with_traces
    with AUGMENTED.open() as f:
        for line in f:
            r = json.loads(line)
            q = r.get("question_distracted") or r.get("question_original")
            trace = r.get("gold_trace") or []
            final = r.get("answer")
            if not q or not trace or final is None:
                continue
            user = f"Q: {q}\nA:"
            assistant = build_cot_from_gold_trace(trace, final)
            samples.append({
                "id": r["id"],
                "base_id": r["base_id"],
                "category": r["category"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ],
                "prompt": user,
                "response": assistant,
            })
            n_var += 1

    import random
    random.seed(42)
    random.shuffle(samples)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    from collections import Counter
    cat_counts = Counter(s["category"] for s in samples)
    print(f"[sft_v2] wrote {len(samples)} samples -> {OUT}")
    print(f"  original: {n_orig}, variant: {n_var}, total: {len(samples)}")
    print(f"  by category: {dict(cat_counts.most_common())}")
    print()
    print("=== sample (first 2) ===")
    for s in samples[:2]:
        print(f"id={s['id']} cat={s['category']}")
        print(f"  prompt: {s['prompt'][:150]}...")
        print(f"  response: {s['response'][:200]}...")
        print()


if __name__ == "__main__":
    main()
