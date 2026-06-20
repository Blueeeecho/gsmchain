"""Evaluate a SFT model on a ChainGSM test set using the same messages
protocol as all_sft.jsonl (system + STEP-template user). The test set
provides question_original / question_distracted / answer / category /
id; we render each example with the project's prompt template and let
the model generate a STEP-format response, then extract the final
"ANSWER: N" line for scoring.

Reuses gsm_answer_extractor + the chat_template from the model dir.
Writes predictions.jsonl and summary_*.jsonl incrementally so progress
is observable on long test sets.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))
from gsm_answer_extractor import extract_answer as extract_text_answer  # noqa: E402
from gsm_answer_extractor import is_correct  # noqa: E402

DEFAULT_TEST_DATA = "/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl"
DEFAULT_TRAIN_DATA = "/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/sft/all_sft.jsonl"


@dataclass
class EvalExample:
    id: str
    base_id: str
    category: str
    question: str
    gold_answer: str


def derive_system_prompt_from_train(train_jsonl: str) -> str:
    with open(train_jsonl) as f:
        for line in f:
            o = json.loads(line)
            return o["messages"][0]["content"]
    raise RuntimeError(f"Could not derive system prompt from {train_jsonl}")


def derive_user_template_from_train(train_jsonl: str) -> str:
    """Pull the user template (everything up to 'Problem:\\n') from the first
    row of all_sft.jsonl, so the eval-time template cannot drift from training.
    """
    with open(train_jsonl) as f:
        for line in f:
            o = json.loads(line)
            content = o["messages"][1]["content"]
            idx = content.find("Problem:\n")
            if idx < 0:
                continue
            return content[: idx + len("Problem:\n")] + "{question}\n"
    raise RuntimeError(f"Could not derive user template from {train_jsonl}")


def load_examples(data_path: str, limit: int | None = None) -> list[EvalExample]:
    out = []
    with open(data_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cat = row["category"]
            q = row["question_original"] if cat == "original" else row["question_distracted"]
            out.append(EvalExample(
                id=row["id"], base_id=row.get("base_id", ""),
                category=cat, question=q, gold_answer=str(row["answer"]),
            ))
            if limit is not None and len(out) >= limit:
                break
    return out


def build_messages(system_prompt: str, user_template: str, question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(question=question.strip())},
    ]


def extract_pred_answer(output: str) -> str | None:
    return extract_text_answer(output)


def chunked(items, n):
    for i in range(0, len(items), n):
        yield items[i: i + n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-path", default=DEFAULT_TEST_DATA)
    ap.add_argument("--train-data", default=DEFAULT_TRAIN_DATA,
                    help="all_sft.jsonl to derive the prompt template from.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    ap.add_argument("--gpu-memory-utilization-candidate", action="append", type=float, default=None)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--no-trust-remote-code", action="store_true")
    args = ap.parse_args()

    system_prompt = derive_system_prompt_from_train(args.train_data)
    user_template = derive_user_template_from_train(args.train_data)
    print(f"system prompt (head): {system_prompt[:120]!r}...")
    print(f"user template (tail): ...{user_template[-160:]!r}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=not args.no_trust_remote_code)
    examples = load_examples(args.data_path, args.limit)
    print(f"loaded {len(examples)} test examples from {args.data_path}")

    prompts = []
    for ex in examples:
        msgs = build_messages(system_prompt, user_template, ex.question)
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    sp = SamplingParams(
        temperature=0.0, top_k=args.top_k, top_p=args.top_p,
        max_tokens=args.max_tokens, seed=args.seed,
    )

    candidates = args.gpu_memory_utilization_candidate or [args.gpu_memory_utilization]
    llm = None
    last_err: Exception | None = None
    for gm in candidates:
        try:
            llm = LLM(
                model=args.model_path,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=gm,
                dtype=args.dtype,
                max_model_len=args.max_model_len,
                trust_remote_code=not args.no_trust_remote_code,
                enforce_eager=True,
                seed=args.seed,
            )
            print(f"vLLM loaded with gpu_memory_utilization={gm}", flush=True)
            last_err = None
            break
        except Exception as e:
            print(f"vLLM load failed with gpu_memory_utilization={gm}: {e}", flush=True)
            last_err = e
    if llm is None:
        raise RuntimeError(f"All gpu_memory_utilization candidates failed: {last_err}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary_overall.jsonl"
    latest_path = output_dir / "latest_metrics.json"
    bycat_path = output_dir / "summary_by_category.jsonl"
    for f in (preds_path, summary_path, bycat_path, latest_path):
        if f.exists():
            f.unlink()

    cat_total: Counter = Counter()
    cat_correct: Counter = Counter()
    overall_correct = 0
    overall_total = 0

    def write_summary(verbose: bool) -> None:
        by_category = [
            {
                "category": cat, "total": cat_total[cat], "correct": cat_correct[cat],
                "accuracy": (cat_correct[cat] / cat_total[cat]) if cat_total[cat] else 0.0,
            }
            for cat in sorted(cat_total)
        ]
        summary_overall = {
            "model_path": args.model_path, "data_path": args.data_path,
            "total": overall_total, "correct": overall_correct,
            "accuracy": (overall_correct / overall_total) if overall_total else 0.0,
            "by_category": by_category,
            "gpu_memory_utilization": gm,
            "max_tokens": args.max_tokens, "seed": args.seed,
        }
        with summary_path.open("w") as f:
            f.write(json.dumps(summary_overall, ensure_ascii=False) + "\n")
        with bycat_path.open("w") as f:
            for row in by_category:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with latest_path.open("w") as f:
            json.dump(summary_overall, f, indent=2, ensure_ascii=False)
        if verbose:
            print(
                f"[eval] progress overall={overall_correct}/{overall_total} "
                f"acc={(overall_correct / overall_total) if overall_total else 0.0:.4f}",
                flush=True,
            )

    n_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    print(f"[eval] running {len(prompts)} prompts in {n_batches} batches of {args.batch_size}", flush=True)
    with preds_path.open("a") as f_preds:
        batch_idx = 0
        for batch, ex_batch in zip(chunked(prompts, args.batch_size), chunked(examples, args.batch_size)):
            batch_idx += 1
            outs = llm.generate(batch, sp, use_tqdm=False)
            for out, ex in zip(outs, ex_batch):
                text = out.outputs[0].text
                pred = extract_pred_answer(text)
                correct = is_correct(pred, ex.gold_answer)
                cat_total[ex.category] += 1
                cat_correct[ex.category] += int(correct)
                overall_total += 1
                overall_correct += int(correct)
                f_preds.write(json.dumps({
                    "id": ex.id, "base_id": ex.base_id, "category": ex.category,
                    "question": ex.question, "gold_answer": ex.gold_answer,
                    "raw_output": text, "pred_answer": pred, "correct": correct,
                }, ensure_ascii=False) + "\n")
            f_preds.flush()
            write_summary(verbose=(batch_idx == n_batches or batch_idx % 4 == 0))

    print(f"\n=== overall: {overall_correct}/{overall_total} = {overall_correct / overall_total:.4f} ===")
    for cat in sorted(cat_total):
        total = cat_total[cat]
        cor = cat_correct[cat]
        print(f"  {cat:30s} {cor:4d}/{total:4d}  {cor / total:.4f}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
