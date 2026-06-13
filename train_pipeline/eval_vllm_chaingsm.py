from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from gsm_answer_extractor import extract_answer as extract_text_answer
from gsm_answer_extractor import is_correct
from train_pipeline.eval_constants import DEFAULT_TEST_DATA
from train_pipeline.preprocess_chaingsm import SYSTEM_PROMPT, USER_TEMPLATE


@dataclass
class EvalExample:
    id: str
    base_id: str
    category: str
    question: str
    gold_answer: str


DIRECT_SYSTEM_PROMPT = "You are a helpful assistant."
ZERO_SHOT_SYSTEM_PROMPT = "You are a helpful assistant."


def load_examples(data_path: str | Path, limit: int | None = None) -> list[EvalExample]:
    examples = []
    with Path(data_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            category = row["category"]
            question_key = "question_original" if category == "original" else "question_distracted"
            examples.append(
                EvalExample(
                    id=row["id"],
                    base_id=row.get("base_id", ""),
                    category=category,
                    question=row[question_key],
                    gold_answer=str(row["answer"]),
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"No eval examples loaded from {data_path}")
    return examples


def build_messages(method: str, question: str) -> list[dict[str, str]]:
    if method == "train_json_prompt":
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(question=question.strip())},
        ]
    if method == "direct":
        return [
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Q: {question}\nA:"},
        ]
    if method == "zero_shot_cot":
        return [
            {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Q: {question}\nA: Let's think step by step."},
        ]
    raise ValueError(f"Unsupported train-time eval method: {method}")


def fallback_chat_prompt(messages: list[dict[str, str]]) -> str:
    pieces = [f"{message['role'].capitalize()}: {message['content']}" for message in messages]
    pieces.append("Assistant:")
    return "\n".join(pieces)


def build_prompt(tokenizer: Any, method: str, question: str) -> str:
    messages = build_messages(method, question)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return fallback_chat_prompt(messages)


def extract_answer(output: str) -> str | None:
    text = str(output or "").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and parsed.get("answer") is not None:
            return extract_text_answer(str(parsed["answer"]))
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict) and parsed.get("answer") is not None:
                    return extract_text_answer(str(parsed["answer"]))
            except json.JSONDecodeError:
                pass
    return extract_text_answer(output)


def chunked(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cleanup_vllm() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
    except Exception:
        pass


def available_gpu_memory_ratio(safety_margin: float = 0.03) -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if total_bytes <= 0:
            return None
        ratio = (free_bytes / total_bytes) - safety_margin
        return max(ratio, 0.0)
    except Exception:
        return None


def filter_gpu_memory_candidates(
    candidates: list[float],
    min_candidate: float = 0.25,
    safety_margin: float = 0.03,
) -> list[float]:
    max_ratio = available_gpu_memory_ratio(safety_margin=safety_margin)
    if max_ratio is None:
        return candidates
    filtered = [candidate for candidate in candidates if candidate <= max_ratio]
    fallback = max(min(max_ratio, max(candidates)), min_candidate)
    if not filtered:
        filtered = [fallback]
    elif fallback not in filtered and fallback < max(filtered):
        filtered.append(fallback)
    filtered = sorted(set(round(x, 3) for x in filtered), reverse=True)
    print(
        f"[eval] GPU memory precheck: usable_ratio~{max_ratio:.3f}, "
        f"candidates={filtered}",
        flush=True,
    )
    return filtered


def summarize(predictions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})
    overall = {"correct": 0, "total": 0}
    for row in predictions:
        category = row["category"]
        by_category[category]["correct"] += int(row["correct"])
        by_category[category]["total"] += 1
        overall["correct"] += int(row["correct"])
        overall["total"] += 1

    category_rows = []
    for category, stats in sorted(by_category.items()):
        total = stats["total"]
        correct = stats["correct"]
        category_rows.append(
            {
                "category": category,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
            }
        )

    overall_rows = [
        {
            "correct": overall["correct"],
            "total": overall["total"],
            "accuracy": overall["correct"] / overall["total"] if overall["total"] else 0.0,
        }
    ]
    return category_rows, overall_rows


def evaluate_with_vllm(
    model_path: str | Path,
    data_path: str | Path = DEFAULT_TEST_DATA,
    output_dir: str | Path = "eval",
    method: str = "train_json_prompt",
    limit: int | None = None,
    batch_size: int = 64,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    gpu_memory_utilization_candidates: list[float] | None = None,
    dtype: str = "auto",
    seed: int = 42,
    max_model_len: int | None = None,
    max_tokens: int = 2048,
    top_k: int = 1,
    top_p: float = 1.0,
    trust_remote_code: bool = True,
) -> dict[str, Any]:
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = load_examples(data_path, limit)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)
    sampling_params = SamplingParams(temperature=0.0, top_k=top_k, top_p=top_p, max_tokens=max_tokens)
    candidates = gpu_memory_utilization_candidates or [gpu_memory_utilization]
    seen = set()
    candidates = [float(x) for x in candidates if not (float(x) in seen or seen.add(float(x)))]
    candidates = filter_gpu_memory_candidates(candidates)

    predictions: list[dict[str, Any]] = []
    prompts = [build_prompt(tokenizer, method, example.question) for example in examples]
    used_gpu_memory_utilization = None
    attempt_errors = []
    predictions_path = output_dir / "predictions.jsonl"
    # 清空旧的 predictions 文件
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text("")
    try:
        for candidate in candidates:
            predictions = []
            predictions_path.write_text("")
            llm = None
            try:
                llm_kwargs: dict[str, Any] = {
                    "model": str(model_path),
                    "trust_remote_code": trust_remote_code,
                    "tensor_parallel_size": tensor_parallel_size,
                    "gpu_memory_utilization": candidate,
                    "dtype": dtype,
                    "seed": seed,
                }
                if max_model_len is not None:
                    llm_kwargs["max_model_len"] = max_model_len
                print(f"[eval] Loading vLLM with gpu_memory_utilization={candidate}", flush=True)
                llm = LLM(**llm_kwargs)
                for prompt_batch, example_batch in zip(chunked(prompts, batch_size), chunked(examples, batch_size)):
                    outputs = llm.generate(prompt_batch, sampling_params, use_tqdm=False)
                    for example, output in zip(example_batch, outputs):
                        raw_output = output.outputs[0].text if output.outputs else ""
                        pred_answer = extract_answer(raw_output)
                        correct = is_correct(pred_answer, example.gold_answer)
                        row = {
                            "id": example.id,
                            "base_id": example.base_id,
                            "category": example.category,
                            "question": example.question,
                            "gold_answer": example.gold_answer,
                            "raw_output": raw_output,
                            "pred_answer": pred_answer,
                            "correct": correct,
                        }
                        predictions.append(row)
                        # 实时写入 predictions.jsonl
                        append_jsonl(predictions_path, row)
                used_gpu_memory_utilization = candidate
                break
            except Exception as exc:
                attempt_errors.append({"gpu_memory_utilization": candidate, "error": repr(exc)})
                print(f"[eval] vLLM eval failed at gpu_memory_utilization={candidate}: {exc!r}", flush=True)
            finally:
                if llm is not None:
                    del llm
                cleanup_vllm()
                time.sleep(2)
        if used_gpu_memory_utilization is None:
            with (output_dir / "eval_errors.json").open("w", encoding="utf-8") as f:
                json.dump(attempt_errors, f, indent=2, ensure_ascii=False)
            raise RuntimeError(f"vLLM eval failed for all gpu_memory_utilization candidates: {attempt_errors}")
    finally:
        del tokenizer
        cleanup_vllm()

    category_rows, overall_rows = summarize(predictions)
    write_jsonl(
        output_dir / "summary_by_category.jsonl",
        category_rows,
    )
    write_jsonl(
        output_dir / "summary_overall.jsonl",
        overall_rows,
    )
    result = {
        "overall_accuracy": overall_rows[0]["accuracy"],
        "overall": overall_rows,
        "by_category": category_rows,
        "prediction_count": len(predictions),
        "output_dir": str(output_dir),
        "gpu_memory_utilization": used_gpu_memory_utilization,
        "attempt_errors": attempt_errors,
    }
    # 写入 eval_result.json，供子进程调用时传递结果
    with (output_dir / "eval_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one ChainGSM model checkpoint with vLLM.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", default=DEFAULT_TEST_DATA)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", default="train_json_prompt", choices=["train_json_prompt", "direct", "zero_shot_cot"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--gpu-memory-utilization-candidate",
        type=float,
        action="append",
        default=None,
        help="Retry vLLM eval with these gpu memory utilization values.",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")
    args = parser.parse_args()
    result = evaluate_with_vllm(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        method=args.method,
        limit=args.limit,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        gpu_memory_utilization_candidates=args.gpu_memory_utilization_candidate,
        dtype=args.dtype,
        seed=args.seed,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        trust_remote_code=args.trust_remote_code,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
