#!/usr/bin/env python3
"""Evaluate four Qwen Instruct models on ChainGSM with validated 8-shot prompts."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from eval_official_gsm import (  # noqa: E402
    build_lm_eval_llama_messages,
    configure_runtime_env,
    extract_answer,
    is_correct,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = (
    REPO_ROOT / "chaingsm_data" / "data" /  "gsmchain" / "gsm8k_test_clean.jsonl"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "code" / "results" / "chaingsm_base_8shot"

QWEN_MULTITURN_PROFILE = "qwen_multiturn_8shot_chat"
QWEN_MATH_COMPLETION_PROFILE = "qwen_math_completion_8shot"
LLAMA_PROFILE = "llama_lm_eval_multiturn_8shot_single_bos"
EIGHT_SHOT_SYSTEM_PROMPT = (
    "As an expert problem solver, solve step by step the following mathematical questions."
)
EIGHT_SHOT_EXAMPLES = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. How many trees did the grove workers "
        "plant today?",
        "Let's think step by step. There are 15 trees originally. Then there were 21 trees "
        "after some more were planted. So there must have been 21 - 15 = 6. "
        "The final answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are "
        "in the parking lot?",
        "Let's think step by step. There are originally 3 cars. 2 more cars arrive. "
        "3 + 2 = 5. The final answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do "
        "they have left in total?",
        "Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. "
        "So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. "
        "The final answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
        "How many lollipops did Jason give to Denny?",
        "Let's think step by step. Jason started with 20 lollipops. Then he had 12 after "
        "giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
        "How many toys does he have now?",
        "Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from "
        "his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed "
        "each day, from monday to thursday. How many computers are now in the server room?",
        "Let's think step by step. There were originally 9 computers. For each of 4 days, "
        "5 more computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 = 29. The final answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he "
        "lost 2 more. How many golf balls did he have at the end of wednesday?",
        "Let's think step by step. Michael started with 58 golf balls. After losing 23 on "
        "tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. "
        "The final answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be "
        "5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. "
        "The final answer is 8.",
    ),
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path


DEFAULT_MODELS = (
    ModelSpec(
        "Qwen2.5-0.5B-Instruct",
        Path("/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct"),
    ),
    ModelSpec(
        "Qwen2.5-1.5B-Instruct",
        Path("/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-1.5B-Instruct"),
    ),
    ModelSpec(
        "Qwen2.5-Math-1.5B-Instruct",
        Path("/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-Math-1.5B-Instruct"),
    ),
    ModelSpec(
        "Qwen2.5-3B-Instruct",
        Path("/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-3B-Instruct"),
    ),
)


@dataclass(frozen=True)
class EvalExample:
    id: str
    base_id: str
    category: str
    question: str
    gold_answer: str


def select_prompt_profile(model_name: str) -> str:
    lowered = model_name.lower()
    if "llama" in lowered:
        return LLAMA_PROFILE
    if "qwen" in lowered and "math" in lowered:
        return QWEN_MATH_COMPLETION_PROFILE
    if "qwen" in lowered:
        return QWEN_MULTITURN_PROFILE
    raise ValueError(f"No validated 8-shot prompt profile for model: {model_name}")


def build_qwen_messages(question: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": EIGHT_SHOT_SYSTEM_PROMPT}]
    for example_question, example_answer in EIGHT_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": f"Q: {example_question}\nA:"})
        messages.append({"role": "assistant", "content": example_answer})
    messages.append({"role": "user", "content": f"Q: {question}\nA: Let's think step by step."})
    return messages


def build_qwen_math_completion_prompt(question: str) -> str:
    pieces = [
        f"Q: {example_question}\nA: {example_answer}"
        for example_question, example_answer in EIGHT_SHOT_EXAMPLES
    ]
    pieces.append(f"Q: {question}\nA: Let's think step by step.")
    return "\n\n".join(pieces)


def build_llama_messages(question: str) -> list[dict[str, str]]:
    return build_lm_eval_llama_messages(question)


def _as_token_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError(f"Expected one tokenized prompt, got batch size {len(encoded)}")
        encoded = encoded[0]
    return [int(token_id) for token_id in encoded]


def build_model_input(tokenizer: Any, question: str, profile: str) -> str | list[int]:
    if profile == QWEN_MATH_COMPLETION_PROFILE:
        return build_qwen_math_completion_prompt(question)

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"Tokenizer has no chat template for profile {profile}")

    if profile == QWEN_MULTITURN_PROFILE:
        return tokenizer.apply_chat_template(
            build_qwen_messages(question),
            tokenize=False,
            add_generation_prompt=True,
        )

    if profile == LLAMA_PROFILE:
        token_ids = _as_token_ids(
            tokenizer.apply_chat_template(
                build_llama_messages(question),
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is None:
            raise ValueError("Llama tokenizer has no BOS token ID")
        bos_count = token_ids.count(int(bos_token_id))
        if bos_count != 1:
            raise ValueError(f"Llama prompt must contain exactly one BOS token, found {bos_count}")
        return token_ids

    raise ValueError(f"Unknown prompt profile: {profile}")


def stop_sequences_for_profile(profile: str) -> list[str]:
    if profile == QWEN_MULTITURN_PROFILE:
        return ["<|im_end|>"]
    if profile == QWEN_MATH_COMPLETION_PROFILE:
        return ["\nQ:", "\nQuestion:", "<|im_end|>"]
    if profile == LLAMA_PROFILE:
        return [
            "<|eot_id|>",
            "<|start_header_id|>user<|end_header_id|>",
            "Q:",
            "</s>",
            "<|im_end|>",
        ]
    raise ValueError(f"Unknown prompt profile: {profile}")


def load_examples(path: Path, limit: int | None = None) -> list[EvalExample]:
    examples: list[EvalExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            category = str(row["category"])
            question_key = "question_original" if category == "original" else "question_distracted"
            examples.append(
                EvalExample(
                    id=str(row.get("id") or f"row_{line_number:06d}"),
                    base_id=str(row.get("base_id") or ""),
                    category=category,
                    question=str(row[question_key]),
                    gold_answer=str(row["answer"]),
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"No examples loaded from {path}")
    return examples


def chunked(items: Sequence[Any], size: int) -> Sequence[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def append_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def summarize_rows(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total = len(rows)
    correct = sum(int(row["correct"]) for row in rows)
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        stats = grouped[str(row["category"])]
        stats["correct"] += int(row["correct"])
        stats["total"] += 1

    by_category = []
    for category, stats in sorted(grouped.items()):
        by_category.append(
            {
                "category": category,
                "correct": stats["correct"],
                "total": stats["total"],
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
            }
        )
    return (
        {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
            "accuracy_percent": 100.0 * correct / total if total else 0.0,
        },
        by_category,
    )


def write_model_summary(model_dir: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    overall, by_category = summarize_rows(rows)
    summary = {"overall": overall, "by_category": by_category}
    write_json(model_dir / "summary.json", summary)
    return summary


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


def available_gpu_memory_ratio() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if total_bytes <= 0:
            return None
        return free_bytes / total_bytes
    except Exception:
        return None


def filter_gpu_memory_candidates(
    candidates: Sequence[float],
    available_ratio: float | None = None,
    safety_margin: float = 0.03,
    min_candidate: float = 0.1,
) -> list[float]:
    unique = sorted({float(candidate) for candidate in candidates}, reverse=True)
    ratio = available_gpu_memory_ratio() if available_ratio is None else available_ratio
    if ratio is None:
        return unique

    usable_ratio = max(float(ratio) - safety_margin, 0.0)
    filtered = [candidate for candidate in unique if candidate <= usable_ratio]
    if filtered:
        return filtered
    if usable_ratio >= min_candidate:
        return [round(usable_ratio, 3)]
    return []


def evaluate_model(
    spec: ModelSpec,
    examples: Sequence[EvalExample],
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    profile = select_prompt_profile(spec.name)
    model_dir = run_dir / "model_outputs" / safe_name(spec.name)
    model_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = model_dir / "predictions.jsonl"
    existing_rows = read_jsonl(predictions_path)
    completed_ids = {str(row["id"]) for row in existing_rows}
    pending = [example for example in examples if example.id not in completed_ids]

    if not pending:
        print(f"[skip] {spec.name}: all {len(existing_rows)} examples already completed", flush=True)
        return write_model_summary(model_dir, existing_rows)

    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.path),
        trust_remote_code=args.trust_remote_code,
    )
    diagnostic_input = build_model_input(tokenizer, pending[0].question, profile)
    diagnostics = {
        "model_name": spec.name,
        "model_path": str(spec.path),
        "prompt_profile": profile,
        "input_type": "token_ids" if isinstance(diagnostic_input, list) else "text",
        "first_prompt_tokens": (
            len(diagnostic_input)
            if isinstance(diagnostic_input, list)
            else len(tokenizer(diagnostic_input, add_special_tokens=False)["input_ids"])
        ),
        "bos_token_id": tokenizer.bos_token_id,
        "bos_count": (
            diagnostic_input.count(tokenizer.bos_token_id)
            if isinstance(diagnostic_input, list) and tokenizer.bos_token_id is not None
            else None
        ),
        "stop_sequences": stop_sequences_for_profile(profile),
    }
    write_json(model_dir / "prompt_diagnostics.json", diagnostics)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=stop_sequences_for_profile(profile),
        seed=args.seed,
    )
    llm_kwargs: dict[str, Any] = {
        "model": str(spec.path),
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    print(
        f"[start] {spec.name}: profile={profile}, pending={len(pending)}, "
        f"completed={len(existing_rows)}",
        flush=True,
    )
    llm = None
    try:
        llm = LLM(**llm_kwargs)
        completed_now = 0
        for example_batch in chunked(pending, args.batch_size):
            model_inputs = [
                build_model_input(tokenizer, example.question, profile) for example in example_batch
            ]
            outputs = llm.generate(list(model_inputs), sampling_params, use_tqdm=False)
            batch_rows = []
            for example, output in zip(example_batch, outputs, strict=True):
                candidate = output.outputs[0] if output.outputs else None
                raw_output = candidate.text if candidate is not None else ""
                pred_answer = extract_answer(raw_output)
                batch_rows.append(
                    {
                        **asdict(example),
                        "model_name": spec.name,
                        "model_path": str(spec.path),
                        "prompt_profile": profile,
                        "raw_output": raw_output,
                        "pred_answer": pred_answer,
                        "correct": is_correct(pred_answer, example.gold_answer),
                        "finish_reason": getattr(candidate, "finish_reason", None),
                        "stop_reason": getattr(candidate, "stop_reason", None),
                    }
                )
            append_jsonl(predictions_path, batch_rows)
            existing_rows.extend(batch_rows)
            completed_now += len(batch_rows)
            print(
                f"[progress] {spec.name}: {len(existing_rows)}/{len(examples)} "
                f"(this run +{completed_now})",
                flush=True,
            )
    finally:
        if llm is not None:
            del llm
        del tokenizer
        cleanup_vllm()
        time.sleep(args.cleanup_sleep)

    return write_model_summary(model_dir, existing_rows)


def write_combined_summaries(run_dir: Path, specs: Sequence[ModelSpec]) -> None:
    overall_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    for spec in specs:
        model_dir = run_dir / "model_outputs" / safe_name(spec.name)
        rows = read_jsonl(model_dir / "predictions.jsonl")
        if not rows:
            continue
        overall, by_category = summarize_rows(rows)
        profile = select_prompt_profile(spec.name)
        overall_rows.append(
            {
                "model_name": spec.name,
                "model_path": str(spec.path),
                "prompt_profile": profile,
                **overall,
            }
        )
        for row in by_category:
            category_rows.append(
                {
                    "model_name": spec.name,
                    "model_path": str(spec.path),
                    "prompt_profile": profile,
                    **row,
                }
            )

    write_json(run_dir / "summary_overall.json", overall_rows)
    write_json(run_dir / "summary_by_category.json", category_rows)
    _write_csv(run_dir / "summary_overall.csv", overall_rows)
    _write_csv(run_dir / "summary_by_category.csv", category_rows)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_model_specs(values: Sequence[str]) -> list[ModelSpec]:
    if not values:
        return list(DEFAULT_MODELS)
    defaults = {spec.name: spec for spec in DEFAULT_MODELS}
    specs: list[ModelSpec] = []
    for value in values:
        if "=" in value:
            name, raw_path = value.split("=", 1)
            specs.append(ModelSpec(name=name, path=Path(raw_path).expanduser()))
        elif value in defaults:
            specs.append(defaults[value])
        else:
            path = Path(value).expanduser()
            specs.append(ModelSpec(name=path.name, path=path))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Reuse this directory to resume an interrupted run.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model name, path, or name=/path. Repeat to select a subset; default runs four Qwen models.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Use one fixed ratio. By default, try an automatically filtered candidate list.",
    )
    parser.add_argument(
        "--gpu-memory-utilization-candidate",
        type=float,
        action="append",
        default=[],
        help="Candidate ratio for automatic retry. Repeat to provide a custom list.",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--cleanup-sleep", type=float, default=2.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _candidate_values(args: argparse.Namespace) -> list[float]:
    if args.gpu_memory_utilization is not None:
        requested = [args.gpu_memory_utilization]
    elif args.gpu_memory_utilization_candidate:
        requested = args.gpu_memory_utilization_candidate
    else:
        requested = [0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25]
    available = available_gpu_memory_ratio()
    filtered = filter_gpu_memory_candidates(requested, available_ratio=available)
    if available is not None:
        print(
            f"[memory] free_ratio={available:.3f}, candidates={filtered}",
            flush=True,
        )
    if not filtered:
        raise RuntimeError(
            "GPU free-memory ratio is too low for the minimum safe vLLM candidate. "
            "Wait for other GPU jobs to finish."
        )
    return filtered


def _worker_command(
    args: argparse.Namespace,
    spec: ModelSpec,
    run_dir: Path,
    candidate: float,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--run-dir",
        str(run_dir),
        "--data-path",
        str(args.data_path),
        "--model",
        f"{spec.name}={spec.path}",
        "--batch-size",
        str(args.batch_size),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(candidate),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--max-tokens",
        str(args.max_tokens),
        "--cleanup-sleep",
        str(args.cleanup_sleep),
        "--fail-fast",
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.max_model_len is not None:
        command.extend(["--max-model-len", str(args.max_model_len)])
    if args.enforce_eager:
        command.append("--enforce-eager")
    if not args.trust_remote_code:
        command.append("--no-trust-remote-code")
    return command


def run_model_in_subprocess(
    args: argparse.Namespace,
    spec: ModelSpec,
    run_dir: Path,
) -> None:
    model_dir = run_dir / "model_outputs" / safe_name(spec.name)
    predictions_path = model_dir / "predictions.jsonl"
    if len(read_jsonl(predictions_path)) >= len(load_examples(args.data_path, args.limit)):
        print(f"[skip] {spec.name}: already complete", flush=True)
        return

    attempts: list[dict[str, Any]] = []
    for candidate in _candidate_values(args):
        print(
            f"[launcher] {spec.name}: trying gpu_memory_utilization={candidate}",
            flush=True,
        )
        command = _worker_command(args, spec, run_dir, candidate)
        proc = subprocess.run(command)
        if proc.returncode == 0:
            print(
                f"[launcher] {spec.name}: completed with gpu_memory_utilization={candidate}",
                flush=True,
            )
            return
        attempts.append(
            {
                "gpu_memory_utilization": candidate,
                "returncode": proc.returncode,
            }
        )
        print(
            f"[launcher] {spec.name}: candidate {candidate} failed; retrying in a fresh process",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(args.cleanup_sleep)

    raise RuntimeError(f"All GPU memory candidates failed for {spec.name}: {attempts}")


def main() -> int:
    args = parse_args()
    configure_runtime_env(sys.executable)
    specs = parse_model_specs(args.model)
    examples = load_examples(args.data_path, args.limit)
    run_dir = args.run_dir or (
        args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    missing = [str(spec.path) for spec in specs if not spec.path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Model directories not found: {missing}")

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(args.data_path),
        "example_count": len(examples),
        "models": [
            {
                "model_name": spec.name,
                "model_path": str(spec.path),
                "prompt_profile": select_prompt_profile(spec.name),
            }
            for spec in specs
        ],
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "runtime": {
            "batch_size": args.batch_size,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "gpu_memory_utilization_candidates": (
                args.gpu_memory_utilization_candidate
                or [0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25]
            ),
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "enforce_eager": args.enforce_eager,
        },
        "resume_command": (
            f"{sys.executable} {Path(__file__).resolve()} --run-dir {run_dir}"
        ),
    }
    if not args.worker:
        write_json(run_dir / "run_config.json", run_config)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Examples: {len(examples)}", flush=True)

    if args.worker:
        if len(specs) != 1:
            raise ValueError("Worker mode requires exactly one model")
        if args.gpu_memory_utilization is None:
            raise ValueError("Worker mode requires one fixed --gpu-memory-utilization")
        evaluate_model(specs[0], examples, run_dir, args)
        write_combined_summaries(run_dir, specs)
        return 0

    for spec in specs:
        try:
            run_model_in_subprocess(args, spec, run_dir)
        except Exception as exc:
            append_jsonl(
                run_dir / "errors.jsonl",
                [
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "model_name": spec.name,
                        "model_path": str(spec.path),
                        "error": repr(exc),
                    }
                ],
            )
            print(f"[error] {spec.name}: {exc!r}", file=sys.stderr, flush=True)
            if args.fail_fast:
                raise
        finally:
            write_combined_summaries(run_dir, specs)

    print(f"Results written to {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
