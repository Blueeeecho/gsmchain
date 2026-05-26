#!/usr/bin/env python3
"""Evaluate local instruct models on ChainGSM variants with vLLM."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "chaingsm_data" / "data" / "final" / "test.jsonl"
DEFAULT_MODEL_ROOT = Path("/home/wwq416/snap/wwq/model")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "code" / "results" / "chaingsm_test"
DEFAULT_EXCLUDED_MODEL_KEYWORDS = ("ministral",)

DIRECT_SYSTEM_PROMPT = "You are a helpful assistant."
ZERO_SHOT_SYSTEM_PROMPT = "You are a helpful assistant."
EIGHT_SHOT_SYSTEM_PROMPT = (
    "As an expert problem solver, solve step by step the following mathematical questions."
)

EIGHT_SHOT_USER_PREFIX = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The final answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Let's think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8.
"""

METHODS = ("direct", "zero_shot_cot", "eight_shot_cot")
SAMPLING_CONFIG = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 512,
}

NUMBER_PATTERN = re.compile(
    r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|(?:\d+))(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?"
)


@dataclass
class EvalExample:
    id: str
    base_id: str
    category: str
    question: str
    gold_answer: str


@dataclass
class PredictionRecord:
    model_name: str
    model_path: str
    method: str
    id: str
    base_id: str
    category: str
    question: str
    gold_answer: str
    raw_output: str
    pred_answer: str | None
    correct: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate local instruct models on ChainGSM test data."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--model-filter",
        action="append",
        default=[],
        help="Case-insensitive substring filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        action="append",
        default=[],
        help="Method to run. Defaults to all methods. Can be passed multiple times.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit examples.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_false",
        dest="trust_remote_code",
    )
    parser.add_argument("--max-model-len", type=int, default=None)
    return parser.parse_args()


def load_examples(data_path: Path, limit: int | None = None) -> list[EvalExample]:
    examples: list[EvalExample] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
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
        raise ValueError(f"No examples loaded from {data_path}")
    return examples


def discover_instruct_models(model_root: Path, filters: Iterable[str]) -> list[Path]:
    filters_lower = [item.lower() for item in filters]
    model_paths: list[Path] = []
    for config_path in sorted(model_root.rglob("config.json")):
        model_path = config_path.parent
        model_name = model_path.name
        model_path_lower = str(model_path).lower()
        if "instruct" not in model_name.lower():
            continue
        if any(keyword in model_path_lower for keyword in DEFAULT_EXCLUDED_MODEL_KEYWORDS):
            continue
        if filters_lower and not any(item in model_path_lower for item in filters_lower):
            continue
        model_paths.append(model_path)
    if not model_paths:
        raise ValueError(f"No instruct models found under {model_root}")
    return model_paths


def build_messages(method: str, question: str) -> list[dict[str, str]]:
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
    if method == "eight_shot_cot":
        user_prompt = (
            f"{EIGHT_SHOT_USER_PREFIX}\n"
            f"Q: {question}\n"
            "A: Let's think step by step."
        )
        return [
            {"role": "system", "content": EIGHT_SHOT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    raise ValueError(f"Unknown method: {method}")


def fallback_chat_prompt(messages: list[dict[str, str]]) -> str:
    pieces: list[str] = []
    for message in messages:
        role = message["role"].capitalize()
        pieces.append(f"{role}: {message['content']}")
    pieces.append("Assistant:")
    return "\n".join(pieces)


def build_prompt(tokenizer: Any, method: str, question: str) -> str:
    messages = build_messages(method, question)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return fallback_chat_prompt(messages)


def extract_number(text: str) -> str | None:
    matches = NUMBER_PATTERN.findall(text.replace(",", ""))
    return matches[-1].strip() if matches else None


def extract_answer(output: str) -> str | None:
    patterns = [
        re.compile(r"The\s+final\s+answer\s+is\s*[:=]?\s*([^\n\.]+)", re.IGNORECASE),
        re.compile(r"####\s*([^\n]+)"),
        re.compile(r"\\boxed\s*\{([^{}]+)\}"),
        re.compile(r"boxed\s*\{([^{}]+)\}", re.IGNORECASE),
    ]
    for pattern in patterns:
        match = pattern.search(output)
        if match:
            number = extract_number(match.group(1))
            if number is not None:
                return number
    return extract_number(output)


def normalize_answer(answer: str | None) -> Fraction | None:
    if answer is None:
        return None
    value = answer.strip()
    value = value.replace(",", "").replace("$", "").replace("%", "")
    value = value.rstrip(".。;:，,")
    value = re.sub(r"\s+", "", value)
    if not value:
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        number = extract_number(value)
        if number is None:
            return None
        try:
            return Fraction(number.replace(" ", ""))
        except (ValueError, ZeroDivisionError):
            return None


def is_correct(pred_answer: str | None, gold_answer: str, tolerance: float = 1e-6) -> bool:
    pred_value = normalize_answer(pred_answer)
    gold_value = normalize_answer(gold_answer)
    if pred_value is None or gold_value is None:
        return False
    tolerance_value = Fraction(str(tolerance))
    diff = abs(pred_value - gold_value)
    allowed = max(tolerance_value, tolerance_value * abs(gold_value))
    return diff <= allowed


def chunked(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def write_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def safe_path_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[PredictionRecord]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_category: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    overall: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {"correct": 0, "total": 0})

    for record in records:
        category_key = (record.model_name, record.method, record.category)
        overall_key = (record.model_name, record.method)
        by_category[category_key]["correct"] += int(record.correct)
        by_category[category_key]["total"] += 1
        overall[overall_key]["correct"] += int(record.correct)
        overall[overall_key]["total"] += 1

    category_rows = []
    for (model_name, method, category), stats in sorted(by_category.items()):
        total = stats["total"]
        correct = stats["correct"]
        category_rows.append(
            {
                "model_name": model_name,
                "method": method,
                "category": category,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
            }
        )

    overall_rows = []
    for (model_name, method), stats in sorted(overall.items()):
        total = stats["total"]
        correct = stats["correct"]
        overall_rows.append(
            {
                "model_name": model_name,
                "method": method,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
            }
        )

    return category_rows, overall_rows


def save_summaries(output_dir: Path, records: list[PredictionRecord]) -> None:
    category_rows, overall_rows = summarize(records)
    write_json(output_dir / "summary_by_category.json", category_rows)
    write_json(output_dir / "summary_overall.json", overall_rows)
    write_csv(
        output_dir / "summary_by_category.csv",
        category_rows,
        ["model_name", "method", "category", "correct", "total", "accuracy"],
    )
    write_csv(
        output_dir / "summary_overall.csv",
        overall_rows,
        ["model_name", "method", "correct", "total", "accuracy"],
    )


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


def load_llm(args: argparse.Namespace, model_path: Path) -> LLM:
    kwargs: dict[str, Any] = {
        "model": str(model_path),
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "seed": args.seed,
    }
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    return LLM(**kwargs)


def evaluate_model(
    args: argparse.Namespace,
    model_path: Path,
    methods: list[str],
    examples: list[EvalExample],
    output_dir: Path,
    records: list[PredictionRecord],
) -> None:
    model_name = model_path.name
    model_output_dir = output_dir / "model_outputs" / safe_path_name(model_name)
    errors_path = model_output_dir / "errors.jsonl"

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=args.trust_remote_code,
        )
        llm = load_llm(args, model_path)
    except Exception as exc:
        write_jsonl_record(
            errors_path,
            {
                "stage": "load_model",
                "model_name": model_name,
                "model_path": str(model_path),
                "error": repr(exc),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        )
        cleanup_vllm()
        return

    sampling_params = SamplingParams(**SAMPLING_CONFIG)
    try:
        for method in tqdm(methods, desc=f"{model_name} methods", leave=False):
            method_output_dir = model_output_dir / safe_path_name(method)
            method_output_dir.mkdir(parents=True, exist_ok=True)
            predictions_path = method_output_dir / "predictions.jsonl"
            method_errors_path = method_output_dir / "errors.jsonl"
            prompts = [build_prompt(tokenizer, method, example.question) for example in examples]
            try:
                for prompt_batch, example_batch in tqdm(
                    zip(chunked(prompts, args.batch_size), chunked(examples, args.batch_size)),
                    total=math.ceil(len(examples) / args.batch_size),
                    desc=f"{model_name} {method}",
                    leave=False,
                ):
                    outputs = llm.generate(prompt_batch, sampling_params, use_tqdm=False)
                    for example, output in zip(example_batch, outputs):
                        raw_output = output.outputs[0].text if output.outputs else ""
                        pred_answer = None
                        correct = False
                        try:
                            pred_answer = extract_answer(raw_output)
                            correct = is_correct(pred_answer, example.gold_answer)
                        except Exception as exc:
                            write_jsonl_record(
                                method_errors_path,
                                {
                                    "stage": "score_example",
                                    "model_name": model_name,
                                    "model_path": str(model_path),
                                    "method": method,
                                    "id": example.id,
                                    "base_id": example.base_id,
                                    "category": example.category,
                                    "gold_answer": example.gold_answer,
                                    "pred_answer": pred_answer,
                                    "error": repr(exc),
                                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                                },
                            )
                        record = PredictionRecord(
                            model_name=model_name,
                            model_path=str(model_path),
                            method=method,
                            id=example.id,
                            base_id=example.base_id,
                            category=example.category,
                            question=example.question,
                            gold_answer=example.gold_answer,
                            raw_output=raw_output,
                            pred_answer=pred_answer,
                            correct=correct,
                        )
                        records.append(record)
                        write_jsonl_record(predictions_path, asdict(record))
            except Exception as exc:
                write_jsonl_record(
                    method_errors_path,
                    {
                        "stage": "generate",
                        "model_name": model_name,
                        "model_path": str(model_path),
                        "method": method,
                        "error": repr(exc),
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                write_jsonl_record(
                    errors_path,
                    {
                        "stage": "generate",
                        "model_name": model_name,
                        "model_path": str(model_path),
                        "method": method,
                        "error": repr(exc),
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                continue
    except Exception as exc:
        write_jsonl_record(
            errors_path,
            {
                "stage": "model_loop",
                "model_name": model_name,
                "model_path": str(model_path),
                "error": repr(exc),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        )
    finally:
        del llm
        del tokenizer
        cleanup_vllm()
        save_summaries(output_dir, records)


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = args.method or list(METHODS)
    examples = load_examples(args.data_path, args.limit)
    model_paths = discover_instruct_models(args.model_root, args.model_filter)

    run_config = {
        "timestamp": timestamp,
        "data_path": str(args.data_path),
        "model_root": str(args.model_root),
        "output_dir": str(output_dir),
        "models": [{"model_name": path.name, "model_path": str(path)} for path in model_paths],
        "excluded_model_keywords": list(DEFAULT_EXCLUDED_MODEL_KEYWORDS),
        "process_file_layout": "model_outputs/<model_name>/<method>/predictions.jsonl",
        "methods": methods,
        "sampling": SAMPLING_CONFIG,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "seed": args.seed,
        "trust_remote_code": args.trust_remote_code,
        "max_model_len": args.max_model_len,
        "example_count": len(examples),
    }
    write_json(output_dir / "run_config.json", run_config)

    print(f"Output dir: {output_dir}", flush=True)
    print(f"Loaded {len(examples)} examples from {args.data_path}", flush=True)
    print(f"Discovered {len(model_paths)} instruct models", flush=True)

    records: list[PredictionRecord] = []
    for model_path in tqdm(model_paths, desc="models"):
        evaluate_model(args, model_path, methods, examples, output_dir, records)

    save_summaries(output_dir, records)
    print(f"Done. Results written to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
