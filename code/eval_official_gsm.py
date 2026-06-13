#!/usr/bin/env python3
"""Reproduce official GSM8K/GSM-Plus 8-shot CoT baselines with local models."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from gsm_answer_extractor import (
    extract_answer,
    extract_balanced_brace_contents,
    extract_conclusion_answer,
    extract_first_number,
    extract_number,
    is_correct,
    normalize_answer,
    normalize_numeric_text,
    split_answer_sentences,
    truncate_generated_continuation,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GSM8K_LOCAL = REPO_ROOT / "chaingsm_data" / "data" / "raw" / "test-00000-of-00001.jsonl"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "code" / "results" / "official_gsm"
DEFAULT_MODELS = {
    "Qwen2.5-0.5B-Instruct": Path("/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct"),
}
SUPPORTED_PROMPT_STYLES = ("chat", "lm_eval_llama_chat_multiturn")

GSMPLUS_TYPES = {
    "rephrase": {"problem understanding"},
    "distract": {"distraction insertion"},
}


def configure_runtime_env(python_executable: str | Path | None = None) -> None:
    """Mirror the repo's local vLLM/FlashInfer runtime environment."""
    executable = Path(python_executable or sys.executable)
    env_bin = executable.parent
    env_root = env_bin.parent
    current_path = os.environ.get("PATH", "")
    path_parts = current_path.split(":") if current_path else []
    if str(env_bin) not in path_parts:
        os.environ["PATH"] = f"{env_bin}:{current_path}" if current_path else str(env_bin)
    os.environ.setdefault("CUDA_HOME", str(env_root))
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "12.0f")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    lib_dir = env_root / "lib"
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = current_ld.split(":") if current_ld else []
    if lib_dir.exists() and str(lib_dir) not in ld_parts:
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{current_ld}" if current_ld else str(lib_dir)
    libstdcxx = lib_dir / "libstdc++.so.6"
    if libstdcxx.exists():
        try:
            ctypes.CDLL(str(libstdcxx), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

FEWSHOT_EXAMPLES = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 21 trees now and there were 15 trees before. So the workers planted "
        "21 - 15 = 6 trees.\n#### 6",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are 3 cars at first. 2 more cars arrive. Now there are 3 + 2 = 5 cars.\n#### 5",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Leah and her sister had 32 + 42 = 74 chocolates. After eating 35, they have "
        "74 - 35 = 39 chocolates left.\n#### 39",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
        "How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops and now has 12. He gave away 20 - 12 = 8 lollipops.\n#### 8",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
        "How many toys does he have now?",
        "Shawn started with 5 toys. He got 2 from his mom and 2 from his dad, so he has "
        "5 + 2 + 2 = 9 toys.\n#### 9",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, "
        "from Monday to Thursday. How many computers are now in the server room?",
        "Five computers were installed on each of 4 days, so 5 * 4 = 20 computers were added. "
        "There are now 9 + 20 = 29 computers.\n#### 29",
    ),
    (
        "Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of Wednesday?",
        "Michael lost 23 + 2 = 25 golf balls. He had 58 - 25 = 33 golf balls left.\n#### 33",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Five bagels cost 5 * 3 = 15 dollars. Olivia has 23 - 15 = 8 dollars left.\n#### 8",
    ),
]

LM_EVAL_COT_EXAMPLES = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
        "How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
        "So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
        "How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. "
        "5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, "
        "from monday to thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. "
        "So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
        "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
]

LM_EVAL_COT_FINAL_ANSWERS = ["6", "5", "39", "8", "9", "29", "33", "8"]
LM_EVAL_LLAMA_DOC_TEMPLATE = (
    "Given the following problem, reason and give a final answer to the problem.\n"
    "Problem: {question}\n"
    'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n'
)


@dataclass
class EvalRow:
    id: str
    split: str
    question: str
    answer: str
    perturbation_type: str = ""


def build_fewshot_prompt(question: str) -> str:
    parts: list[str] = []
    for shot_question, shot_answer in FEWSHOT_EXAMPLES:
        parts.append(f"Question: {shot_question}\nAnswer: Let's think step by step. {shot_answer}")
    answer_prefix = "Answer: Let's think step by step."
    parts.append(f"Question: {question.strip()}\n{answer_prefix}")
    return "\n\n".join(parts)


def build_lm_eval_llama_messages(question: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for (shot_question, shot_answer), final_answer in zip(LM_EVAL_COT_EXAMPLES, LM_EVAL_COT_FINAL_ANSWERS):
        target = re.sub(
            r"\s*The answer is [-+]?\d+(?:\.\d+)?\.\s*$",
            f" The final answer is {final_answer}",
            shot_answer,
        )
        messages.append(
            {
                "role": "user",
                "content": LM_EVAL_LLAMA_DOC_TEMPLATE.format(question=shot_question),
            }
        )
        messages.append({"role": "assistant", "content": target})
    messages.append(
        {
            "role": "user",
            "content": LM_EVAL_LLAMA_DOC_TEMPLATE.format(question=question.strip()),
        }
    )
    return messages


def build_qwen_chat_prompt(tokenizer: Any, question: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("The Qwen chat baseline requires a tokenizer chat template")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": build_fewshot_prompt(question)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_model_input(tokenizer: Any, question: str, prompt_style: str) -> str | list[int]:
    if prompt_style not in SUPPORTED_PROMPT_STYLES:
        raise ValueError(
            f"Unsupported prompt style: {prompt_style}. "
            f"Choose one of: {', '.join(SUPPORTED_PROMPT_STYLES)}"
        )
    if prompt_style == "lm_eval_llama_chat_multiturn":
        encoded = tokenizer.apply_chat_template(
            build_lm_eval_llama_messages(question),
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(encoded, Mapping):
            return list(encoded["input_ids"])
        return list(encoded)
    return build_qwen_chat_prompt(tokenizer, question)


def load_gsm8k_jsonl(path: Path, limit: int | None = None) -> list[EvalRow]:
    rows: list[EvalRow] = []
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                EvalRow(
                    id=str(item.get("base_id") or item.get("id") or f"gsm8k_{index:06d}"),
                    split="original",
                    question=str(item["question"]),
                    answer=str(item.get("final_answer") or extract_answer(item.get("answer")) or item["answer"]),
                )
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_hf_gsm8k(limit: int | None = None) -> list[EvalRow]:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    rows = [
        EvalRow(id=f"gsm8k_test_{idx:06d}", split="original", question=row["question"], answer=row["answer"])
        for idx, row in enumerate(dataset)
    ]
    return rows[:limit] if limit is not None else rows


def load_gsmplus_rows(raw_rows: Iterable[dict[str, Any]], split_name: str, limit: int | None = None) -> list[EvalRow]:
    wanted = GSMPLUS_TYPES[split_name]
    rows: list[EvalRow] = []
    for index, item in enumerate(raw_rows):
        perturbation_type = str(item.get("perturbation_type", "")).strip()
        if perturbation_type not in wanted:
            continue
        rows.append(
            EvalRow(
                id=f"gsmplus_{split_name}_{len(rows):06d}",
                split=split_name,
                question=str(item["question"]),
                answer=str(item["answer"]),
                perturbation_type=perturbation_type,
            )
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def load_hf_gsmplus(split_name: str, limit: int | None = None, mini: bool = False) -> list[EvalRow]:
    from datasets import load_dataset

    hf_split = "testmini" if mini else "test"
    return load_gsmplus_rows(load_dataset("qintongli/GSM-Plus", split=hf_split), split_name, limit=limit)


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


def chunked(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(1 for row in predictions if row["correct"])
    total = len(predictions)
    by_type = Counter(row.get("perturbation_type") or row["split"] for row in predictions)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "accuracy_percent": 100 * correct / total if total else 0.0,
        "counts_by_type": dict(by_type),
    }


def evaluate_model(
    model_name: str,
    model_path: Path,
    rows: list[EvalRow],
    output_dir: Path,
    batch_size: int,
    gpu_memory_utilization: float,
    dtype: str,
    seed: int,
    max_model_len: int | None,
    prompt_style: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)
    prompts = [build_model_input(tokenizer, row.question, prompt_style=prompt_style) for row in rows]
    if prompt_style == "lm_eval_llama_chat_multiturn":
        stops = ["<|eot_id|>", "<|start_header_id|>user<|end_header_id|>", "Q:", "</s>", "<|im_end|>"]
    else:
        stops = None
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512, stop=stops)
    llm_kwargs: dict[str, Any] = {
        "model": str(model_path),
        "trust_remote_code": trust_remote_code,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": gpu_memory_utilization,
        "dtype": dtype,
        "seed": seed,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len

    predictions: list[dict[str, Any]] = []
    llm = None
    try:
        llm = LLM(**llm_kwargs)
        for prompt_batch, row_batch in zip(chunked(prompts, batch_size), chunked(rows, batch_size)):
            outputs = llm.generate(list(prompt_batch), sampling_params, use_tqdm=False)
            for row, output in zip(row_batch, outputs):
                raw_output = output.outputs[0].text if output.outputs else ""
                pred_answer = extract_answer(raw_output)
                predictions.append(
                    {
                        **asdict(row),
                        "model_name": model_name,
                        "model_path": str(model_path),
                        "raw_output": raw_output,
                        "pred_answer": pred_answer,
                        "correct": is_correct(pred_answer, row.answer),
                    }
                )
    finally:
        if llm is not None:
            del llm
        del tokenizer
        cleanup_vllm()
        time.sleep(2)

    summary = summarize(predictions)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_model_args(model_args: list[str]) -> dict[str, Path]:
    if not model_args:
        return dict(DEFAULT_MODELS)
    models: dict[str, Path] = {}
    for item in model_args:
        if "=" in item:
            name, path = item.split("=", 1)
            models[name] = Path(path)
        else:
            path = Path(item)
            models[path.name] = path
    return models


def load_split(split_name: str, limit: int | None, gsm8k_path: Path, gsmplus_mini: bool) -> list[EvalRow]:
    if split_name == "original":
        return load_gsm8k_jsonl(gsm8k_path, limit=limit) if gsm8k_path.exists() else load_hf_gsm8k(limit=limit)
    if split_name in GSMPLUS_TYPES:
        return load_hf_gsmplus(split_name, limit=limit, mini=gsmplus_mini)
    raise ValueError(f"Unsupported split: {split_name}")


def main() -> None:
    configure_runtime_env()
    parser = argparse.ArgumentParser(description="Official GSM8K/GSM-Plus 8-shot CoT vLLM evaluator.")
    parser.add_argument("--split", choices=["original", "rephrase", "distract"], action="append", default=[])
    parser.add_argument("--model", action="append", default=[], help="Path or name=/path. Defaults to local Qwen.")
    parser.add_argument("--gsm8k-path", type=Path, default=DEFAULT_GSM8K_LOCAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument(
        "--prompt-style",
        choices=SUPPORTED_PROMPT_STYLES,
        default="chat",
    )
    parser.add_argument("--gsmplus-mini", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")
    args = parser.parse_args()

    splits = args.split or ["original"]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_id
    models = parse_model_args(args.model)
    all_summaries: list[dict[str, Any]] = []

    config = {
        "splits": splits,
        "models": {name: str(path) for name, path in models.items()},
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 512,
        "fewshot": "8-shot CoT",
        "extractor": "shared_numeric_extractor",
        "prompt_style": args.prompt_style,
        "gsmplus_types": {key: sorted(value) for key, value in GSMPLUS_TYPES.items()},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    for split_name in splits:
        rows = load_split(split_name, limit=args.limit, gsm8k_path=args.gsm8k_path, gsmplus_mini=args.gsmplus_mini)
        if not rows:
            raise RuntimeError(f"No rows loaded for split {split_name}")
        for model_name, model_path in models.items():
            print(f"[official-gsm] split={split_name} model={model_name} rows={len(rows)}", flush=True)
            summary = evaluate_model(
                model_name=model_name,
                model_path=model_path,
                rows=rows,
                output_dir=run_dir / split_name / model_name,
                batch_size=args.batch_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                dtype=args.dtype,
                seed=args.seed,
                max_model_len=args.max_model_len,
                prompt_style=args.prompt_style,
                trust_remote_code=args.trust_remote_code,
            )
            all_summaries.append({"split": split_name, "model_name": model_name, **summary})
            print(json.dumps(all_summaries[-1], ensure_ascii=False), flush=True)

    (run_dir / "summary.json").write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[official-gsm] output_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
