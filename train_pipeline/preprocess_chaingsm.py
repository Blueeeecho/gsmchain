from __future__ import annotations

import argparse
import ast
import json
import math
import operator
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


DEFAULT_INPUT = (
    "/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/train_balanced_one_variant/"
    "gsm8k_train_balanced_one_variant/gsm8k_train_balanced_one_variant_14946.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/rl_preprocessed/"
    "gsm8k_train_balanced_one_variant_14946"
)

SYSTEM_PROMPT = (
    "You are a careful mathematical reasoning assistant. Select only the computation chain that answers "
    "the question, ignore distractor chains, and return JSON only."
)

USER_TEMPLATE = """Solve the following grade-school math problem. Return exactly one JSON object with this schema:
{{
  "target": "short target quantity",
  "selected_steps": [
    {{"variable": "name", "description": "short explanation", "expression": "arithmetic expression", "value": "computed value"}}
  ],
  "final_expression": "arithmetic expression",
  "answer": "final answer"
}}

Problem:
{question}
"""


# Neutral prompt (target-conditioned reasoning, no distractor/ignore terms).
# Added 2026-06-08: matches SFT neutral_prompts data and the
# "LB-PRM-style chain-structure reward" approach recommended by the advisor.
# We keep the original SYSTEM_PROMPT / USER_TEMPLATE unchanged for backward
# compat with other callers; the new constants are what `eval_vllm_chaingsm`
# (method=train_json_prompt) and the patched GRPO parquet use.
NEUTRAL_SYSTEM_PROMPT = (
    "You are an expert grade-school math solver. Identify the quantity being asked, "
    "solve it carefully step by step, and follow the requested output format."
)

NEUTRAL_USER_TEMPLATE = """Solve the following grade-school math problem.
First identify the target quantity asked by the question. Then write the calculation steps needed to compute that target.
Return one valid JSON object with this schema:
{{
  "target": "short description of the quantity being asked",
  "selected_steps": [
    {{
      "variable": "short variable name",
      "description": "mathematical meaning of this step",
      "expression": "arithmetic expression",
      "value": "computed value"
    }}
  ],
  "final_expression": "arithmetic expression that computes the answer",
  "answer": "final numeric answer"
}}
Requirements:
- The JSON must be parseable.
- Use meaningful step descriptions.
- Do not include calculations that are not needed for the target quantity.
- Do not add text outside the JSON object.
Problem:
{question}
"""

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


@dataclass
class TraceStep:
    expression: str
    value: float


def read_jsonl(path: str | os.PathLike[str], max_samples: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return rows


def write_jsonl(path: str | os.PathLike[str], rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_expression(expr: str) -> str:
    cleaned = str(expr or "").strip()
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("×", "*").replace("÷", "/").replace("^", "**")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def eval_ast(node: ast.AST, steps: list[TraceStep]) -> float:
    if isinstance(node, ast.Expression):
        return eval_ast(node.body, steps)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -eval_ast(node.operand, steps)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return eval_ast(node.operand, steps)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = eval_ast(node.left, steps)
        right = eval_ast(node.right, steps)
        value = float(_OPS[type(node.op)](left, right))
        if not math.isfinite(value):
            raise ValueError("non-finite expression value")
        steps.append(TraceStep(expression=ast.unparse(node), value=value))
        return value
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def parse_expression_trace(expr: str) -> tuple[str, list[TraceStep]]:
    normalized = clean_expression(expr)
    tree = ast.parse(normalized, mode="eval")
    steps: list[TraceStep] = []
    final_value = eval_ast(tree, steps)
    if not steps:
        steps.append(TraceStep(expression=normalized, value=final_value))
    return normalized, steps


def format_number(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if math.isfinite(number) and abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.10g}"


def chain_targets(chain: Any) -> list[str]:
    targets = []
    if isinstance(chain, list):
        for triple in chain:
            if isinstance(triple, list | tuple) and len(triple) >= 2:
                targets.append(str(triple[1]))
    return targets


def assign_variables(steps: list[TraceStep], targets: list[str], prefix: str) -> list[str]:
    variables = [f"{prefix}_step_{i + 1}" for i in range(len(steps))]
    if not targets:
        return variables
    if len(targets) >= len(steps):
        return targets[-len(steps) :]
    start = len(steps) - len(targets)
    variables[start:] = targets
    return variables


def build_trace(record: dict[str, Any], expression_key: str, chain_key: str, prefix: str) -> tuple[str, list[dict[str, str]]]:
    normalized, raw_steps = parse_expression_trace(record.get(expression_key, ""))
    variables = assign_variables(raw_steps, chain_targets(record.get(chain_key)), prefix)
    trace = []
    for idx, (step, variable) in enumerate(zip(raw_steps, variables), start=1):
        trace.append(
            {
                "variable": variable,
                "description": f"Compute step {idx} for the {'correct' if prefix == 'gold' else 'distractor'} chain.",
                "expression": clean_expression(step.expression),
                "value": format_number(step.value),
            }
        )
    return normalized, trace


def infer_target(record: dict[str, Any], trace: list[dict[str, str]]) -> str:
    if trace:
        return trace[-1]["variable"]
    question = record.get("question_distracted") or record.get("question_original") or "answer"
    return str(question).strip().split("?")[0][-80:]


def response_from_trace(target: str, trace: list[dict[str, str]], final_expression: str, answer: str) -> dict[str, Any]:
    return {
        "target": target,
        "selected_steps": trace,
        "final_expression": final_expression,
        "answer": str(answer),
    }


def build_messages(question: str, response: dict[str, Any] | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(question=question.strip())},
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": json.dumps(response, ensure_ascii=False)})
    return messages


def augment_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized_gold, gold_trace = build_trace(record, "gold_expression", "core_chain", "gold")
    normalized_distractor, distractor_trace = build_trace(record, "distractor_expression", "distractor_chain", "distractor")
    gold_target = infer_target(record, gold_trace)
    distractor_target = infer_target(record, distractor_trace)
    gold_response = response_from_trace(gold_target, gold_trace, normalized_gold, str(record.get("answer", "")))
    distractor_answer = distractor_trace[-1]["value"] if distractor_trace else ""
    rejected_response = response_from_trace(distractor_target, distractor_trace, normalized_distractor, distractor_answer)
    out = dict(record)
    out.update(
        {
            "normalized_gold_expression": normalized_gold,
            "normalized_distractor_expression": normalized_distractor,
            "gold_trace": gold_trace,
            "distractor_trace": distractor_trace,
            "gold_response": gold_response,
            "rejected_response": rejected_response,
            "trace_build_status": "ok",
        }
    )
    return out


def to_sft_row(record: dict[str, Any]) -> dict[str, Any]:
    question = record.get("question_distracted") or record.get("question_original") or ""
    return {
        "id": record["id"],
        "messages": build_messages(question, record["gold_response"]),
        "prompt": USER_TEMPLATE.format(question=question.strip()),
        "response": json.dumps(record["gold_response"], ensure_ascii=False),
    }


def to_dpo_row(record: dict[str, Any]) -> dict[str, Any]:
    question = record.get("question_distracted") or record.get("question_original") or ""
    return {
        "id": record["id"],
        "prompt": USER_TEMPLATE.format(question=question.strip()),
        "chosen": json.dumps(record["gold_response"], ensure_ascii=False),
        "rejected": json.dumps(record["rejected_response"], ensure_ascii=False),
    }


def reward_reference(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "gold_answer": str(record.get("answer", "")),
        "gold_expression": record.get("normalized_gold_expression", record.get("gold_expression", "")),
        "gold_trace": record.get("gold_trace", []),
        "distractor_expression": record.get(
            "normalized_distractor_expression", record.get("distractor_expression", "")
        ),
        "distractor_trace": record.get("distractor_trace", []),
        "category": record.get("category", ""),
        "difficulty_tags": record.get("difficulty_tags", {}),
    }


def to_grpo_row(record: dict[str, Any]) -> dict[str, Any]:
    question = record.get("question_distracted") or record.get("question_original") or ""
    prompt = USER_TEMPLATE.format(question=question.strip())
    return {
        "id": record["id"],
        "prompt": prompt,
        "messages": build_messages(question),
        "reward_reference": reward_reference(record),
    }


def to_verl_grpo_row(record: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    grpo = to_grpo_row(record)
    return {
        "data_source": "chaingsm",
        "prompt": grpo["messages"],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": grpo["reward_reference"]},
        "extra_info": {"split": split, "index": index, "id": record["id"], "category": record.get("category", "")},
    }


def to_verl_sft_row(record: dict[str, Any]) -> dict[str, Any]:
    question = record.get("question_distracted") or record.get("question_original") or ""
    return {
        "question": USER_TEMPLATE.format(question=question.strip()),
        "answer": json.dumps(record["gold_response"], ensure_ascii=False),
    }


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if pd is None:
        raise RuntimeError("pandas is required to write parquet files.")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ChainGSM JSONL for SFT, DPO, GRPO, and verl GRPO.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-parquet", action="store_true", help="Only write JSONL outputs.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = read_jsonl(args.input, args.max_samples)
    augmented: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for idx, record in enumerate(raw_rows):
        try:
            augmented.append(augment_record(record))
        except Exception as exc:
            errors.append({"index": idx, "id": record.get("id"), "error": str(exc), "record": record})

    sft_rows = [to_sft_row(row) for row in augmented]
    dpo_rows = [to_dpo_row(row) for row in augmented]
    grpo_rows = [to_grpo_row(row) for row in augmented]

    write_jsonl(output_dir / "source_augmented_with_traces.jsonl", augmented)
    write_jsonl(output_dir / "sft_train.jsonl", sft_rows)
    write_jsonl(output_dir / "dpo_train.jsonl", dpo_rows)
    write_jsonl(output_dir / "grpo_train.jsonl", grpo_rows)
    write_jsonl(output_dir / "trace_build_errors.jsonl", errors)
    shutil.copyfile(output_dir / "sft_train.jsonl", output_dir / "sft_then_grpo_stage1_sft.jsonl")
    shutil.copyfile(output_dir / "grpo_train.jsonl", output_dir / "sft_then_grpo_stage2_grpo.jsonl")

    if not args.skip_parquet:
        write_parquet(output_dir / "verl_sft_train.parquet", [to_verl_sft_row(row) for row in augmented])
        write_parquet(output_dir / "verl_grpo_train.parquet", [to_verl_grpo_row(row, "train", i) for i, row in enumerate(augmented)])

    stats = {
        "input_path": str(args.input),
        "output_dir": str(output_dir),
        "input_count": len(raw_rows),
        "kept_count": len(augmented),
        "error_count": len(errors),
        "files": {
            "sft": "sft_train.jsonl",
            "dpo": "dpo_train.jsonl",
            "grpo": "grpo_train.jsonl",
            "verl_sft": "verl_sft_train.parquet",
            "verl_grpo": "verl_grpo_train.parquet",
        },
    }
    with open(output_dir / "preprocess_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
