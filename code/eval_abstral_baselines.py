#!/usr/bin/env python3
"""Evaluate GranulAR/AbstRaL-style prompting baselines on ChainGSM."""

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
from pathlib import Path
from typing import Any, Iterable

import sympy as sp
from tqdm import tqdm

from gsm_answer_extractor import extract_answer, extract_number, is_correct, normalize_answer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = (
    REPO_ROOT
    / "chaingsm_data"
    / "data"
    / "final"
    / "gsm8k_test_full"
    / "gsm8k_test_all.jsonl"
)
DEFAULT_MODEL_ROOT = Path("/home/wwq416/snap/wwq/model")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "code" / "results" / "baseline_test"
DEFAULT_EXCLUDED_MODEL_KEYWORDS = ("ministral",)

GRANULAR_METHOD = "granular_style_prompting"
ABSTRAL_METHOD = "abstral_style_two_stage_prompting"
METHODS = (GRANULAR_METHOD, ABSTRAL_METHOD)

SAMPLING_CONFIG = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 512,
}

ONE_CALL_SYSTEM_PROMPT = """You are a careful math word problem solver.

Your task is to solve the problem using a granular abstract reasoning format.

Important rules:
1. First identify the exact quantity asked by the question.
2. Use only the reasoning chain needed to answer that target quantity.
3. The problem may contain extra arithmetic chains that are valid but irrelevant.
4. Do not use an irrelevant chain in the final answer.
5. Write each necessary arithmetic derivation inside double angle brackets, such as <<20+10=out0>>.
6. Use output variables out0, out1, out2, ... for derived quantities.
7. The final answer must be a single number.

Return your response exactly in the following format:

<target>
the exact quantity being asked
</target>

<subquestions>
Q1: ...
Q2: ...
</subquestions>

<abstract_reasoning>
Q1: ... <<expression=out0>> ...
Q2: ... <<expression=out1>> ...
</abstract_reasoning>

<final_expression>
...
</final_expression>

<answer>
...
</answer>"""

STAGE1_SYSTEM_PROMPT = """You are a careful condition recognition parser for math word problems.

Your task is to replace every explicit or implicit numerical value in the problem with an abstract input symbol.

Rules:
1. Replace each numerical value with [in0], [in1], [in2], ... in the order it appears.
2. Convert implicit numerical values into explicit values before assigning symbols.
   For example, "twice" should be treated as 2, "half" as 1/2, "three times" as 3.
3. Do not solve the problem.
4. Do not decide which conditions are relevant.
5. Preserve all non-numerical wording as much as possible.
6. Return only the abstract question and the condition equations.
7. Do not explain your work.
8. Do not include Markdown, bullet points, LaTeX, code, examples, or any text outside the XML tags.
9. Your first character must be "<" and your response must start with <abstract_question>.
10. Your response must end with </conditions>.
11. Every condition equation must be inside the <conditions> block.
12. Condition names must be plain in0, in1, in2, ... without square brackets.

Return exactly in the following format:

<abstract_question>
...
</abstract_question>

<conditions>
in0 = ...
in1 = ...
in2 = ...
</conditions>"""

STAGE2_SYSTEM_PROMPT = """You are a careful abstract math reasoner.

You are given:
1. An abstract math word problem where numerical values have been replaced by [in0], [in1], ...
2. A list of input conditions defining each abstract symbol.

Your task is to solve the problem in a granular abstract reasoning format.

Important rules:
1. First identify the exact target quantity asked by the question.
2. The problem may contain extra arithmetic chains that are valid but irrelevant.
3. Use only the reasoning chain needed to answer the target quantity.
4. Do not use irrelevant chains in the final answer.
5. List the necessary sub-questions before solving.
6. For each sub-question, quote the relevant input or previous output variables.
7. Put every arithmetic derivation inside double angle brackets, such as <<in0+in1=out0>>.
8. Use out0, out1, out2, ... for derived quantities.
9. At the end, state which output variable is the final answer.
10. Return the final numerical answer using the given conditions.

Return exactly in the following format:

<target>
...
</target>

<subquestions>
Q1: ...
Q2: ...
</subquestions>

<abstract_reasoning>
Q1: ...
Q2: ...
</abstract_reasoning>

<final_var>
outN
</final_var>

<final_expression>
...
</final_expression>

<answer>
...
</answer>"""

TAG_PATTERN_TEMPLATE = r"<\s*{tag}\s*>\s*(.*?)\s*</\s*{tag}\s*>"
DERIVATION_PATTERN = re.compile(r"<<\s*(.+?)\s*=\s*(out\d+)\s*>>", re.DOTALL)
CONDITION_PATTERN = re.compile(r"^\s*\[?(in\d+)\]?\s*=\s*(.+?)\s*$")


@dataclass
class EvalExample:
    id: str
    base_id: str
    category: str
    question: str
    gold_answer: str
    gold_expression: str
    distractor_expression: str


@dataclass
class PredictionRecord:
    model_name: str
    model_path: str
    method: str
    dataset: str
    id: str
    base_id: str
    category: str
    question: str
    gold_answer: str
    gold_expression: str
    distractor_expression: str
    raw_output: str
    stage1_raw_output: str | None
    abstract_question: str | None
    conditions: str | None
    target: str | None
    subquestions: str | None
    abstract_reasoning: str | None
    derivations: list[dict[str, str | None]]
    final_var: str | None
    final_expression: str | None
    model_answer: str | None
    sympy_answer: str | None
    pred_answer: str | None
    correct: bool
    gold_expression_value: str | None
    distractor_expression_value: str | None
    pred_matches_gold_expression: bool | None
    pred_matches_distractor_expression: bool | None
    parse_error: str | None
    sympy_error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GranulAR/AbstRaL-style prompting baselines on ChainGSM."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--dataset-name", default=None)
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
        help="Baseline method to run. Defaults to both methods. Can be passed multiple times.",
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


def infer_dataset_name(data_path: Path) -> str:
    if data_path.name == "gsm8k_test_all.jsonl" and data_path.parent.name == "gsm8k_test_full":
        return "chaingsm_gsm8k_test_full"
    parent = safe_path_name(data_path.parent.name)
    stem = safe_path_name(data_path.stem)
    return f"chaingsm_{parent}_{stem}" if parent else f"chaingsm_{stem}"


def load_examples(data_path: Path, limit: int | None = None) -> list[EvalExample]:
    examples: list[EvalExample] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
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
                    gold_expression=str(row.get("gold_expression") or ""),
                    distractor_expression=str(row.get("distractor_expression") or ""),
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


def one_call_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ONE_CALL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Problem:\n"
                f"{question}\n\n"
                "Solve the problem using the required granular abstract reasoning format."
            ),
        },
    ]


def stage1_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Problem:\n"
                f"{question}\n\n"
                "Return XML only. Do not solve. Do not explain.\n"
                "Start immediately with <abstract_question> and end with </conditions>."
            ),
        },
    ]


def stage2_messages(abstract_question: str, conditions: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": STAGE2_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Abstract problem:\n"
                f"{abstract_question}\n\n"
                "Conditions:\n"
                f"{conditions}\n\n"
                "Solve the abstract problem using the required granular abstract reasoning format."
            ),
        },
    ]


def fallback_chat_prompt(messages: list[dict[str, str]]) -> str:
    pieces: list[str] = []
    for message in messages:
        role = message["role"].capitalize()
        pieces.append(f"{role}: {message['content']}")
    pieces.append("Assistant:")
    return "\n".join(pieces)


def build_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return fallback_chat_prompt(messages)


def extract_tag(text: str, tag: str) -> str | None:
    pattern = re.compile(TAG_PATTERN_TEMPLATE.format(tag=re.escape(tag)), re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def extract_answer_text(output: str) -> str | None:
    tagged_answer = extract_tag(output, "answer")
    if tagged_answer:
        return extract_answer(tagged_answer)
    return extract_answer(output)


def answers_equal(left: str | None, right: str | None, tolerance: float = 1e-6) -> bool | None:
    if left is None or right is None:
        return None
    return is_correct(left, right, tolerance=tolerance)


def clean_expression(expression: str) -> str:
    cleaned = expression.strip()
    cleaned = re.sub(r"\[([A-Za-z]+\d+)\]", r"\1", cleaned)
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("×", "*").replace("÷", "/")
    cleaned = cleaned.replace("−", "-").replace("–", "-")
    cleaned = re.sub(r"\b[xX]\b", "*", cleaned)
    cleaned = cleaned.replace("^", "**")
    cleaned = re.sub(r"(?<=\d)\s+(?=\d)", "", cleaned)
    return cleaned


def sympy_to_answer(value: Any) -> str:
    simplified = sp.simplify(value)
    if simplified.is_Integer:
        return str(int(simplified))
    if simplified.is_Rational:
        return f"{int(simplified.p)}/{int(simplified.q)}"
    numeric = sp.N(simplified, 15)
    text = str(numeric)
    return text.rstrip("0").rstrip(".") if "." in text else text


def eval_sympy_expression(expression: str, env: dict[str, Any] | None = None) -> Any:
    cleaned = clean_expression(expression)
    local_dict = dict(env or {})
    local_dict.update({"Rational": sp.Rational})
    return sp.simplify(sp.sympify(cleaned, locals=local_dict))


def eval_expression_to_answer(expression: str, env: dict[str, Any] | None = None) -> tuple[str | None, str | None]:
    if not expression.strip():
        return None, "empty expression"
    try:
        return sympy_to_answer(eval_sympy_expression(expression, env)), None
    except Exception as exc:
        number = extract_number(expression)
        if number is None:
            return None, repr(exc)
        try:
            return sympy_to_answer(eval_sympy_expression(number, env)), None
        except Exception as fallback_exc:
            return None, f"{repr(exc)}; fallback={repr(fallback_exc)}"


def parse_conditions(conditions: str | None) -> tuple[dict[str, Any], list[dict[str, str | None]], str | None]:
    env: dict[str, Any] = {}
    parsed: list[dict[str, str | None]] = []
    errors: list[str] = []
    if not conditions:
        return env, parsed, None
    for line in conditions.splitlines():
        line = line.strip()
        if not line:
            continue
        match = CONDITION_PATTERN.match(line)
        if not match:
            continue
        name, raw_value = match.groups()
        value, error = eval_expression_to_answer(raw_value, env)
        parsed.append({"name": name, "expression": raw_value, "value": value, "error": error})
        if value is None:
            errors.append(f"{name}: {error}")
            continue
        env[name] = eval_sympy_expression(value, env)
    return env, parsed, "; ".join(errors) if errors else None


def extract_derivations(raw_output: str) -> list[dict[str, str | None]]:
    derivations: list[dict[str, str | None]] = []
    for match in DERIVATION_PATTERN.finditer(raw_output):
        expression, name = match.groups()
        derivations.append(
            {
                "name": name.strip(),
                "expression": expression.strip(),
                "value": None,
                "error": None,
            }
        )
    return derivations


def evaluate_derivations(
    derivations: list[dict[str, str | None]], env: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str | None]], str | None]:
    errors: list[str] = []
    working_env = dict(env)
    evaluated: list[dict[str, str | None]] = []
    for derivation in derivations:
        name = derivation["name"] or ""
        expression = derivation["expression"] or ""
        value, error = eval_expression_to_answer(expression, working_env)
        row = dict(derivation)
        row["value"] = value
        row["error"] = error
        evaluated.append(row)
        if value is None:
            errors.append(f"{name}: {error}")
            continue
        try:
            working_env[name] = eval_sympy_expression(value, working_env)
        except Exception as exc:
            errors.append(f"{name}: env assignment failed: {repr(exc)}")
    return working_env, evaluated, "; ".join(errors) if errors else None


def parse_stage1_output(raw_output: str) -> tuple[str | None, str | None, str | None]:
    abstract_question = extract_tag(raw_output, "abstract_question")
    conditions = extract_tag(raw_output, "conditions")
    errors: list[str] = []
    if not abstract_question:
        errors.append("missing abstract_question")
    if not conditions:
        errors.append("missing conditions")
    return abstract_question, conditions, "; ".join(errors) if errors else None


def parse_reasoning_output(
    raw_output: str,
    method: str,
    conditions: str | None = None,
) -> dict[str, Any]:
    parse_errors: list[str] = []
    sympy_errors: list[str] = []

    target = extract_tag(raw_output, "target")
    subquestions = extract_tag(raw_output, "subquestions")
    abstract_reasoning = extract_tag(raw_output, "abstract_reasoning")
    final_var = extract_tag(raw_output, "final_var")
    final_expression = extract_tag(raw_output, "final_expression")
    model_answer = extract_answer_text(raw_output)

    if not target:
        parse_errors.append("missing target")
    if not abstract_reasoning:
        parse_errors.append("missing abstract_reasoning")
    if method == ABSTRAL_METHOD and not final_var:
        parse_errors.append("missing final_var")
    if not final_expression:
        parse_errors.append("missing final_expression")

    condition_env, parsed_conditions, condition_error = parse_conditions(conditions)
    if condition_error:
        sympy_errors.append(f"conditions: {condition_error}")

    derivations = extract_derivations(raw_output)
    if not derivations:
        parse_errors.append("missing derivations")

    env, evaluated_derivations, derivation_error = evaluate_derivations(derivations, condition_env)
    if derivation_error:
        sympy_errors.append(f"derivations: {derivation_error}")

    sympy_answer: str | None = None
    final_error: str | None = None
    if method == ABSTRAL_METHOD and final_var:
        final_var_clean = clean_expression(final_var)
        if final_var_clean in env:
            sympy_answer = sympy_to_answer(env[final_var_clean])
        else:
            final_error = f"final_var {final_var_clean!r} not found"

    if sympy_answer is None and final_expression:
        sympy_answer, final_error = eval_expression_to_answer(final_expression, env)

    if final_error:
        sympy_errors.append(f"final: {final_error}")

    pred_answer = sympy_answer or model_answer

    return {
        "target": target,
        "subquestions": subquestions,
        "abstract_reasoning": abstract_reasoning,
        "parsed_conditions": parsed_conditions,
        "derivations": evaluated_derivations,
        "final_var": final_var.strip() if final_var else None,
        "final_expression": final_expression,
        "model_answer": model_answer,
        "sympy_answer": sympy_answer,
        "pred_answer": pred_answer,
        "parse_error": "; ".join(parse_errors) if parse_errors else None,
        "sympy_error": "; ".join(sympy_errors) if sympy_errors else None,
    }


def expression_value(expression: str) -> str | None:
    if not expression.strip():
        return None
    value, _ = eval_expression_to_answer(expression)
    return value


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[PredictionRecord]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_category: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"correct": 0, "total": 0, "distractor_matches": 0}
    )
    overall: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"correct": 0, "total": 0, "distractor_matches": 0}
    )

    for record in records:
        category_key = (record.model_name, record.method, record.category)
        overall_key = (record.model_name, record.method)
        for stats in (by_category[category_key], overall[overall_key]):
            stats["correct"] += int(record.correct)
            stats["total"] += 1
            stats["distractor_matches"] += int(record.pred_matches_distractor_expression is True)

    category_rows = []
    for (model_name, method, category), stats in sorted(by_category.items()):
        total = stats["total"]
        correct = stats["correct"]
        distractor_matches = stats["distractor_matches"]
        category_rows.append(
            {
                "model_name": model_name,
                "method": method,
                "category": category,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
                "distractor_matches": distractor_matches,
                "distractor_match_rate": distractor_matches / total if total else 0.0,
            }
        )

    overall_rows = []
    for (model_name, method), stats in sorted(overall.items()):
        total = stats["total"]
        correct = stats["correct"]
        distractor_matches = stats["distractor_matches"]
        overall_rows.append(
            {
                "model_name": model_name,
                "method": method,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
                "distractor_matches": distractor_matches,
                "distractor_match_rate": distractor_matches / total if total else 0.0,
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
        [
            "model_name",
            "method",
            "category",
            "correct",
            "total",
            "accuracy",
            "distractor_matches",
            "distractor_match_rate",
        ],
    )
    write_csv(
        output_dir / "summary_overall.csv",
        overall_rows,
        [
            "model_name",
            "method",
            "correct",
            "total",
            "accuracy",
            "distractor_matches",
            "distractor_match_rate",
        ],
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


def load_llm(args: argparse.Namespace, model_path: Path) -> Any:
    from vllm import LLM

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


def generate_texts(
    llm: Any,
    prompts: list[str],
    sampling_params: Any,
    batch_size: int,
    desc: str,
) -> list[str]:
    outputs_text: list[str] = []
    for prompt_batch in tqdm(
        chunked(prompts, batch_size),
        total=math.ceil(len(prompts) / batch_size),
        desc=desc,
        leave=False,
    ):
        outputs = llm.generate(prompt_batch, sampling_params, use_tqdm=False)
        for output in outputs:
            outputs_text.append(output.outputs[0].text if output.outputs else "")
    return outputs_text


def build_prediction_record(
    model_name: str,
    model_path: Path,
    method: str,
    dataset_name: str,
    example: EvalExample,
    raw_output: str,
    parsed: dict[str, Any],
    stage1_raw_output: str | None = None,
    abstract_question: str | None = None,
    conditions: str | None = None,
    extra_parse_error: str | None = None,
) -> PredictionRecord:
    gold_value = expression_value(example.gold_expression)
    distractor_value = expression_value(example.distractor_expression)
    pred_answer = parsed["pred_answer"]
    parse_error = parsed["parse_error"]
    if extra_parse_error:
        parse_error = f"{extra_parse_error}; {parse_error}" if parse_error else extra_parse_error

    return PredictionRecord(
        model_name=model_name,
        model_path=str(model_path),
        method=method,
        dataset=dataset_name,
        id=example.id,
        base_id=example.base_id,
        category=example.category,
        question=example.question,
        gold_answer=example.gold_answer,
        gold_expression=example.gold_expression,
        distractor_expression=example.distractor_expression,
        raw_output=raw_output,
        stage1_raw_output=stage1_raw_output,
        abstract_question=abstract_question,
        conditions=conditions,
        target=parsed["target"],
        subquestions=parsed["subquestions"],
        abstract_reasoning=parsed["abstract_reasoning"],
        derivations=parsed["derivations"],
        final_var=parsed["final_var"],
        final_expression=parsed["final_expression"],
        model_answer=parsed["model_answer"],
        sympy_answer=parsed["sympy_answer"],
        pred_answer=pred_answer,
        correct=is_correct(pred_answer, example.gold_answer),
        gold_expression_value=gold_value,
        distractor_expression_value=distractor_value,
        pred_matches_gold_expression=answers_equal(pred_answer, gold_value),
        pred_matches_distractor_expression=answers_equal(pred_answer, distractor_value),
        parse_error=parse_error,
        sympy_error=parsed["sympy_error"],
    )


def evaluate_granular(
    args: argparse.Namespace,
    tokenizer: Any,
    llm: Any,
    sampling_params: Any,
    model_path: Path,
    examples: list[EvalExample],
    dataset_name: str,
    output_dir: Path,
) -> list[PredictionRecord]:
    model_name = model_path.name
    model_output_dir = output_dir / "model_outputs" / safe_path_name(model_name)
    predictions_path = model_output_dir / "predictions.jsonl"
    prompts = [build_prompt(tokenizer, one_call_messages(example.question)) for example in examples]
    raw_outputs = generate_texts(
        llm,
        prompts,
        sampling_params,
        args.batch_size,
        f"{model_name} granular",
    )

    records: list[PredictionRecord] = []
    for example, raw_output in zip(examples, raw_outputs):
        parsed = parse_reasoning_output(raw_output, GRANULAR_METHOD)
        record = build_prediction_record(
            model_name,
            model_path,
            GRANULAR_METHOD,
            dataset_name,
            example,
            raw_output,
            parsed,
        )
        records.append(record)
        write_jsonl_record(predictions_path, asdict(record))
    return records


def evaluate_abstral_two_stage(
    args: argparse.Namespace,
    tokenizer: Any,
    llm: Any,
    sampling_params: Any,
    model_path: Path,
    examples: list[EvalExample],
    dataset_name: str,
    output_dir: Path,
) -> list[PredictionRecord]:
    model_name = model_path.name
    model_output_dir = output_dir / "model_outputs" / safe_path_name(model_name)
    predictions_path = model_output_dir / "predictions.jsonl"
    stage1_path = model_output_dir / "stage1_outputs.jsonl"

    stage1_prompts = [build_prompt(tokenizer, stage1_messages(example.question)) for example in examples]
    stage1_outputs = generate_texts(
        llm,
        stage1_prompts,
        sampling_params,
        args.batch_size,
        f"{model_name} abstral stage1",
    )

    stage1_rows: list[dict[str, Any]] = []
    stage2_prompts: list[str] = []
    for example, raw_output in zip(examples, stage1_outputs):
        abstract_question, conditions, parse_error = parse_stage1_output(raw_output)
        row = {
            "model_name": model_name,
            "model_path": str(model_path),
            "method": ABSTRAL_METHOD,
            "dataset": dataset_name,
            "id": example.id,
            "base_id": example.base_id,
            "category": example.category,
            "question": example.question,
            "raw_output": raw_output,
            "abstract_question": abstract_question,
            "conditions": conditions,
            "parse_error": parse_error,
        }
        stage1_rows.append(row)
        write_jsonl_record(stage1_path, row)
        stage2_prompts.append(
            build_prompt(tokenizer, stage2_messages(abstract_question or "", conditions or ""))
        )

    stage2_outputs = generate_texts(
        llm,
        stage2_prompts,
        sampling_params,
        args.batch_size,
        f"{model_name} abstral stage2",
    )

    records: list[PredictionRecord] = []
    for example, stage1_row, raw_output in zip(examples, stage1_rows, stage2_outputs):
        parsed = parse_reasoning_output(
            raw_output,
            ABSTRAL_METHOD,
            conditions=stage1_row["conditions"],
        )
        record = build_prediction_record(
            model_name,
            model_path,
            ABSTRAL_METHOD,
            dataset_name,
            example,
            raw_output,
            parsed,
            stage1_raw_output=stage1_row["raw_output"],
            abstract_question=stage1_row["abstract_question"],
            conditions=stage1_row["conditions"],
            extra_parse_error=stage1_row["parse_error"],
        )
        records.append(record)
        write_jsonl_record(predictions_path, asdict(record))
    return records


def save_method_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    timestamp: str,
    method: str,
    dataset_name: str,
    examples: list[EvalExample],
    model_paths: list[Path],
) -> None:
    run_config = {
        "timestamp": timestamp,
        "method": method,
        "dataset_name": dataset_name,
        "data_path": str(args.data_path),
        "model_root": str(args.model_root),
        "output_dir": str(output_dir),
        "models": [{"model_name": path.name, "model_path": str(path)} for path in model_paths],
        "excluded_model_keywords": list(DEFAULT_EXCLUDED_MODEL_KEYWORDS),
        "process_file_layout": "model_outputs/<model_name>/predictions.jsonl",
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
    if method == ABSTRAL_METHOD:
        run_config["stage1_file"] = "model_outputs/<model_name>/stage1_outputs.jsonl"
    write_json(output_dir / "run_config.json", run_config)


def evaluate_model(
    args: argparse.Namespace,
    model_path: Path,
    methods: list[str],
    examples: list[EvalExample],
    dataset_name: str,
    method_output_dirs: dict[str, Path],
    records_by_method: dict[str, list[PredictionRecord]],
) -> None:
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    model_name = model_path.name
    model_errors = {
        method: method_output_dirs[method]
        / "model_outputs"
        / safe_path_name(model_name)
        / "errors.jsonl"
        for method in methods
    }

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=args.trust_remote_code,
        )
        llm = load_llm(args, model_path)
    except Exception as exc:
        for method in methods:
            write_jsonl_record(
                model_errors[method],
                {
                    "stage": "load_model",
                    "model_name": model_name,
                    "model_path": str(model_path),
                    "method": method,
                    "error": repr(exc),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                },
            )
        cleanup_vllm()
        return

    sampling_params = SamplingParams(**SAMPLING_CONFIG)
    try:
        for method in tqdm(methods, desc=f"{model_name} methods", leave=False):
            try:
                if method == GRANULAR_METHOD:
                    records = evaluate_granular(
                        args,
                        tokenizer,
                        llm,
                        sampling_params,
                        model_path,
                        examples,
                        dataset_name,
                        method_output_dirs[method],
                    )
                elif method == ABSTRAL_METHOD:
                    records = evaluate_abstral_two_stage(
                        args,
                        tokenizer,
                        llm,
                        sampling_params,
                        model_path,
                        examples,
                        dataset_name,
                        method_output_dirs[method],
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")
                records_by_method[method].extend(records)
                save_summaries(method_output_dirs[method], records_by_method[method])
            except Exception as exc:
                write_jsonl_record(
                    model_errors[method],
                    {
                        "stage": "method_loop",
                        "model_name": model_name,
                        "model_path": str(model_path),
                        "method": method,
                        "error": repr(exc),
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                continue
    finally:
        del llm
        del tokenizer
        cleanup_vllm()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = args.dataset_name or infer_dataset_name(args.data_path)
    methods = args.method or list(METHODS)
    examples = load_examples(args.data_path, args.limit)
    model_paths = discover_instruct_models(args.model_root, args.model_filter)

    method_output_dirs = {
        method: args.output_root / method / dataset_name / timestamp for method in methods
    }
    for method, output_dir in method_output_dirs.items():
        output_dir.mkdir(parents=True, exist_ok=True)
        save_method_run_config(output_dir, args, timestamp, method, dataset_name, examples, model_paths)

    print(f"Dataset: {dataset_name}", flush=True)
    print(f"Loaded {len(examples)} examples from {args.data_path}", flush=True)
    print(f"Discovered {len(model_paths)} instruct models", flush=True)
    for method, output_dir in method_output_dirs.items():
        print(f"{method} output dir: {output_dir}", flush=True)

    records_by_method: dict[str, list[PredictionRecord]] = {method: [] for method in methods}
    for model_path in tqdm(model_paths, desc="models"):
        evaluate_model(
            args,
            model_path,
            methods,
            examples,
            dataset_name,
            method_output_dirs,
            records_by_method,
        )

    for method in methods:
        save_summaries(method_output_dirs[method], records_by_method[method])
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
