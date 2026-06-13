"""Standalone reward module for verl GRPO training.

This file is a self-contained copy of reward_chaingsm.py with the verl-compatible
``compute_reward`` entry point. It is intended to be loaded by verl via
``custom_reward_function.path`` in the Hydra config.

Usage in verl YAML config::

    custom_reward_function:
      path: /home/wwq416/snap/wwq/math-chain/train_pipeline/reward_chaingsm_verl.py
      name: compute_reward
      reward_kwargs:
        format_weight: 0.2
        answer_weight: 2.5
        expression_weight: 1.0
        trace_weight: 1.0
        distractor_penalty: 0.5
        invalid_reward: -0.5
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any


# ---------------------------------------------------------------------------
# Helper functions (same as reward_chaingsm.py)
# ---------------------------------------------------------------------------

def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _safe_eval(expr: str) -> float | None:
    try:
        node = ast.parse(str(expr).replace("^", "**"), mode="eval")
        return _eval_node(node)
    except Exception:
        return None


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _eval_node(node.operand)
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
    raise ValueError(type(node).__name__)


def _parse_response(completion: str) -> dict[str, Any] | None:
    text = str(completion or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _norm_expr(expr: str) -> str:
    return re.sub(r"\s+", "", str(expr or "").replace("^", "**").replace(",", ""))


def _trace_expressions(trace: Any) -> set[str]:
    if not isinstance(trace, list):
        return set()
    return {_norm_expr(step.get("expression", "")) for step in trace if isinstance(step, dict)}


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def score_response(
    completion: str,
    reference: dict[str, Any],
    format_weight: float = 0.2,
    answer_weight: float = 0.4,
    expression_weight: float = 0.2,
    trace_weight: float = 0.2,
    distractor_penalty: float = 0.3,
    invalid_reward: float = -0.5,
) -> tuple[float, dict[str, Any]]:
    parsed = _parse_response(completion)
    if not isinstance(parsed, dict):
        return invalid_reward, {"format": 0.0, "reason": "invalid_json"}

    required = {"target", "selected_steps", "final_expression", "answer"}
    selected_steps = parsed.get("selected_steps")
    step_format = isinstance(selected_steps, list) and all(
        isinstance(step, dict) and {"variable", "description", "expression", "value"} <= set(step)
        for step in selected_steps
    )
    format_ok = required <= set(parsed) and step_format

    pred_answer = _to_float(parsed.get("answer"))
    gold_answer = _to_float(reference.get("gold_answer"))
    answer_ok = pred_answer is not None and gold_answer is not None and abs(pred_answer - gold_answer) < 1e-6

    pred_expr_value = _safe_eval(parsed.get("final_expression", ""))
    gold_expr_value = _safe_eval(reference.get("gold_expression", ""))
    expression_ok = (
        pred_expr_value is not None and gold_expr_value is not None and abs(pred_expr_value - gold_expr_value) < 1e-6
    )

    pred_exprs = _trace_expressions(selected_steps)
    gold_exprs = _trace_expressions(reference.get("gold_trace"))
    distractor_exprs = _trace_expressions(reference.get("distractor_trace"))
    trace_overlap = len(pred_exprs & gold_exprs) / max(len(gold_exprs), 1)
    distractor_overlap = len(pred_exprs & distractor_exprs) / max(len(distractor_exprs), 1)
    if _norm_expr(parsed.get("final_expression", "")) == _norm_expr(reference.get("distractor_expression", "")):
        distractor_overlap = max(distractor_overlap, 1.0)

    reward = (
        format_weight * float(format_ok)
        + answer_weight * float(answer_ok)
        + expression_weight * float(expression_ok)
        + trace_weight * trace_overlap
        - distractor_penalty * distractor_overlap
    )
    return float(reward), {
        "format": float(format_ok),
        "answer": float(answer_ok),
        "expression": float(expression_ok),
        "trace_overlap": trace_overlap,
        "distractor_overlap": distractor_overlap,
        "reward": reward,
    }


# ---------------------------------------------------------------------------
# verl entry point
# ---------------------------------------------------------------------------

def compute_reward(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: dict | None = None,
    extra_info: dict | None = None,
    **kwargs,
) -> dict[str, Any]:
    """verl-compatible reward function.

    Called by ``NaiveRewardManager`` for each sample during GRPO training.
    ``reward_kwargs`` from the YAML config are merged into ``**kwargs`` by
    ``verl.trainer.ppo.reward._call_with_kwargs``.

    Returns:
        dict with ``"score"`` (float) and ``"metrics"`` (dict) keys.
    """
    reference = ground_truth or {}
    if isinstance(reference, dict) and "ground_truth" in reference:
        reference = reference["ground_truth"]
    reward, metrics = score_response(solution_str or "", reference or {}, **kwargs)
    return {
        "score": reward,
        "accuracy": float(metrics.get("answer", 0.0)),
        "format": float(metrics.get("format", 0.0)),
        "answer": float(metrics.get("answer", 0.0)),
        "expression": float(metrics.get("expression", 0.0)),
        "trace_overlap": float(metrics.get("trace_overlap", 0.0)),
        "distractor_overlap": float(metrics.get("distractor_overlap", 0.0)),
    }
