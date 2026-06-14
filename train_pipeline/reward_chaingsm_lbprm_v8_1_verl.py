"""LB-PRM v8.1 reward - ChainGSM 适配 5 机制.

v8.1 公式:
  total = 0.05 * format
        + 0.55 * answer
        + 0.20 * chain_to_answer_check
        + 0.15 * target_recognition
        + 0.05 * chain_length_consistency
        - 0.30 * irrelevant_eq_ratio

irrelevant_eq 算法 (v8.1 改进):
  S = gold_answer ∪ core_chain eval ∪ gold_expression eval + 中间值
  irrelevant 当: 算式 (left, op, right, expected) **任一值在 S 中** → 算跟 gold chain 有关
                **全部不在 S** → 算无关
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

_CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from gsm_answer_extractor import extract_answer, is_correct  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EQUATION_PATTERN = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/x×÷])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)"
)
EXPRESSION_PATTERN = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/x×÷])\s*(-?\d+(?:\.\d+)?)"
)
NUMBER_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)")

QUESTION_UNIT_PATTERNS = [
    (r"how\s+much\s+in\s+dollars", ["dollar", "$"]),
    (r"how\s+many\s+dollars", ["dollar", "$"]),
    (r"how\s+many\s+(?:total\s+)?meters?", ["meter", "m "]),
    (r"how\s+many\s+eggs?", ["egg"]),
    (r"how\s+many\s+(?:total\s+)?books?", ["book"]),
    (r"how\s+many\s+cookies?", ["cookie"]),
    (r"how\s+much\s+profit", ["profit", "dollar", "$"]),
    (r"how\s+much\s+(?:does|do)", ["dollar", "$", "money"]),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("$", "").replace("%", "")
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _floats_close(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) < tol


def _eval_op(left: float, op: str, right: float) -> float | None:
    try:
        if op == "+":
            return left + right
        elif op == "-":
            return left - right
        elif op in ("*", "x", "×"):
            return left * right
        elif op in ("/", "÷"):
            if right == 0:
                return None
            return left / right
    except (ValueError, ZeroDivisionError):
        return None
    return None


def _parse_equations(text: str) -> list[tuple[float, str, float, float]]:
    out = []
    for left_s, op, right_s, result_s in EQUATION_PATTERN.findall(text):
        try:
            l = float(left_s)
            r = float(right_s)
            res = float(result_s)
            out.append((l, op, r, res))
        except ValueError:
            continue
    return out


def _eval_simple(expr: str) -> float | None:
    """Eval 不含括号的表达式 (按 op 优先级 * / + -)."""
    if not expr:
        return None
    expr = expr.strip()
    v = _to_float(expr)
    if v is not None and not any(op in expr for op in "+-*/x×÷"):
        return v
    for op in ["*/x×÷", "+-"]:
        while True:
            m = re.search(r"(-?\d+(?:\.\d+)?)\s*([*/x×÷])\s*(-?\d+(?:\.\d+)?)" if "/" in op or "*" in op
                          else r"(-?\d+(?:\.\d+)?)\s*([+\-])\s*(-?\d+(?:\.\d+)?)", expr)
            if not m:
                break
            l = _to_float(m.group(1))
            r = _to_float(m.group(3))
            op_char = m.group(2)
            if l is None or r is None:
                return None
            actual = _eval_op(l, op_char, r)
            if actual is None:
                return None
            expr = expr[:m.start()] + str(round(actual, 6)) + expr[m.end():]
    return _to_float(expr)


def _eval_nested_expr(expr: str) -> set[float]:
    """Eval 嵌套表达式, 收集所有 (操作数 + 中间值)."""
    S: set[float] = set()
    if not expr:
        return S
    for n in NUMBER_PATTERN.findall(expr):
        v = _to_float(n)
        if v is not None:
            S.add(round(v, 2))
    for m in EXPRESSION_PATTERN.finditer(expr):
        l = _to_float(m.group(1))
        op = m.group(2)
        r = _to_float(m.group(3))
        if l is not None and r is not None:
            actual = _eval_op(l, op, r)
            if actual is not None:
                S.add(round(actual, 2))
    # 括号嵌套 eval
    while "(" in expr:
        m = re.search(r"\(([^()]+)\)", expr)
        if not m:
            break
        inner_val = _eval_simple(m.group(1))
        if inner_val is not None:
            S.add(round(inner_val, 2))
        expr = expr[:m.start()] + str(inner_val) + expr[m.end():]
    rem = _eval_simple(expr)
    if rem is not None:
        S.add(round(rem, 2))
    return S


def _eval_core_chain(core_chain: list) -> set[float]:
    S: set[float] = set()
    if not core_chain:
        return S
    for step in core_chain:
        if len(step) >= 4:
            v1 = _to_float(step[0])
            op = step[2]
            v2 = _to_float(step[3])
            if v1 is not None and v2 is not None:
                actual = _eval_op(v1, op, v2)
                if actual is not None:
                    S.add(round(actual, 2))
    return S


def _build_should_appear_set(reference: dict[str, Any]) -> set[float]:
    S: set[float] = set()
    gold = reference.get("gold_answer")
    if gold is not None:
        g = _to_float(gold)
        if g is not None:
            S.add(round(g, 2))
    S |= _eval_core_chain(reference.get("core_chain", []))
    S |= _eval_nested_expr(reference.get("gold_expression", ""))
    return S


# ---------------------------------------------------------------------------
# 子项评分
# ---------------------------------------------------------------------------

def _numeric_correctness_score(text: str) -> tuple[float, int]:
    equations = EQUATION_PATTERN.findall(text)
    if not equations:
        return 0.0, 0
    correct = 0
    for left_s, op, right_s, result_s in equations:
        l = _to_float(left_s)
        r = _to_float(right_s)
        expected = _to_float(result_s)
        if l is None or r is None or expected is None:
            continue
        actual = _eval_op(l, op, r)
        if actual is not None and _floats_close(actual, expected):
            correct += 1
    return correct / len(equations) if equations else 0.0, len(equations)


def _chain_to_answer_check(text: str) -> float:
    equations = _parse_equations(text)
    if not equations:
        return 0.0
    expected = equations[-1][3]
    pred_answer = extract_answer(text)
    if pred_answer is None:
        return 0.0
    try:
        pred_val = float(pred_answer)
    except (ValueError, TypeError):
        return 0.0
    return 1.0 if _floats_close(pred_val, expected) else 0.0


def _target_recognition(text: str, question: str | None) -> float:
    if not question:
        return 0.5
    text_lower = text.lower()
    question_lower = question.lower()
    for pattern, expected_keywords in QUESTION_UNIT_PATTERNS:
        if re.search(pattern, question_lower):
            for kw in expected_keywords:
                if kw in text_lower:
                    return 1.0
            return 0.0
    return 0.5


def _chain_length_consistency(text: str, question: str | None) -> float:
    n_eq = len(EQUATION_PATTERN.findall(text))
    if not question:
        return 0.5
    # 改进: 数字数包括 question 里的 + 题面上下文 (如果有 question_text 字段)
    # 现在 question 字段只存问句, 不含题面正文 — 用问句数字数不够
    # fallback: 算 model text 里的数字数, 跟 n_eq 对比 (粗略)
    # 更稳定: 给 0.5 中性 (避免误判)
    return 1.0  # v8.1 v4: 默认 1.0: 暂不信任 length_consistency, 给中性分


def _count_irrelevant_eq(text: str, reference: dict[str, Any]) -> int:
    """v8.1 v3 严格算法: 算式 4 个值 (left, right, expected) **全部不在 S** 才算无关.

    关键修正: 不是 "expected_result 不在 S" 而是 "all 3 values 不在 S".
    """
    equations = _parse_equations(text)
    if not equations:
        return 0
    S = _build_should_appear_set(reference)
    if not S:
        return 0
    n_irrelevant = 0
    for left, op, right, expected in equations:
        if (round(left, 2) not in S and
            round(right, 2) not in S and
            round(expected, 2) not in S):
            n_irrelevant += 1
    return n_irrelevant


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

_METRICS_SCHEMA = {
    "format": 0.0,
    "answer": 0.0,
    "chain_to_answer_check": 0.0,
    "target_recognition": 0.0,
    "chain_length_consistency": 0.5,
    "n_irrelevant": 0,
    "n_equations": 0,
    "penalty": 0.0,
    "reward": 0.0,
    "reason": "",
}


def _empty_metrics(reason: str, reward: float) -> dict:
    m = dict(_METRICS_SCHEMA)
    m["reason"] = reason
    m["reward"] = reward
    return m


def score_response(
    completion,
    reference: dict[str, Any],
    format_weight: float = 0.05,
    answer_weight: float = 0.55,
    chain_to_answer_check_weight: float = 0.20,
    target_recognition_weight: float = 0.15,
    chain_length_consistency_weight: float = 0.05,
    irrelevant_eq_penalty_weight: float = 0.30,
    invalid_reward: float = -0.5,
    **kwargs,
) -> tuple[float, dict[str, Any]]:
    text = str(completion or "").strip()
    if not text:
        return invalid_reward, _empty_metrics("empty_response", invalid_reward)

    gold_answer = reference.get("gold_answer")
    question = reference.get("question")

    pred_answer_raw = extract_answer(text)
    format_ok = 1.0 if pred_answer_raw is not None else 0.0
    answer_ok = 1.0 if (pred_answer_raw is not None and is_correct(pred_answer_raw, gold_answer)) else 0.0
    c2a_score = _chain_to_answer_check(text)
    target_score = _target_recognition(text, question)
    length_score = _chain_length_consistency(text, question)
    n_irrelevant = _count_irrelevant_eq(text, reference)
    n_eq = len(EQUATION_PATTERN.findall(text))
    penalty = (n_irrelevant / max(n_eq, 1)) * irrelevant_eq_penalty_weight

    total = (
        format_weight * format_ok
        + answer_weight * answer_ok
        + chain_to_answer_check_weight * c2a_score
        + target_recognition_weight * target_score
        + chain_length_consistency_weight * length_score
        - penalty
    )

    m = dict(_METRICS_SCHEMA)
    m.update({
        "format": float(format_ok),
        "answer": float(answer_ok),
        "chain_to_answer_check": float(c2a_score),
        "target_recognition": float(target_score),
        "chain_length_consistency": float(length_score),
        "n_irrelevant": int(n_irrelevant),
        "n_equations": int(n_eq),
        "penalty": float(penalty),
        "reward": float(total),
        "reason": "ok" if format_ok else "no_final_answer_marker",
    })
    return float(total), m


def compute_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if isinstance(ground_truth, dict):
        reference = ground_truth
    else:
        reference = {"gold_answer": ground_truth}
    r, m = score_response(solution_str, reference, **kwargs)
    return {"score": r, **m}
