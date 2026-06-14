"""LB-PRM v6 reward for verl GRPO training (8-shot CoT protocol).

基于 v3/v5 reward 改写,适配 8-shot CoT 协议。spec 在 docs/superpowers/specs/2026-06-14-lbprm-v6-design.md

v6 改动 (2026-06-14):
  1. **协议换为 8-shot CoT**: 不再要求 JSON schema (selected_steps / final_expression),
     模型自由推理, 答案以 "The final answer is N." 收尾
  2. **reward 重写**:
     - format = 1.0 if 收尾于 "The final answer is N." else 0.0
     - answer = 1.0 if N 数值 == gold_answer else 0.0
     - reasoning_quality = 0.5·step_count + 0.3·numeric_correctness + 0.2·no_contradiction
  3. **权重**:
     - format=0.15, answer=0.60, reasoning_quality=0.25
     - 相对 v3 (0.20/0.55/0.25): format 降低, answer 升高, reasoning_quality 保留
  4. **训练-评测 prompt 100% 对齐**: 训练 prompt = 8-shot CoT 完整模板, 评测 prompt = 训练 prompt
  5. **不复用 v3 任何字段**: chain_to_answer_ok / causal_liveness / step_calc / no_degenerate
     全部废弃, 改为 reasoning_quality 3 子项
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# 复用 code/gsm_answer_extractor 的 answer 提取与判等
_CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from gsm_answer_extractor import extract_answer, is_correct  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (8-shot CoT 协议)
# ---------------------------------------------------------------------------

# 8-shot CoT 收尾 marker
FINAL_ANSWER_PATTERN = re.compile(
    r"(?:the\s+)?final answer(?:\s+is)?\s*[:=]?\s*([^\n]+)",
    re.IGNORECASE,
)
# 末步提取 (用于 format check)
LAST_LINE_PATTERN = re.compile(r"[^\n]+\Z", re.MULTILINE)

# 算式匹配: "X op Y = Z" 或 "X op Y" 形式
# 简单算式: 数字 + 操作符 + 数字
EQUATION_PATTERN = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/x×÷])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)"
)
# 也匹配 "X op Y" (无 = Z)
EQUATION_NO_RESULT_PATTERN = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/x×÷])\s*(-?\d+(?:\.\d+)?)"
)

# 推理步骤分句 (基于句号 + 数字)
STEP_DELIMITER_PATTERN = re.compile(r"(?<=[.!?。])\s+|\n+")


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


def _safe_eval(expr: str) -> float | None:
    import ast as _ast
    text = str(expr).strip().replace("^", "**").replace("x", "*").replace("×", "*").replace("÷", "/")
    text = re.sub(r"(?<=\d)\s+(?=\d)", "*", text)  # 隐式乘法 "2 3" -> "2*3" (保守处理)
    try:
        node = _ast.parse(text, mode="eval")
        return _eval_node(node)
    except Exception:
        return None


def _eval_node(node) -> float:
    import ast as _ast
    if isinstance(node, _ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, _ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.USub):
        return -_eval_node(node.operand)
    if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.UAdd):
        return _eval_node(node.operand)
    if isinstance(node, _ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, _ast.Add):
            return left + right
        if isinstance(node.op, _ast.Sub):
            return left - right
        if isinstance(node.op, _ast.Mult):
            return left * right
        if isinstance(node.op, _ast.Div):
            return left / right
        if isinstance(node.op, _ast.Pow):
            return left**right
    raise ValueError(type(node).__name__)


def _floats_close(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# v6 reasoning_quality 3 子项
# ---------------------------------------------------------------------------

def _split_steps(text: str) -> list[str]:
    """分句: 基于句号/问号/感叹号/换行"""
    return [s.strip() for s in STEP_DELIMITER_PATTERN.split(text) if s.strip()]


def _step_count_score(text: str) -> tuple[float, int]:
    """推理步骤数评分: 3-7 步满分 1.0, <3 步扣 0.3, >7 步扣 0.1"""
    steps = _split_steps(text)
    n = len(steps)
    if n == 0:
        return 0.0, 0
    if 3 <= n <= 7:
        return 1.0, n
    if n < 3:
        # 1-2 步: 扣 0.3
        return max(0.0, 1.0 - 0.3 * (3 - n)), n
    # n > 7: 扣 0.1 per extra step, 最低 0.5
    return max(0.5, 1.0 - 0.1 * (n - 7)), n


def _numeric_correctness_score(text: str) -> tuple[float, int]:
    """数值计算正确率: 抓所有 "X op Y = Z" 算式, 验证 eval(X op Y) == Z (容差 1e-3)"""
    equations = EQUATION_PATTERN.findall(text)
    if not equations:
        # 没找到带 = 的算式, 退到无 = 算式, 视为 0 分 (鼓励"写出算式")
        return 0.0, 0
    correct = 0
    for left, op, right, result in equations:
        try:
            l = float(left)
            r = float(right)
            expected = float(result)
            if op == "+":
                actual = l + r
            elif op == "-":
                actual = l - r
            elif op in ("*", "x", "×"):
                actual = l * r
            elif op in ("/", "÷"):
                if r == 0:
                    continue
                actual = l / r
            else:
                continue
            if _floats_close(actual, expected):
                correct += 1
        except (ValueError, ZeroDivisionError):
            continue
    return correct / len(equations), len(equations)


def _no_contradiction_score(text: str, final_answer: Any) -> tuple[float, bool]:
    """无自相矛盾: 最终答案 N 是否在推理过程出现的数值中"""
    if final_answer is None:
        return 0.0, False
    final_num = _to_float(final_answer)
    if final_num is None:
        return 0.0, False
    # 抓推理过程中所有数字
    all_numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not all_numbers:
        return 0.0, False
    # 检查最终答案 N 是否在过程中出现 (容差 1e-3)
    for n_str in all_numbers:
        try:
            n = float(n_str)
            if _floats_close(n, final_num):
                return 1.0, True
        except ValueError:
            continue
    return 0.0, False


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

# 固定的 metrics schema. 所有 sample(无论是否成功解析)都返回同样的 key 集合,
# 否则 verl _postprocess 会因为 KeyError 崩溃.
_METRICS_SCHEMA = {
    "format": 0.0,
    "answer": 0.0,
    "step_count_score": 0.0,
    "n_steps": 0,
    "numeric_correctness_score": 0.0,
    "n_equations": 0,
    "no_contradiction_score": 0.0,
    "final_in_process": 0,
    "reasoning_quality_score": 0.0,
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
    format_weight: float = 0.15,
    answer_weight: float = 0.60,
    reasoning_quality_weight: float = 0.25,
    invalid_reward: float = -0.5,
    **kwargs,
) -> tuple[float, dict[str, Any]]:
    text = str(completion or "").strip()
    if not text:
        return invalid_reward, _empty_metrics("empty_response", invalid_reward)

    gold_answer = reference.get("gold_answer")

    # 1. format check: 收尾是否含 "The final answer is N." 或 "#### N" 或 "boxed{N}"
    # 用 extract_answer 来判定: 如果能提取出 N, 视为 format ok
    pred_answer_raw = extract_answer(text)
    format_ok = 1.0 if pred_answer_raw is not None else 0.0

    # 2. answer check: N 数值是否匹配 gold
    answer_ok = 1.0 if (pred_answer_raw is not None and is_correct(pred_answer_raw, gold_answer)) else 0.0

    # 3. reasoning_quality 3 子项
    step_count, n_steps = _step_count_score(text)
    numeric_correctness, n_equations = _numeric_correctness_score(text)
    no_contradiction, final_in_process = _no_contradiction_score(text, pred_answer_raw)

    reasoning_quality_score = (
        0.5 * step_count
        + 0.3 * numeric_correctness
        + 0.2 * no_contradiction
    )

    total = (
        format_weight * format_ok
        + answer_weight * answer_ok
        + reasoning_quality_weight * reasoning_quality_score
    )

    m = dict(_METRICS_SCHEMA)
    m.update({
        "format": float(format_ok),
        "answer": float(answer_ok),
        "step_count_score": float(step_count),
        "n_steps": int(n_steps),
        "numeric_correctness_score": float(numeric_correctness),
        "n_equations": int(n_equations),
        "no_contradiction_score": float(no_contradiction),
        "final_in_process": int(final_in_process),
        "reasoning_quality_score": float(reasoning_quality_score),
        "reward": float(total),
        "reason": "ok" if format_ok else "no_final_answer_marker",
    })
    return float(total), m


def compute_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """verl entry point (single-sample kwargs mode, verl 0.8.0 contract).

    返回 dict with "score" key, 其他 key 走 reward_extra_info
    """
    if isinstance(ground_truth, dict):
        reference = ground_truth
    else:
        reference = {"gold_answer": ground_truth}
    r, m = score_response(solution_str, reference, **kwargs)
    return {"score": r, **m}
