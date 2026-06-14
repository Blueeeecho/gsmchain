"""LB-PRM v8 reward for verl GRPO training (8-shot CoT protocol, 4 子项, 砍 numeric).

相对 v7 (3 子项 0.10/0.70/0.20), v8 改为 4 子项简化:
  - format 0.05 (降, base 0.5B 格式基本 ok)
  - answer 0.85 (升, 主线目标 original 数字对, 抢回主线梯度)
  - numeric_correctness 0.05 (大幅砍, v7 抢 answer 梯度的元凶)
  - step_count 0.05 (新, 鼓励展开算式, 0/1/2+ -> 0/0.5/1.0)

总公式:
  total = 0.05·format + 0.85·answer + 0.05·numeric_correctness + 0.05·step_count

v7 失败根因 (0.4428 vs 目标 0.46):
- numeric 子项 0.20 抢了 answer 0.70 梯度, step_200+ policy 退到 answer 0.10
- numeric 是 0-1 连续信号, answer 是 0/1 稀疏信号, GRPO 偏好连续信号
- v8 砍 numeric 0.05, 提 answer 0.85, 主线信号清晰

verl 接口契约 (同 v7):
  compute_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs) kwargs 模式
  返回 {"score": r, **metrics}, 所有 early-return 走 _METRICS_SCHEMA
"""
from __future__ import annotations

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
# Constants
# ---------------------------------------------------------------------------

# 算式匹配: "X op Y = Z" (带 =)
EQUATION_PATTERN = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/x×÷])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)"
)


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


# ---------------------------------------------------------------------------
# v8 子项评分
# ---------------------------------------------------------------------------

def _numeric_correctness_score(text: str) -> tuple[float, int]:
    """算式正确率: 抓 "X op Y = Z", 验证 eval(X op Y) == Z (容差 1e-3)."""
    equations = EQUATION_PATTERN.findall(text)
    if not equations:
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


def _step_count_score(n_equations: int) -> float:
    """算式数量评分: 鼓励展开算式 (v8 新加).

    0 算式: 0.0
    1 算式: 0.5
    2+ 算式: 1.0
    """
    if n_equations == 0:
        return 0.0
    elif n_equations == 1:
        return 0.5
    else:  # 2+
        return 1.0


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

# 固定的 metrics schema. 所有 sample (无论是否成功解析) 都返回同样的 key 集合.
# v8 schema (vs v7):
#   + step_count_score (新)
#   - 无其他变化
_METRICS_SCHEMA = {
    "format": 0.0,
    "answer": 0.0,
    "numeric_correctness_score": 0.0,
    "step_count_score": 0.0,
    "n_equations": 0,
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
    answer_weight: float = 0.85,
    numeric_correctness_weight: float = 0.05,
    step_count_weight: float = 0.05,
    invalid_reward: float = -0.5,
    **kwargs,
) -> tuple[float, dict[str, Any]]:
    text = str(completion or "").strip()
    if not text:
        return invalid_reward, _empty_metrics("empty_response", invalid_reward)

    gold_answer = reference.get("gold_answer")

    # 1. format check: 收尾是否含 "The final answer is N." 或 "#### N" 或 "boxed{N}"
    pred_answer_raw = extract_answer(text)
    format_ok = 1.0 if pred_answer_raw is not None else 0.0

    # 2. answer check: N 数值是否匹配 gold
    answer_ok = 1.0 if (pred_answer_raw is not None and is_correct(pred_answer_raw, gold_answer)) else 0.0

    # 3. numeric_correctness: 算式正确率
    numeric_correctness, n_equations = _numeric_correctness_score(text)

    # 4. step_count: 鼓励展开算式
    step_count = _step_count_score(n_equations)

    total = (
        format_weight * format_ok
        + answer_weight * answer_ok
        + numeric_correctness_weight * numeric_correctness
        + step_count_weight * step_count
    )

    m = dict(_METRICS_SCHEMA)
    m.update({
        "format": float(format_ok),
        "answer": float(answer_ok),
        "numeric_correctness_score": float(numeric_correctness),
        "step_count_score": float(step_count),
        "n_equations": int(n_equations),
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
