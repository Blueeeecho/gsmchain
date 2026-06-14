"""LB-PRM v7 reward for verl GRPO training (8-shot CoT protocol, 简化 3 子项).

相对 v6 (0.15/0.60/0.25) 和 v6b (0.10/0.55/0.35, 4 子项), v7 改为 3 子项简化:
  - format 0.10 (降, base 0.5B 格式基本 ok)
  - answer 0.70 (升, 主线目标是 original 数字对)
  - numeric_correctness 0.20 (保留, 0.5B base 算式正确率仅 22%, 重点拉)
  - 砍: step_count_score (0.74 区分度低), no_contradiction_score (0.875 满分噪声),
        equation_count_bonus (鼓励凑算式, 不利于推理)

总公式:
  total = 0.10·format + 0.70·answer + 0.20·numeric_correctness

verl 接口契约 (同 v6/v6b):
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
# v7 子项评分
# ---------------------------------------------------------------------------

def _numeric_correctness_score(text: str) -> tuple[float, int]:
    """算式正确率: 抓 "X op Y = Z", 验证 eval(X op Y) == Z (容差 1e-3).

    等同 v6b 的 _numeric_correctness_score (1 个子项, 不要 step_count 不要 eq_count_bonus).
    """
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


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

# 固定的 metrics schema. 所有 sample (无论是否成功解析) 都返回同样的 key 集合,
# 否则 verl _postprocess 会因为 KeyError 崩溃.
# v7 schema: 砍掉 v6b 的 step_count_score, n_steps, no_contradiction_score,
#            final_in_process, equation_count_bonus, n_equations_total,
#            reasoning_v2_score
_METRICS_SCHEMA = {
    "format": 0.0,
    "answer": 0.0,
    "numeric_correctness_score": 0.0,
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
    format_weight: float = 0.10,
    answer_weight: float = 0.70,
    numeric_correctness_weight: float = 0.20,
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

    # 3. numeric_correctness: 算式正确率
    numeric_correctness, n_equations = _numeric_correctness_score(text)

    total = (
        format_weight * format_ok
        + answer_weight * answer_ok
        + numeric_correctness_weight * numeric_correctness
    )

    m = dict(_METRICS_SCHEMA)
    m.update({
        "format": float(format_ok),
        "answer": float(answer_ok),
        "numeric_correctness_score": float(numeric_correctness),
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
