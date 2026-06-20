"""NatTrace-GRPO v11 reward.

v10 -> v11 关键改进 (基于真实评测数据诊断):

公式 (signed, 不 max(0, ·)):
  R = 3.0 * r_answer
    + 1.5 * r_step_value      # NEW: 跟 gold step values 对齐 (vs v10 self-consistency)
    + 0.5 * r_core            # 权重下调 (vs v10 1.2) - 模仿 trace != 算对
    - 0.5 * r_distractor

变化:
  - 删除 r_format (0.999 永远饱和, 浪费 0.2 weight)
  - r_calc 替换为 r_step_value (用 gold_trace_tokens 拆出 gold values, 跟 pred 比)
  - r_answer 2.5 -> 3.0 (更强信号, 把错答拉开)
  - r_core 1.2 -> 0.5 (避免 trace 模仿覆盖 answer 信号)
  - core_trace_w / core_final_w 改回 0.8 / 0.2 (v9 风格, 更聚焦 trace)
  - r_distractor 0.5 保留, 但在 original 仍 = 0 (没变)

验证 (288 个真实评测预测):
  - v10 vs v11 on original (n=196):
    corr(is_correct): 0.985 -> 0.989
    wrong mean reward: 0.69 -> 0.13 (-81%)
  - v10 vs v11 on decoy (n=92):
    corr(is_correct): 0.973 -> 0.978
    wrong mean reward: 0.83 -> 0.18 (-78%)
  - 错答 median reward: 0.672 -> 0.107 (-84%)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import Levenshtein  # python-Levenshtein

_CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from gsm_answer_extractor import extract_answer, is_correct  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOKEN_PATTERN = re.compile(r"\*\*|[()]|(?<![\d.])-?\d+(?:\.\d+)?|[+\-*/]")

STEP_PATTERN = re.compile(r"<<\s*([^>]+?)\s*>>")
FINAL_PATTERN = re.compile(r"<<\s*FINAL\s*:\s*([^>]+?)\s*>>", re.IGNORECASE)
ANSWER_PATTERN = re.compile(r"^\s*ANSWER\s*:\s*([^\n]+?)\s*$", re.MULTILINE | re.IGNORECASE)
TARGET_PATTERN = re.compile(r"^\s*TARGET\s*:\s*([^\n]+?)\s*$", re.MULTILINE | re.IGNORECASE)

_METRICS_SCHEMA = {
    "format": 0.0,
    "answer": 0.0,
    "step_value": 0.0,
    "core_trace_sim": 0.0,
    "core_final_sim": 0.0,
    "core": 0.0,
    "step_value_exact": 0.0,
    "distractor_sim": 0.0,
    "distractor": 0.0,
    "reason": "ok",
    "n_steps": 0,
    "n_gold_steps": 0,
}


# ---------------------------------------------------------------------------
# Tokenize / similarity
# ---------------------------------------------------------------------------

def _tokenize_expr(expr: str) -> list[str]:
    if not expr:
        return []
    e = expr.replace(" ", "")
    return TOKEN_PATTERN.findall(e)


def _normalized_edit_similarity(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    dist = Levenshtein.distance(a, b)
    denom = max(len(a), len(b))
    if denom == 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - dist / denom))


# ---------------------------------------------------------------------------
# Parsing (新 CoT 协议)
# ---------------------------------------------------------------------------

def _extract_pred_target(text: str) -> str:
    m = TARGET_PATTERN.search(text)
    return m.group(1).strip() if m else ""


def _extract_pred_final_expr(text: str) -> str:
    m = FINAL_PATTERN.search(text)
    if not m:
        return ""
    body = m.group(1).strip()
    m2 = re.match(r"(.+?)\s*=\s*(.+)$", body)
    if m2:
        return m2.group(1).strip()
    return body


def _extract_pred_steps(text: str) -> list[tuple[str, str]]:
    steps = []
    for m in STEP_PATTERN.finditer(text):
        body = m.group(1).strip()
        if body.upper().startswith("FINAL"):
            continue
        m2 = re.match(r"(.+?)\s*=\s*(.+)$", body)
        if m2:
            steps.append((m2.group(1).strip(), m2.group(2).strip()))
    return steps


def _has_any_step(text: str) -> bool:
    return bool(STEP_PATTERN.search(text))


def _extract_pred_trace_tokens(text: str) -> list[str]:
    steps = _extract_pred_steps(text)
    flat: list[str] = []
    for i, (expr, val) in enumerate(steps):
        if i > 0:
            flat.append("<step>")
        flat.extend(_tokenize_expr(expr))
        flat.append("=")
        flat.extend(_tokenize_expr(val))
    return flat


# ---------------------------------------------------------------------------
# gold trace token -> gold step values (NEW for v11)
# ---------------------------------------------------------------------------

def _extract_gold_step_values(gold_trace_tokens) -> list[str]:
    """从 gold_trace_tokens (e.g. ['8','*','3','=','24','<step>',...,'=','5'])
    提取每一步的 value 字符串. 返回 ['24', '5']."""
    if gold_trace_tokens is None:
        return []
    if hasattr(gold_trace_tokens, "tolist"):
        gold_trace_tokens = list(gold_trace_tokens.tolist())
    else:
        gold_trace_tokens = list(gold_trace_tokens)
    values: list[str] = []
    cur: list[str] = []
    in_val = False
    for tok in gold_trace_tokens:
        if tok == "=":
            cur = []  # 丢弃 expr 部分
            in_val = True
        elif tok == "<step>":
            if in_val and cur:
                values.append(''.join(cur))
            cur = []
            in_val = False
        else:
            cur.append(tok)
    if in_val and cur:
        values.append(''.join(cur))
    return values


# ---------------------------------------------------------------------------
# Main score
# ---------------------------------------------------------------------------

def score_response(
    completion: str,
    reference: dict[str, Any],
    answer_weight: float = 3.0,
    step_value_weight: float = 1.5,
    core_weight: float = 0.5,
    core_trace_w: float = 0.8,
    core_final_w: float = 0.2,
    distractor_weight: float = 0.5,
    invalid_reward: float = -0.5,
    **kwargs,
) -> tuple[float, dict[str, Any]]:
    """v11 reward: ans + step_value (vs gold) + core - distractor.

    关键变化:
      - 删 r_format (饱和)
      - 删 r_calc self-consistency -> 替换为 r_step_value (vs gold values)
      - r_answer 2.5 -> 3.0
      - r_core 1.2 -> 0.5
      - core_trace_w / core_final_w 0.7/0.3 -> 0.8/0.2
    """
    text = str(completion or "").strip()
    if not text:
        return invalid_reward, _dict_metrics("empty_response")

    # ----- 解析 reference -----
    gold_answer = str(reference.get("answer") or reference.get("gold_answer") or "")
    gold_expr = reference.get("gold_expression") or ""
    _gtt = reference.get("gold_trace_tokens")
    gold_trace_tokens = list(_gtt.tolist()) if hasattr(_gtt, "tolist") else list(_gtt or [])
    _dt = reference.get("distractor_trace_tokens")
    distractor_tokens = list(_dt.tolist()) if hasattr(_dt, "tolist") else list(_dt or [])
    category = str(reference.get("category") or "")
    is_original = (category == "original") or not reference.get("distractor_enabled", False)

    # ----- r_answer -----
    pred_answer = extract_answer(text)
    r_answer = 1.0 if (pred_answer is not None and is_correct(pred_answer, gold_answer)) else 0.0

    # ----- r_step_value (NEW: 跟 gold step values 对齐, 不是 self-consistency) -----
    gold_values = _extract_gold_step_values(gold_trace_tokens)
    pred_steps = _extract_pred_steps(text)
    pred_value_strs = [v.strip() for _, v in pred_steps]
    n_gold_steps = len(gold_values)
    n_matched = 0
    if gold_values and pred_value_strs:
        for gv in gold_values:
            if any(pv == gv for pv in pred_value_strs):
                n_matched += 1
        r_step_value = n_matched / len(gold_values)
    else:
        r_step_value = 0.0

    # ----- r_core (trace + final sim, 跟 v9/v10 类似) -----
    pred_trace_tokens = _extract_pred_trace_tokens(text)
    sim_trace = _normalized_edit_similarity(pred_trace_tokens, list(gold_trace_tokens))
    pred_final = _extract_pred_final_expr(text)
    sim_final = _normalized_edit_similarity(_tokenize_expr(pred_final), _tokenize_expr(gold_expr))
    r_core = core_trace_w * sim_trace + core_final_w * sim_final

    # ----- r_distractor (跟 v10 一样) -----
    distractor_expr_str = reference.get("distractor_expression") or ""
    if is_original or not distractor_expr_str:
        sim_distractor = 0.0
        r_distractor = 0.0
    else:
        distractor_toks = _tokenize_expr(distractor_expr_str)
        sim_distractor = _normalized_edit_similarity(_tokenize_expr(pred_final), distractor_toks)
        if pred_steps:
            step_max = max(
                _normalized_edit_similarity(_tokenize_expr(e), distractor_toks)
                for e, _ in pred_steps
            )
            sim_distractor = max(sim_distractor, step_max)
        r_distractor = max(0.0, sim_distractor - r_core)

    # ----- total (signed) -----
    total = (
        answer_weight * r_answer
        + step_value_weight * r_step_value
        + core_weight * r_core
        - distractor_weight * r_distractor
    )

    m = _dict_metrics("ok")
    m.update({
        "answer": r_answer,
        "step_value": r_step_value,
        "step_value_exact": r_step_value,
        "core_trace_sim": sim_trace,
        "core_final_sim": sim_final,
        "core": r_core,
        "distractor_sim": sim_distractor,
        "distractor": r_distractor,
        "n_steps": len(pred_steps),
        "n_gold_steps": n_gold_steps,
    })
    return float(total), m


def _dict_metrics(reason: str) -> dict[str, Any]:
    m = dict(_METRICS_SCHEMA); m["reason"] = reason; return m


# ---------------------------------------------------------------------------
# Verl entry point
# ---------------------------------------------------------------------------

def compute_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if isinstance(ground_truth, dict):
        reference = ground_truth
    else:
        reference = {"answer": str(ground_truth)}
    r, m = score_response(solution_str, reference, **kwargs)
    return {"score": r, **m}


if __name__ == "__main__":
    import argparse
    import json
    p = argparse.ArgumentParser(description="Test v11 reward on single sample.")
    p.add_argument("--text", required=True, help="模型输出文本")
    p.add_argument("--reference-json", required=True, help="reference dict 的 JSON 字符串")
    args = p.parse_args()
    ref = json.loads(args.reference_json)
    r, m = score_response(args.text, ref)
    print(f"reward: {r}")
    print(f"metrics: {json.dumps(m, indent=2, ensure_ascii=False)}")
