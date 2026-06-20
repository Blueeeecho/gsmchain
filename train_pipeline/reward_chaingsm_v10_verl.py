"""NatTrace-GRPO v10-signed reward.

公式 (signed, **不** max(0, ·)):
  R = 0.2 * r_format + 2.5 * r_answer + 1.2 * r_core + 0.3 * r_calc - 0.5 * r_distractor

  r_format  = (I_TARGET + I_STEP + I_FINAL + I_ANSWER) / 4
  r_answer  = 1 if ANSWER == gold (Fraction compare) else 0
  r_core    = 0.7 * sim_trace + 0.3 * sim_final
  r_calc    = 0.8 * r_step_calc + 0.2 * r_final_calc
  r_distractor = max(0, sim_distractor - r_core)   # original 类别 = 0

解析协议: 新 CoT 协议
  - 普通 step: <<expr = value>> (单步)
  - FINAL:     <<FINAL: expr = answer>>
  - ANSWER:    ANSWER: N
  - TARGET:    TARGET: ...

数据来源: chaingsm_data/data/final/grpo/all_grpo_cot.parquet
  - ground_truth 含 gold_trace_tokens (带括号) / distractor_expression / distractor_trace_tokens=[] / category
  - 复用 v9 的 tokenize/edit-similarity 实现, 但保留括号 (跟 GRPO 数据一致)

设计: docs/superpowers/specs/2026-06-17-grpo-v10-signed-design.md
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
from fractions import Fraction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 跟 v9 一样, 但加括号 (跟 GRPO 数据带括号对齐)
TOKEN_PATTERN = re.compile(r"\*\*|[()]|(?<![\d.])-?\d+(?:\.\d+)?|[+\-*/]")

# 协议 marker 正则 (新 CoT 协议)
STEP_PATTERN = re.compile(r"<<\s*([^>]+?)\s*>>")
FINAL_PATTERN = re.compile(r"<<\s*FINAL\s*:\s*([^>]+?)\s*>>", re.IGNORECASE)
ANSWER_PATTERN = re.compile(r"^\s*ANSWER\s*:\s*([^\n]+?)\s*$", re.MULTILINE | re.IGNORECASE)
TARGET_PATTERN = re.compile(r"^\s*TARGET\s*:\s*([^\n]+?)\s*$", re.MULTILINE | re.IGNORECASE)

_METRICS_SCHEMA = {
    "format": 0.0,
    "answer": 0.0,
    "core_trace_sim": 0.0,
    "core_final_sim": 0.0,
    "core": 0.0,
    "calc_step": 0.0,
    "calc_final": 0.0,
    "calc": 0.0,
    "distractor_sim": 0.0,
    "distractor": 0.0,
    "reason": "ok",
    "n_steps": 0,
}


# ---------------------------------------------------------------------------
# Tokenize / similarity (复用 v9, 不删括号)
# ---------------------------------------------------------------------------

def _tokenize_expr(expr: str) -> list[str]:
    """跟 GRPO 数据带括号一致, 不删括号."""
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
# 解析 (新 CoT 协议)
# ---------------------------------------------------------------------------

def _extract_pred_target(text: str) -> str:
    m = TARGET_PATTERN.search(text)
    return m.group(1).strip() if m else ""


def _extract_pred_final_expr(text: str) -> str:
    m = FINAL_PATTERN.search(text)
    if not m:
        return ""
    body = m.group(1).strip()
    # 拆 "expr = answer" 形式, 留 expr
    m2 = re.match(r"(.+?)\s*=\s*(.+)$", body)
    if m2:
        return m2.group(1).strip()
    return body


def _extract_pred_steps(text: str) -> list[tuple[str, str]]:
    """抽所有 <<expr = value>>, 排除 FINAL 块. 返回 [(expr, value), ...]"""
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
    """是否有任何 <<...>> 块 (包括 FINAL)"""
    return bool(STEP_PATTERN.search(text))


def _extract_pred_trace_tokens(text: str) -> list[str]:
    """跟 GRPO 数据格式对齐: expr_toks + ['='] + value_toks, step 间 <step>"""
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
# 等式自洽性 (新机制)
# ---------------------------------------------------------------------------

def _safe_frac(s: str):
    try:
        return Fraction(str(s).strip().replace(" ", ""))
    except (ZeroDivisionError, ValueError, TypeError):
        return None


def _eval_expr(expr: str):
    """安全求值表达式. 失败返 None."""
    e = expr.strip()
    if not e:
        return None
    # 只允许数字 / 运算符 / 括号 / 小数点 / 负号开头
    if not re.match(r"^[\d\s+\-*/().]+$", e):
        return None
    try:
        return Fraction(eval(e, {"__builtins__": {}}, {}))
    except Exception:
        return None


def _is_step_calc_correct(expr: str, value: str) -> bool:
    fe, fv = _eval_expr(expr), _safe_frac(value)
    if fe is not None and fv is not None:
        return fe == fv
    return False


def _is_final_calc_correct(expr: str, answer: str) -> bool:
    fe, fv = _eval_expr(expr), _safe_frac(answer)
    if fe is not None and fv is not None:
        return fe == fv
    return False


# ---------------------------------------------------------------------------
# Main score
# ---------------------------------------------------------------------------

def score_response(
    completion: str,
    reference: dict[str, Any],
    format_weight: float = 0.2,
    answer_weight: float = 2.5,
    core_weight: float = 1.2,
    core_trace_w: float = 0.7,
    core_final_w: float = 0.3,
    calc_weight: float = 0.3,
    calc_step_w: float = 0.8,
    calc_final_w: float = 0.2,
    distractor_weight: float = 0.5,
    invalid_reward: float = -0.5,
    **kwargs,
) -> tuple[float, dict[str, Any]]:
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

    # ----- r_format (4 marker / 4) -----
    has_target = 1.0 if _extract_pred_target(text) else 0.0
    has_step = 1.0 if _has_any_step(text) else 0.0
    has_final = 1.0 if _extract_pred_final_expr(text) else 0.0
    has_answer = 1.0 if ANSWER_PATTERN.search(text) else 0.0
    r_format = (has_target + has_step + has_final + has_answer) / 4.0

    # ----- r_answer -----
    pred_answer = extract_answer(text)
    r_answer = 1.0 if (pred_answer is not None and is_correct(pred_answer, gold_answer)) else 0.0

    # ----- r_core (trace + final sim) -----
    pred_trace_tokens = _extract_pred_trace_tokens(text)
    sim_trace = _normalized_edit_similarity(pred_trace_tokens, list(gold_trace_tokens))
    pred_final = _extract_pred_final_expr(text)
    sim_final = _normalized_edit_similarity(_tokenize_expr(pred_final), _tokenize_expr(gold_expr))
    r_core = core_trace_w * sim_trace + core_final_w * sim_final

    # ----- r_calc (等式自洽, 新机制) -----
    steps = _extract_pred_steps(text)
    if steps:
        n_correct = sum(1 for e, v in steps if _is_step_calc_correct(e, v))
        r_step_calc = n_correct / len(steps)
    else:
        r_step_calc = 0.0
    r_final_calc = 1.0 if (pred_final and _is_final_calc_correct(pred_final, pred_answer or gold_answer)) else 0.0
    r_calc = calc_step_w * r_step_calc + calc_final_w * r_final_calc

    # ----- r_distractor (only meaningful for non-original) -----
    # 拿 distractor_expression (字符串), 跟 pred_final / step exprs 比相似度
    distractor_expr_str = reference.get("distractor_expression") or ""

    if is_original or not distractor_expr_str:
        sim_distractor = 0.0
        r_distractor = 0.0
    else:
        # v10: 跟 distractor_expression 字符串比 (不是 distractor_trace_tokens, 后者永远空)
        distractor_toks = _tokenize_expr(distractor_expr_str)
        sim_distractor = _normalized_edit_similarity(_tokenize_expr(pred_final), distractor_toks)
        # 也算 step exprs vs distractor 取 max
        if steps:
            step_max = max(
                _normalized_edit_similarity(_tokenize_expr(e), distractor_toks)
                for e, _ in steps
            )
            sim_distractor = max(sim_distractor, step_max)
        r_distractor = max(0.0, sim_distractor - r_core)

    # ----- signed total (不 max(0, ·), v10 关键) -----
    total = (
        format_weight * r_format
        + answer_weight * r_answer
        + core_weight * r_core
        + calc_weight * r_calc
        - distractor_weight * r_distractor
    )

    m = _dict_metrics("ok")
    m.update({
        "format": r_format,
        "answer": r_answer,
        "core_trace_sim": sim_trace,
        "core_final_sim": sim_final,
        "core": r_core,
        "calc_step": r_step_calc,
        "calc_final": r_final_calc,
        "calc": r_calc,
        "distractor_sim": sim_distractor,
        "distractor": r_distractor,
        "n_steps": len(steps),
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
    p = argparse.ArgumentParser(description="Test v10 reward on single sample.")
    p.add_argument("--text", required=True, help="模型输出文本")
    p.add_argument("--reference-json", required=True, help="reference dict 的 JSON 字符串")
    args = p.parse_args()
    ref = json.loads(args.reference_json)
    r, m = score_response(args.text, ref)
    print(f"reward: {r}")
    print(f"metrics: {json.dumps(m, indent=2, ensure_ascii=False)}")
