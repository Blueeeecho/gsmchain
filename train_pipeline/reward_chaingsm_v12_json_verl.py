"""NatTrace-GRPO v12 reward (JSON output compatible).

与 v11 公式相同:
  R = 3.0 * r_answer
    + 1.5 * r_step_value
    + 0.5 * r_core
    - 0.5 * r_distractor

区别: pred 解析层从 <<expr=val>> 切换到 JSON.steps[].{expression, value}。
- r_step_value: 抽 JSON.steps[].value, 跟 gold values 对齐
- r_core (trace + final): 抽 JSON.steps[].expression 拼成 trace tokens,
  final_expression 抽 JSON.final_expression 字段
- r_distractor: 同 v11
- r_answer: 抽 JSON.answer 字段
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import Levenshtein  # python-Levenshtein

_CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from gsm_answer_extractor import extract_answer, is_correct  # noqa: E402

TOKEN_PATTERN = re.compile(r"\*\*|[()]|(?<![\d.])-?\d+(?:\.\d+)?|[+\-*/]")

_METRICS_SCHEMA = {
    "answer": 0.0,
    "step_value": 0.0,
    "step_value_exact": 0.0,
    "core_trace_sim": 0.0,
    "core_final_sim": 0.0,
    "core": 0.0,
    "distractor_sim": 0.0,
    "distractor": 0.0,
    "n_steps": 0,
    "n_gold_steps": 0,
}


# ---------------------------------------------------------------------------
# JSON parsing (v12 new)
# ---------------------------------------------------------------------------

def _parse_json_output(text: str) -> dict | None:
    """Extract the first complete JSON object from text.
    Returns parsed dict or None.
    """
    text = str(text or "").strip()
    # Strategy 1: find first {...} block (DOTALL)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    # Strategy 2: full text parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def _extract_pred_answer_from_json(text: str) -> str | None:
    """从 JSON 抽 answer 字段, fallback 到 extract_answer (旧协议)."""
    parsed = _parse_json_output(text)
    if parsed is not None:
        if parsed.get("answer") is not None:
            return extract_answer(str(parsed["answer"]))
        if parsed.get("final_expression"):
            return extract_answer(str(parsed["final_expression"]))
    return extract_answer(text)


def _extract_pred_steps_from_json(text: str) -> list[tuple[str, str]]:
    """从 JSON.steps[] 抽 (expression, value) 对."""
    parsed = _parse_json_output(text)
    if parsed is None:
        return []
    steps_raw = parsed.get("steps", [])
    if not isinstance(steps_raw, list):
        return []
    out = []
    for s in steps_raw:
        if not isinstance(s, dict):
            continue
        expr = str(s.get("expression", "")).strip()
        val = str(s.get("value", "")).strip()
        if expr and val:
            out.append((expr, val))
    return out


def _extract_pred_final_expr_from_json(text: str) -> str:
    """从 JSON.final_expression 抽 final 表达式."""
    parsed = _parse_json_output(text)
    if parsed is None:
        return ""
    return str(parsed.get("final_expression", "")).strip()


def _extract_pred_trace_tokens_from_json(text: str) -> list[str]:
    """从 JSON.steps[] 抽 trace tokens (跟 gold_trace_tokens 同格式)."""
    steps = _extract_pred_steps_from_json(text)
    flat = []
    for i, (expr, val) in enumerate(steps):
        if i > 0:
            flat.append("<step>")
        flat.extend(_tokenize_expr(expr))
        flat.append("=")
        flat.extend(_tokenize_expr(val))
    return flat


# ---------------------------------------------------------------------------
# Reused from v11
# ---------------------------------------------------------------------------

def _tokenize_expr(expr: str) -> list[str]:
    if not expr:
        return []
    s = re.sub(r"\s+", "", expr)
    if not s:
        return []
    toks = TOKEN_PATTERN.findall(s)
    if not toks:
        return []
    if len(toks) == 1:
        return toks
    if len(toks) % 2 == 0:
        return []
    return toks


def _normalized_edit_similarity(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    sa = " ".join(a)
    sb = " ".join(b)
    if not sa and not sb:
        return 1.0
    dist = Levenshtein.distance(sa, sb)
    return 1.0 - dist / max(len(sa), len(sb))


def _extract_gold_step_values(gold_trace_tokens) -> list[str]:
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
            cur = []
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
    """v12 reward: 公式同 v11, pred 解析层切到 JSON."""
    text = str(completion or "").strip()
    if not text:
        return invalid_reward, _dict_metrics("empty_response")

    # ----- 解析 reference -----
    gold_answer = str(reference.get("answer") or reference.get("gold_answer") or "")
    gold_expr = reference.get("gold_expression") or ""
    _gtt = reference.get("gold_trace_tokens")
    gold_trace_tokens = list(_gtt.tolist()) if hasattr(_gtt, "tolist") else list(_gtt or [])
    category = str(reference.get("category") or "")
    is_original = (category == "original") or not reference.get("distractor_enabled", False)

    # ----- r_answer (JSON-aware) -----
    pred_answer = _extract_pred_answer_from_json(text)
    r_answer = 1.0 if (pred_answer is not None and is_correct(pred_answer, gold_answer)) else 0.0

    # ----- r_step_value (JSON-aware) -----
    gold_values = _extract_gold_step_values(gold_trace_tokens)
    pred_steps = _extract_pred_steps_from_json(text)
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

    # ----- r_core (JSON-aware) -----
    pred_trace_tokens = _extract_pred_trace_tokens_from_json(text)
    sim_trace = _normalized_edit_similarity(pred_trace_tokens, list(gold_trace_tokens))
    pred_final = _extract_pred_final_expr_from_json(text)
    sim_final = _normalized_edit_similarity(_tokenize_expr(pred_final), _tokenize_expr(gold_expr))
    r_core = core_trace_w * sim_trace + core_final_w * sim_final

    # ----- r_distractor -----
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
