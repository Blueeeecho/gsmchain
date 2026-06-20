"""LB-PRM v9 reward for ChainGSM.

v9 公式:
  R = max(0, 0.2 * r_format + 2.5 * r_answer + 1.5 * r_core - 0.5 * r_distractor)

  r_format = (I_TARGET + I_STEP_EXPR_VALUE + I_FINAL_EXPR + I_ANSWER) / 4
  r_answer = 1 if numeric_equal(pred_answer, gold_answer) else 0
  r_core   = 0.8 * sim_trace + 0.2 * sim_final
  r_distractor = max(0, sim_distractor - r_core)

  sim_trace  = normalized_edit_similarity(pred_trace_tokens, gold_trace_tokens)
  sim_final  = normalized_edit_similarity(tokens(FINAL_EXPR), tokens(gold_expression))
  sim_distractor = normalized_edit_similarity(tokens(pred_expr), distractor_trace_tokens)

Notes:
  - pred_trace_tokens are extracted from STEP blocks; we tokenize the
    expression AND keep the =VALUE tail so the gold-vs-pred sim_trace
    is structurally aligned.
  - distractor_trace_tokens in the parquet are *expression-only* (no
    =VALUE tail); we compare pred expression tokens to them.
  - r_distractor is forced to 0 for `original` (no distractor field).
  - edit distance uses python-Levenshtein (C implementation).
  - numeric comparison uses gsm_answer_extractor.is_correct (Fraction-based).
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

# matches "VAR: ...", "QTY: ...", "EXPR: ...", "VALUE: ...", "FINAL_EXPR: ...",
# "ANSWER: ...", "TARGET: ..."
KV_PATTERN = re.compile(r"^(TARGET|FINAL_EXPR|ANSWER|STEP\s+\d+|VAR|QTY|EXPR|VALUE)\s*:?\s*(.*)$", re.MULTILINE)
STEP_HEADER = re.compile(r"^STEP\s+(\d+)\s*:\s*$", re.MULTILINE)

TOKEN_PATTERN = re.compile(r"\*\*|(?<![\d.])-?\d+(?:\.\d+)?|[+\-*/]")

_METRICS_SCHEMA = {
    "format": 0.0,
    "answer": 0.0,
    "core_trace_sim": 0.0,
    "core_final_sim": 0.0,
    "core": 0.0,
    "distractor_sim": 0.0,
    "distractor": 0.0,
    "reason": "ok",
    "n_steps": 0,
}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _extract_pred_trace_tokens(text: str) -> list[str]:
    """Walk STEP blocks; for each STEP, take the EXPR/VALUE pair, tokenize
    expr, append =value, and join steps with <step>.
    """
    blocks: list[list[str]] = []
    cur: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m = STEP_HEADER.match(s)
        if m:
            if cur:
                blocks.append(cur); cur = []
            continue
        if s.upper().startswith("FINAL_EXPR"):
            if cur:
                blocks.append(cur); cur = []
            continue
        if s.upper().startswith("ANSWER"):
            if cur:
                blocks.append(cur); cur = []
            continue
        if s.startswith("EXPR"):
            expr = s.split(":", 1)[1].strip() if ":" in s else ""
            cur.append(("expr", expr))
        elif s.startswith("VALUE"):
            val = s.split(":", 1)[1].strip() if ":" in s else ""
            cur.append(("value", val))
    if cur:
        blocks.append(cur)

    flat: list[str] = []
    for i, block in enumerate(blocks):
        expr_val = ""
        val_val = ""
        for k, v in block:
            if k == "expr":
                expr_val = v
            elif k == "value":
                val_val = v
        if not expr_val:
            continue
        toks = _tokenize_expr(expr_val)
        toks.append("=")
        toks.append(val_val if val_val else "")
        if i > 0:
            flat.append("<step>")
        flat.extend(toks)
    return flat


def _extract_pred_final_expr(text: str) -> str:
    m = re.search(r"^FINAL_EXPR\s*:?\s*(.+?)\s*$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _extract_pred_target(text: str) -> str:
    m = re.search(r"^TARGET\s*:?\s*(.+?)\s*$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _has_any_step(text: str) -> bool:
    return bool(STEP_HEADER.search(text))


def _has_step_with_expr_value(text: str) -> bool:
    """A STEP block with non-empty EXPR and VALUE."""
    blocks = _extract_pred_trace_tokens(text)
    return any(t for t in blocks if t and t != "<step>")


def _tokenize_expr(expr: str) -> list[str]:
    if not expr:
        return []
    e = re.sub(r"[()]", "", expr.replace(" ", ""))
    return TOKEN_PATTERN.findall(e)


# ---------------------------------------------------------------------------
# Edit similarity
# ---------------------------------------------------------------------------


def _normalized_edit_similarity(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    dist = Levenshtein.distance(a, b)
    denom = max(len(a), len(b))
    if denom == 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - dist / denom))


# ---------------------------------------------------------------------------
# Main score
# ---------------------------------------------------------------------------


def score_response(
    completion: str,
    reference: dict[str, Any],
    format_weight: float = 0.2,
    answer_weight: float = 2.5,
    core_weight: float = 1.5,
    core_trace_w: float = 0.8,
    core_final_w: float = 0.2,
    distractor_weight: float = 0.5,
    invalid_reward: float = -0.5,
    **kwargs,
) -> tuple[float, dict[str, Any]]:
    text = str(completion or "").strip()
    if not text:
        return invalid_reward, _dict_metrics("empty_response")

    gold_answer = str(reference.get("answer") or reference.get("gold_answer") or "")
    gold_expr = reference.get("gold_expression") or ""
    _gtt = reference.get("gold_trace_tokens")
    gold_trace_tokens = list(_gtt.tolist()) if hasattr(_gtt, "tolist") else list(_gtt or [])
    _dt = reference.get("distractor_trace_tokens")
    distractor_tokens = list(_dt.tolist()) if hasattr(_dt, "tolist") else list(_dt or [])
    category = str(reference.get("category") or "")
    is_original = (category == "original") or not distractor_tokens

    # ----- format (low weight, never gates) -----
    has_target = 1.0 if _extract_pred_target(text) else 0.0
    has_step = 1.0 if _has_step_with_expr_value(text) else 0.0
    has_final = 1.0 if _extract_pred_final_expr(text) else 0.0
    has_answer = 1.0 if re.search(r"^ANSWER\s*:", text, re.MULTILINE | re.IGNORECASE) else 0.0
    r_format = (has_target + has_step + has_final + has_answer) / 4.0

    # ----- answer -----
    pred_answer = extract_answer(text)
    r_answer = 1.0 if (pred_answer is not None and is_correct(pred_answer, gold_answer)) else 0.0

    # ----- core (trace + final expression sim) -----
    pred_trace_tokens = _extract_pred_trace_tokens(text)
    sim_trace = _normalized_edit_similarity(pred_trace_tokens, list(gold_trace_tokens))

    pred_final = _extract_pred_final_expr(text)
    sim_final = _normalized_edit_similarity(_tokenize_expr(pred_final), _tokenize_expr(gold_expr))

    r_core = core_trace_w * sim_trace + core_final_w * sim_final

    # ----- distractor (only meaningful for non-original) -----
    if is_original:
        sim_distractor = 0.0
        r_distractor = 0.0
    else:
        sim_distractor = _normalized_edit_similarity(_tokenize_expr(pred_final), list(distractor_tokens))
        r_distractor = max(0.0, sim_distractor - r_core)

    total = (
        format_weight * r_format
        + answer_weight * r_answer
        + core_weight * r_core
        - distractor_weight * r_distractor
    )
    total = max(0.0, total)

    m = _dict_metrics("ok")
    m.update({
        "format": r_format,
        "answer": r_answer,
        "core_trace_sim": sim_trace,
        "core_final_sim": sim_final,
        "core": r_core,
        "distractor_sim": sim_distractor,
        "distractor": r_distractor,
        "n_steps": pred_trace_tokens.count("<step>") + (1 if pred_trace_tokens else 0),
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
    # tiny CLI smoke (not the full TDD test)
    sample = "TARGET: money_left\n\nSTEP 1:\nEXPR: 2*2\nVALUE: 4\n\nFINAL_EXPR: 15-(2+3+2*2)\nANSWER: 6"
    ref = {"answer": "6", "gold_expression": "15-(2+3+2*2)",
            "gold_trace_tokens": ["2", "*", "2", "=", "4", "<step>", "2", "+", "3", "+", "4", "=", "9", "<step>", "15", "-", "9", "=", "6"],
            "distractor_trace_tokens": [], "category": "original"}
    print(score_response(sample, ref))
