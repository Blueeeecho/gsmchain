"""Patch reward_chaingsm_lbprm_v2_verl.py to add the 4 advisor-required checks.

Additions (per advisor feedback 2026-06-08):
  1. step_calculation_consistency: eval(step.expression) == step.value
  2. final_consistency: eval(final_expression) == answer
                        AND last_step.value == answer
                        AND eval(last_step.expression) == eval(final_expression)
  3. terminal_step_liveness: last step is live if its expression/value matches
     final_expression or answer (handled inside _compute_liveness, already present;
     we just also add: eval(last_step.expression) == eval(final_expression))
  4. anti_degenerate_penalty: penalize empty steps, too many steps, repeated
     expressions, un-evaluable expressions, non-numeric values, final_expression
     == answer literal

New total formula:
    total = 0.2 * format + 0.4 * answer + 0.4 * chain_quality_score

chain_quality_score is a weighted blend of:
  - liveness_score (existing)
  - step_consistency_score (new)
  - final_consistency_score (new)
  - anti_degenerate_score (new)

This script rewrites specific functions in the v2 file via regex.
"""
from __future__ import annotations

import re
from pathlib import Path

PATH = Path("train_pipeline/reward_chaingsm_lbprm_v2_verl.py")
text = PATH.read_text()

# ---------------------------------------------------------------------------
# 1) Add helper functions right after _is_correct (before _compute_liveness)
# ---------------------------------------------------------------------------
HELPERS = '''
# ---------------------------------------------------------------------------
# Helpers for v2 (advisor-required checks)
# ---------------------------------------------------------------------------

def _safe_eval(expr: str) -> float | None:
    """Evaluate a simple arithmetic expression using ast (handles + - * / **, parens).

    Returns None on any parse/eval error.
    """
    try:
        import ast as _ast
        node = _ast.parse(str(expr).replace("^", "**"), mode="eval")
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
        if isinstance(node.op, _ast.Add):   return left + right
        if isinstance(node.op, _ast.Sub):   return left - right
        if isinstance(node.op, _ast.Mult):  return left * right
        if isinstance(node.op, _ast.Div):   return left / right
        if isinstance(node.op, _ast.Pow):   return left ** right
    raise ValueError(type(node).__name__)


def _step_consistency(steps: list[dict]) -> tuple[float, int, int]:
    """Check eval(step.expression) == step.value for every step.

    Returns (score, n_consistent, n_total).
    Each step contributes equally: score = n_consistent / n_total.
    Non-dict steps or un-evaluable expressions: that step counts as 0.
    """
    if not steps:
        return 0.0, 0, 0
    n_ok = 0
    n_total = 0
    for step in steps:
        if not isinstance(step, dict):
            n_total += 1
            continue
        n_total += 1
        expr = str(step.get("expression", "") or "")
        val = str(step.get("value", "") or "")
        ef = _safe_eval(expr)
        vf = _to_float(val)
        if ef is not None and vf is not None and abs(ef - vf) < 1e-3:
            n_ok += 1
    return (n_ok / n_total) if n_total else 0.0, n_ok, n_total


def _final_consistency(steps: list[dict], final_expr: str, pred_answer: Any) -> tuple[float, dict]:
    """Three sub-checks:
      (i)   eval(final_expression) == answer
      (ii)  last_step.value == answer (float-tolerant)
      (iii) eval(last_step.expression) == eval(final_expression)
    Each contributes 1/3 of the score. Returns (score, details).
    """
    details = {"fc_eval_fe_ans": False, "fc_last_val_ans": False, "fc_last_expr_fe": False}
    if not steps:
        return 0.0, details
    ans_f = _to_float(pred_answer)
    fe_val = _safe_eval(final_expr)
    if fe_val is not None and ans_f is not None and abs(fe_val - ans_f) < 1e-6:
        details["fc_eval_fe_ans"] = True
    last = steps[-1] if isinstance(steps[-1], dict) else None
    if last is not None:
        last_val = _to_float(last.get("value", ""))
        if last_val is not None and ans_f is not None and abs(last_val - ans_f) < 1e-6:
            details["fc_last_val_ans"] = True
        last_expr_val = _safe_eval(str(last.get("expression", "") or ""))
        if last_expr_val is not None and fe_val is not None and abs(last_expr_val - fe_val) < 1e-6:
            details["fc_last_expr_fe"] = True
    return sum(details.values()) / 3.0, details


def _anti_degenerate(steps: list[dict], final_expr: str, pred_answer: Any) -> tuple[float, dict]:
    """Penalty score in [0, 1]; 1.0 = no degeneracy, lower = more degenerate.

    Degenerate behaviors detected:
      - empty steps                                 -> 0
      - step count > 12 (excessive)                -> -0.2
      - repeated identical expressions              -> -0.3 per repeat (capped -0.6)
      - any step expression that cannot be parsed  -> -0.2
      - any step value that is non-numeric         -> -0.2
      - final_expression equals the literal answer -> -0.3 (bypass via "10" only)
    """
    details: dict[str, Any] = {}
    if not steps:
        return 0.0, {"reason": "empty_steps"}
    n = len(steps)
    score = 1.0

    if n > 12:
        score -= 0.2
        details["too_many_steps"] = True

    # Repeated identical expressions
    exprs = [str(s.get("expression", "")) for s in steps if isinstance(s, dict)]
    seen: dict[str, int] = {}
    for e in exprs:
        seen[e] = seen.get(e, 0) + 1
    repeats = sum(max(0, c - 1) for c in seen.values())
    if repeats > 0:
        score -= min(0.6, 0.3 * repeats)
        details["repeated_expressions"] = repeats

    # Un-evaluable expressions
    n_uneval = sum(1 for e in exprs if _safe_eval(e) is None)
    if n_uneval > 0:
        score -= 0.2
        details["unevaluable_expressions"] = n_uneval

    # Non-numeric values
    n_nonnum = sum(1 for s in steps if isinstance(s, dict) and _to_float(s.get("value", "")) is None)
    if n_nonnum > 0:
        score -= 0.2
        details["non_numeric_values"] = n_nonnum

    # final_expression == literal answer (the "10" bypass)
    fe_f = _safe_eval(final_expr)
    ans_f = _to_float(pred_answer)
    if fe_f is not None and ans_f is not None and abs(fe_f - ans_f) < 1e-6:
        # If the final_expression has length <= len(str(ans)) + 2, it's a literal bypass
        if len(final_expr.strip()) <= len(str(ans_f)) + 2:
            score -= 0.3
            details["final_is_literal_answer"] = True

    return max(0.0, score), details


'''

# Insert helpers just after the _is_correct function (before _compute_liveness)
marker = "def _compute_liveness(steps: list[dict], final_expr: str, pred_answer: Any) -> list[bool]:"
text = text.replace(marker, HELPERS + "\n" + marker)

# ---------------------------------------------------------------------------
# 2) Augment terminal-step liveness rule inside _compute_liveness
#    Add: if last step and eval(last.expression) == eval(final_expression): live=True
# ---------------------------------------------------------------------------
old_terminal = """            elif val.strip() == str(pred_answer).strip():
                live = True

        is_live.append(live)"""
new_terminal = """            elif val.strip() == str(pred_answer).strip():
                live = True

        # (c') v2 addition: last step + eval(step.expression) == eval(final_expression)
        if (
            not live
            and k == n - 1
            and final_expr
        ):
            expr_val = _safe_eval(str(step.get("expression", "") or ""))
            fe_val = _safe_eval(final_expr)
            if expr_val is not None and fe_val is not None and abs(expr_val - fe_val) < 1e-6:
                live = True

        is_live.append(live)"""
text = text.replace(old_terminal, new_terminal)

# ---------------------------------------------------------------------------
# 3) Replace the body of score_response to use chain_quality_score
# ---------------------------------------------------------------------------
# Find from "# ---- format ----" through "liveness_score = 1.0" and the total calc
old_score_block = """    # ---- format ----
    format_ok = 1.0 if (isinstance(steps, list) and steps and final_expr and pred_answer is not None) else 0.0
    if not format_ok:
        return invalid_reward * 0.5, {"format": 0.0, "answer": 0.0, "liveness": 0.0, "reason": "empty_steps"}

    # ---- answer ----
    gold_answer = reference.get("gold_answer")
    answer_ok = 1.0 if _is_correct(pred_answer, gold_answer) else 0.0

    # ---- liveness ----
    n = len(steps)
    is_live = _compute_liveness(steps, final_expr, pred_answer)
    n_live = sum(is_live)
    first_dead = next((k for k, x in enumerate(is_live) if not x), None)

    # per-step rewards
    per_step: list[float] = []
    for k, live in enumerate(is_live):
        if live:
            per_step.append(0.1)
        else:
            per_step.append(-0.5 if k == first_dead else -0.1)
    per_step_sum = sum(per_step)

    # normalize to [0, 1]
    max_sum = 0.1 * n
    min_sum = -0.5 - 0.1 * (n - 1) if n > 0 else 0.0
    if max_sum > min_sum:
        liveness_score = (per_step_sum - min_sum) / (max_sum - min_sum)
        liveness_score = max(0.0, min(1.0, liveness_score))
    else:
        liveness_score = 1.0

    total = (
        format_weight * format_ok
        + answer_weight * answer_ok
        + liveness_weight * liveness_score
    )

    return float(total), {
        "format": float(format_ok),
        "answer": float(answer_ok),
        "liveness_ratio": n_live / n,
        "liveness_score": float(liveness_score),
        "first_dead_step": first_dead,
        "n_live": n_live,
        "n_steps": n,
        "reward": float(total),
    }"""

new_score_block = """    # ---- format ----
    format_ok = 1.0 if (isinstance(steps, list) and steps and final_expr and pred_answer is not None) else 0.0
    if not format_ok:
        return invalid_reward * 0.5, {"format": 0.0, "answer": 0.0, "liveness": 0.0, "reason": "empty_steps"}

    # ---- answer ----
    gold_answer = reference.get("gold_answer")
    answer_ok = 1.0 if _is_correct(pred_answer, gold_answer) else 0.0

    # ---- liveness (existing) ----
    n = len(steps)
    is_live = _compute_liveness(steps, final_expr, pred_answer)
    n_live = sum(is_live)
    first_dead = next((k for k, x in enumerate(is_live) if not x), None)

    per_step: list[float] = []
    for k, live in enumerate(is_live):
        if live:
            per_step.append(0.1)
        else:
            per_step.append(-0.5 if k == first_dead else -0.1)
    per_step_sum = sum(per_step)
    max_sum = 0.1 * n
    min_sum = -0.5 - 0.1 * (n - 1) if n > 0 else 0.0
    if max_sum > min_sum:
        liveness_score = (per_step_sum - min_sum) / (max_sum - min_sum)
        liveness_score = max(0.0, min(1.0, liveness_score))
    else:
        liveness_score = 1.0

    # ---- v2 additions: step consistency + final consistency + anti-degenerate ----
    step_cons_score, n_cons, n_total = _step_consistency(steps)
    final_cons_score, final_cons_details = _final_consistency(steps, final_expr, pred_answer)
    anti_degen_score, anti_degen_details = _anti_degenerate(steps, final_expr, pred_answer)

    # chain_quality_score: weighted blend (0.4 / 0.3 / 0.3)
    chain_quality_score = (
        0.4 * liveness_score
        + 0.3 * step_cons_score
        + 0.3 * final_cons_score
    ) * anti_degen_score  # anti-degenerate multiplies (0..1)

    total = (
        format_weight * format_ok
        + answer_weight * answer_ok
        + liveness_weight * chain_quality_score
    )

    return float(total), {
        "format": float(format_ok),
        "answer": float(answer_ok),
        "liveness_ratio": n_live / n,
        "liveness_score": float(liveness_score),
        "step_consistency_score": float(step_cons_score),
        "final_consistency_score": float(final_cons_score),
        "anti_degenerate_score": float(anti_degen_score),
        "chain_quality_score": float(chain_quality_score),
        "first_dead_step": first_dead,
        "n_live": n_live,
        "n_steps": n,
        "n_step_consistent": n_cons,
        "n_step_total": n_total,
        "final_consistency_details": final_cons_details,
        "anti_degenerate_details": anti_degen_details,
        "reward": float(total),
    }"""

if old_score_block not in text:
    raise SystemExit("Could not find old score_response block to replace")
text = text.replace(old_score_block, new_score_block)

# ---------------------------------------------------------------------------
# 4) Update compute_reward to surface new metrics
# ---------------------------------------------------------------------------
old_compute_return = """    return {
        "score": reward,
        "accuracy": metrics.get("answer", 0.0),
        "format": metrics.get("format", 0.0),
        "answer": metrics.get("answer", 0.0),
        "liveness_score": metrics.get("liveness_score", 0.0),
        "liveness_ratio": metrics.get("liveness_ratio", 0.0),
        "n_live": metrics.get("n_live", 0),
        "n_steps": metrics.get("n_steps", 0),
        "first_dead_step": metrics.get("first_dead_step"),
    }"""

new_compute_return = """    return {
        "score": reward,
        "accuracy": metrics.get("answer", 0.0),
        "format": metrics.get("format", 0.0),
        "answer": metrics.get("answer", 0.0),
        "liveness_score": metrics.get("liveness_score", 0.0),
        "liveness_ratio": metrics.get("liveness_ratio", 0.0),
        "step_consistency_score": metrics.get("step_consistency_score", 0.0),
        "final_consistency_score": metrics.get("final_consistency_score", 0.0),
        "anti_degenerate_score": metrics.get("anti_degenerate_score", 0.0),
        "chain_quality_score": metrics.get("chain_quality_score", 0.0),
        "n_live": metrics.get("n_live", 0),
        "n_steps": metrics.get("n_steps", 0),
        "n_step_consistent": metrics.get("n_step_consistent", 0),
        "n_step_total": metrics.get("n_step_total", 0),
        "first_dead_step": metrics.get("first_dead_step"),
    }"""

if old_compute_return not in text:
    raise SystemExit("Could not find old compute_reward return block")
text = text.replace(old_compute_return, new_compute_return)

# ---------------------------------------------------------------------------
# 5) Update header docstring
# ---------------------------------------------------------------------------
text = text.replace(
    "LB-PRM design (see docs/superpowers/specs/2026-06-08-lb-prm-design.md):\n\n    total = 0.2 * format_ok + 0.4 * answer_ok + 0.4 * liveness_score",
    "LB-PRM v2 design (advisor feedback 2026-06-08):\n\n    total = 0.2 * format + 0.4 * answer + 0.4 * chain_quality_score\n\nchain_quality_score = (0.4 * liveness + 0.3 * step_consistency + 0.3 * final_consistency) * anti_degenerate",
)

PATH.write_text(text)
print(f"Patched {PATH}")
