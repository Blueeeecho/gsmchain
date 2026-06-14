"""LB-PRM (Liveness-Based Process Reward) for verl GRPO training.

Standalone reward module for verl GRPO training. Mirrors the entry-point
contract of reward_chaingsm_verl.py so NaiveRewardManager can load it via
``custom_reward_function.path`` without any pipeline change.

LB-PRM v2 design (advisor feedback 2026-06-08):

    total = 0.2 * format + 0.4 * answer + 0.4 * chain_quality_score

chain_quality_score = (0.4 * liveness + 0.3 * step_consistency + 0.3 * final_consistency) * anti_degenerate

The liveness_score measures chain structural coherence: a step is "live"
if its value (or variable) is referenced by a later step's expression, by
the final_expression, or (for the last step) matches the predicted answer.
The first dead step receives an extra penalty so policy gradient can
localize "where the chain went wrong".

Compared to reward_chaingsm_verl.py:
    * No reference to gold_trace / distractor_trace / gold_expression /
      distractor_expression. Only the question text and the model's own
      selected_steps are read.
    * Sympy sub-expression scan handles the "re-compute" style chains
      produced by the current prompt (where variable names are semantic
      but expressions re-derive prior sub-computations).

Usage in verl YAML config::

    custom_reward_function:
      path: /home/wwq416/snap/wwq/math-chain/train_pipeline/reward_chaingsm_lbprm_verl.py
      name: compute_reward
      reward_kwargs:
        format_weight: 0.2
        answer_weight: 0.4
        liveness_weight: 0.4
        invalid_reward: -0.5
"""

from __future__ import annotations

import json
import re
from typing import Any


# ---------------------------------------------------------------------------
# Ast-based sub-expression enumeration (the one piece of new logic vs the old reward)
# ---------------------------------------------------------------------------

def _val_in_subexpr(val: float, expr_str: str) -> bool:
    """Check whether ``val`` equals the evaluation of any sub-expression in ``expr_str``.

    Uses Python's ``ast`` to walk every sub-expression node and ``eval`` to
    compute each one. This is faster and more reliable than sympy for our
    use case: sympy's ``evaluate=False`` does not fully preserve sub-
    expression structure (e.g., ``12/60*50`` is rearranged to
    ``12*50*Pow(60,-1)`` and the intermediate ``0.2`` is lost), whereas
    ast.walk + eval captures every prefix-suffix sub-value natively.

    Safety: we only evaluate ast sub-trees, not raw user input. ``eval``
    of a compiled ast.Expression cannot execute arbitrary statements.
    """
    try:
        import ast as _ast
    except Exception:
        return False
    try:
        tree = _ast.parse(str(expr_str).replace("^", "**"), mode="eval")
    except Exception:
        return False
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Expression):
            continue
        try:
            v = eval(compile(_ast.Expression(node), "<expr>", "eval"))
        except Exception:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and v == v:  # not NaN
            # 1e-3 tolerance handles 3-decimal rounding common in SFT outputs
            # (e.g. step value "3.333" should still match subexpr 10/3 = 3.3333...)
            if abs(float(v) - val) < 1e-3:
                return True
    return False


# ---------------------------------------------------------------------------
# JSON + step parsing
# ---------------------------------------------------------------------------

def _parse_response(completion: str) -> dict[str, Any] | None:
    """Parse the model's JSON output. Mirrors reward_chaingsm_verl._parse_response."""
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


def _to_float(value: Any) -> float | None:
    """Best-effort float parse (mirrors reward_chaingsm_verl._to_float)."""
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


def _is_correct(pred: Any, gold: Any, tolerance: float = 1e-6) -> bool:
    """Numeric equality on float-extracted values, mirroring gsm_answer_extractor.is_correct."""
    p = _to_float(pred)
    g = _to_float(gold)
    if p is None or g is None:
        return False
    return abs(p - g) < tolerance


# ---------------------------------------------------------------------------
# Liveness analysis (the LB-PRM core)
# ---------------------------------------------------------------------------


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



def _compute_liveness(steps: list[dict], final_expr: str, pred_answer: Any) -> list[bool]:
    """Return a per-step is_live list of length len(steps).

    A step k is live if ANY of the following hold:
        (a) variable_k or value_k appears as substring in any later step's expression
        (a') sympy sub-expression of any later step's expression evaluates to value_k
        (b) variable_k or value_k appears in final_expression
        (c) k is the last step AND value_k (string) == pred_answer (string)
    """
    n = len(steps)
    is_live: list[bool] = []
    for k, step in enumerate(steps):
        if not isinstance(step, dict):
            is_live.append(False)
            continue
        v = str(step.get("variable", ""))
        val = str(step.get("value", ""))
        live = False

        # (a) and (a'): scan later steps
        for j in range(k + 1, n):
            if not isinstance(steps[j], dict):
                continue
            expr = str(steps[j].get("expression", ""))
            if (v and v in expr) or (val and val in expr):
                live = True
                break
            # (a') sympy value match
            if not live and val:
                val_f = _to_float(val)
                if val_f is not None and _val_in_subexpr(val_f, expr):
                    live = True
                    break

        # (b) final_expression contains variable/value
        if not live and final_expr:
            if (v and v in final_expr) or (val and val in final_expr):
                live = True

        # (c) last step + value matches pred_answer (float-tolerant)
        if not live and k == n - 1 and val and pred_answer is not None:
            val_f = _to_float(val)
            pred_f = _to_float(pred_answer)
            if val_f is not None and pred_f is not None and abs(val_f - pred_f) < 1e-6:
                live = True
            elif val.strip() == str(pred_answer).strip():
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

        is_live.append(live)
    return is_live


# ---------------------------------------------------------------------------
# Core scoring function (replacement for reward_chaingsm_verl.score_response)
# ---------------------------------------------------------------------------

def score_response(
    completion: str,
    reference: dict[str, Any],
    format_weight: float = 0.2,
    answer_weight: float = 0.4,
    liveness_weight: float = 0.4,
    invalid_reward: float = -0.5,
    **kwargs,  # absorb old reward kwargs (expression_weight, trace_weight, distractor_penalty) silently
) -> tuple[float, dict[str, Any]]:
    """Compute the LB-PRM reward for a single completion.

    Returns:
        (reward, metrics) where reward is a float in [-0.5, 1.0] and
        metrics is a dict of per-component scores for logging.
    """
    parsed = _parse_response(completion)
    if not isinstance(parsed, dict):
        return invalid_reward, {"format": 0.0, "answer": 0.0, "liveness": 0.0, "reason": "invalid_json"}

    required = {"selected_steps", "final_expression", "answer"}
    if not required <= set(parsed):
        return invalid_reward, {"format": 0.0, "answer": 0.0, "liveness": 0.0, "reason": "missing_fields"}

    steps = parsed.get("selected_steps")
    final_expr = str(parsed.get("final_expression", "") or "")
    pred_answer = parsed.get("answer")

    # ---- format ----
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
    }


# ---------------------------------------------------------------------------
# verl entry point (drop-in replacement for reward_chaingsm_verl.compute_reward)
# ---------------------------------------------------------------------------

def compute_reward(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: dict | None = None,
    extra_info: dict | None = None,
    **kwargs,
) -> dict[str, Any]:
    """verl-compatible reward function.

    Mirrors the signature of reward_chaingsm_verl.compute_reward so
    NaiveRewardManager can load this file with no other change.
    """
    reference = ground_truth or {}
    if isinstance(reference, dict) and "ground_truth" in reference:
        reference = reference["ground_truth"]
    reward, metrics = score_response(solution_str or "", reference or {}, **kwargs)
    return {
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
    }


# ---------------------------------------------------------------------------
# Self-test (runnable as `python reward_chaingsm_lbprm_verl.py`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    cases = [
        ("A perfect AbstRaL-style chain", {
            "selected_steps": [
                {"variable": "out0", "expression": "12/60",      "value": "0.2"},
                {"variable": "out1", "expression": "out0 * 50",  "value": "10"},
            ],
            "final_expression": "out1",
            "answer": "10",
        }, 10, 1.00),
        ("B perfect re-compute chain", {
            "selected_steps": [
                {"variable": "minute_rate", "expression": "12 / 60",       "value": "0.2"},
                {"variable": "earnings",    "expression": "12 / 60 * 50",  "value": "10"},
            ],
            "final_expression": "12 / 60 * 50",
            "answer": "10",
        }, 10, 1.00),
        ("C dead step in the middle", {
            "selected_steps": [
                {"variable": "minute_rate", "expression": "12 / 60",   "value": "0.2"},
                {"variable": "weather",     "expression": "70 + 5",    "value": "75"},
                {"variable": "earnings",    "expression": "0.2 * 50",  "value": "10"},
            ],
            "final_expression": "0.2 * 50",
            "answer": "10",
        }, 10, 0.76),
        ("D self-consistent but wrong (followed distractor)", {
            "selected_steps": [
                {"variable": "minute_rate", "expression": "12 / 60",     "value": "0.2"},
                {"variable": "earnings",    "expression": "0.2 * 50",    "value": "10"},
                {"variable": "after_snack", "expression": "10 - 3",      "value": "7"},
            ],
            "final_expression": "10 - 3",
            "answer": "7",
        }, 10, 0.60),
        ("E polluted chain (gold start, garbage end)", {
            "selected_steps": [
                {"variable": "minute_rate", "expression": "12 / 60",   "value": "0.2"},
                {"variable": "earnings",    "expression": "0.2 * 50",  "value": "10"},
                {"variable": "weird",       "expression": "10 + 999",  "value": "1009"},
            ],
            "final_expression": "10 + 999",
            "answer": "1009",
        }, 10, 0.60),
        ("F all-garbage chain", {
            "selected_steps": [
                {"variable": "x", "expression": "1 + 1", "value": "2"},
                {"variable": "y", "expression": "3 + 3", "value": "6"},
                {"variable": "z", "expression": "4 + 4", "value": "8"},
            ],
            "final_expression": "4 + 4",
            "answer": "8",
        }, 8, 0.68),
        ("G invalid JSON", "not json at all", 10, -0.50),
    ]

    print(f"{'Case':<48} {'expected':>10} {'got':>10}   ok?")
    print("-" * 76)
    all_ok = True
    for name, completion, gold, expected in cases:
        if isinstance(completion, str) and not completion.startswith("{"):
            raw = completion
        else:
            raw = json.dumps(completion) if not isinstance(completion, str) else completion
        reward, metrics = score_response(raw, {"gold_answer": str(gold)})
        ok = abs(reward - expected) < 1e-6
        all_ok = all_ok and ok
        print(f"{name:<48} {expected:>10.2f} {reward:>10.4f}   {'OK' if ok else 'FAIL'}")
        if not ok:
            print(f"    metrics: {metrics}")
    print("-" * 76)
    print("ALL OK" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)
