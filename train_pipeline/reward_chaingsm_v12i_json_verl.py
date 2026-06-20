"""V12-improved reward (V12 公式同构, 信号源重做, 4 项保持).

改进 (vs V12):
1. r_step_value 改 LCS + 0.5% 容差 (治 0.1+0.2=0.300000004 浮点失败)
2. 砍 r_core (软相似度, 鼓励"看起来像"), 加 r_step_value_exact (精确匹配额外加分)
3. r_distractor 改 per-step 检查 (V12 只看末态, 改看每一行, 抓中间步骤走分心链)

公式:
  R = 3.0 * r_answer
    + 1.5 * r_step_value (LCS + 0.5% 容差)
    + 0.5 * r_step_value_exact (精确匹配)
    - 0.5 * r_distractor (per-step)
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
    "step_value_lcs": 0.0,
    "step_value_exact": 0.0,
    "distractor_max_sim": 0.0,
    "distractor": 0.0,
    "n_steps": 0,
    "n_gold_steps": 0,
    "reason": "ok",
}


# ---------------------------------------------------------------------------
# JSON parsing (跟 V12 一样)
# ---------------------------------------------------------------------------

def _parse_json_output(text: str) -> dict | None:
    text = str(text or "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def _extract_pred_answer_from_json(text: str) -> str | None:
    parsed = _parse_json_output(text)
    if parsed is not None:
        if parsed.get("answer") is not None:
            return extract_answer(str(parsed["answer"]))
        if parsed.get("final_expression"):
            return extract_answer(str(parsed["final_expression"]))
    return extract_answer(text)


def _extract_pred_steps_from_json(text: str) -> list[tuple[str, str]]:
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
    parsed = _parse_json_output(text)
    if parsed is None:
        return ""
    return str(parsed.get("final_expression", "")).strip()


# ---------------------------------------------------------------------------
# Tokenize + 相似度
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


# ---------------------------------------------------------------------------
# V12-improved reward components
# ---------------------------------------------------------------------------

def _safe_eval(s: str) -> float | None:
    if s is None:
        return None
    s = str(s).strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _r_step_value_lcs(pred_values: list[str], gold_values: list[str], tol: float = 0.005) -> float:
    """LCS-style: 允许 pred 多写, 按 gold 顺序匹配, 0.5% 容差.

    改进 vs V12:
    - V12: strict equal, 容差 0, 不能多写
    - V12i: 0.5% 容差, LCS 顺序匹配
    """
    if not gold_values:
        return 0.0
    pred_nums = []
    for v in pred_values:
        n = _safe_eval(v)
        if n is not None:
            pred_nums.append(n)
    if not pred_nums:
        return 0.0
    matched = 0
    gi = 0
    for pv in pred_nums:
        if gi >= len(gold_values):
            break
        gv = _safe_eval(gold_values[gi])
        if gv is None:
            gi += 1
            continue
        if abs(pv - gv) < max(tol * abs(gv), 1e-3):
            matched += 1
            gi += 1
    return matched / len(gold_values)


def _r_step_value_exact(pred_values: list[str], gold_values: list[str]) -> float:
    """严格相等匹配率 (V12 风格). 跟 _r_step_value_lcs 互补."""
    if not gold_values:
        return 0.0
    if not pred_values:
        return 0.0
    n_exact = 0
    for pv, gv in zip(pred_values, gold_values):
        if str(pv).strip() == str(gv).strip():
            n_exact += 1
    return n_exact / len(gold_values)


def _r_distractor_per_step(
    pred_steps: list[tuple[str, str]],
    pred_final_expr: str,
    distractor_expression: str,
    gold_expression: str = "",
) -> float:
    """任一 step.expression 或 final_expression 跟 distractor_expression 高 sim → 扣分.

    改进 vs V12:
    - V12: 只看末行 (pred_final), 中间步骤走分心链不扣分
    - V12i: 看每一行, 任一行高 sim 都扣分
    - 进一步: 跟 gold_expression (最终算式) 比 sim, distractor_sim > gold_sim 才是"走错链"
      (避免短算式天然高 sim 的假阳性)
    """
    if not distractor_expression:
        return 0.0
    dist_toks = _tokenize_expr(distractor_expression)
    if not dist_toks:
        return 0.0
    gold_toks = _tokenize_expr(gold_expression) if gold_expression else []
    
    # 把 final_expression 也算进来
    candidates = [(expr, _) for expr, _ in pred_steps]
    if pred_final_expr:
        candidates.append((pred_final_expr, ""))
    
    sims = []
    for expr, _ in candidates:
        expr_toks = _tokenize_expr(expr)
        if not expr_toks:
            continue
        sim_d = _normalized_edit_similarity(expr_toks, dist_toks)
        if gold_toks:
            sim_g = _normalized_edit_similarity(expr_toks, gold_toks)
            # 走错链: distractor 比 gold 更像
            leak = max(0.0, sim_d - sim_g)
        else:
            leak = sim_d
        sims.append(leak)
    return max(sims) if sims else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def score_response(
    completion: str,
    reference: dict[str, Any],
    answer_weight: float = 3.0,
    step_value_weight: float = 1.5,
    step_value_exact_weight: float = 0.5,
    distractor_weight: float = 0.5,
    invalid_reward: float = -0.5,
    **kwargs,
) -> tuple[float, dict[str, Any]]:
    """V12-improved reward: 4 项, 跟 V12 公式同构."""
    text = str(completion or "").strip()
    if not text:
        return invalid_reward, _dict_metrics("empty_response")

    # ----- 解析失败: parse 失败给 invalid_reward -----
    parsed = _parse_json_output(text)
    if parsed is None:
        return invalid_reward, _dict_metrics("json_parse_fail")

    # ----- 解析 reference -----
    gold_answer = str(reference.get("answer") or reference.get("gold_answer") or "")
    gold_expr = reference.get("gold_expression") or ""
    category = str(reference.get("category") or "")
    is_original = (category == "original") or not reference.get("distractor_enabled", False)

    # ----- r_answer (跟 V12 一样) -----
    pred_answer = _extract_pred_answer_from_json(text)
    r_answer = 1.0 if (pred_answer is not None and is_correct(pred_answer, gold_answer)) else 0.0

    # ----- r_step_value (LCS + 0.5% 容差, NEW) -----
    _gtt = reference.get("gold_trace_tokens")
    gold_trace_tokens = list(_gtt.tolist()) if hasattr(_gtt, "tolist") else list(_gtt or [])
    gold_values = _extract_gold_step_values(gold_trace_tokens)
    pred_steps = _extract_pred_steps_from_json(text)
    pred_value_strs = [v for _, v in pred_steps]
    pred_final = _extract_pred_final_expr_from_json(text)
    n_gold_steps = len(gold_values)
    r_step_value_lcs = _r_step_value_lcs(pred_value_strs, gold_values)

    # ----- r_step_value_exact (V12 严格相等风格, NEW 项) -----
    r_step_value_exact = _r_step_value_exact(pred_value_strs, gold_values)

    # ----- r_distractor (per-step, 改良) -----
    distractor_expr_str = reference.get("distractor_expression") or ""
    if is_original or not distractor_expr_str:
        sim_distractor = 0.0
    else:
        sim_distractor = _r_distractor_per_step(pred_steps, pred_final, distractor_expr_str, gold_expr)

    # ----- total -----
    total = (
        answer_weight * r_answer
        + step_value_weight * r_step_value_lcs
        + step_value_exact_weight * r_step_value_exact
        - distractor_weight * sim_distractor
    )

    m = _dict_metrics("ok")
    m.update({
        "answer": r_answer,
        "step_value_lcs": r_step_value_lcs,
        "step_value_exact": r_step_value_exact,
        "distractor_max_sim": sim_distractor,
        "distractor": sim_distractor,
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


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test 1: 完全对
    sol1 = '''{
      "target": "x",
      "use_facts": ["a", "b"],
      "exclude_facts": [],
      "steps": [
        {"explanation": "e1", "expression": "1 + 1", "value": "2"},
        {"explanation": "e2", "expression": "2 * 3", "value": "6"}
      ],
      "final_expression": "(1 + 1) * 3",
      "answer": "6"
    }'''
    gt1 = {
        "answer": "6",
        "gold_trace_tokens": ["1", "+", "1", "=", "2", "<step>", "2", "*", "3", "=", "6"],
        "distractor_expression": "",
        "distractor_enabled": False,
        "category": "original",
    }
    r, m = score_response(sol1, gt1)
    print(f"Test 1 (完全对): score={r:.3f}  {m}")
    # 期望: r_a=1.0(3.0) + r_sv_lcs=1.0(1.5) + r_sv_exact=1.0(0.5) - r_dist=0 = 5.0

    # Test 2: 浮点容差 0.1+0.2=0.300000004
    sol2 = '''{
      "target": "x",
      "use_facts": [],
      "exclude_facts": [],
      "steps": [
        {"explanation": "e1", "expression": "0.1 + 0.2", "value": "0.30000000000000004"}
      ],
      "final_expression": "0.1 + 0.2",
      "answer": "0.3"
    }'''
    gt2 = {
        "answer": "0.3",
        "gold_trace_tokens": ["0.1", "+", "0.2", "=", "0.3"],
        "distractor_expression": "",
        "distractor_enabled": False,
        "category": "original",
    }
    r, m = score_response(sol2, gt2)
    print(f"Test 2 (浮点 0.300000004): score={r:.3f}  r_sv_lcs={m['step_value_lcs']:.3f} r_sv_exact={m['step_value_exact']:.3f}")
    # 期望: V12 严格 equal 失败 (0 分), V12i LCS 0.5% 容差通过 (满分 5.0)

    # Test 3: parse fail
    r, m = score_response("sorry i cannot", gt1)
    print(f"Test 3 (parse fail): score={r:.3f}  reason={m['reason']}")
    # 期望: -0.5

    # Test 4: Hannah 医生账单 (steps 错, 重复 200 两次)
    sol4 = '''{
      "target": "x",
      "use_facts": [],
      "exclude_facts": [],
      "steps": [
        {"explanation": "cast", "expression": "200", "value": "200"},
        {"explanation": "visit", "expression": "300 * 0.5", "value": "150"},
        {"explanation": "pills", "expression": "4 * 30", "value": "120"},
        {"explanation": "parking", "expression": "6 * 2", "value": "12"},
        {"explanation": "sum", "expression": "200 + 150 + 120 + 12", "value": "582"}
      ],
      "final_expression": "582",
      "answer": "582"
    }'''
    gt4 = {
        "answer": "482",
        "gold_trace_tokens": ["200", "=", "200", "<step>", "300", "*", "0.5", "=", "150", "<step>", "4", "*", "30", "=", "120", "<step>", "6", "*", "2", "=", "12"],
        "distractor_expression": "",
        "distractor_enabled": False,
        "category": "original",
    }
    r, m = score_response(sol4, gt4)
    print(f"Test 4 (Hannah 重复 200): score={r:.3f}  r_a={m['answer']} r_sv_lcs={m['step_value_lcs']:.3f} r_sv_exact={m['step_value_exact']:.3f}")
    # 期望: r_a=0 + r_sv_lcs=0.5 (200+150+120+12 都对, 5 步 gold 4 步) + r_sv_exact=0.5 (200 重复) - r_dist=0 = ~1.25

    # Test 5: 走分心链 (decoy 类题)
    sol5 = '''{
      "target": "x",
      "use_facts": [],
      "exclude_facts": [],
      "steps": [
        {"explanation": "weight", "expression": "2 * 3", "value": "6"},
        {"explanation": "weight2", "expression": "1 * 2", "value": "2"},
        {"explanation": "sum_weight", "expression": "6 + 2", "value": "8"}
      ],
      "final_expression": "2 * 3 + 1 * 2",
      "answer": "8"
    }'''
    gt5 = {
        "answer": "3",
        "gold_trace_tokens": ["2", "=", "2", "<step>", "2", "/", "2", "=", "1", "<step>", "2", "+", "1", "=", "3"],
        "distractor_expression": "2 * 3 + 1 * 2",
        "distractor_enabled": True,
        "category": "attribute_mismatch",
    }
    r, m = score_response(sol5, gt5)
    print(f"Test 5 (走分心链, decoy 算式): score={r:.3f}  r_a={m['answer']} r_dist_max={m['distractor_max_sim']:.3f}")
    # 期望: r_a=0 + r_sv_lcs=0 + r_sv_exact=0 - r_dist=1.0 (steps 全是 distractor) = -0.5

    # Test 6: 走对链 (decoy 类, 正确路径)
    sol6 = '''{
      "target": "x",
      "use_facts": [],
      "exclude_facts": [],
      "steps": [
        {"explanation": "blue", "expression": "2", "value": "2"},
        {"explanation": "white", "expression": "2 / 2", "value": "1"},
        {"explanation": "total", "expression": "2 + 1", "value": "3"}
      ],
      "final_expression": "2 + 2 / 2",
      "answer": "3"
    }'''
    r, m = score_response(sol6, gt5)
    print(f"Test 6 (走 gold chain, decoy 题): score={r:.3f}  r_a={m['answer']} r_dist_max={m['distractor_max_sim']:.3f}")
    # 期望: r_a=1.0(3.0) + r_sv_lcs=1.0(1.5) + r_sv_exact=1.0(0.5) - r_dist=0 = 5.0
