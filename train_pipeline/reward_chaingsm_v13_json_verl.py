"""V13 GRPO reward (5 字段版): Hard per-step 数值匹配 + 仅信 final_expression.

设计要点 (vs V12):
- 砍 r_answer (3.0): R11 违反 (fe 算对但 answer 字段错) 4.3% 失分
- 砍 r_core: 软相似度鼓励"生成相似"而非"算对", 治不了 64% 算式错
- 改 r_step_value: 软 trace sim → hard per-step 数值匹配
- 保留 exclude_facts: 给 decoy 显式位置, 加 r_exclude 检查
- 砍 r_distractor: 不再是 (1 - sim), 改为 decoy fact 不出现在 steps.expression 中

新公式:
  r_final    = 1.0 if eval(final_expression) == gold else 0.0
  r_step_val = per_step_value_match / n_steps  (硬数值匹配)
  r_step_fmt = 1.0 if all steps have 3 valid fields
  r_format   = 1.0 if JSON parse ok and 5 fields present
  r_exclude  = 1.0 - fraction(gold_decoys_in_steps_expression)
  score      = 3.0 * r_final + 1.5 * r_step_val + 0.5 * r_step_fmt + 0.5 * r_format + 0.5 * r_exclude
"""

from __future__ import annotations

import json
import re
from typing import Any


def _safe_eval(expr: str) -> float | None:
    """Safely evaluate arithmetic expression. Returns None on failure."""
    if not expr or not isinstance(expr, str):
        return None
    expr = expr.strip().replace("×", "*").replace("÷", "/").replace(",", "")
    if not re.match(r"^[\d\s\+\-\*\/\.\(\)]+$", expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError):
        return None


def _try_parse_json(text: str) -> dict | None:
    """Brace counting JSON extract (find outermost {...})."""
    text = str(text or "").strip()
    if not text:
        return None
    s = text.find("{")
    if s < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(s, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"' and not esc:
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[s:i + 1])
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    pass
                return None
    return None


def _per_step_value_match(steps: list, gold_steps: list) -> tuple[float, int]:
    """Per-step numeric value matching (hard match)."""
    if not isinstance(steps, list) or not steps:
        return 0.0, 0
    if not isinstance(gold_steps, list) or not gold_steps:
        return 0.0, len(steps)
    
    n = min(len(steps), len(gold_steps))
    matches = 0
    for i in range(n):
        s = steps[i]
        g = gold_steps[i]
        if not isinstance(s, dict) or not isinstance(g, dict):
            continue
        sv = _safe_eval(str(s.get("value", "")))
        gv = _safe_eval(str(g.get("value", "")))
        if sv is not None and gv is not None:
            if abs(sv - gv) < 1e-3:
                matches += 1
                continue
        se = _safe_eval(str(s.get("expression", "")))
        ge = _safe_eval(str(g.get("expression", "")))
        if se is not None and ge is not None:
            if abs(se - ge) < 1e-3:
                matches += 1
    
    return matches / max(n, 1), max(n, 1)


def _decoy_in_steps_exclude(decoys: list, steps: list) -> float:
    """Return fraction of gold decoys whose numeric value appears in any step expression.
    0 = perfect (no decoy leaked), 1 = all decoys leaked.
    """
    if not isinstance(decoys, list) or not decoys:
        return 0.0
    if not isinstance(steps, list) or not steps:
        return 0.0
    
    # 抽 decoy 中的数字
    decoy_nums = set()
    for d in decoys:
        if not isinstance(d, str):
            continue
        for m in re.finditer(r"[-+]?\d+(?:\.\d+)?", d):
            v = float(m.group(0))
            if v != 0:
                decoy_nums.add(v)
    
    if not decoy_nums:
        return 0.0
    
    # 检查 steps.expression 里是否出现
    leaked = 0
    for s in steps:
        if not isinstance(s, dict):
            continue
        e = str(s.get("expression", ""))
        for m in re.finditer(r"[-+]?\d+(?:\.\d+)?", e):
            v = float(m.group(0))
            if v in decoy_nums:
                leaked += 1
                break
    
    return leaked / len(decoy_nums)


def compute_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """V13 GRPO reward (verl-compatible single-sample signature, 5 字段版).

    Returns:
        dict with key ``score`` (float) and metric sub-dict (compatible with verl).
    """
    answer_weight = float(kwargs.get("answer_weight", 3.0))
    step_value_weight = float(kwargs.get("step_value_weight", 1.5))
    step_format_weight = float(kwargs.get("step_format_weight", 0.5))
    format_weight = float(kwargs.get("format_weight", 0.5))
    exclude_weight = float(kwargs.get("exclude_weight", 0.5))

    metrics = {
        "r_final": 0.0,
        "r_step_val": 0.0,
        "r_step_fmt": 0.0,
        "r_format": 0.0,
        "r_exclude": 0.0,
        "reason": "ok",
    }

    # 1) Unpack ground truth (verl passes the row dict as ground_truth)
    if isinstance(ground_truth, dict):
        gt = ground_truth
    else:
        gt = {"answer": str(ground_truth)}
    if not isinstance(extra_info, dict):
        extra_info = {}

    gold_answer = str(
        gt.get("answer", gt.get("final_answer", extra_info.get("gold_answer", "")))
    )
    gold_steps = gt.get("steps") or gt.get("gold_steps") or extra_info.get("gold_steps") or []
    gold_decoys = gt.get("exclude_facts") or extra_info.get("gold_exclude_facts") or []

    gold_val = _safe_eval(gold_answer)
    if gold_val is None:
        m = re.search(r"[-+]?\d+(?:\.\d+)?", str(gold_answer))
        if m:
            gold_val = float(m.group(0))

    # 2) Parse model output
    obj = _try_parse_json(str(solution_str))
    if obj is None:
        metrics["reason"] = "json_parse_fail"
        return {"score": 0.0, **metrics}

    # r_format: 5 字段都存在
    required_fields = ["target", "steps", "exclude_facts", "final_expression", "final_answer"]
    metrics["r_format"] = 1.0 if all(f in obj for f in required_fields) else 0.0

    # r_step_fmt
    steps = obj.get("steps", [])
    if not isinstance(steps, list) or not steps:
        metrics["r_step_fmt"] = 0.0
    else:
        valid = sum(
            1
            for s in steps
            if isinstance(s, dict)
            and all(k in s for k in ["explanation", "expression", "value"])
        )
        metrics["r_step_fmt"] = valid / max(len(steps), 1)

    # r_step_val: per-step 硬数值匹配
    if isinstance(steps, list) and gold_steps:
        metrics["r_step_val"], _ = _per_step_value_match(steps, gold_steps)

    # r_final: final_expression eval == gold
    fe = str(obj.get("final_expression", ""))
    fe_val = _safe_eval(fe)
    if fe_val is not None and gold_val is not None and abs(fe_val - gold_val) < 1e-3:
        metrics["r_final"] = 1.0

    # r_exclude: decoy 没出现在 steps.expression 里
    if isinstance(steps, list):
        leak_ratio = _decoy_in_steps_exclude(gold_decoys, steps)
        metrics["r_exclude"] = 1.0 - leak_ratio

    score = (
        answer_weight * metrics["r_final"]
        + step_value_weight * metrics["r_step_val"]
        + step_format_weight * metrics["r_step_fmt"]
        + format_weight * metrics["r_format"]
        + exclude_weight * metrics["r_exclude"]
    )
    return {"score": float(score), **metrics}


def _to_list(x):
    if x is None: return []
    if hasattr(x, "__iter__") and not isinstance(x, str): return list(x)
    return [x]


def _verl_reward_fn(data, **kwargs):
    solution_strs = _to_list(data.get("response", data.get("solution_str", "")))
    ground_truths = _to_list(data.get("ground_truth", data.get("gt", "")))
    extra_infos = _to_list(data.get("extra_info", {}))
    return compute_reward(None, solution_strs, ground_truths, extra_infos, **kwargs)


if __name__ == "__main__":
    # Test 1: 完全答对
    sol = '''{
      "target": "dollars per day",
      "steps": [
        {"explanation": "remaining", "expression": "16 - 3 - 4", "value": 9},
        {"explanation": "dollars", "expression": "9 * 2", "value": 18}
      ],
      "exclude_facts": [],
      "final_expression": "(16 - 3 - 4) * 2",
      "final_answer": 18
    }'''
    gt = {"answer": 18, "steps": [
        {"explanation": "remaining", "expression": "16 - 3 - 4", "value": 9},
        {"explanation": "dollars", "expression": "9 * 2", "value": 18}
    ], "exclude_facts": []}
    r = compute_reward(None, [sol], [gt], [{}])
    print(f"Test 1 (完全答对): reward = {r[0]:.3f}  (期望 ~5.0)")
    
    # Test 2: final 对 steps 错
    sol2 = '''{
      "target": "x",
      "steps": [{"explanation": "wrong", "expression": "1+1", "value": 99}],
      "exclude_facts": [],
      "final_expression": "(16 - 3 - 4) * 2",
      "final_answer": 18
    }'''
    r = compute_reward(None, [sol2], [gt], [{}])
    print(f"Test 2 (final 对 steps 错): reward = {r[0]:.3f}")
    
    # Test 3: JSON parse fail
    r = compute_reward(None, ["Sorry"], [gt], [{}])
    print(f"Test 3 (parse fail): reward = {r[0]:.3f}  (期望 0.0)")
    
    # Test 4: R11 违反 - fe 对 answer 错 → 应该仍然高分
    sol4 = '''{
      "target": "x",
      "steps": [
        {"explanation": "remaining", "expression": "16 - 3 - 4", "value": 9},
        {"explanation": "dollars", "expression": "9 * 2", "value": 18}
      ],
      "exclude_facts": [],
      "final_expression": "(16 - 3 - 4) * 2",
      "final_answer": 999
    }'''
    r = compute_reward(None, [sol4], [gt], [{}])
    print(f"Test 4 (R11 fe 对 answer 错): reward = {r[0]:.3f}  (期望高分)")
    
    # Test 5: decoy leaked into steps (R10 违反)
    gt_with_decoy = {
        "answer": 3, 
        "steps": [{"explanation": "white", "expression": "2/2", "value": 1}, {"explanation": "total", "expression": "2+1", "value": 3}],
        "exclude_facts": ["Each blue bolt weighs 3 pounds", "Each white bolt weighs 2 pounds"]
    }
    sol5 = '''{
      "target": "total bolts",
      "steps": [
        {"explanation": "white", "expression": "2/2", "value": 1},
        {"explanation": "wrong", "expression": "3 + 2", "value": 5}
      ],
      "exclude_facts": ["Each blue bolt weighs 3 pounds"],
      "final_expression": "2 + 2 / 2",
      "final_answer": 3
    }'''
    r = compute_reward(None, [sol5], [gt_with_decoy], [{}])
    print(f"Test 5 (decoy 3 出现在 steps): reward = {r[0]:.3f}  (期望 r_exclude 扣分)")
