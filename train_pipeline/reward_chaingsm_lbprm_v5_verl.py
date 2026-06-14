"""LB-PRM v5 reward for verl GRPO training.

基于 v3 改写,接力 v3 best 起点。spec 在 docs/superpowers/specs/2026-06-14-lbprm-v5-design.md

v5 改动(2026-06-14):
  1. **causal_liveness 松绑**: v3 收紧过头(0.33-0.37),liveness 偏低挡了真引用信号
     - v3 (a) variable \\b 匹配 → 保留
     - v3 (b) variable 出现在 final_expression → 保留
     - v3 (c) 末步:eval(step.expression) == eval(final_expression) → 保留
     - v3 (d) 末步:value_k 与 pred_answer 严格相等 → 保留
     - v5 新增 (a') value 子表达式匹配: 遍历后续 step expression 的 ast.walk 子表达式,
       eval 出数值, abs(eval_val - step.value) < 1e-3 视为活
     - v5 新增 (a'') value 出现在 final_expression 子表达式里: 同上容差
     - **v2 失败原因不是 (a') 子串匹配本身, 而是其他 reward hacking 漏洞(v3 已修), 所以 v5 重启 (a') 安全**
  2. **length_bonus**: n_steps ∈ [3, 6] 时 +0.05
     - 鼓励"格式规整的 chain"
     - 防止 chain 越写越短(no_degenerate 已经在长 chain 上扣分, 但没奖励"格式规整")
  3. **格式计分/权重**: 沿用 v3 (format=0.20, answer=0.55, chain_quality=0.25)
"""
from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Reused helpers from v2
# ---------------------------------------------------------------------------

def _to_float(value: Any) -> float | None:
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


def _safe_eval(expr: str) -> float | None:
    import ast as _ast
    try:
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
        if isinstance(node.op, _ast.Add):
            return left + right
        if isinstance(node.op, _ast.Sub):
            return left - right
        if isinstance(node.op, _ast.Mult):
            return left * right
        if isinstance(node.op, _ast.Div):
            return left / right
        if isinstance(node.op, _ast.Pow):
            return left**right
    raise ValueError(type(node).__name__)


def _is_correct(pred: Any, gold: Any, tolerance: float = 1e-6) -> bool:
    pf = _to_float(pred)
    gf = _to_float(gold)
    if pf is None or gf is None:
        return False
    return abs(pf - gf) < tolerance


def _floats_close(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) < tol


def _parse_response(completion: str) -> dict[str, Any] | None:
    text = str(completion or "").strip()
    if not text:
        return None
    if isinstance(completion, dict):  # 测试方便
        return completion if "selected_steps" in completion else None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                cand = text[start:i+1]
                try:
                    return json.loads(cand)
                except Exception:
                    return None
    return None


# ---------------------------------------------------------------------------
# v3 新定义
# ---------------------------------------------------------------------------

def _chain_to_answer_ok(steps: list, final_expr: str, gold_answer: Any) -> int:
    """硬门:0 或 1。容差 1e-3。

    测的是"chain 能否独立推出 gold_answer",**不**是"推出 pred_answer"。
    这样:
      - 答对 + chain 真推出 gold → 1(高分)
      - 答错 + chain 推出 pred(pred ≠ gold)→ 0(被掐)
      - 答对 + chain 推出 gold 巧合(可能)/或真推出 → 1

    满足任一即 1:
      A. eval(final_expression) ≈ gold_answer
      B. 末步:value_k 严格等于 gold_answer(数字)
      C. 任意 step expression 求值 ≈ gold_answer
    """
    ans_f = _to_float(gold_answer)
    if ans_f is None:
        return 0

    # 路径 A
    fe_f = _safe_eval(final_expr)
    if fe_f is not None and _floats_close(fe_f, ans_f):
        return 1

    # 路径 B(末步 value == gold)
    if steps and isinstance(steps[-1], dict):
        last_val = str(steps[-1].get("value", ""))
        vf = _to_float(last_val)
        if vf is not None and _floats_close(vf, ans_f):
            return 1
        if last_val.strip() == str(gold_answer).strip():
            return 1

    return 0


def _ast_subexpr_values(expr: str) -> list[float]:
    """用 ast.walk 取 expression 字符串里所有"算得出来的子表达式数值"。

    例: expr = "a + b * 2" → 可能产出 a, b, b*2, a+b*2 的子表达式值
    只保留能 _safe_eval 算出来的,失败/None 跳过。
    """
    vals: list[float] = []
    if not expr:
        return vals
    try:
        import ast as _ast
        tree = _ast.parse(str(expr).replace("^", "**"), mode="eval")
    except Exception:
        return vals
    # 收集所有 BinOp / UnaryOp / Constant 节点
    for node in _ast.walk(tree):
        # 跳过整个表达式自己
        if isinstance(node, _ast.Expression):
            continue
        # 评估"算得出来"的节点 (BinOp / UnaryOp / Constant)
        if isinstance(node, (_ast.BinOp, _ast.UnaryOp, _ast.Constant)):
            try:
                if isinstance(node, _ast.Constant) and isinstance(node.value, (int, float)):
                    v = float(node.value)
                else:
                    v = _eval_node(node)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            except Exception:
                pass
    return vals


def _causal_liveness(steps: list, final_expr: str, pred_answer: Any) -> list[bool]:
    """v5 松绑版 liveness: variable 真引用 + value 子表达式匹配 + 末步对齐。"""
    n = len(steps)
    is_live: list[bool] = []
    for k, step in enumerate(steps):
        if not isinstance(step, dict):
            is_live.append(False)
            continue
        v = str(step.get("variable", ""))
        val = str(step.get("value", ""))
        val_f = _to_float(val)
        live = False

        # (a) variable 作为单词出现在后续 step 的 expression 里
        for j in range(k + 1, n):
            if not isinstance(steps[j], dict):
                continue
            expr = str(steps[j].get("expression", "") or "")
            if v and re.search(rf"\b{re.escape(v)}\b", expr):
                live = True
                break

        # (b) variable 出现在 final_expression 里
        if not live and v and final_expr and re.search(rf"\b{re.escape(v)}\b", final_expr):
            live = True

        # (a') [v5 新增] value 作为子表达式数值出现在后续 step 的 expression 里
        #   用 ast.walk 取所有算得出来的子表达式数值,容差 1e-3 比对
        if not live and val_f is not None:
            for j in range(k + 1, n):
                if not isinstance(steps[j], dict):
                    continue
                expr = str(steps[j].get("expression", "") or "")
                if not expr:
                    continue
                sub_vals = _ast_subexpr_values(expr)
                for sub_v in sub_vals:
                    if abs(sub_v - val_f) < 1e-3:
                        live = True
                        break
                if live:
                    break

        # (a'') [v5 新增] value 作为子表达式数值出现在 final_expression 里
        if not live and val_f is not None and final_expr:
            sub_vals = _ast_subexpr_values(final_expr)
            for sub_v in sub_vals:
                if abs(sub_v - val_f) < 1e-3:
                    live = True
                    break

        # (c) 末步:eval(step.expression) ≈ eval(final_expression)
        if not live and k == n - 1 and final_expr:
            ef = _safe_eval(str(step.get("expression", "") or ""))
            fef = _safe_eval(final_expr)
            if ef is not None and fef is not None and _floats_close(ef, fef):
                live = True

        # (d) 末步:value 严格等于 pred_answer
        if not live and k == n - 1 and val and pred_answer is not None:
            vf = _to_float(val)
            af = _to_float(pred_answer)
            if vf is not None and af is not None and _floats_close(vf, af):
                live = True
            elif val.strip() == str(pred_answer).strip():
                live = True

        is_live.append(live)
    return is_live


def _causal_liveness_score(steps: list, final_expr: str, pred_answer: Any) -> tuple[float, list[bool]]:
    is_live = _causal_liveness(steps, final_expr, pred_answer)
    if not steps:
        return 0.0, is_live
    return sum(is_live) / len(steps), is_live


def _step_calc_score(steps: list) -> float:
    if not steps:
        return 0.0
    n_ok = 0
    n_total = 0
    for step in steps:
        if not isinstance(step, dict):
            n_total += 1
            continue
        n_total += 1
        ef = _safe_eval(str(step.get("expression", "") or ""))
        vf = _to_float(str(step.get("value", "") or ""))
        if ef is not None and vf is not None and _floats_close(ef, vf):
            n_ok += 1
    return n_ok / n_total if n_total else 0.0


def _no_degenerate_score(steps: list, final_expr: str) -> float:
    if not steps:
        return 0.0
    n = len(steps)
    score = 1.0
    if n > 12:
        score -= 0.2
    exprs = [str(s.get("expression", "")) for s in steps if isinstance(s, dict)]
    seen: dict[str, int] = {}
    for e in exprs:
        seen[e] = seen.get(e, 0) + 1
    repeats = sum(max(0, c - 1) for c in seen.values())
    if repeats > 0:
        score -= min(0.6, 0.3 * repeats)
    n_uneval = sum(1 for e in exprs if _safe_eval(e) is None)
    if n_uneval > 0:
        score -= 0.2
    n_nonnum = sum(1 for s in steps if isinstance(s, dict) and _to_float(s.get("value", "")) is None)
    if n_nonnum > 0:
        score -= 0.2
    return max(0.0, score)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

# 固定的 metrics schema。所有 sample(无论是否成功解析)都返回同样的 key 集合,
# 否则 verl _postprocess 会因为 KeyError 崩溃(reward_extra_infos 列表里
# 某些 sample 缺某些 key)。
_METRICS_SCHEMA = {
    "format": 0.0,
    "answer": 0.0,
    "chain_to_answer_ok": 0,
    "causal_liveness_score": 0.0,
    "step_calc_score": 0.0,
    "no_degenerate_score": 0.0,
    "chain_quality_score": 0.0,
    "gated_chain_quality": 0.0,
    "n_live": 0,
    "n_steps": 0,
    "length_bonus": 0.0,        # v5 新增
    "length_bonus_flag": 0,     # v5 新增: n_steps ∈ [3, 6] 时 = 1
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
    format_weight: float = 0.20,
    answer_weight: float = 0.55,
    chain_quality_weight: float = 0.25,
    length_bonus_value: float = 0.05,   # v5 新增
    length_bonus_lo: int = 3,           # v5 新增
    length_bonus_hi: int = 6,           # v5 新增
    invalid_reward: float = -0.5,
    **kwargs,
) -> tuple[float, dict[str, Any]]:
    parsed = _parse_response(completion)
    if not isinstance(parsed, dict):
        return invalid_reward, _empty_metrics("invalid_json", invalid_reward)

    required = {"selected_steps", "final_expression", "answer"}
    if not required <= set(parsed):
        return invalid_reward * 0.5, _empty_metrics("missing_fields", invalid_reward * 0.5)

    steps = parsed.get("selected_steps")
    final_expr = str(parsed.get("final_expression", "") or "")
    pred_answer = parsed.get("answer")

    format_ok = 1.0 if (isinstance(steps, list) and steps and final_expr and pred_answer is not None) else 0.0
    if not format_ok:
        m = _empty_metrics("empty_steps_or_fields", invalid_reward * 0.5)
        m["format"] = 0.0
        return invalid_reward * 0.5, m

    gold_answer = reference.get("gold_answer")
    answer_ok = 1.0 if _is_correct(pred_answer, gold_answer) else 0.0

    c2a = _chain_to_answer_ok(steps, final_expr, gold_answer)

    liv_score, is_live = _causal_liveness_score(steps, final_expr, pred_answer)
    step_calc = _step_calc_score(steps)
    no_degen = _no_degenerate_score(steps, final_expr)

    chain_quality_score = (
        0.5 * liv_score
        + 0.3 * step_calc
        + 0.2 * no_degen
    )
    gated_chain_quality = c2a * chain_quality_score

    # v5 新增 length_bonus
    n_steps = len(steps)
    length_bonus_flag = 1 if (length_bonus_lo <= n_steps <= length_bonus_hi) else 0
    length_bonus = length_bonus_flag * length_bonus_value

    total = (
        format_weight * format_ok
        + answer_weight * answer_ok
        + chain_quality_weight * gated_chain_quality
        + length_bonus
    )

    m = dict(_METRICS_SCHEMA)
    m.update({
        "format": float(format_ok),
        "answer": float(answer_ok),
        "chain_to_answer_ok": int(c2a),
        "causal_liveness_score": float(liv_score),
        "step_calc_score": float(step_calc),
        "no_degenerate_score": float(no_degen),
        "chain_quality_score": float(chain_quality_score),
        "gated_chain_quality": float(gated_chain_quality),
        "n_live": int(sum(is_live)),
        "n_steps": int(n_steps),
        "length_bonus": float(length_bonus),
        "length_bonus_flag": int(length_bonus_flag),
        "reward": float(total),
        "reason": "ok",
    })
    return float(total), m


def compute_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """verl entry point(single-sample kwargs mode, verl 0.8.0 contract).

    verl 实验版 naive reward manager 按以下方式调用本函数:
        compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=..., **extra_reward_kwargs)
    返回值可以是:
      - 单个 float(reward)
      - (reward, metrics_dict) 二元组
      - 任意可被 trainer 处理的 dict
    """
    if isinstance(ground_truth, dict):
        reference = ground_truth
    else:
        reference = {"gold_answer": ground_truth}
    r, m = score_response(solution_str, reference, **kwargs)
    # verl naive reward manager expects dict with "score" key, other keys go to reward_extra_info
    return {"score": r, **m}
