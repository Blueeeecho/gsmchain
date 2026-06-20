"""V14 GRPO reward: 4 项, 跟 V12 公式同构, 但信号源重做.

R = 3.0 * r_answer
  + 1.5 * r_step_value
  + 0.5 * r_path_alignment
  - 0.5 * r_distractor_leak

设计要点:
- r_answer: reasoning 末行 = gold answer (0.5% 容差)
- r_step_value: 从 reasoning 抽 "= number" 行, LCS 对齐到 gold values
- r_path_alignment: reasoning 表达式行 tokenize 跟 gold_trace_tokens vs distractor_trace_tokens 比 sim
- r_distractor_leak: reasoning 任何表达式行高 sim distractor_expression → 扣分
"""
from __future__ import annotations

import json
import re
from typing import Any

TOKEN_RE = re.compile(r"\*\*|(?<![\d.])-?\d+(?:\.\d+)?|[+\-*/]")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _try_parse_json(text: str) -> dict | None:
    """Brace counting JSON extract."""
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
        if c == "\\" and in_str:
            esc = True
            continue
        if c == '"':
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
                    return json.loads(text[s:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _looks_like_math(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    has_num = bool(re.search(r"\d", line))
    has_op = bool(re.search(r"[+\-*/=]", line))
    math_chars = sum(1 for c in line if c in "0123456789+-*/().= ")
    return has_num and has_op and math_chars / max(len(line), 1) > 0.4


def _safe_eval_number(s: str):
    """从字符串抽数字. 接受 'var = 3.5' / '3.5' / '= 12'."""
    if s is None:
        return None
    s = str(s).strip()
    # 1) 末段 = number
    m = re.search(r"[=]\s*([-+]?\d+(?:\.\d+)?)\s*$", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    # 2) 整个字符串 = number
    try:
        return float(s)
    except ValueError:
        return None


def _expression_from_line(line: str) -> str:
    """从 'var = expr' 行抽 expr 部分."""
    line = line.strip()
    if "=" in line:
        # 找第一个 =, 后面的就是 expr (除非是值行)
        parts = line.split("=", 1)
        rhs = parts[1].strip()
        # 如果 rhs 是纯数字, 算"值行", 返回空
        try:
            float(rhs)
            return ""
        except ValueError:
            return rhs
    return line


def _tokenize_expr(expr: str) -> list[str]:
    if not expr:
        return []
    return TOKEN_RE.findall(expr.replace(" ", ""))


def _normalized_edit_sim(a: list, b: list) -> float:
    """1 - levenshtein(a, b) / max(len(a), len(b))."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # Lazy Levenshtein (动态规划), 避免依赖 python-Levenshtein
    m, n = len(a), len(b)
    if m > n:
        a, b = b, a
        m, n = n, m
    prev = list(range(m + 1))
    for j in range(1, n + 1):
        cur = [j] + [0] * m
        for i in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[i] = min(cur[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost)
        prev = cur
    dist = prev[m]
    return 1.0 - dist / max(m, n)


# ---------------------------------------------------------------------------
# Reward components
# ---------------------------------------------------------------------------
def _r_answer(reasoning: list, gold_answer) -> float:
    if not reasoning:
        return 0.0
    last_val = _safe_eval_number(reasoning[-1])
    if last_val is None or gold_answer is None:
        return 0.0
    try:
        gold = float(gold_answer)
    except (ValueError, TypeError):
        return 0.0
    if abs(last_val - gold) < max(0.005 * abs(gold), 1e-3):
        return 1.0
    return 0.0


def _r_step_value(reasoning: list, gold_values: list) -> float:
    """LCS-style: pred 抽所有 "= number" 行的数字, 按 gold 顺序匹配."""
    if not gold_values:
        return 0.0
    pred_values = []
    for line in reasoning:
        v = _safe_eval_number(line)
        if v is not None:
            pred_values.append(v)
    if not pred_values:
        return 0.0
    matched = 0
    gi = 0
    for pv in pred_values:
        if gi >= len(gold_values):
            break
        gv = float(gold_values[gi])
        if abs(pv - gv) < max(0.005 * abs(gv), 1e-3):
            matched += 1
            gi += 1
    return matched / len(gold_values)


def _r_path_alignment(reasoning: list, gold_trace_tokens: list, distractor_trace_tokens: list) -> float:
    """把 reasoning 表达式行 tokenize, 跟 gold / distractor 都比 sim, 越高越像 gold 越好."""
    if not gold_trace_tokens:
        # original 类别: 走 baseline (always 1.0)
        return 1.0
    pred_tokens = []
    for line in reasoning:
        if _looks_like_math(line):
            expr = _expression_from_line(line)
            if expr:
                pred_tokens.extend(_tokenize_expr(expr))
    if not pred_tokens:
        return 0.0
    sim_gold = _normalized_edit_sim(pred_tokens, list(gold_trace_tokens))
    if not distractor_trace_tokens:
        return sim_gold
    sim_dist = _normalized_edit_sim(pred_tokens, list(distractor_trace_tokens))
    if sim_gold >= sim_dist:
        return sim_gold
    # 走错链: 0.3 * sim_gold, 弱探索分
    return 0.3 * sim_gold


def _r_distractor_leak(reasoning: list, distractor_expression: str, distractor_enabled: bool) -> float:
    """reasoning 任何表达式行高 sim distractor_expression → 扣分 (返回正数, 主函数用 -weight)."""
    if not distractor_enabled or not distractor_expression:
        return 0.0
    dist_toks = _tokenize_expr(distractor_expression)
    if not dist_toks:
        return 0.0
    sims = []
    for line in reasoning:
        if _looks_like_math(line):
            expr = _expression_from_line(line)
            if expr:
                sims.append(_normalized_edit_sim(_tokenize_expr(expr), dist_toks))
    return max(sims) if sims else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def compute_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """V14 GRPO reward (verl-compatible single-sample signature).

    Returns:
        dict with key ``score`` (float) and metric sub-dict.
    """
    answer_weight = float(kwargs.get("answer_weight", 3.0))
    step_value_weight = float(kwargs.get("step_value_weight", 1.5))
    path_alignment_weight = float(kwargs.get("path_alignment_weight", 0.5))
    distractor_leak_weight = float(kwargs.get("distractor_leak_weight", 0.5))

    metrics = {
        "r_answer": 0.0,
        "r_step_value": 0.0,
        "r_path_alignment": 0.0,
        "r_distractor_leak": 0.0,
        "reason": "ok",
    }

    # 1) Unpack ground truth
    if isinstance(ground_truth, dict):
        gt = ground_truth
    else:
        gt = {"answer": str(ground_truth)}
    if not isinstance(extra_info, dict):
        extra_info = {}

    gold_answer = gt.get("answer", gt.get("gold_answer", extra_info.get("gold_answer", "")))
    gold_values = gt.get("gold_values") or extra_info.get("gold_values") or []
    gold_trace_tokens = gt.get("gold_trace_tokens", []) or []
    distractor_expression = gt.get("distractor_expression") or ""
    distractor_trace_tokens = gt.get("distractor_trace_tokens", []) or []
    distractor_enabled = bool(gt.get("distractor_enabled", False)) or bool(distractor_expression)

    # 2) Parse model output
    obj = _try_parse_json(str(solution_str))
    if obj is None:
        metrics["reason"] = "json_parse_fail"
        # parse 失败: 给一个温和 invalid reward (跟 V12 风格)
        return {"score": -0.5, **metrics}

    reasoning = obj.get("reasoning", [])
    if not isinstance(reasoning, list):
        reasoning = []

    # 3) Compute 4 components
    metrics["r_answer"] = _r_answer(reasoning, gold_answer)
    metrics["r_step_value"] = _r_step_value(reasoning, gold_values)
    metrics["r_path_alignment"] = _r_path_alignment(reasoning, gold_trace_tokens, distractor_trace_tokens)
    metrics["r_distractor_leak"] = _r_distractor_leak(reasoning, distractor_expression, distractor_enabled)

    score = (
        answer_weight * metrics["r_answer"]
        + step_value_weight * metrics["r_step_value"]
        + path_alignment_weight * metrics["r_path_alignment"]
        - distractor_leak_weight * metrics["r_distractor_leak"]
    )
    return {"score": float(score), **metrics}


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test 1: 完全答对 (走 gold chain, 算对)
    sol1 = '''{
      "target": "total bolts",
      "use_facts": ["2 blue bolts", "half as many white as blue"],
      "exclude_facts": ["blue weighs 3 lb; not asked for weight"],
      "reasoning": [
        "We need the total number of bolts, not their weight.",
        "blue_bolts = 2",
        "white is half of blue.",
        "white_bolts = blue_bolts / 2 = 2 / 2 = 1",
        "Add them.",
        "total_bolts = blue_bolts + white_bolts = 2 + 1",
        "total_bolts = 3"
      ]
    }'''
    gt1 = {
        "answer": 3,
        "gold_values": ["2", "1", "3"],
        "gold_trace_tokens": ["2", "=", "2", "<step>", "2", "/", "2", "=", "1", "<step>", "2", "+", "1", "=", "3"],
        "distractor_expression": "",
        "distractor_trace_tokens": [],
        "distractor_enabled": False,
    }
    r = compute_reward(None, sol1, gt1, {})
    print(f"Test 1 (gold chain, correct): score={r['score']:.3f}  {r}")
    # 期望: r_answer=1.0 (3.0) + r_step_value=1.0 (1.5) + r_path_alignment~0.7 (0.35) = ~4.85
    
    # Test 2: 走分心链 (用了 decoy 算式)
    sol2 = '''{
      "target": "total bolts",
      "use_facts": ["2 blue bolts", "half as many white as blue", "blue weighs 3 lb"],
      "exclude_facts": [],
      "reasoning": [
        "We need the total weight.",
        "blue_weight = 2 * 3 = 6",
        "white_weight = 1 * 2 = 2",
        "total_weight = 6 + 2",
        "total_weight = 8"
      ]
    }'''
    gt2 = {
        "answer": 3,
        "gold_values": ["2", "1", "3"],
        "gold_trace_tokens": ["2", "=", "2", "<step>", "2", "/", "2", "=", "1", "<step>", "2", "+", "1", "=", "3"],
        "distractor_expression": "2 * 3 + 1 * 2",
        "distractor_trace_tokens": ["2", "*", "3", "+", "1", "*", "2"],
        "distractor_enabled": True,
    }
    r = compute_reward(None, sol2, gt2, {})
    print(f"Test 2 (distractor chain, wrong): score={r['score']:.3f}  {r}")
    # 期望: r_answer=0 + r_step_value=0 + r_path_alignment~0.2 + r_distractor_leak high → 负分
    
    # Test 3: parse fail
    r = compute_reward(None, "sorry i cannot", gt1, {})
    print(f"Test 3 (parse fail): score={r['score']:.3f}  {r}")
    # 期望: score = -0.5
    
    # Test 4: 中间步骤对, 末行数字错
    sol4 = '''{
      "target": "total bolts",
      "use_facts": ["2 blue bolts", "half as many white as blue"],
      "exclude_facts": ["blue weighs 3 lb; not asked for weight"],
      "reasoning": [
        "We need the total number of bolts, not their weight.",
        "blue_bolts = 2",
        "white is half of blue.",
        "white_bolts = 2 / 2 = 1",
        "Add them.",
        "total_bolts = 2 + 1 = 999"
      ]
    }'''
    r = compute_reward(None, sol4, gt1, {})
    print(f"Test 4 (steps right, final wrong): score={r['score']:.3f}  {r}")
    # 期望: r_answer=0 + r_step_value=1.0 (1.5) + r_path_alignment high → ~2.0
    
    # Test 5: original 类别, 部分对
    sol5 = '''{
      "target": "x",
      "use_facts": [],
      "exclude_facts": [],
      "reasoning": [
        "Try something.",
        "a = 1 + 1 = 2",
        "b = 2 * 2 = 4"
      ]
    }'''
    gt5 = {
        "answer": 4,
        "gold_values": ["2", "4"],
        "gold_trace_tokens": ["1", "+", "1", "=", "2", "<step>", "2", "*", "2", "=", "4"],
        "distractor_expression": "",
        "distractor_trace_tokens": [],
        "distractor_enabled": False,
    }
    r = compute_reward(None, sol5, gt5, {})
    print(f"Test 5 (original, partial right): score={r['score']:.3f}  {r}")
