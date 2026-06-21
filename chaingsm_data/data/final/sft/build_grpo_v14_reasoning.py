"""V14 GRPO data builder.

Self-contained: reads raw + supplementary jsonl directly (same input as
build_grpo_v12_json.py), constructs the v12 reward_meta fields internally
via build_grpo.py helpers, and writes the verl-compatible V14 parquet.
No intermediate V12 jsonl is required.

One-step pipeline (mirrors the v6a_LTLT.py pattern): prompt templates are
defined as module-level constants at the top, the data is loaded, and the
verl-compatible parquet is written directly. To change the V14 prompt,
edit V14_SYSTEM_PROMPT / V14_USER_TEMPLATE below and rerun this script.

Output: chaingsm_data/data/final/grpo/grpo_v14_reasoning.parquet (3051 rows,
5 columns: data_source / prompt / ability / reward_model / extra_info)

Key changes vs V13:
- 4-field schema: target / use_facts / exclude_facts / reasoning
- reasoning = prose 表达式 物理交错 数组, derived from gold_trace
- distractor_trace_tokens: 从 distractor_expression 线性 tokenize 生成
- exclude_facts: 从 distractor_values / distractor_expression 推断
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# V14 prompt (from docs/prompts/v14_long_prompt.md, system + user template)
# ---------------------------------------------------------------------------
V14_SYSTEM_PROMPT = """You are a careful grade-school math reasoning assistant.
Your task is to solve the problem correctly and return only one valid JSON object.
Correctness, target selection, step alignment, and exclusion of irrelevant facts are judged. Reasoning style is not judged.

Output rules:
- Output valid JSON only. No markdown. No prose before/after. No comments. No trailing commas.
- The "reasoning" array is the only place where you record your derivation. Every calculation used for the final answer must appear as a math-expression line in "reasoning".
- Reasoning must physically interleave natural-language COT lines and math-expression lines.
- The very last element of "reasoning" must be the final numeric answer.
- Decoy facts (not needed to answer the target) must be listed in "exclude_facts" with a brief reason. Do not use them in "reasoning".

Required JSON schema:
{
  "target": "<quantity and unit being asked>",
  "use_facts": [
    "<fact used to solve the target>"
  ],
  "exclude_facts": [
    "<fact excluded and brief reason>"
  ],
  "reasoning": [
    "<natural-language COT line>",
    "<math expression line: e.g. \'var = expr\' or just \'expr\'>",
    "<natural-language COT line>",
    "<math expression line>",
    "...",
    "<final numeric value line, e.g. \'total_bolts = 3\' or \'3\'>"
  ]
}

Rules:
R1. Every reasoning line must introduce or combine a useful quantity. Do not add number-padding lines.
R2. "increased by X%" means multiply by 1 + X/100. Identify whether a percentage means "X% of Y", "increased by X%", "decreased by X%", or "X% of the remaining" before computing.
R3. "but instead", "instead", "however", and "after that" usually indicate a path switch. Earlier alternative scenarios are excluded unless the question asks about them.
R4. If a fact is excluded in "exclude_facts", it must not appear as a math-expression in "reasoning".
R5. The target must match the question exactly (quantity and unit).
R6. A decoy is a fact not needed to answer the target. Put it in "exclude_facts"; do not compute it in "reasoning".
R7. In multiplication or division steps, units must match the quantity being computed.
R8. "How many X" must not mix units such as counts, weights, prices, or rates in the same summed expression.
R9. Do not write self-cancelling expressions such as "A / X * X" or "A + B - B" unless explicitly required.
R10. Reasoning must end with a numeric value; that value is the final answer.
R11. Math-expression lines contain numbers, operators, parentheses, and the equals sign. They do not contain prose.
R12. At least one natural-language COT line must appear in "reasoning" (do not skip prose entirely)."""


V14_USER_TEMPLATE = """Solve the following grade-school math problem. Return only one valid JSON object.

{{
  "target": "<quantity and unit>",
  "use_facts": ["<fact used>"],
  "exclude_facts": ["<fact excluded + reason>"],
  "reasoning": [
    "<prose line>",
    "<math expression line>",
    "<prose line>",
    "<math expression line>",
    "...",
    "<final numeric value>"
  ]
}}

Important:
- Output JSON only. No markdown. No prose outside JSON.
- Every calculation used for the final answer must appear as a math-expression line in "reasoning".
- Decoy / alternative-path / separate-scope facts must appear in "exclude_facts", never in "reasoning".
- The last element of "reasoning" is the final answer (a number).

Examples:

Example 1 (attribute_mismatch, decoy exclusion):
Problem: A robe takes 2 bolts of blue fiber and half that much white fiber. Each bolt of blue fiber weighs 3 pounds and each bolt of white fiber weighs 2 pounds. How many bolts in total does it take?
JSON:
{{
  "target": "total number of bolts",
  "use_facts": [
    "The robe takes 2 bolts of blue fiber.",
    "The robe takes half as many white bolts as blue."
  ],
  "exclude_facts": [
    "Each blue bolt weighs 3 pounds; the question asks for bolts, not weight.",
    "Each white bolt weighs 2 pounds; the question asks for bolts, not weight."
  ],
  "reasoning": [
    "We need the total number of bolts, not their weight, so bolt weight facts are decoys.",
    "blue_bolts = 2",
    "The white bolts count is half the blue bolts count.",
    "white_bolts = blue_bolts / 2 = 2 / 2 = 1",
    "Now add the two counts to get the total.",
    "total_bolts = blue_bolts + white_bolts = 2 + 1",
    "total_bolts = 3"
  ]
}}

Example 2 (path_competition, but-instead switch):
Problem: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She could sell all 16 eggs for $32, but instead she sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
JSON:
{{
  "target": "dollars made at the farmers' market per day",
  "use_facts": [
    "Janet's ducks lay 16 eggs per day.",
    "She eats 3 eggs for breakfast.",
    "She uses 4 eggs for muffins.",
    "She sells the remaining eggs at $2 each."
  ],
  "exclude_facts": [
    "She could sell all 16 eggs for $32; this is an alternative scenario replaced by \'but instead\'."
  ],
  "reasoning": [
    "\'but instead\' switches the path; the $32 alternative is excluded.",
    "remaining_eggs = 16 - 3 - 4",
    "remaining_eggs = 9",
    "Multiply remaining eggs by $2 each.",
    "dollars_per_day = remaining_eggs * 2 = 9 * 2",
    "dollars_per_day = 18"
  ]
}}

Example 3 (percent, increased-by mapping):
Problem: Josh buys a house for $80,000 and puts in $50,000 in repairs. This increased the value of the house by 150%. After selling, he donates 10% of his profit to charity. How much profit did he make from the flip?
JSON:
{{
  "target": "profit in dollars",
  "use_facts": [
    "The original house price is $80,000.",
    "The repair cost is $50,000.",
    "The house value increased by 150%."
  ],
  "exclude_facts": [
    "The 10% charity donation; the question asks for profit, not donation."
  ],
  "reasoning": [
    "\'Increased by 150%\' means multiply by 1 + 150/100 = 2.5, not 1.5.",
    "new_value = 80000 * 2.5",
    "new_value = 200000",
    "Profit subtracts purchase and repair costs.",
    "profit = new_value - 80000 - 50000 = 200000 - 130000",
    "profit = 70000"
  ]
}}

Now solve the following problem.

Problem:
__QUESTION__"""


def _extract_question(user_content: str) -> str:
    """从 V12 user content 末尾抽 'Problem: ...' 后的内容."""
    if "Problem:" in user_content:
        return user_content.split("Problem:")[-1].strip()
    return user_content.strip()



ROOT = Path("/home/wwq416/snap/wwq/math-chain")
# Self-contained: read raw + supplementary jsonl directly (same as v12/v13
# build scripts), no intermediate V12 jsonl required.
SRC = ROOT / "chaingsm_data/data/final/train_balanced_one_variant/gsm8k_train_balanced_one_variant/gsm8k_train_balanced_one_variant_14946_clean.jsonl"
SUP = ROOT / "chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/grpo_train_clean_v2.jsonl"
OUT = ROOT / "chaingsm_data/data/final/grpo/grpo_v14_reasoning.parquet"

# V12 tokenize 规则 (跟 build_grpo.py 一致)
TOKEN_RE = re.compile(r"\*\*|(?<![\d.])-?\d+(?:\.\d+)?|[+\-*/]")


def expression_to_tokens(expression: str) -> list[str]:
    """线性 tokenize, 去括号, 跟 V12 gold_trace_tokens 风格一致."""
    if not expression:
        return []
    expr = re.sub(r"[()]", "", expression.replace(" ", ""))
    if not expr:
        return []
    return TOKEN_RE.findall(expr)


def _build_use_facts(question: str, gold_steps: list) -> list[str]:
    """从 gold_steps 提炼 use_facts. 简化: 用 quantity 字段."""
    out = []
    for s in gold_steps:
        q = s.get("quantity", "").strip()
        if q and q not in out:
            out.append(q)
    return out[:6]  # 截断, 避免 prompt 过长


def _build_exclude_facts(category: str, distractor_expression: str, distractor_values: list, question: str) -> list[str]:
    """根据 category 推断 exclude_facts 描述."""
    if not distractor_expression:
        return []
    # 简单模板: 由 category 选一条
    templates = {
        "attribute_mismatch": "Some facts refer to a different attribute (e.g. weight vs count); the question asks for a specific attribute.",
        "independent_decoy": "Some facts are from a separate scope and are not needed to answer the target.",
        "path_competition": "An earlier alternative path is replaced by 'but instead' / 'instead' / 'however'; only the later path counts.",
        "target_scope_misalignment": "Some facts concern a related but different target; the question asks for a specific quantity.",
    }
    msg = templates.get(category, "Some facts are decoys and not needed to answer the target.")
    # 加上 decoy 涉及的 value (V12 distractor_values)
    extras = []
    for v in (distractor_values or [])[:2]:
        if v and str(v) not in msg:
            extras.append(f"Decoy value '{v}' is not used in the target computation.")
    return [msg] + extras


def _build_reasoning(gold_steps: list) -> list[str]:
    """从 gold_trace 转 prose 表达式 交错 数组.

    每个 step 产生 2-3 行:
      - 1 行 prose (说"算什么", 可选, 用 quantity 字段)
      - 1 行 math expression: "var = expr"
      - 1 行 value: "var = value"  (可选, 视教学需要)
    """
    out = []
    for i, s in enumerate(gold_steps):
        var = s.get("variable", f"step_{i+1}")
        qty = s.get("quantity", "")
        expr = s.get("expression", "")
        val = s.get("value", "")

        # prose 行 (解释这一步算什么)
        if i == 0:
            prose = f"First, compute the {qty}." if qty else "Compute the first quantity."
        elif i == len(gold_steps) - 1:
            prose = f"Now combine to get the final {qty}." if qty else "Now combine to get the final answer."
        else:
            prose = f"Then, compute the {qty}." if qty else "Then, compute the next quantity."
        out.append(prose)

        # 表达式行
        out.append(f"{var} = {expr}")

        # 值行 (在 multi-step 教学场景里, 显式给值)
        if val:
            out.append(f"{var} = {val}")

    return out


def _is_last_value_a_number(reasoning: list) -> bool:
    """末行必须是数字 (或 var = number)."""
    if not reasoning:
        return False
    last = reasoning[-1].strip()
    m = re.search(r"[=]\s*([-+]?\d+(?:\.\d+)?)\s*$", last)
    if m:
        try:
            float(m.group(1))
            return True
        except ValueError:
            return False
    try:
        float(last)
        return True
    except ValueError:
        return False


def _last_value(reasoning: list):
    if not reasoning:
        return None
    last = reasoning[-1].strip()
    m = re.search(r"[=]\s*([-+]?\d+(?:\.\d+)?)\s*$", last)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    try:
        return float(last)
    except ValueError:
        return None


def main():
    if not SRC.exists() or not SUP.exists():
        sys.exit(f"Missing input file: SRC={SRC.exists()} SUP={SUP.exists()}")

    # Build v12 reward_meta fields inline (same as build_grpo_v12_json.py).
    from build_grpo import build_gold_trace, build_gold_trace_tokens, build_distractor_values

    src_by_id = {json.loads(l)["id"]: json.loads(l) for l in SRC.open()}
    sup_by_id = {json.loads(l)["id"]: json.loads(l) for l in SUP.open()}
    print(f"loaded src rows: {len(src_by_id)}, sup rows: {len(sup_by_id)}")

    rows = []
    n = 0
    n_bad = 0
    n_with_dist = 0
    for sid, src in src_by_id.items():
        sup = sup_by_id.get(sid)
        if not sup:
            n_bad += 1
            continue
        ref = sup.get("reward_reference") or {}
        gold_trace_raw = ref.get("gold_trace") or []
        if not all(isinstance(t, dict) for t in gold_trace_raw):
            n_bad += 1
            continue
        if not gold_trace_raw:
            n_bad += 1
            continue
        n += 1

        gold_trace = build_gold_trace(gold_trace_raw)
        gold_expression = ref.get("gold_expression") or src.get("gold_expression") or ""
        gold_trace_tokens = build_gold_trace_tokens(gold_trace) or []
        distractor_trace_raw = ref.get("distractor_trace") or []
        distractor_trace = [t for t in distractor_trace_raw if isinstance(t, dict)]
        distractor_expression = ref.get("distractor_expression") or src.get("distractor_expression") or ""
        distractor_values = build_distractor_values(gold_trace, distractor_trace) if distractor_trace else []
        category = src.get("category", "original")
        gold_answer = str(ref.get("gold_answer") or src.get("answer") or "")
        # V14 prompt user template uses __QUESTION__ placeholder
        question = src["question_distracted"].strip()

        # reasoning 数组
        reasoning = _build_reasoning(gold_trace)
        if not _is_last_value_a_number(reasoning):
            # 修补: 末行是表达式 (无值), 追加一行值
            if gold_answer:
                try:
                    reasoning.append(f"final_answer = {float(gold_answer)}")
                except ValueError:
                    pass
            if not _is_last_value_a_number(reasoning):
                n_bad += 1
                continue  # 跳过

        # use_facts
        use_facts = _build_use_facts(sid, gold_trace)

        # exclude_facts
        exclude_facts = _build_exclude_facts(
            category, distractor_expression, distractor_values, ""
        )

        # distractor_trace_tokens (从 distractor_expression tokenize)
        distractor_trace_tokens = expression_to_tokens(distractor_expression) if distractor_expression else []
        if distractor_trace_tokens:
            n_with_dist += 1

        # target (从原始问题里抽 "How much/many X" 模板, 这里简化)
        _v12_user_content = f"Problem: {question}"  # mimic v12 user template tail for _extract_target
        target = _extract_target(_v12_user_content)

        ground_truth = {
            "answer": str(gold_answer),
            "gold_answer": str(gold_answer),
            "gold_expression": gold_expression,
            "gold_trace_tokens": list(gold_trace_tokens),
            "distractor_expression": distractor_expression,
            "distractor_trace_tokens": distractor_trace_tokens,
            "distractor_enabled": bool(distractor_expression),
            "category": category,
            # V14 新增: 把期望的 reasoning 也存到 ground_truth
            # (reward 不会直接用, 仅做 sanity check)
            "gold_reasoning": reasoning,
            "use_facts": use_facts,
            "exclude_facts": exclude_facts,
            "target": target,
        }

        # V14 prompt 注入: V14 system + user 模板
        v14_messages = [
            {"role": "system", "content": V14_SYSTEM_PROMPT},
            {"role": "user", "content": V14_USER_TEMPLATE.replace("__QUESTION__", question)},
        ]

        rows.append({
            "data_source": "chaingsm_cot_grpo_v14_reasoning",
            "prompt": v14_messages,
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "id": f"{sid}_grpo_v14",
                "source_id": sid,
                "category": category,
                "split": "train",
                "index": n - 1,
            },
        })
    print(f"loaded {n} rows, kept {len(rows)}, bad={n_bad}, with_distractor={n_with_dist}", flush=True)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {len(df)} rows to {OUT}", flush=True)
    # 2026-06-21: 同步 V14_SYSTEM_PROMPT 给评测, 避免 train-test prompt mismatch.
    _system_prompt_path = OUT.parent / "_system_prompt.txt"
    _system_prompt_path.write_text(V14_SYSTEM_PROMPT, encoding="utf-8")
    print(f"wrote system prompt to {_system_prompt_path} ({len(V14_SYSTEM_PROMPT)} chars)", flush=True)


def _extract_target(user_content: str) -> str:
    """从 user prompt 末尾的 'Problem:' 后面抽 'How much/many X?' 模板."""
    # 取 Problem: 之后的内容
    if "Problem:" in user_content:
        q = user_content.split("Problem:")[-1].strip()
    else:
        q = user_content.strip()
    # 找 "How much/many ... ?" 模板
    patterns = [
        r"How (?:much|many)\s+(.+?)\s+(?:does|did|do)\s+[^?]+\?",
        r"How (?:much|many)\s+(.+?)\s*\?",
        r"What (?:is|are)\s+the\s+(.+?)\s*\?",
        r"What (?:is|are)\s+(.+?)\s*\?",
    ]
    for pat in patterns:
        m = re.search(pat, q, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return "the requested quantity"


if __name__ == "__main__":
    main()
