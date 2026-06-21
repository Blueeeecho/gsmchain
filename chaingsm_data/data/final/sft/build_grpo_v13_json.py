"""Build v13 GRPO training parquet directly from raw + supplementary data.

One-step pipeline (mirrors the v6a_LTLT.py pattern): prompt templates are
defined as module-level constants at the top, the data is loaded from the
raw + supplementary jsonl files, and the verl-compatible parquet is
written directly under chaingsm_data/data/final/grpo/. No intermediate
jsonl is produced. To change the prompt, edit SYSTEM / USER_TEMPLATE
below and rerun this script.

Differences from v12:
- 5 fields (target / steps / exclude_facts / final_expression / final_answer)
  instead of 6 (no use_facts / no answer).
- 3 rules (R1 useful / R2 percent / R3 path switch) instead of 12.
- final_answer is numeric (not string) for v13 reward hard numeric check.
- ground_truth includes "gold_steps" [{explanation, expression, value}, ...]
  so v13 reward can do per-step hard numeric matching.
- ground_truth includes "exclude_facts" (gold decoy facts) for r_exclude.

Output: chaingsm_data/data/final/grpo/grpo_v13_json.parquet (3051 rows,
5 columns: data_source / prompt / ability / reward_model / extra_info)
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/home/wwq416/snap/wwq/math-chain")
# 2026-06-21: 改用 unified jsonl (raw 14946 + supp grpo_train_clean_v2 合并), 1 个文件就够.
SRC = ROOT / "chaingsm_data/data/final/sft/gsm8k_train_unified_6102.jsonl"
# Direct parquet output (no intermediate jsonl). Matches submit_grpo_v13_verl.sh
# symlink target name expected by train_configs/remote/grpo_verl_v13_vllm.yaml.
DST = ROOT / "chaingsm_data/data/final/grpo/grpo_v13_json.parquet"

# 2026-06-19: v12_long_prompt v4 system prompt (与 docs/prompts/v12_long_prompt.md / eval_vllm_chaingsm.py 同步)
SYSTEM = """You are a careful grade-school math reasoning assistant.
Your task is to solve the problem correctly and return only one valid JSON object.
Reasoning style is not judged; correctness, target selection, step validity, and exclusion of irrelevant facts are judged.

Output rules:
- Output valid JSON only.
- No markdown. No prose before/after. No comments. No trailing commas.
- "steps" is the only place where you record arithmetic derivations. Every calculation used for the final answer must appear in "steps".
- Each step has three fields:
  "explanation": one short sentence describing what this step computes.
  "expression": arithmetic only (numbers, operators, parentheses). No prose.
  "value": the evaluated result.
- The "final_answer" must equal the numeric value of "final_expression".
- Decoy facts (not needed to answer the target) must be listed in "exclude_facts" with a brief reason. Do not compute them in "steps".

Required JSON schema:
{
  "target": "<quantity and unit being asked>",
  "steps": [
    {"explanation": "", "expression": "", "value": ""}
  ],
  "exclude_facts": [
    "<fact excluded and brief reason>"
  ],
  "final_expression": "<single expression for the final answer>",
  "final_answer": <number>
}

Rules:
R1. Every step must introduce or combine a useful quantity. Do not add number-padding steps.
R2. "increased by X%" means multiply by 1 + X/100. Identify whether a percentage means "X% of Y", "increased by X%", or "decreased by X%" before computing.
R3. "but instead", "instead", "however", and "after that" usually indicate a path switch. Earlier alternative scenarios are excluded unless the question asks about them.


---"""

USER_TEMPLATE = """Solve the following grade-school math problem. Return only one valid JSON object.

{
  "target": "<quantity and unit being asked>",
  "steps": [
    {"explanation": "<what this step computes>", "expression": "<arithmetic>", "value": "<result>"}
  ],
  "exclude_facts": [
    "<fact excluded and brief reason>"
  ],
  "final_expression": "<single expression for the final answer>",
  "final_answer": <number>
}

Examples:

Example 1 (simple arithmetic):
Problem: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins. She sells the rest at $2 each. How much does she make per day at the farmers' market?
JSON:
{
  "target": "dollars per day at the farmers' market",
  "steps": [
    {"explanation": "Compute remaining eggs after breakfast and muffins.", "expression": "16 - 3 - 4", "value": 9},
    {"explanation": "Compute dollars from selling the remaining eggs at $2 each.", "expression": "9 * 2", "value": 18}
  ],
  "exclude_facts": [],
  "final_expression": "(16 - 3 - 4) * 2",
  "final_answer": 18
}

Example 2 (percent, no decoy):
Problem: Josh buys a house for $80,000 and spends $50,000 on repairs. The house value increases by 150%. What is the profit?
JSON:
{
  "target": "profit in dollars",
  "steps": [
    {"explanation": "Compute new value after 150% increase.", "expression": "80000 * (1 + 150/100)", "value": 200000},
    {"explanation": "Subtract purchase and repair costs.", "expression": "200000 - 80000 - 50000", "value": 70000}
  ],
  "exclude_facts": [],
  "final_expression": "80000 * (1 + 150/100) - 80000 - 50000",
  "final_answer": 70000
}

Example 3 (path switch + decoy exclusion):
Problem: A robe takes 2 bolts of blue fiber and half that much white fiber. Each bolt of blue fiber weighs 3 pounds and each bolt of white fiber weighs 2 pounds. How many bolts in total does it take?
JSON:
{
  "target": "total number of bolts",
  "steps": [
    {"explanation": "Compute the number of white bolts as half the number of blue bolts.", "expression": "2 / 2", "value": 1},
    {"explanation": "Add the blue bolts and white bolts.", "expression": "2 + 1", "value": 3}
  ],
  "exclude_facts": [
    "Each blue bolt weighs 3 pounds; the question asks for bolts, not weight.",
    "Each white bolt weighs 2 pounds; the question asks for bolts, not weight."
  ],
  "final_expression": "2 + 2 / 2",
  "final_answer": 3
}

Problem:
{question}"""


def main():
    if not SRC.exists():
        sys.exit(f"Missing input file: SRC={SRC}")
    DST.parent.mkdir(parents=True, exist_ok=True)

    src_by_id = {json.loads(l)["id"]: json.loads(l) for l in SRC.open()}
    # 2026-06-21: 改用 unified jsonl, sup_by_id 不再需要
    print(f"loaded src rows: {len(src_by_id)} (unified jsonl, raw+supp 合并)")

    written, skipped = 0, 0
    rows = []
    for sid, src in src_by_id.items():
        gold_trace_raw = src.get("gold_trace") or []
        if not all(isinstance(t, dict) for t in gold_trace_raw):
            skipped += 1
            continue
        if not gold_trace_raw:
            skipped += 1
            continue

        # Reuse all reward_meta fields from build_grpo.py
        from build_grpo import build_gold_trace, build_gold_trace_tokens, build_distractor_values
        answer = str(src.get("answer") or "")
        gold_expr = src.get("gold_expression") or ""
        gold_trace = build_gold_trace(gold_trace_raw)
        tokens = build_gold_trace_tokens(gold_trace) or []
        distractor_trace_raw = src.get("distractor_trace") or []
        distractor_trace = [t for t in distractor_trace_raw if isinstance(t, dict)]
        distractor_expr = src.get("distractor_expression") or ""
        distractor_values = build_distractor_values(gold_trace, distractor_trace) if distractor_trace else []

        # Build the verl-compatible parquet row directly.
        # v13-specific: include gold_steps and exclude_facts for the
        # v13 reward's per-step hard numeric matching and r_exclude.
        # (Equivalent to the old v13_json_to_parquet.py step, inlined here.)
        messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TEMPLATE.replace("{question}", src["question_distracted"].strip())},
        ]
        gold_steps = [
                {
                    "explanation": step.get("quantity", ""),
                    "expression": step.get("expression", ""),
                    "value": step.get("value", ""),
                }
                for step in gold_trace
        ]
        exclude_facts = [
                d.strip() for d in (distractor_values or []) if isinstance(d, str) and d.strip()
        ]
        ground_truth = {
                "answer": answer,
                "gold_answer": answer,
                "gold_expression": gold_expr,
                "gold_steps": gold_steps,
                "exclude_facts": exclude_facts,
                "gold_trace_tokens": tokens,
                "distractor_expression": distractor_expr,
                "distractor_trace_tokens": [],
                "distractor_enabled": bool(distractor_expr),
                "category": src["category"],
        }
        rows.append({
                "data_source": "chaingsm_cot_grpo_v13_json",
                "prompt": [messages[0], messages[1]],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "id": f"{sid}_grpo_v13",
                    "source_id": sid,
                    "category": src["category"],
                    "split": "train",
                    "index": len(rows),
                },
        })
        written += 1

    DST.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(DST, index=False)
    # 2026-06-21: 同步 SYSTEM 给评测, 避免 train-test prompt mismatch.
    _system_prompt_path = DST.parent / "_system_prompt.txt"
    _system_prompt_path.write_text(SYSTEM, encoding="utf-8")
    print(f"wrote system prompt to {_system_prompt_path} ({len(SYSTEM)} chars)", flush=True)
    print(f"written={written} skipped={skipped}")
    print(f"out: {DST}")


if __name__ == "__main__":
    main()
