"""Build v12 GRPO training parquet directly from raw + supplementary data.

One-step pipeline (mirrors the v6a_LTLT.py pattern): prompt templates are
defined as module-level constants at the top, the data is loaded from the
raw + supplementary jsonl files, and the verl-compatible parquet is
written directly under chaingsm_data/data/final/grpo/. To change the
prompt, edit SYSTEM / USER_TEMPLATE below and rerun this script.

Output: chaingsm_data/data/final/grpo/grpo_v12_json.parquet (3051 rows,
5 columns: data_source / prompt / ability / reward_model / extra_info)
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/home/wwq416/snap/wwq/math-chain")
SRC = ROOT / "chaingsm_data/data/final/train_balanced_one_variant/gsm8k_train_balanced_one_variant/gsm8k_train_balanced_one_variant_14946_clean.jsonl"
SUP = ROOT / "chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/grpo_train_clean_v2.jsonl"
# Direct parquet output (no intermediate jsonl). Matches submit_grpo_v12_verl.sh
# symlink target name expected by train_configs/remote/grpo_verl_v12_vllm.yaml.
DST = ROOT / "chaingsm_data/data/final/grpo/grpo_v12_json.parquet"
SYSTEM = """You are a careful grade-school math reasoning assistant.
Your task is to solve the problem correctly and return only one valid JSON object.
Reasoning style is not judged; correctness, target selection, calculation validity, and exclusion of irrelevant facts are judged.
Output requirements:
Output valid JSON only.
Do not use markdown.
Do not write any text before or after the JSON.
Do not include comments.
Do not include trailing commas.
All arithmetic derivations used for the final answer must appear in the "steps" array.
Each step must contain exactly three fields:
"explanation": a short natural-language explanation of the quantity being computed.
"expression": the arithmetic expression.
"value": the evaluated result.
The final answer must be represented by "final_expression" and "answer".
Do not include irrelevant, hypothetical, optional, alternative-path, or separate-scope calculations in "steps".
If a fact is excluded, list it in "exclude_facts" with a brief reason, but do not compute it in "steps".
Required JSON schema:
{
"target": "",
"use_facts": [
""
],
"exclude_facts": [
""
],
"steps": [
{
"explanation": "",
"expression": "",
"value": ""
}
],
"final_expression": "",
"answer": ""
}
Rules:
R1. No self-cancelling expressions such as "A / X * X", "A * X / X", "A + B - B", or "A - B + B" unless they are explicitly required by the problem.
R2. In multiplication or division steps, the units must match the quantity being computed.
R3. Every step must introduce or combine a useful quantity. Do not add number-padding steps.
R4. If "exclude_facts" says a fact is excluded, its value must not appear in any step expression or final_expression.
R5. "increased by X%" means multiply by 1 + X/100. For example, increased by 150% means multiply by 2.5.
R6. Identify whether a percentage means "X% of Y", "increased by X%", "decreased by X%", or "X% of the remaining" before computing.
R7. If the question asks "how many X", do not mix units such as counts, weights, prices, or rates in the same summed expression.
R8. "but instead", "instead", and "however" usually indicate a path switch; earlier alternative scenarios are excluded unless the question asks about them.
R9. The target must match the question exactly.
R10. A decoy is a fact not needed to answer the target. Put it in "exclude_facts"; do not compute it in "steps".
R11. The "answer" must equal the value of "final_expression".
R12. Use only arithmetic expressions in "expression" and "final_expression"; do not put prose there.
"""

USER_TEMPLATE = """Solve the following grade-school math problem.

Return only one valid JSON object following this schema:

{
"target": "<quantity and unit being asked>",
"use_facts": [
"<fact used to solve the target>"
],
"exclude_facts": [
"<fact excluded and reason>"
],
"steps": [
{
"explanation": "<what this step computes>",
"expression": "<arithmetic expression>",
"value": "<result>"
}
],
"final_expression": "<single expression for the final answer>",
"answer": "<number>"
}

Important:

* Output JSON only.
* Do not use markdown.
* Do not write explanations outside the JSON.
* Every calculation used for the final answer must appear in "steps".
* Irrelevant, hypothetical, optional, alternative-path, or separate-scope facts must appear only in "exclude_facts", not in "steps".
* "final_expression" must compute the final answer directly.
* "answer" must be the numeric value of "final_expression".

Examples:

Example 1:
Problem: A robe takes 2 bolts of blue fiber and half that much white fiber. Each bolt of blue fiber weighs 3 pounds and each bolt of white fiber weighs 2 pounds. How many bolts in total does it take?

JSON:
{
"target": "total number of bolts",
"use_facts": [
"The robe takes 2 bolts of blue fiber.",
"The robe takes half as many bolts of white fiber as blue fiber."
],
"exclude_facts": [
"Each blue bolt weighs 3 pounds; the question asks for bolts, not weight.",
"Each white bolt weighs 2 pounds; the question asks for bolts, not weight."
],
"steps": [
{
"explanation": "Compute the number of white bolts as half the number of blue bolts.",
"expression": "2 / 2",
"value": "1"
},
{
"explanation": "Add the blue bolts and white bolts.",
"expression": "2 + 1",
"value": "3"
}
],
"final_expression": "2 + 2 / 2",
"answer": "3"
}

Example 2:
Problem: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She could sell all 16 eggs for $32, but instead she sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

JSON:
{
"target": "dollars made at the farmers' market per day",
"use_facts": [
"Janet's ducks lay 16 eggs per day.",
"She eats 3 eggs for breakfast.",
"She uses 4 eggs for baking.",
"She sells the remaining eggs for $2 each."
],
"exclude_facts": [
"She could sell all 16 eggs for $32; this is an alternative scenario replaced by 'but instead'."
],
"steps": [
{
"explanation": "Compute the number of eggs remaining after breakfast and baking.",
"expression": "16 - 3 - 4",
"value": "9"
},
{
"explanation": "Compute the dollars earned from selling the remaining eggs at $2 each.",
"expression": "9 * 2",
"value": "18"
}
],
"final_expression": "(16 - 3 - 4) * 2",
"answer": "18"
}

Example 3:
Problem: Josh buys a house for $80,000 and puts in $50,000 in repairs. This increased the value of the house by 150%. After selling, he donates 10% of his profit to charity. How much profit did he make from the flip?

JSON:
{
"target": "profit in dollars",
"use_facts": [
"The original house price is $80,000.",
"The repair cost is $50,000.",
"The house value increased by 150%, meaning the value becomes 2.5 times the original price."
],
"exclude_facts": [
"The 10% donation is excluded because the question asks for profit, not donation."
],
"steps": [
{
"explanation": "Compute the selling value after a 150% increase.",
"expression": "80000 * 2.5",
"value": "200000"
},
{
"explanation": "Compute profit by subtracting purchase cost and repair cost from selling value.",
"expression": "200000 - 80000 - 50000",
"value": "70000"
}
],
"final_expression": "80000 * 2.5 - 80000 - 50000",
"answer": "70000"
}

Now solve the following problem.

Problem:
{question}
"""


def main():
    if not SRC.exists() or not SUP.exists():
        sys.exit(f"Missing input file: SRC={SRC.exists()} SUP={SUP.exists()}")
    DST.parent.mkdir(parents=True, exist_ok=True)

    src_by_id = {json.loads(l)["id"]: json.loads(l) for l in SRC.open()}
    sup_by_id = {json.loads(l)["id"]: json.loads(l) for l in SUP.open()}
    print(f"loaded src rows: {len(src_by_id)}, sup rows: {len(sup_by_id)}")

    written, skipped = 0, 0
    rows = []
    for sid, src in src_by_id.items():
        sup = sup_by_id.get(sid)
        if not sup:
            skipped += 1
            continue
        ref = sup.get("reward_reference") or {}
        gold_trace_raw = ref.get("gold_trace") or []
        if not all(isinstance(t, dict) for t in gold_trace_raw):
            skipped += 1
            continue
        if not gold_trace_raw:
            skipped += 1
            continue

        # Reuse all reward_meta fields from build_grpo.py
        from build_grpo import build_gold_trace, build_gold_trace_tokens, build_distractor_values
        answer = str(ref.get("gold_answer") or src.get("answer") or "")
        gold_expr = ref.get("gold_expression") or src.get("gold_expression") or ""
        gold_trace = build_gold_trace(gold_trace_raw)
        tokens = build_gold_trace_tokens(gold_trace) or []
        distractor_trace_raw = ref.get("distractor_trace") or []
        distractor_trace = [t for t in distractor_trace_raw if isinstance(t, dict)]
        distractor_expr = ref.get("distractor_expression") or src.get("distractor_expression") or ""
        distractor_values = build_distractor_values(gold_trace, distractor_trace) if distractor_trace else []

        # Build the verl-compatible parquet row directly. Equivalent to the
        # old v12_json_to_parquet.py step, inlined here.
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TEMPLATE.replace("{question}", src["question_distracted"].strip())},
        ]
        ground_truth = {
            "answer": answer,
            "gold_answer": answer,
            "gold_expression": gold_expr,
            "gold_trace_tokens": tokens,
            "distractor_expression": distractor_expr,
            "distractor_trace_tokens": [],
            "distractor_enabled": bool(distractor_expr),
            "category": src["category"],
        }
        rows.append({
            "data_source": "chaingsm_cot_grpo_v12_json",
            "prompt": [messages[0], messages[1]],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "id": f"{sid}_grpo_v12",
                "source_id": sid,
                "category": src["category"],
                "split": "train",
                "index": len(rows),
            },
        })
        written += 1

    df = pd.DataFrame(rows)
    df.to_parquet(DST, index=False)
    # 2026-06-21: 同步 SYSTEM 给评测, 避免 train-test prompt mismatch.
    # eval_vllm_chaingsm.py method=parquet_prompt 读这个文件当 system prompt.
    _system_prompt_path = DST.parent / "_system_prompt.txt"
    _system_prompt_path.write_text(SYSTEM, encoding="utf-8")
    print(f"wrote system prompt to {_system_prompt_path} ({len(SYSTEM)} chars)", flush=True)
    print(f"written={written} skipped={skipped}")
    print(f"out: {DST}")


if __name__ == "__main__":
    main()
