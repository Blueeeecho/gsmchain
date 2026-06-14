"""Patch sft_train.jsonl: only system + user prompts. Response (incl. descriptions) untouched.

Per advisor feedback (2026-06-08):
  - system: remove "ignore distractor chains" and "return JSON only"
  - user: keep JSON schema, but prepend "First identify the target quantity ..."
  - response: NOT modified (keeps "Compute step X for the correct chain." descriptions intact
    since advisor said "不要修改 description" / "用方案 C 但 description 暂不改")

This satisfies: 保持数据结构 + 调整提示词
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "chaingsm_data" / "data" / "final" / "rl_preprocessed" / "gsm8k_train_balanced_one_variant_14946"
SOURCE_SFT = DATA_DIR / "sft_train.jsonl"
OUTPUT_FILE = DATA_DIR / "sft_train_neutral_promptsonly.jsonl"

NEUTRAL_SYSTEM = (
    "You are an expert grade-school math solver. Identify the quantity being asked, "
    "solve it carefully step by step, and follow the requested output format."
)

NEUTRAL_USER_PREAMBLE = """Solve the following grade-school math problem.
First identify the target quantity asked by the question. Then write the calculation steps needed to compute that target.
Return one valid JSON object with this schema:
{
  "target": "short description of the quantity being asked",
  "selected_steps": [
    {
      "variable": "short variable name",
      "description": "mathematical meaning of this step",
      "expression": "arithmetic expression",
      "value": "computed value"
    }
  ],
  "final_expression": "arithmetic expression that computes the answer",
  "answer": "final numeric answer"
}
Requirements:
- The JSON must be parseable.
- Use meaningful step descriptions.
- Do not include calculations that are not needed for the target quantity.
- Do not add text outside the JSON object.
Problem:
"""


def _extract_question_from_old_user(old_user: str) -> str:
    m = re.search(r"Problem:\s*\n(.+?)\s*$", old_user, re.DOTALL)
    return m.group(1).strip() if m else ""


def main() -> None:
    n_total = 0
    n_written = 0
    cat_counter: dict[str, int] = {}

    with SOURCE_SFT.open("r", encoding="utf-8") as fin, OUTPUT_FILE.open("w", encoding="utf-8") as fout:
        for line in fin:
            n_total += 1
            d = json.loads(line)
            cat_counter[d.get("category", "unknown")] = cat_counter.get(d.get("category", "unknown"), 0) + 1

            old_system = d["messages"][0]["content"]
            old_user = d["messages"][1]["content"]
            old_response = d["messages"][2]["content"]  # UNCHANGED

            new_user = NEUTRAL_USER_PREAMBLE + _extract_question_from_old_user(old_user)

            new_messages = [
                {"role": "system", "content": NEUTRAL_SYSTEM},
                {"role": "user", "content": new_user},
                {"role": "assistant", "content": old_response},  # keep response as-is
            ]

            out = {
                "id": d["id"],
                "base_id": d.get("base_id"),
                "category": d.get("category"),
                "messages": new_messages,
                "prompt": new_messages[1]["content"],
                "response": new_messages[2]["content"],
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Input rows: {n_total}")
    print(f"Written: {n_written}")
    print(f"Category distribution: {cat_counter}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
