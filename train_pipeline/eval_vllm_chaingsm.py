from __future__ import annotations
import re

import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from gsm_answer_extractor import extract_answer as extract_text_answer
from gsm_answer_extractor import is_correct
from train_pipeline.eval_constants import DEFAULT_TEST_DATA
from train_pipeline.preprocess_chaingsm import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    NEUTRAL_SYSTEM_PROMPT,
    NEUTRAL_USER_TEMPLATE,
)


@dataclass
class EvalExample:
    id: str
    base_id: str
    category: str
    question: str
    gold_answer: str


DIRECT_SYSTEM_PROMPT = "You are a helpful assistant."
ZERO_SHOT_SYSTEM_PROMPT = "You are a helpful assistant."

# 2026-06-17: 新 CoT 协议 prompt, 与 SFT 训练时 all_sft_cot.jsonl 同源。
# 任何修改必须与 chaingsm_data/data/final/sft/all_sft_cot.jsonl 第一条 system 字段同步。
COT_BRACKETS_SYSTEM_PROMPT = (
    "You are a careful grade-school math reasoning assistant. Solve the problem "
    "using natural reasoning. Put every arithmetic derivation that is used for the "
    "final answer inside double angle brackets in the exact form <<expression = value>>. "
    "Put the final derivation inside <<FINAL: expression = answer>>. Do not put prose "
    "inside the double angle brackets. Do not put ignored, hypothetical, optional, "
    "separate-scope, or distractor calculations inside the double angle brackets; "
    "mention them only in prose if needed."
)

# 2026-06-17: 与 SFT 训练时 all_sft_cot.jsonl 同源 user 模板, {question} 占位。
COT_BRACKETS_USER_TEMPLATE = (
    "Solve the following grade-school math problem.\n\n"
    "Use natural language reasoning, but put each arithmetic derivation that "
    "contributes to the final answer inside double angle brackets.\n\n"
    "Format:\nTARGET: ...\n\n"
    "[Brief natural-language reasoning.]\n<<expression = value>>\n\n"
    "[Brief natural-language reasoning.]\n<<expression = value>>\n\n"
    "Add more derivations as needed.\n\n"
    "<<FINAL: final_expression = answer>>\nANSWER: answer\n\n"
    "Rules:\n"
    "- Only calculations used to answer the actual question should appear inside <<...>>.\n"
    "- Do not put unused, hypothetical, optional, separate-scope, or distractor "
    "calculations inside <<...>>.\n"
    "- If a fact is not used, you may explain in prose why it is excluded, but do not "
    "calculate it inside <<...>>.\n"
    "- Keep all prose outside <<...>>.\n"
    "- The final line must contain ANSWER: answer.\n\n"
    "Problem:\n{question}\n"
)

# 2026-06-19: v12_long_prompt v4 (纯 JSON 输出, 6 字段结构)
# 与 docs/prompts/v12_long_prompt.md 同步
COT_BRACKETS_V12_JSON_SYSTEM_PROMPT = (
    "You are a careful grade-school math reasoning assistant.\n"
    "Your task is to solve the problem correctly and return only one valid JSON object.\n"
    "Reasoning style is not judged; correctness, target selection, calculation validity, and exclusion of irrelevant facts are judged.\n"
    "Output requirements:\n"
    "Output valid JSON only.\n"
    "Do not use markdown.\n"
    "Do not write any text before or after the JSON.\n"
    "Do not include comments.\n"
    "Do not include trailing commas.\n"
    "All arithmetic derivations used for the final answer must appear in the \"steps\" array.\n"
    "Each step must contain exactly three fields:\n"
    "\"explanation\": a short natural-language explanation of the quantity being computed.\n"
    "\"expression\": the arithmetic expression.\n"
    "\"value\": the evaluated result.\n"
    "The final answer must be represented by \"final_expression\" and \"answer\".\n"
    "Do not include irrelevant, hypothetical, optional, alternative-path, or separate-scope calculations in \"steps\".\n"
    "If a fact is excluded, list it in \"exclude_facts\" with a brief reason, but do not compute it in \"steps\".\n"
    "Required JSON schema:\n"
    "{\n"
    "\"target\": \"\",\n"
    "\"use_facts\": [\n"
    "\"\"\n"
    "],\n"
    "\"exclude_facts\": [\n"
    "\"\"\n"
    "],\n"
    "\"steps\": [\n"
    "{\n"
    "\"explanation\": \"\",\n"
    "\"expression\": \"\",\n"
    "\"value\": \"\"\n"
    "}\n"
    "],\n"
    "\"final_expression\": \"\",\n"
    "\"answer\": \"\"\n"
    "}\n"
    "Rules:\n"
    "R1. No self-cancelling expressions such as \"A / X * X\", \"A * X / X\", \"A + B - B\", or \"A - B + B\" unless they are explicitly required by the problem.\n"
    "R2. In multiplication or division steps, the units must match the quantity being computed.\n"
    "R3. Every step must introduce or combine a useful quantity. Do not add number-padding steps.\n"
    "R4. If \"exclude_facts\" says a fact is excluded, its value must not appear in any step expression or final_expression.\n"
    "R5. \"increased by X%\" means multiply by 1 + X/100. For example, increased by 150% means multiply by 2.5.\n"
    "R6. Identify whether a percentage means \"X% of Y\", \"increased by X%\", \"decreased by X%\", or \"X% of the remaining\" before computing.\n"
    "R7. If the question asks \"how many X\", do not mix units such as counts, weights, prices, or rates in the same summed expression.\n"
    "R8. \"but instead\", \"instead\", and \"however\" usually indicate a path switch; earlier alternative scenarios are excluded unless the question asks about them.\n"
    "R9. The target must match the question exactly.\n"
    "R10. A decoy is a fact not needed to answer the target. Put it in \"exclude_facts\"; do not compute it in \"steps\".\n"
    "R11. The \"answer\" must equal the value of \"final_expression\".\n"
    "R12. Use only arithmetic expressions in \"expression\" and \"final_expression\"; do not put prose there.\n"
)

COT_BRACKETS_V12_JSON_USER_TEMPLATE = (
    "Solve the following grade-school math problem.\n\n"
    "Return only one valid JSON object following this schema:\n\n"
    "{\n"
    "\"target\": \"<quantity and unit being asked>\",\n"
    "\"use_facts\": [\n"
    "\"<fact used to solve the target>\"\n"
    "],\n"
    "\"exclude_facts\": [\n"
    "\"<fact excluded and reason>\"\n"
    "],\n"
    "\"steps\": [\n"
    "{\n"
    "\"explanation\": \"<what this step computes>\",\n"
    "\"expression\": \"<arithmetic expression>\",\n"
    "\"value\": \"<result>\"\n"
    "}\n"
    "],\n"
    "\"final_expression\": \"<single expression for the final answer>\",\n"
    "\"answer\": \"<number>\"\n"
    "}\n\n"
    "Important:\n\n"
    "* Output JSON only.\n"
    "* Do not use markdown.\n"
    "* Do not write explanations outside the JSON.\n"
    "* Every calculation used for the final answer must appear in \"steps\".\n"
    "* Irrelevant, hypothetical, optional, alternative-path, or separate-scope facts must appear only in \"exclude_facts\", not in \"steps\".\n"
    "* \"final_expression\" must compute the final answer directly.\n"
    "* \"answer\" must be the numeric value of \"final_expression\".\n\n"
    "Examples:\n\n"
    "Example 1:\n"
    "Problem: A robe takes 2 bolts of blue fiber and half that much white fiber. Each bolt of blue fiber weighs 3 pounds and each bolt of white fiber weighs 2 pounds. How many bolts in total does it take?\n\n"
    "JSON:\n"
    "{\n"
    "\"target\": \"total number of bolts\",\n"
    "\"use_facts\": [\n"
    "\"The robe takes 2 bolts of blue fiber.\",\n"
    "\"The robe takes half as many bolts of white fiber as blue fiber.\"\n"
    "],\n"
    "\"exclude_facts\": [\n"
    "\"Each blue bolt weighs 3 pounds; the question asks for bolts, not weight.\",\n"
    "\"Each white bolt weighs 2 pounds; the question asks for bolts, not weight.\"\n"
    "],\n"
    "\"steps\": [\n"
    "{\n"
    "\"explanation\": \"Compute the number of white bolts as half the number of blue bolts.\",\n"
    "\"expression\": \"2 / 2\",\n"
    "\"value\": \"1\"\n"
    "},\n"
    "{\n"
    "\"explanation\": \"Add the blue bolts and white bolts.\",\n"
    "\"expression\": \"2 + 1\",\n"
    "\"value\": \"3\"\n"
    "}\n"
    "],\n"
    "\"final_expression\": \"2 + 2 / 2\",\n"
    "\"answer\": \"3\"\n"
    "}\n\n"
    "Example 2:\n"
    "Problem: Janet\'s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She could sell all 16 eggs for $32, but instead she sells the remainder at the farmers\' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers\' market?\n\n"
    "JSON:\n"
    "{\n"
    "\"target\": \"dollars made at the farmers\' market per day\",\n"
    "\"use_facts\": [\n"
    "\"Janet\'s ducks lay 16 eggs per day.\",\n"
    "\"She eats 3 eggs for breakfast.\",\n"
    "\"She uses 4 eggs for baking.\",\n"
    "\"She sells the remaining eggs for $2 each.\"\n"
    "],\n"
    "\"exclude_facts\": [\n"
    "\"She could sell all 16 eggs for $32; this is an alternative scenario replaced by \'but instead\'.\"\n"
    "],\n"
    "\"steps\": [\n"
    "{\n"
    "\"explanation\": \"Compute the number of eggs remaining after breakfast and baking.\",\n"
    "\"expression\": \"16 - 3 - 4\",\n"
    "\"value\": \"9\"\n"
    "},\n"
    "{\n"
    "\"explanation\": \"Compute the dollars earned from selling the remaining eggs at $2 each.\",\n"
    "\"expression\": \"9 * 2\",\n"
    "\"value\": \"18\"\n"
    "}\n"
    "],\n"
    "\"final_expression\": \"(16 - 3 - 4) * 2\",\n"
    "\"answer\": \"18\"\n"
    "}\n\n"
    "Example 3:\n"
    "Problem: Josh buys a house for $80,000 and puts in $50,000 in repairs. This increased the value of the house by 150%. After selling, he donates 10% of his profit to charity. How much profit did he make from the flip?\n\n"
    "JSON:\n"
    "{\n"
    "\"target\": \"profit in dollars\",\n"
    "\"use_facts\": [\n"
    "\"The original house price is $80,000.\",\n"
    "\"The repair cost is $50,000.\",\n"
    "\"The house value increased by 150%, meaning the value becomes 2.5 times the original price.\"\n"
    "],\n"
    "\"exclude_facts\": [\n"
    "\"The 10% donation is excluded because the question asks for profit, not donation.\"\n"
    "],\n"
    "\"steps\": [\n"
    "{\n"
    "\"explanation\": \"Compute the selling value after a 150% increase.\",\n"
    "\"expression\": \"80000 * 2.5\",\n"
    "\"value\": \"200000\"\n"
    "},\n"
    "{\n"
    "\"explanation\": \"Compute profit by subtracting purchase cost and repair cost from selling value.\",\n"
    "\"expression\": \"200000 - 80000 - 50000\",\n"
    "\"value\": \"70000\"\n"
    "}\n"
    "],\n"
    "\"final_expression\": \"80000 * 2.5 - 80000 - 50000\",\n"
    "\"answer\": \"70000\"\n"
    "}\n\n"
    "Now solve the following problem.\n\n"
    "Problem:\n{question}\n"
)



# 2026-06-19: v13_long_prompt v1 (5 字段, 3 rules)
# 砍 V12 的 use_facts / answer 字段, 保留 exclude_facts (给 decoy 显式位置)
# 砍 V12 的 12 rules 到 3 条 (R1 有用量 / R2 percent / R3 path switch)
# Prompt: V12 2256 tokens → V13 1328 tokens (-41%)
COT_BRACKETS_V13_JSON_SYSTEM_PROMPT = (
    "You are a careful grade-school math reasoning assistant.\\n"
    "Your task is to solve the problem correctly and return only one valid JSON object.\\n"
    "Reasoning style is not judged; correctness, target selection, step validity, and exclusion of irrelevant facts are judged.\\n"
    "Output rules:\\n"
    "Output valid JSON only.\\n"
    "Do not use markdown.\\n"
    "Do not write any text before or after the JSON.\\n"
    "Do not include comments.\\n"
    "Do not include trailing commas.\\n"
    "All arithmetic derivations used for the final answer must appear in the \\\"steps\\\" array.\\n"
    "Each step must contain exactly three fields:\\n"
    "\\\"explanation\\\": a short natural-language explanation of the quantity being computed.\\n"
    "\\\"expression\\\": the arithmetic expression.\\n"
    "\\\"value\\\": the evaluated result.\\n"
    "The final answer must be represented by \\\"final_expression\\\" and \\\"final_answer\\\".\\n"
    "Do not include irrelevant, hypothetical, optional, alternative-path, or separate-scope calculations in \\\"steps\\\".\\n"
    "Decoy facts (not needed to answer the target) must be listed in \\\"exclude_facts\\\" with a brief reason.\\n"
    "Required JSON schema:\\n"
    "{\\n"
    "\\\"target\\\": \\\"\\\",\\n"
    "\\\"steps\\\": [\\n"
    "{\\n"
    "\\\"explanation\\\": \\\"\\\",\\n"
    "\\\"expression\\\": \\\"\\\",\\n"
    "\\\"value\\\": \\\"\\\"\\n"
    "}\\n"
    "],\\n"
    "\\\"exclude_facts\\\": [\\n"
    "\\\"\\\"\\n"
    "],\\n"
    "\\\"final_expression\\\": \\\"\\\",\\n"
    "\\\"final_answer\\\": 0\\n"
    "}\\n"
    "Rules:\\n"
    "R1. Every step must introduce or combine a useful quantity. Do not add number-padding steps.\\n"
    "R2. \\\"increased by X%\\\" means multiply by 1 + X/100. Identify whether a percentage means \\\"X% of Y\\\", \\\"increased by X%\\\", or \\\"decreased by X%\\\" before computing.\\n"
    "R3. \\\"but instead\\\", \\\"instead\\\", \\\"however\\\", and \\\"after that\\\" usually indicate a path switch. Earlier alternative scenarios are excluded unless the question asks about them.\\n"
)

# 2026-06-19: v14_long_prompt v1 (4 字段, 12 rules, prose 表达式 reasoning)
# 砍 V13 的 steps 数组, 改 prose/表达式 行交错放在 reasoning 数组
# 末行 = 最终数值
COT_BRACKETS_V14_REASONING_SYSTEM_PROMPT = (
    "You are a careful grade-school math reasoning assistant.\n"
    "Your task is to solve the problem correctly and return only one valid JSON object.\n"
    "Correctness, target selection, step alignment, and exclusion of irrelevant facts are judged. Reasoning style is not judged.\n\n"
    "Output rules:\n"
    "Output valid JSON only.\n"
    "Do not use markdown.\n"
    "Do not write any text before or after the JSON.\n"
    "Do not include comments.\n"
    "Do not include trailing commas.\n"
    "The \"reasoning\" array is the only place where you record your derivation. Every calculation used for the final answer must appear as a math-expression line in \"reasoning\".\n"
    "Reasoning must physically interleave natural-language COT lines and math-expression lines.\n"
    "The very last element of \"reasoning\" must be the final numeric answer.\n"
    "Decoy facts (not needed to answer the target) must be listed in \"exclude_facts\" with a brief reason. Do not use them in \"reasoning\".\n\n"
    "Required JSON schema:\n"
    "{\n"
    "\"target\": \"\",\n"
    "\"use_facts\": [\n"
    "\"\"\n"
    "],\n"
    "\"exclude_facts\": [\n"
    "\"\"\n"
    "],\n"
    "\"reasoning\": [\n"
    "\"<natural-language COT line>\",\n"
    "\"<math expression line>\",\n"
    "\"...\",\n"
    "\"<final numeric value line>\"\n"
    "]\n"
    "}\n\n"
    "Rules:\n"
    "R1. Every reasoning line must introduce or combine a useful quantity. Do not add number-padding lines.\n"
    "R2. \"increased by X%\" means multiply by 1 + X/100. Identify whether a percentage means \"X% of Y\", \"increased by X%\", \"decreased by X%\", or \"X% of the remaining\" before computing.\n"
    "R3. \"but instead\", \"instead\", \"however\", and \"after that\" usually indicate a path switch. Earlier alternative scenarios are excluded unless the question asks about them.\n"
    "R4. If a fact is excluded in \"exclude_facts\", it must not appear as a math-expression in \"reasoning\".\n"
    "R5. The target must match the question exactly (quantity and unit).\n"
    "R6. A decoy is a fact not needed to answer the target. Put it in \"exclude_facts\"; do not compute it in \"reasoning\".\n"
    "R7. In multiplication or division steps, units must match the quantity being computed.\n"
    "R8. \"How many X\" must not mix units such as counts, weights, prices, or rates in the same summed expression.\n"
    "R9. Do not write self-cancelling expressions such as \"A / X * X\" or \"A + B - B\" unless explicitly required.\n"
    "R10. Reasoning must end with a numeric value; that value is the final answer.\n"
    "R11. Math-expression lines contain numbers, operators, parentheses, and the equals sign. They do not contain prose.\n"
    "R12. At least one natural-language COT line must appear in \"reasoning\" (do not skip prose entirely).\n"
)


COT_BRACKETS_V14_REASONING_USER_TEMPLATE = (
    "Solve the following grade-school math problem.\n\n"
    "Return only one valid JSON object following this schema:\n\n"
    "{\n"
    "\"target\": \"<quantity and unit>\",\n"
    "\"use_facts\": [\n"
    "\"<fact used>\"\n"
    "],\n"
    "\"exclude_facts\": [\n"
    "\"<fact excluded + reason>\"\n"
    "],\n"
    "\"reasoning\": [\n"
    "\"<prose line>\",\n"
    "\"<math expression line>\",\n"
    "\"...\",\n"
    "\"<final numeric value>\"\n"
    "]\n"
    "}\n\n"
    "Important:\n"
    "* Output JSON only.\n"
    "* Do not use markdown.\n"
    "* Do not write explanations outside the JSON.\n"
    "* Every calculation used for the final answer must appear as a math-expression line in \"reasoning\".\n"
    "* Decoy / alternative-path / separate-scope facts must appear in \"exclude_facts\", never in \"reasoning\".\n"
    "* The last element of \"reasoning\" is the final answer (a number).\n\n"
    "Examples:\n\n"
    "Example 1 (attribute_mismatch, decoy exclusion):\n"
    "Problem: A robe takes 2 bolts of blue fiber and half that much white fiber. Each bolt of blue fiber weighs 3 pounds and each bolt of white fiber weighs 2 pounds. How many bolts in total does it take?\n\n"
    "JSON:\n"
    "{\n"
    "\"target\": \"total number of bolts\",\n"
    "\"use_facts\": [\n"
    "\"The robe takes 2 bolts of blue fiber.\",\n"
    "\"The robe takes half as many white bolts as blue.\"\n"
    "],\n"
    "\"exclude_facts\": [\n"
    "\"Each blue bolt weighs 3 pounds; the question asks for bolts, not weight.\",\n"
    "\"Each white bolt weighs 2 pounds; the question asks for bolts, not weight.\"\n"
    "],\n"
    "\"reasoning\": [\n"
    "\"We need the total number of bolts, not their weight, so bolt weight facts are decoys.\",\n"
    "\"blue_bolts = 2\",\n"
    "\"The white bolts count is half the blue bolts count.\",\n"
    "\"white_bolts = blue_bolts / 2 = 2 / 2 = 1\",\n"
    "\"Now add the two counts to get the total.\",\n"
    "\"total_bolts = blue_bolts + white_bolts = 2 + 1\",\n"
    "\"total_bolts = 3\"\n"
    "]\n"
    "}\n\n"
    "Example 2 (path_competition, but-instead switch):\n"
    "Problem: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She could sell all 16 eggs for $32, but instead she sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\n\n"
    "JSON:\n"
    "{\n"
    "\"target\": \"dollars made at the farmers' market per day\",\n"
    "\"use_facts\": [\n"
    "\"Janet's ducks lay 16 eggs per day.\",\n"
    "\"She eats 3 eggs for breakfast.\",\n"
    "\"She uses 4 eggs for muffins.\",\n"
    "\"She sells the remaining eggs at $2 each.\"\n"
    "],\n"
    "\"exclude_facts\": [\n"
    "\"She could sell all 16 eggs for $32; this is an alternative scenario replaced by 'but instead'.\"\n"
    "],\n"
    "\"reasoning\": [\n"
    "\"'but instead' switches the path; the $32 alternative is excluded.\",\n"
    "\"remaining_eggs = 16 - 3 - 4\",\n"
    "\"remaining_eggs = 9\",\n"
    "\"Multiply remaining eggs by $2 each.\",\n"
    "\"dollars_per_day = remaining_eggs * 2 = 9 * 2\",\n"
    "\"dollars_per_day = 18\"\n"
    "]\n"
    "}\n\n"
    "Now solve the following problem.\n\n"
    "Problem:\n{question}\n"
)


COT_BRACKETS_V13_JSON_USER_TEMPLATE = (
    "Solve the following grade-school math problem.\\n\\n"
    "Return only one valid JSON object following this schema:\\n\\n"
    "{\\n"
    "\\\"target\\\": \\\"<quantity and unit being asked>\\\",\\n"
    "\\\"steps\\\": [\\n"
    "{\\n"
    "\\\"explanation\\\": \\\"<what this step computes>\\\",\\n"
    "\\\"expression\\\": \\\"<arithmetic>\\\",\\n"
    "\\\"value\\\": \\\"<result>\\\"\\n"
    "}\\n"
    "],\\n"
    "\\\"exclude_facts\\\": [\\n"
    "\\\"<fact excluded and brief reason>\\\"\\n"
    "],\\n"
    "\\\"final_expression\\\": \\\"<single expression for the final answer>\\\",\\n"
    "\\\"final_answer\\\": <number>\\n"
    "}\\n\\n"
    "Important:\\n\\n"
    "* Output JSON only.\\n"
    "* Do not use markdown.\\n"
    "* Do not write explanations outside the JSON.\\n"
    "* Every calculation used for the final answer must appear in \\\"steps\\\".\\n"
    "* Decoy facts (not needed to answer the target) must appear only in \\\"exclude_facts\\\", not in \\\"steps\\\".\\n"
    "* \\\"final_expression\\\" must compute the final answer directly.\\n"
    "* \\\"final_answer\\\" must be the numeric value of \\\"final_expression\\\".\\n\\n"
    "Examples:\\n\\n"
    "Example 1 (simple arithmetic):\\n"
    "Problem: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for muffins. She sells the rest at $2 each. How much does she make per day at the farmers\' market?\\n\\n"
    "JSON:\\n"
    "{\\n"
    "\\\"target\\\": \\\"dollars per day at the farmers\\' market\\\",\\n"
    "\\\"steps\\\": [\\n"
    "{\\n"
    "\\\"explanation\\\": \\\"Compute remaining eggs after breakfast and muffins.\\\",\\n"
    "\\\"expression\\\": \\\"16 - 3 - 4\\\",\\n"
    "\\\"value\\\": 9\\n"
    "},\\n"
    "{\\n"
    "\\\"explanation\\\": \\\"Compute dollars from selling the remaining eggs at $2 each.\\\",\\n"
    "\\\"expression\\\": \\\"9 * 2\\\",\\n"
    "\\\"value\\\": 18\\n"
    "}\\n"
    "],\\n"
    "\\\"exclude_facts\\\": [],\\n"
    "\\\"final_expression\\\": \\\"(16 - 3 - 4) * 2\\\",\\n"
    "\\\"final_answer\\\": 18\\n"
    "}\\n\\n"
    "Example 2 (percent):\\n"
    "Problem: Josh buys a house for $80,000 and spends $50,000 on repairs. The house value increases by 150%. What is the profit?\\n\\n"
    "JSON:\\n"
    "{\\n"
    "\\\"target\\\": \\\"profit in dollars\\\",\\n"
    "\\\"steps\\\": [\\n"
    "{\\n"
    "\\\"explanation\\\": \\\"Compute new value after 150% increase.\\\",\\n"
    "\\\"expression\\\": \\\"80000 * (1 + 150/100)\\\",\\n"
    "\\\"value\\\": 200000\\n"
    "},\\n"
    "{\\n"
    "\\\"explanation\\\": \\\"Subtract purchase and repair costs.\\\",\\n"
    "\\\"expression\\\": \\\"200000 - 80000 - 50000\\\",\\n"
    "\\\"value\\\": 70000\\n"
    "}\\n"
    "],\\n"
    "\\\"exclude_facts\\\": [],\\n"
    "\\\"final_expression\\\": \\\"80000 * (1 + 150/100) - 80000 - 50000\\\",\\n"
    "\\\"final_answer\\\": 70000\\n"
    "}\\n\\n"
    "Example 3 (path switch + decoy exclusion):\\n"
    "Problem: A robe takes 2 bolts of blue fiber and half that much white fiber. Each bolt of blue fiber weighs 3 pounds and each bolt of white fiber weighs 2 pounds. How many bolts in total does it take?\\n\\n"
    "JSON:\\n"
    "{\\n"
    "\\\"target\\\": \\\"total number of bolts\\\",\\n"
    "\\\"steps\\\": [\\n"
    "{\\n"
    "\\\"explanation\\\": \\\"Compute the number of white bolts as half the number of blue bolts.\\\",\\n"
    "\\\"expression\\\": \\\"2 / 2\\\",\\n"
    "\\\"value\\\": 1\\n"
    "},\\n"
    "{\\n"
    "\\\"explanation\\\": \\\"Add the blue bolts and white bolts.\\\",\\n"
    "\\\"expression\\\": \\\"2 + 1\\\",\\n"
    "\\\"value\\\": 3\\n"
    "}\\n"
    "],\\n"
    "\\\"exclude_facts\\\": [\\n"
    "\\\"Each blue bolt weighs 3 pounds; the question asks for bolts, not weight.\\\",\\n"
    "\\\"Each white bolt weighs 2 pounds; the question asks for bolts, not weight.\\\"\\n"
    "],\\n"
    "\\\"final_expression\\\": \\\"2 + 2 / 2\\\",\\n"
    "\\\"final_answer\\\": 3\\n"
    "}\\n\\n"
    "Problem:\\n{question}\\n"
)


def load_examples(data_path: str | Path, limit: int | None = None) -> list[EvalExample]:
    examples = []
    with Path(data_path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            category = row["category"]
            question_key = "question_original" if category == "original" else "question_distracted"
            examples.append(
                EvalExample(
                    id=row["id"],
                    base_id=row.get("base_id", ""),
                    category=category,
                    question=row[question_key],
                    gold_answer=str(row["answer"]),
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    if not examples:
        raise ValueError(f"No eval examples loaded from {data_path}")
    return examples



# --- 2026-06-21: parquet_prompt 模式的 prompt / 答案抽取路由 ---
# _USER_TEMPLATE_VERSION 由 main() 在解析 --parquet-dir 时设置:
#   /.../grpo_v12_json.parquet        -> "v12"  (default: 含 V12 USER_TEMPLATE)
#   /.../grpo_v13_json.parquet        -> "v13"
#   /.../grpo_v14_reasoning.parquet   -> "v14"
#   其它                                -> "v12" (兜底)
_USER_TEMPLATE_VERSION = "v12"
_SYSTEM_PROMPT_CACHE: str | None = None


def _detect_parquet_version(parquet_dir: str | Path) -> str:
    """从 --parquet-dir 路径猜是 V12 / V13 / V14, 决定 USER_TEMPLATE + answer 抽取器."""
    p = Path(parquet_dir)
    name = p.name.lower()
    if "v14" in name:
        return "v14"
    if "v13" in name:
        return "v13"
    if "v12" in name:
        return "v12"
    return "v12"


def _load_training_system_prompt() -> str:
    """读 build_grpo_vXX_*.py 写出的 _system_prompt.txt (与 grpo_vXX_*.parquet 同目录)."""
    global _SYSTEM_PROMPT_CACHE
    if _SYSTEM_PROMPT_CACHE is not None:
        return _SYSTEM_PROMPT_CACHE
    candidates = []
    if _PARQUET_DIR_FOR_PROMPT is not None:
        candidates.append(Path(_PARQUET_DIR_FOR_PROMPT) / "_system_prompt.txt")
    # 兜底: 当前工作目录
    candidates.append(Path.cwd() / "_system_prompt.txt")
    for path in candidates:
        if path.is_file():
            _SYSTEM_PROMPT_CACHE = path.read_text(encoding="utf-8")
            print(f"[eval] loaded training system prompt from {path} ({len(_SYSTEM_PROMPT_CACHE)} chars)", flush=True)
            return _SYSTEM_PROMPT_CACHE
    raise FileNotFoundError(
        f"parquet_prompt mode 需要 _system_prompt.txt, 在以下位置都没找到: "
        f"{[str(p) for p in candidates]}. "
        f"请先跑 build_grpo_vXX_*.py 写出 SYSTEM 快照, 或显式 --parquet-dir 指向 parquet 所在目录."
    )


def _select_user_template_for_parquet_prompt() -> str:
    """按 _USER_TEMPLATE_VERSION 选 USER_TEMPLATE. USER 段不带 few-shot examples."""
    if _USER_TEMPLATE_VERSION == "v14":
        return COT_BRACKETS_V14_REASONING_USER_TEMPLATE
    if _USER_TEMPLATE_VERSION == "v13":
        return COT_BRACKETS_V13_JSON_USER_TEMPLATE
    return COT_BRACKETS_V12_JSON_USER_TEMPLATE


# 全局变量, main() 通过 --parquet-dir 设置
_PARQUET_DIR_FOR_PROMPT: str | None = None


def build_messages(method: str, question: str) -> list[dict[str, str]]:
    if method == "train_json_prompt":
        # 2026-06-08: use the neutral prompt so eval-time prompt matches
        # the SFT/GRPO training-time prompt (no distractor / ignore / "JSON only" terms).
        return [
            {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
            {"role": "user", "content": NEUTRAL_USER_TEMPLATE.replace("{question}", question.strip())},
        ]
    if method == "direct":
        return [
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Q: {question}\nA:"},
        ]
    if method == "zero_shot_cot":
        return [
            {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Q: {question}\nA: Let's think step by step."},
        ]
    if method == "cot_brackets":
        # 2026-06-17: 与 SFT 训练时同源 prompt, 不动。
        return [
            {"role": "system", "content": COT_BRACKETS_SYSTEM_PROMPT},
            {"role": "user", "content": COT_BRACKETS_USER_TEMPLATE.replace("{question}", question.strip())},
        ]
    if method == "cot_brackets_v12_json":
        # 2026-06-19: v12_long_prompt v4, 纯 JSON 输出
        return [
            {"role": "system", "content": COT_BRACKETS_V12_JSON_SYSTEM_PROMPT},
            {"role": "user", "content": COT_BRACKETS_V12_JSON_USER_TEMPLATE.replace("{question}", question.strip())},
        ]
    if method == "cot_brackets_v13_json":
        # 2026-06-19: v13_long_prompt v1, 5 字段 JSON
        return [
            {"role": "system", "content": COT_BRACKETS_V13_JSON_SYSTEM_PROMPT},
            {"role": "user", "content": COT_BRACKETS_V13_JSON_USER_TEMPLATE.replace("{question}", question.strip())},
        ]
    if method == "cot_brackets_v14_reasoning":
        # 2026-06-19: v14_long_prompt v1, 4 字段 + prose 表达式 reasoning
        return [
            {"role": "system", "content": COT_BRACKETS_V14_REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": COT_BRACKETS_V14_REASONING_USER_TEMPLATE.replace("{question}", question.strip())},
        ]
    if method == "parquet_prompt":
        # 2026-06-21: 评测时复用训练时的 SYSTEM prompt (避免 train-test prompt mismatch).
        # 来源: build_grpo_vXX_*.py 写出的 _system_prompt.txt, 路径从 --parquet-dir 拿.
        # USER 段: 按 _USER_TEMPLATE_VERSION 选 V12/V13/V14 USER_TEMPLATE (与训练 schema 对齐).
        system_prompt = _load_training_system_prompt()
        user_template = _select_user_template_for_parquet_prompt()
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_template.replace("{question}", question.strip())},
        ]
    raise ValueError(f"Unsupported train-time eval method: {method}")


def fallback_chat_prompt(messages: list[dict[str, str]]) -> str:
    pieces = [f"{message['role'].capitalize()}: {message['content']}" for message in messages]
    pieces.append("Assistant:")
    return "\n".join(pieces)


def build_prompt(tokenizer: Any, method: str, question: str) -> str:
    messages = build_messages(method, question)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return fallback_chat_prompt(messages)


def extract_answer_with_json_v12(output: str) -> str | None:
    """2026-06-19: v12_long_prompt v4 JSON 抽取 (修复 brace counting 版)。

    关键修复: 用 brace counting 找最外层 { ... }, 替代非贪婪 {.*?}
    (非贪婪在 steps 数组里会抓到 step 内的 {explanation, expression, value} 子结构)

    优先级:
      1) brace counting 找最外层 JSON
         - dict["answer"] 优先
         - dict["final_expression"] 兜底
      2) <<FINAL: expr = N>> 老协议
      3) ANSWER: N 老协议
      4) 全文 brace counting 抓 answer
      5) gsm_answer_extractor.extract_answer 兜底 (抓 last number 等)
    """
    text = str(output or "").strip()
    if not text:
        return extract_answer(output)
    # 1) brace counting 找最外层 {...}
    s = text.find("{")
    if s >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(s, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == "\\":
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
                        parsed = json.loads(text[s:i + 1])
                        if isinstance(parsed, dict):
                            if parsed.get("answer") is not None:
                                return extract_text_answer(str(parsed["answer"]))
                            if parsed.get("final_expression"):
                                return extract_text_answer(str(parsed["final_expression"]))
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break
    # 2) 回退到老协议 (<<FINAL:>> / ANSWER: / 全文 / last number)
    return extract_answer(output)


def extract_answer_with_json_v14(output: str) -> str | None:
    """2026-06-19: v14_long_prompt v1 JSON 抽取 (4 字段, reasoning 末行, brace counting)。

    关键差异 vs v13:
    - 4 字段 (target / use_facts / exclude_facts / reasoning)
    - 无 final_expression / final_answer 字段, 末行 = 最终数值
    - reasoning 末行直接拿数字
    """
    import re as _re
    text = str(output or "").strip()
    if not text:
        return extract_answer(output)
    s = text.find("{")
    if s >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(s, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == "\\":
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
                        obj = json.loads(text[s:i + 1])
                    except json.JSONDecodeError:
                        return extract_answer(output)
                    # 1) reasoning 末行
                    reasoning = obj.get("reasoning")
                    if isinstance(reasoning, list) and reasoning:
                        last = reasoning[-1]
                        if isinstance(last, (int, float)):
                            return str(last)
                        if isinstance(last, str):
                            m = _re.search(r"[=]\\s*([-+]?\\d+(?:\\.\\d+)?)\\s*$", last.strip())
                            if m:
                                return m.group(1)
                            try:
                                return str(float(last.strip()))
                            except (ValueError, TypeError):
                                pass
                    return extract_answer(output)
    return extract_answer(output)


def extract_answer_with_json_v13(output: str) -> str | None:
    """2026-06-19: v13_long_prompt v1 JSON 抽取 (5 字段, brace counting)。

    关键差异 vs v12:
    - 5 字段 (target / steps / exclude_facts / final_expression / final_answer)
    - 优先 final_answer 字段 (V13 取代 V12 的 answer 字段)
    - fallback final_expression
    - 仍走 brace counting, 不再信 <<FINAL:>> 老协议
    """
    text = str(output or "").strip()
    if not text:
        return extract_answer(output)
    s = text.find("{")
    if s >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(s, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == "\\":
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
                        parsed = json.loads(text[s:i + 1])
                        if isinstance(parsed, dict):
                            if parsed.get("final_answer") is not None:
                                return extract_text_answer(str(parsed["final_answer"]))
                            if parsed.get("final_expression"):
                                return extract_text_answer(str(parsed["final_expression"]))
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break
    return extract_answer(output)



def extract_answer(output: str) -> str | None:
    text = str(output or "").strip()
    # 2026-06-17: 新协议 <<FINAL: expr = N>> 优先反扫 (与 SFT 训练时同协议)。
    final_match = re.search(r"<<\s*FINAL\s*:[^>]{0,500}=\s*([^\n>]+?)\s*>>", text)
    if final_match:
        return extract_text_answer(final_match.group(1).strip())
    # 第二级: ANSWER: N (与 SFT 训练时同协议)。
    ans_match = re.search(r"^\s*ANSWER\s*:\s*([^\n]+?)\s*$", text, re.MULTILINE | re.IGNORECASE)
    if ans_match:
        return extract_text_answer(ans_match.group(1).strip())
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and parsed.get("answer") is not None:
            return extract_text_answer(str(parsed["answer"]))
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict) and parsed.get("answer") is not None:
                    return extract_text_answer(str(parsed["answer"]))
            except json.JSONDecodeError:
                pass
    return extract_text_answer(output)


def chunked(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cleanup_vllm() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
    except Exception:
        pass


def available_gpu_memory_ratio(safety_margin: float = 0.03) -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if total_bytes <= 0:
            return None
        ratio = (free_bytes / total_bytes) - safety_margin
        return max(ratio, 0.0)
    except Exception:
        return None


def filter_gpu_memory_candidates(
    candidates: list[float],
    min_candidate: float = 0.25,
    safety_margin: float = 0.03,
) -> list[float]:
    max_ratio = available_gpu_memory_ratio(safety_margin=safety_margin)
    if max_ratio is None:
        return candidates
    filtered = [candidate for candidate in candidates if candidate <= max_ratio]
    fallback = max(min(max_ratio, max(candidates)), min_candidate)
    if not filtered:
        filtered = [fallback]
    elif fallback not in filtered and fallback < max(filtered):
        filtered.append(fallback)
    filtered = sorted(set(round(x, 3) for x in filtered), reverse=True)
    print(
        f"[eval] GPU memory precheck: usable_ratio~{max_ratio:.3f}, "
        f"candidates={filtered}",
        flush=True,
    )
    return filtered


def summarize(predictions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})
    overall = {"correct": 0, "total": 0}
    for row in predictions:
        category = row["category"]
        by_category[category]["correct"] += int(row["correct"])
        by_category[category]["total"] += 1
        overall["correct"] += int(row["correct"])
        overall["total"] += 1

    category_rows = []
    for category, stats in sorted(by_category.items()):
        total = stats["total"]
        correct = stats["correct"]
        category_rows.append(
            {
                "category": category,
                "correct": correct,
                "total": total,
                "accuracy": correct / total if total else 0.0,
            }
        )

    overall_rows = [
        {
            "correct": overall["correct"],
            "total": overall["total"],
            "accuracy": overall["correct"] / overall["total"] if overall["total"] else 0.0,
        }
    ]
    return category_rows, overall_rows


def evaluate_with_vllm(
    model_path: str | Path,
    data_path: str | Path = DEFAULT_TEST_DATA,
    output_dir: str | Path = "eval",
    method: str = "train_json_prompt",
    limit: int | None = None,
    batch_size: int = 64,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    gpu_memory_utilization_candidates: list[float] | None = None,
    dtype: str = "auto",
    seed: int = 42,
    max_model_len: int | None = None,
    max_tokens: int = 2048,
    top_k: int = 1,
    top_p: float = 1.0,
    trust_remote_code: bool = True,
) -> dict[str, Any]:
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = load_examples(data_path, limit)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)
    sampling_params = SamplingParams(temperature=0.0, top_k=top_k, top_p=top_p, max_tokens=max_tokens)
    candidates = gpu_memory_utilization_candidates or [gpu_memory_utilization]
    seen = set()
    candidates = [float(x) for x in candidates if not (float(x) in seen or seen.add(float(x)))]
    candidates = filter_gpu_memory_candidates(candidates)

    predictions: list[dict[str, Any]] = []
    prompts = [build_prompt(tokenizer, method, example.question) for example in examples]
    used_gpu_memory_utilization = None
    attempt_errors = []
    predictions_path = output_dir / "predictions.jsonl"
    # 清空旧的 predictions 文件
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text("")
    try:
        for candidate in candidates:
            predictions = []
            predictions_path.write_text("")
            llm = None
            try:
                llm_kwargs: dict[str, Any] = {
                    "model": str(model_path),
                    "trust_remote_code": trust_remote_code,
                    "tensor_parallel_size": tensor_parallel_size,
                    "gpu_memory_utilization": candidate,
                    "dtype": dtype,
                    "seed": seed,
                }
                if max_model_len is not None:
                    llm_kwargs["max_model_len"] = max_model_len
                print(f"[eval] Loading vLLM with gpu_memory_utilization={candidate}", flush=True)
                llm = LLM(**llm_kwargs)
                for prompt_batch, example_batch in zip(chunked(prompts, batch_size), chunked(examples, batch_size)):
                    outputs = llm.generate(prompt_batch, sampling_params, use_tqdm=False)
                    for example, output in zip(example_batch, outputs):
                        raw_output = output.outputs[0].text if output.outputs else ""
                        if method == "cot_brackets_v13_json" or _USER_TEMPLATE_VERSION == "v13":
                            pred_answer = extract_answer_with_json_v13(raw_output)
                        elif method == "cot_brackets_v14_reasoning" or _USER_TEMPLATE_VERSION == "v14":
                            pred_answer = extract_answer_with_json_v14(raw_output)
                        else:
                            pred_answer = extract_answer_with_json_v12(raw_output)
                        correct = is_correct(pred_answer, example.gold_answer)
                        row = {
                            "id": example.id,
                            "base_id": example.base_id,
                            "category": example.category,
                            "question": example.question,
                            "gold_answer": example.gold_answer,
                            "raw_output": raw_output,
                            "pred_answer": pred_answer,
                            "correct": correct,
                        }
                        predictions.append(row)
                        # 实时写入 predictions.jsonl
                        append_jsonl(predictions_path, row)
                used_gpu_memory_utilization = candidate
                break
            except Exception as exc:
                import traceback as _tb; attempt_errors.append({"gpu_memory_utilization": candidate, "error": repr(exc), "traceback": _tb.format_exc()})
                print(f"[eval] vLLM eval failed at gpu_memory_utilization={candidate}: {exc!r}", flush=True)
            finally:
                if llm is not None:
                    del llm
                cleanup_vllm()
                time.sleep(2)
        if used_gpu_memory_utilization is None:
            with (output_dir / "eval_errors.json").open("w", encoding="utf-8") as f:
                json.dump(attempt_errors, f, indent=2, ensure_ascii=False)
            raise RuntimeError(f"vLLM eval failed for all gpu_memory_utilization candidates: {attempt_errors}")
    finally:
        del tokenizer
        cleanup_vllm()

    category_rows, overall_rows = summarize(predictions)
    write_jsonl(
        output_dir / "summary_by_category.jsonl",
        category_rows,
    )
    write_jsonl(
        output_dir / "summary_overall.jsonl",
        overall_rows,
    )
    result = {
        "overall_accuracy": overall_rows[0]["accuracy"],
        "overall": overall_rows,
        "by_category": category_rows,
        "prediction_count": len(predictions),
        "output_dir": str(output_dir),
        "gpu_memory_utilization": used_gpu_memory_utilization,
        "attempt_errors": attempt_errors,
    }
    # 写入 eval_result.json，供子进程调用时传递结果
    with (output_dir / "eval_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one ChainGSM model checkpoint with vLLM.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", default=DEFAULT_TEST_DATA)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", default="train_json_prompt", choices=["train_json_prompt", "direct", "zero_shot_cot", "cot_brackets", "cot_brackets_v12_json", "cot_brackets_v13_json", "cot_brackets_v14_reasoning", "parquet_prompt"])
    parser.add_argument(
        "--parquet-dir",
        default=None,
        help="parquet_prompt 模式下指向 grpo_vXX_*.parquet 所在目录 (含 _system_prompt.txt). "
             "不传时回退到 cwd. 自动从目录名 v12/v13/v14 选 USER_TEMPLATE + 答案抽取器.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--gpu-memory-utilization-candidate",
        type=float,
        action="append",
        default=None,
        help="Retry vLLM eval with these gpu memory utilization values.",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")
    args = parser.parse_args()
    # 2026-06-21: parquet_prompt 模式设置全局
    global _PARQUET_DIR_FOR_PROMPT, _USER_TEMPLATE_VERSION
    _PARQUET_DIR_FOR_PROMPT = args.parquet_dir
    _USER_TEMPLATE_VERSION = _detect_parquet_version(args.parquet_dir) if args.parquet_dir else "v12"
    if args.method == "parquet_prompt":
        print(f"[eval] parquet_prompt mode: _USER_TEMPLATE_VERSION={_USER_TEMPLATE_VERSION}", flush=True)
    result = evaluate_with_vllm(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        method=args.method,
        limit=args.limit,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        gpu_memory_utilization_candidates=args.gpu_memory_utilization_candidate,
        dtype=args.dtype,
        seed=args.seed,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        trust_remote_code=args.trust_remote_code,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
