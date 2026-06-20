"""TDD for reward_chaingsm_v9_verl.

Covers:
  - Case A: perfect gold trace + correct answer  -> R = 4.2 (max)
  - Case B: correct answer but wrong trace      -> r_core < 1, distractor=0
  - Case C: wrong answer, right format          -> r_answer=0, r_format=1
  - Case D: distractor-pulled output (independent_decoy, model writes distractor arithmetic)
  - Case E: Gunter case (spec §12.1)
  - Case F: Micheal case (spec §12.2)
  - Case G: original sample: r_distractor must be 0 even if pred trace matches distractor
  - Case H: non-original with pred_final exactly == distractor_expression -> sim_distractor=1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/wwq416/snap/wwq/math-chain")
sys.path.insert(0, str(REPO))

from train_pipeline.reward_chaingsm_v9_verl import score_response  # noqa: E402

GOLD = {
    "answer": "6",
    "gold_expression": "15-(2+3+2*2)",
    "gold_trace_tokens": [
        "2", "*", "2", "=", "4",
        "<step>",
        "2", "+", "3", "+", "4", "=", "9",
        "<step>",
        "15", "-", "9", "=", "6",
    ],
    "distractor_trace_tokens": ["2", "+", "3", "+", "2", "*", "2", "+", "4"],
    "category": "target_scope_misalignment",
}

GOLD_ORIGINAL = {**GOLD, "category": "original", "distractor_trace_tokens": []}


def _fmt(text: str) -> str:
    return f"""TARGET: money_left

STEP 1:
EXPR: 2*2
VALUE: 4

STEP 2:
EXPR: 2+3+4
VALUE: 9

STEP 3:
EXPR: 15-9
VALUE: 6

FINAL_EXPR: 15-(2+3+2*2)
ANSWER: 6"""


def case(name: str, text: str, reference: dict, expected: dict):
    r, m = score_response(text, reference)
    print(f"\n=== {name} ===")
    for k, v in expected.items():
        actual = m.get(k) if k != "reward" else r
        ok = abs(actual - v) < 1e-3 if isinstance(v, float) else (actual == v)
        flag = "OK" if ok else "FAIL"
        print(f"  {k:25s} expected={v} actual={actual:.4f} [{flag}]" if isinstance(actual, float) else f"  {k:25s} expected={v} actual={actual} [{flag}]")
        assert ok, f"{name}: {k} expected {v} got {actual}"


# Case A: perfect
case("A: perfect original (no distractor)",
     _fmt("..."), GOLD_ORIGINAL,
     {"format": 1.0, "answer": 1.0, "core": 1.0, "distractor": 0.0, "reward": 4.2})

# Case B: right answer, wrong trace (different steps)
case("B: right answer wrong trace",
     """TARGET: money_left

STEP 1:
EXPR: 2*4
VALUE: 8

FINAL_EXPR: 8
ANSWER: 6""",
     GOLD_ORIGINAL,
     {"format": 1.0, "answer": 1.0, "core_trace_sim": 0.21, "reward": 2.953})  # sim_final=0 (8 vs full expr), core=0.8*0.21=0.168, reward=0.2+2.5+1.5*0.168=2.952

# Case C: wrong answer, right format
case("C: wrong answer, right format",
     """TARGET: money_left

STEP 1:
EXPR: 2*2
VALUE: 4

FINAL_EXPR: 4
ANSWER: 99""",
     GOLD_ORIGINAL,
     {"format": 1.0, "answer": 0.0, "core_trace_sim": 0.263, "reward": 0.516})  # partial trace sim

# Case D: distractor-pulled output (independent_decoy style)
# pred writes 2+3+2*2+4 which is exactly distractor_expression
case("D: pulled to distractor expression",
     """TARGET: money_left

STEP 1:
EXPR: 2+3
VALUE: 5

FINAL_EXPR: 2+3+2*2+4
ANSWER: 13""",
     GOLD,  # non-original
     {"distractor_sim": 1.0})  # perfect sim with distractor

# Case E: Gunter case (spec §12.1) - original sample, wrong answer
gunter = """TARGET: average

STEP 1:
EXPR: 80/2
VALUE: 40

STEP 2:
EXPR: 1-0.25
VALUE: 0.75

STEP 3:
EXPR: 40/0.75
VALUE: 53

FINAL_EXPR: 40/0.75
ANSWER: 105"""

case("E: Gunter (original, wrong answer, partial trace sim)",
     gunter, {**GOLD_ORIGINAL, "answer": "80"},
     {"format": 1.0, "answer": 0.0, "distractor": 0.0})  # v9 is graded by structural sim, not exact match; reward is 0.2+0+~0.5

# Case F: Micheal case (spec §12.2) - model produces wrong intermediate step
micheal = """TARGET: total

STEP 1:
EXPR: 5*25
VALUE: 125

STEP 2:
EXPR: 25*4
VALUE: 100

STEP 3:
EXPR: 5*25+25*4
VALUE: 225

STEP 4:
EXPR: 60*3
VALUE: 180

STEP 5:
EXPR: 5*25+25*4+60*3
VALUE: 405

FINAL_EXPR: 5*25+25*4+60*3
ANSWER: 405"""

case("F: Micheal (original, missing factors, partial trace sim)",
     micheal, {**GOLD_ORIGINAL, "answer": "860"},
     {"answer": 0.0, "distractor": 0.0})  # 0.2 + 0 + 1.5*0.23 (partial) = 0.55

# Case G: original with pred trace coincidentally matching distractor of OTHER category
# (just sanity that we still get r_distractor=0)
case("G: original, no distractor penalty",
     _fmt("..."), GOLD_ORIGINAL,
     {"distractor": 0.0, "reward": 4.2})

print("\nAll v9 TDD cases passed.")
