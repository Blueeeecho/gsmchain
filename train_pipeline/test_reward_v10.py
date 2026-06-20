"""v10 reward 单元测试 (5+ case).

跑: python train_pipeline/test_reward_v10.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_pipeline.reward_chaingsm_v10_verl import score_response


def test_case(name, text, reference, expect_reward_range, expect_keys):
    r, m = score_response(text, reference)
    ok_range = expect_reward_range[0] <= r <= expect_reward_range[1]
    ok_keys = all(k in m for k in expect_keys)
    status = "PASS" if (ok_range and ok_keys) else "FAIL"
    print(f"[{status}] {name}: reward={r:.4f} (expected {expect_reward_range})")
    if not ok_range:
        print(f"  WARN: reward out of range")
    for k in expect_keys:
        v = m.get(k, "MISSING")
        print(f"  {k}: {v}")
    return ok_range and ok_keys


passed = 0
total = 0

# Case 1: 完美 gold chain + 答对 (original, 应拿满)
total += 1
ref = {
    "answer": "5",
    "gold_expression": "120 / (8 * 3)",
    "gold_trace_tokens": ["8","*","3","=","24","<step>","120","/","(","8","*","3",")","=","5"],
    "distractor_trace_tokens": [],
    "distractor_enabled": False,
    "category": "original",
}
text = """TARGET: hours

First, compute pages per hour.
<<8 * 3 = 24>>

Then total hours.
<<120 / (8 * 3) = 5>>

<<FINAL: 120 / (8 * 3) = 5>>
ANSWER: 5
"""
if test_case("Case 1: perfect gold chain", text, ref, (3.5, 4.2),
             ["format", "answer", "core", "calc", "distractor"]):
    passed += 1

# Case 2: 答对 + 走分心链 (变体, 应被 distractor 惩罚, R 可能 < answer 单项)
total += 1
ref = {
    "answer": "72",
    "gold_expression": "48 + 48/2",
    "gold_trace_tokens": ["48","/","2","=","24","<step>","48","+","48","/","2","=","72"],
    "distractor_expression": "48 * 2",
    "distractor_trace_tokens": [],  # v10 走 expression-level
    "distractor_enabled": True,
    "category": "path_competition",
}
text = """TARGET: total
<<48 * 2 = 96>>
<<48 + 48/2 = 72>>
<<FINAL: 48 * 2 = 96>>
ANSWER: 72
"""
if test_case("Case 2: answer correct, distractor + gold both present (mild penalty)", text, ref, (2.5, 4.0),
             ["format", "answer", "core", "calc", "distractor"]):
    passed += 1

# Case 3: 答错 + 自洽的伪算式
total += 1
ref = {
    "answer": "5",
    "gold_expression": "120 / (8 * 3)",
    "gold_trace_tokens": ["8","*","3","=","24","<step>","120","/","(","8","*","3",")","=","5"],
    "distractor_trace_tokens": [],
    "distractor_enabled": False,
    "category": "original",
}
text = """TARGET: hours
<<15 - 30 * 1 = 15>>
<<FINAL: 15 - 30 * 1 = 15>>
ANSWER: 15
"""
if test_case("Case 3: 伪算式 (math incorrect, format OK)", text, ref, (-0.5, 1.5),
             ["format", "answer", "core", "calc", "distractor"]):
    passed += 1

# Case 4: 完全乱写 (应负分)
total += 1
ref = {
    "answer": "5",
    "gold_expression": "120 / (8 * 3)",
    "gold_trace_tokens": ["8","*","3","=","24","<step>","120","/","(","8","*","3",")","=","5"],
    "distractor_trace_tokens": [],
    "distractor_enabled": False,
    "category": "original",
}
text = "I don't know the answer."
if test_case("Case 4: empty / unrelated", text, ref, (-0.5, 0.5),
             ["format", "answer", "core", "calc", "distractor"]):
    passed += 1

# Case 5: 变体样本 + gold chain + 答对 (期望 R 在 +3 到 +4 区间, 走 distractor_enabled=False 因为没真用分心)
total += 1
ref = {
    "answer": "72",
    "gold_expression": "48 + 48/2",
    "gold_trace_tokens": ["48","/","2","=","24","<step>","48","+","48","/","2","=","72"],
    "distractor_expression": "48 * 2",
    "distractor_trace_tokens": [],
    "distractor_enabled": True,
    "category": "path_competition",
}
text = """TARGET: total
<<48 / 2 = 24>>
<<48 + 48/2 = 72>>
<<FINAL: 48 + 48/2 = 72>>
ANSWER: 72
"""
if test_case("Case 5: 变体 + 走 gold chain + 答对", text, ref, (3.5, 4.2),
             ["format", "answer", "core", "calc", "distractor"]):
    passed += 1

# Case 6: 等价 distractor (distractor_enabled=False, 应不扣分)
total += 1
ref = {
    "answer": "10",
    "gold_expression": "100 / 10",
    "gold_trace_tokens": ["100","/","10","=","10"],
    "distractor_expression": "100 / 10",  # 等价
    "distractor_trace_tokens": [],
    "distractor_enabled": False,  # 已禁用
    "category": "path_competition",
}
text = """TARGET: answer
<<100 / 10 = 10>>
<<FINAL: 100 / 10 = 10>>
ANSWER: 10
"""
if test_case("Case 6: 等价 distractor (禁用)", text, ref, (3.5, 4.2),
             ["format", "answer", "core", "calc", "distractor"]):
    passed += 1

print(f"\n=== {passed}/{total} PASS ===")
sys.exit(0 if passed == total else 1)
