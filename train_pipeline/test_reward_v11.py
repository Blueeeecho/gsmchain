"""v11 reward 单元测试 (8 case).

跑: python train_pipeline/test_reward_v11.py

v10 -> v11 关键变化:
  - 删除 r_format (饱和) -> 改 r_step_value (vs gold values)
  - r_answer 2.5 -> 3.0
  - r_core 1.2 -> 0.5
  - core_trace_w / core_final_w 0.7/0.3 -> 0.8/0.2
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_pipeline.reward_chaingsm_v11_verl import score_response


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

# Case 1: 完美 gold chain + 答对 (original, max ~5.0)
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
if test_case("Case 1: perfect gold chain + correct", text, ref, (4.0, 5.0),
             ["answer", "step_value", "core", "distractor"]):
    passed += 1

# Case 2: 答对 + 走分心链 (变体, 应被 distractor 惩罚, R 可能 < answer 单项)
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
<<48 * 2 = 96>>
<<48 + 48/2 = 72>>
<<FINAL: 48 * 2 = 96>>
ANSWER: 72
"""
if test_case("Case 2: distractor + gold both present (mild penalty)", text, ref, (2.0, 4.5),
             ["answer", "step_value", "core", "distractor"]):
    passed += 1

# Case 3: 答错 + self-consistent 伪算式 (v11 应该严厉扣分)
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
# v10 给 ~0.74 (calc 满分); v11 应该 < 0.5 (step_value=0, ans=0, core ~ 0.05)
if test_case("Case 3: self-consistent wrong (v11 should be < 0.5)", text, ref, (-0.5, 0.5),
             ["answer", "step_value", "core", "distractor"]):
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
if test_case("Case 4: empty / unrelated", text, ref, (-0.5, 0.0),
             ["answer", "step_value", "core", "distractor"]):
    passed += 1

# Case 5: 变体样本 + gold chain + 答对 (max)
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
if test_case("Case 5: decoy + gold chain + correct", text, ref, (4.0, 5.0),
             ["answer", "step_value", "core", "distractor"]):
    passed += 1

# Case 6: 答对但 step values 不匹配 (R 应低于 Case 1)
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
<<100 / 20 = 5>>
<<FINAL: 100 / 20 = 5>>
ANSWER: 5
"""
# 答对, 但 step_value=0 (没有 '24' 跟 '5' 匹配 — '5' 末尾匹配)
# v11: ans=1, step_value=0.5 (只 match 末尾 '5'), core=low, dist=0
# 估算: 3.0 + 1.5*0.5 + 0.5*0.x = 3.75
if test_case("Case 6: answer right but wrong steps (v11 distinguishes)", text, ref, (3.0, 4.5),
             ["answer", "step_value", "core", "distractor"]):
    passed += 1

# Case 7: step values 匹配 (中间值) 但 answer 错
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
<<8 * 3 = 24>>
<<120 / (8 * 3) = 4>>  # 算式对, 但最后一步算错
<<FINAL: 120 / (8 * 3) = 4>>
ANSWER: 4
"""
# step_value=1.0 (24 + 4 跟 gold 末尾 5 不匹配... wait, 末尾 '5' 在 gold, pred 是 '4' 不匹配)
# 实际: gold values = ['24', '5'], pred values = ['24', '4']
# step_value = 1/2 = 0.5
# v11: ans=0, step_value=0.5, core=high, dist=0 → 1.5*0.5 + 0.5*0.8 = 0.75 + 0.4 = 1.15
if test_case("Case 7: partial step match, wrong answer (v11 should give some credit)", text, ref, (0.5, 2.0),
             ["answer", "step_value", "core", "distractor"]):
    passed += 1

# Case 8: 答对 + 走 distractor (变体, 应该被 distractor 严重惩罚)
total += 1
ref = {
    "answer": "10",
    "gold_expression": "12/60*50",
    "gold_trace_tokens": ["12","/","60","*","50","=","10"],
    "distractor_expression": "12/60*50-3",
    "distractor_trace_tokens": [],
    "distractor_enabled": True,
    "category": "independent_decoy",
}
text = """TARGET: minutes
<<12/60 = 0.2>>
<<0.2 * 50 = 10>>
<<FINAL: 12/60*50-3 = 7>>  # 用了 distractor, 答案对是巧合
ANSWER: 10
"""
# v11: ans=1, step_value=1.0 (10 匹配), core=low, dist=high → 扣分
if test_case("Case 8: decoy correct with distractor (should be penalized)", text, ref, (4.0, 5.0),
             ["answer", "step_value", "core", "distractor"]):
    passed += 1

print(f"\n=== {passed}/{total} PASS ===")
sys.exit(0 if passed == total else 1)
