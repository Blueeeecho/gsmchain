"""v8 reward 行为测试. 覆盖 4 子项 (format / answer / numeric / step_count) + 边界 + 权重验证.

v8 公式 (vs v7 0.10/0.70/0.20 砍 numeric 抢 answer 梯度):
  total = 0.05 * format
        + 0.85 * answer
        + 0.05 * numeric_correctness
        + 0.05 * step_count (0 算式 0.0, 1 算式 0.5, 2+ 算式 1.0)

  format: extract_answer 能取到值 -> 1.0 else 0.0
  answer: 数值 == gold -> 1.0 else 0.0
  numeric_correctness: 算式 (X op Y = Z) 中 eval(X op Y) == Z 的比例 (0-1)
  step_count: 0 -> 0, 1 -> 0.5, 2+ -> 1.0

用法: cd /home/wwq416/snap/wwq/math-chain && \
      /home/wwq416/miniconda3/envs/math_chain_verl/bin/python \
      train_pipeline/test_lbprm_v8.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "train_pipeline"))

# 故意尝试导入 v8 module, 应该 ImportError (RED 阶段)
try:
    from reward_chaingsm_lbprm_v8_verl import score_response, compute_reward  # noqa: E402
    HAS_V8 = True
except ImportError as e:
    HAS_V8 = False
    print(f"=== TDD RED 阶段: reward module 未实现 ({e}) ===")
    print(f"=== 需要先写 train_pipeline/reward_chaingsm_lbprm_v8_verl.py ===")
    sys.exit(1)

# v8 权重
W = {
    "format_weight": 0.05,
    "answer_weight": 0.85,
    "numeric_correctness_weight": 0.05,
    "step_count_weight": 0.05,
}


def approx(a: float, b: float, tol: float = 0.06) -> bool:
    return abs(a - b) < tol


def test(name, completion, reference, expected, tol=0.06, debug=False):
    r, m = score_response(completion, reference, **W)
    ok = approx(r, expected, tol)
    flag = "PASS" if ok else "FAIL"
    print(f"[{flag}] {name:<55s} r={r:.3f} (expect {expected:.3f} ±{tol})")
    if debug or not ok:
        print(f"        metrics: {m}")
    return ok


passed = 0
total = 0

# === Group 1: 完美链 ===
# format 1, answer 1, numeric 1.0, step_count 1.0 (2+ 算式)
# total = 0.05 + 0.85 + 0.05 + 0.05 = 1.0
total += 1
if test("v8-A-perfect-3steps-correct-arith",
         "First, 16 - 3 = 13 eggs left after breakfast. Then 13 - 4 = 9 eggs after baking. She sells 9 at $2 each, so 9 * 2 = 18 dollars. The final answer is 18.",
         {"gold_answer": 18}, 1.0):
    passed += 1

# === Group 2: 算式对 + 答案错 ===
# format 1, answer 0, numeric 1.0, step_count 1.0
# total = 0.05 + 0 + 0.05 + 0.05 = 0.15
total += 1
if test("v8-C-correct-arith-wrong-final",
         "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 22.",
         {"gold_answer": 18}, 0.15, tol=0.05):
    passed += 1

# === Group 3: 答案对, 算式部分错 ===
# "16 - 3 = 12" (错), "12 - 4 = 8" (对), "8 * 2 = 18" (错) -> numeric 1/3, step_count 1.0
# total = 0.05 + 0.85 + 0.05*(1/3) + 0.05 = 0.967
expected_partial = 0.05 + 0.85 + 0.05 * (1/3) + 0.05
total += 1
if test("v8-D-correct-wrong-half-arith",
         "16 - 3 = 12. 12 - 4 = 8. 8 * 2 = 18. The final answer is 18.",
         {"gold_answer": 18}, expected_partial, tol=0.05, debug=True):
    passed += 1

# === Group 4: 答案对, 0 算式 ===
# format 1, answer 1, numeric 0, step_count 0
# total = 0.05 + 0.85 + 0 + 0 = 0.90
total += 1
if test("v8-E-correct-no-valid-arith",
         "She thinks about the numbers. Adds them up somehow. The final answer is 18.",
         {"gold_answer": 18}, 0.90, tol=0.05):
    passed += 1

# === Group 5: 答案错 + 算式全对 ===
# format 1, answer 0, numeric 1.0, step_count 1.0
# total = 0.05 + 0 + 0.05 + 0.05 = 0.15
total += 1
if test("v8-F-wrong-wrong-wrong",
         "16 + 3 = 19. 19 + 4 = 23. 23 * 2 = 46. The final answer is 46.",
         {"gold_answer": 18}, 0.15, tol=0.05):
    passed += 1

# === Group 6: 1 算式 + 答案对 ===
# format 1, answer 1, numeric 1.0, step_count 0.5 (1 算式)
# total = 0.05 + 0.85 + 0.05 + 0.05*0.5 = 0.975
total += 1
if test("v8-G-correct-1-equation",
         "16 - 3 = 13. The final answer is 13.",
         {"gold_answer": 13}, 0.975, tol=0.05):
    passed += 1

# === Group 7: 2 算式 + 答案对 ===
# format 1, answer 1, numeric 1.0, step_count 1.0 (2 算式)
# total = 0.05 + 0.85 + 0.05 + 0.05 = 1.0
total += 1
if test("v8-H-correct-2-equations",
         "16 - 3 = 13. 13 - 4 = 9. The final answer is 9.",
         {"gold_answer": 9}, 1.0):
    passed += 1

# === Group 8: 答案对, 3 算式, 2 对 1 错 ===
# "5 + 3 = 8" (对), "8 * 2 = 18" (错, 应是 16), "16 / 2 = 8" (对)
# numeric 2/3, step_count 1.0
# total = 0.05 + 0.85 + 0.05*(2/3) + 0.05 = 0.983
expected_3eq = 0.05 + 0.85 + 0.05 * (2/3) + 0.05
total += 1
if test("v8-I-correct-3-equations-partial",
         "5 + 3 = 8. 8 * 2 = 18. 16 / 2 = 8. The final answer is 8.",
         {"gold_answer": 8}, expected_3eq, tol=0.05, debug=True):
    passed += 1

# === Group 9: 答案对, 0 算式, 0 step_count ===
# total = 0.05 + 0.85 + 0 + 0 = 0.90 (跟 Group 4 一样)
total += 1
if test("v8-J-correct-no-arith",
         "She has 18 dollars. The final answer is 18.",
         {"gold_answer": 18}, 0.90, tol=0.05):
    passed += 1

# === Group 10: empty / unrelated ===
total += 1
if test("v8-K-empty",
         "",
         {"gold_answer": 18}, -0.5):
    passed += 1

# === Group 11: 答错 + 无算式 ===
# format 1, answer 0, numeric 0, step_count 0
# total = 0.05 + 0 + 0 + 0 = 0.05
total += 1
if test("v8-L-wrong-no-arith",
         "She has 22 dollars. The final answer is 22.",
         {"gold_answer": 18}, 0.05, tol=0.05):
    passed += 1

# === Group 12: gold 缺失 ===
# format 1 (有 final), answer 0 (gold None 永远不匹配), numeric 0, step_count 0
# total = 0.05
total += 1
r, m = score_response("The final answer is 18.", {"gold_answer": None}, **W)
if approx(r, 0.05, 0.01):
    print(f"[PASS] v8-M-gold-None                                     r={r:.3f} (expect 0.05)")
    passed += 1
else:
    print(f"[FAIL] v8-M-gold-None: r={r:.3f}, m={m}")

# === Group 13: metrics schema ===
# v8 score_response 应该有: format, answer, numeric_correctness_score, step_count_score, n_equations, reward, reason
# v8 score_response 不包含 'score' (那是 compute_reward 才会加)
expected_keys = {"format", "answer", "numeric_correctness_score", "step_count_score",
                 "n_equations", "reward", "reason"}
test_completions = [
    ("normal", "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 18.", {"gold_answer": 18}),
    ("empty", "", {"gold_answer": 18}),
    ("wrong_final", "The final answer is 22.", {"gold_answer": 18}),
]
all_schema_ok = True
for case_name, completion, ref in test_completions:
    r, m = score_response(completion, ref, **W)
    keys = set(m.keys())
    has_v8 = expected_keys.issubset(keys)
    ok = has_v8
    flag = "PASS" if ok else "FAIL"
    extra = keys - expected_keys
    missing = expected_keys - keys
    print(f"[{flag}] v8-schema ({case_name}): present_v8={has_v8}")
    if not ok:
        print(f"        missing={missing} extra={extra}")
        all_schema_ok = False
    total += 1
if all_schema_ok:
    passed += len(test_completions)

# === Group 14: compute_reward entry (kwargs 模式, 包含 score 键) ===
total += 1
r = compute_reward("chaingsm_8shot_cot",
                    "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 18.",
                    {"gold_answer": 18}, None, **W)
if "score" in r and r["score"] > 0.95 and len(r) >= 7:
    print(f"[PASS] v8-compute-reward entry: score={r['score']:.3f}, keys={list(r.keys())[:5]}...")
    passed += 1
else:
    print(f"[FAIL] v8-compute-reward entry: {r}")

# === Group 15: 验证 v8 公式权重 0.05/0.85/0.05/0.05 ===
# 全对 + 全算式对 + 2+ 算式: 0.05 + 0.85 + 0.05 + 0.05 = 1.0
# 全错: -0.5
# format 1 + answer 0 + numeric 1.0 + step_count 1.0: 0.05 + 0 + 0.05 + 0.05 = 0.15
total += 1
r1, _ = score_response("16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 18.", {"gold_answer": 18}, **W)
r2, _ = score_response("", {"gold_answer": 18}, **W)
r3, _ = score_response("16 + 3 = 19. 19 + 4 = 23. 23 * 2 = 46. The final answer is 46.", {"gold_answer": 18}, **W)
if approx(r1, 1.0, 0.01) and approx(r2, -0.5, 0.01) and approx(r3, 0.15, 0.05):
    print(f"[PASS] v8-weights-sanity: r_perfect={r1:.3f}, r_empty={r2:.3f}, r_wrong_wrong={r3:.3f}")
    passed += 1
else:
    print(f"[FAIL] v8-weights-sanity: r1={r1}, r2={r2}, r3={r3}")

# === Group 16: 算式对答错但数字在过程出现 ===
# "16 - 3 = 13. 13 - 4 = 9. The final answer is 22." -> numeric 2/2=1.0, step_count 1.0, answer 0
# total = 0.05 + 0 + 0.05 + 0.05 = 0.15
total += 1
if test("v8-O-contradiction-no-score",
         "16 - 3 = 13. 13 - 4 = 9. The final answer is 22.",
         {"gold_answer": 18}, 0.15, tol=0.05):
    passed += 1

# === Group 17: 分数答案 ===
# "16 / 4 = 4. The final answer is 4." -> 4.0 == 4, numeric 1.0, step_count 0.5 (1 算式)
# total = 0.05 + 0.85 + 0.05 + 0.025 = 0.975
total += 1
if test("v8-P-correct-division",
         "She has 16 eggs, gives 4 away. 16 / 4 = 4 eggs left. The final answer is 4.",
         {"gold_answer": 4}, 0.975, tol=0.05):
    passed += 1

# === Group 18: 负数 ===
# 0.5B 在 8-shot 自由推理下也可能给负数 answer
total += 1
if test("v8-Q-correct-negative",
         "She had 5 dollars, owes 12. 5 - 12 = -7. The final answer is -7.",
         {"gold_answer": -7}, 1.0):
    passed += 1

# === Group 19: step_count 子项验证 ===
# 0 算式 -> step_count_score = 0.0
# 1 算式 -> step_count_score = 0.5
# 2 算式 -> step_count_score = 1.0
# 5 算式 -> step_count_score = 1.0
total += 1
r_0, m_0 = score_response("Just text. The final answer is 5.", {"gold_answer": 5}, **W)
r_1, m_1 = score_response("5 + 0 = 5. The final answer is 5.", {"gold_answer": 5}, **W)
r_2, m_2 = score_response("2 + 3 = 5. 5 * 1 = 5. The final answer is 5.", {"gold_answer": 5}, **W)
r_5, m_5 = score_response("1+1=2. 2+1=3. 3+1=4. 4+1=5. 5*1=5. The final answer is 5.", {"gold_answer": 5}, **W)
if (abs(m_0["step_count_score"] - 0.0) < 0.01 and
    abs(m_1["step_count_score"] - 0.5) < 0.01 and
    abs(m_2["step_count_score"] - 1.0) < 0.01 and
    abs(m_5["step_count_score"] - 1.0) < 0.01):
    print(f"[PASS] v8-step-count: 0_eq={m_0['step_count_score']}, 1_eq={m_1['step_count_score']}, 2_eq={m_2['step_count_score']}, 5_eq={m_5['step_count_score']}")
    passed += 1
else:
    print(f"[FAIL] v8-step-count: 0={m_0['step_count_score']}, 1={m_1['step_count_score']}, 2={m_2['step_count_score']}, 5={m_5['step_count_score']}")

# === Group 20: 算式数 (n_equations) 验证 ===
# 验证 n_equations 字段正确
total += 1
r, m = score_response("5+3=8. 8*2=16. 16/4=4. The final answer is 4.", {"gold_answer": 4}, **W)
if m["n_equations"] == 3 and m["step_count_score"] == 1.0:
    print(f"[PASS] v8-n-equations: n_equations={m['n_equations']}, step_count_score={m['step_count_score']}")
    passed += 1
else:
    print(f"[FAIL] v8-n-equations: n_equations={m['n_equations']}, step_count_score={m['step_count_score']}")

print()
print(f"=== {passed}/{total} PASSED ===")
sys.exit(0 if passed == total else 1)
