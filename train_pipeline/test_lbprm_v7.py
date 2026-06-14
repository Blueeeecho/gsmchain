"""v7 reward 行为测试. 覆盖 3 子项 (format / answer / numeric_correctness) + 边界 + 权重验证.

v7 公式:
  total = 0.10 * format + 0.70 * answer + 0.20 * numeric_correctness

  format: extract_answer 能取到值 -> 1.0 else 0.0
  answer: 数值 == gold -> 1.0 else 0.0
  numeric_correctness: 算式 (X op Y = Z) 中 eval(X op Y) == Z 的比例 (0-1, 0 算式 -> 0.0)

  跟 v6b 区别:
    - 砍 step_count_score (区分度低, 0.5B 自由推理 0.74 已接近满分)
    - 砍 no_contradiction_score (0.875 接近满分, 噪声)
    - 砍 equation_count_bonus (鼓励 "凑算式" 反利于推理)
    - answer 权重 0.55 -> 0.70 (主线目标 original 数字对)

用法: cd /home/wwq416/snap/wwq/math-chain && \
      /home/wwq416/miniconda3/envs/math_chain_verl/bin/python \
      train_pipeline/test_lbprm_v7.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "train_pipeline"))

from reward_chaingsm_lbprm_v7_verl import score_response, compute_reward  # noqa: E402

# v7 权重
W = {"format_weight": 0.10, "answer_weight": 0.70, "numeric_correctness_weight": 0.20}


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
# format 1, answer 1, numeric 1.0 -> total = 1.0
total += 1
if test("v7-A-perfect-3steps-correct-arith",
         "First, 16 - 3 = 13 eggs left after breakfast. Then 13 - 4 = 9 eggs after baking. She sells 9 at $2 each, so 9 * 2 = 18 dollars. The final answer is 18.",
         {"gold_answer": 18}, 1.0):
    passed += 1

# 短链 + 全算式对 + 答案对 -> 1.0
total += 1
if test("v7-B-perfect-2steps-correct-arith",
         "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 18.",
         {"gold_answer": 18}, 1.0):
    passed += 1

# === Group 2: 算式对 + 答案错 (关键: format + numeric 高, answer 0) ===
# format 1, answer 0, numeric 3/3 = 1.0 -> total = 0.10 + 0 + 0.20 = 0.30
total += 1
if test("v7-C-correct-arith-wrong-final",
         "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 22.",
         {"gold_answer": 18}, 0.30, tol=0.05):
    passed += 1

# === Group 3: 答案对, 算式部分错 ===
# "16 - 3 = 12" (错), "12 - 4 = 8" (对), "8 * 2 = 18" (错) -> numeric 1/3
# format 1, answer 1, numeric 1/3 -> total = 0.10 + 0.70 + 0.20*(1/3) = 0.867
expected_partial = 0.10 + 0.70 + 0.20 * (1/3)
total += 1
if test("v7-D-correct-wrong-half-arith",
         "16 - 3 = 12. 12 - 4 = 8. 8 * 2 = 18. The final answer is 18.",
         {"gold_answer": 18}, expected_partial, tol=0.05, debug=True):
    passed += 1

# === Group 4: 答案对, 0 算式 ===
# format 1, answer 1, numeric 0 -> total = 0.10 + 0.70 + 0 = 0.80
total += 1
if test("v7-E-correct-no-valid-arith",
         "She thinks about the numbers. Adds them up somehow. The final answer is 18.",
         {"gold_answer": 18}, 0.80, tol=0.05):
    passed += 1

# === Group 5: 答案错 + 算式全对 ===
# "16 + 3 = 19" (对), "19 + 4 = 23" (对), "23 * 2 = 46" (对) -> numeric 3/3
# format 1, answer 0, numeric 1.0 -> total = 0.10 + 0 + 0.20 = 0.30
total += 1
if test("v7-F-wrong-wrong-wrong",
         "16 + 3 = 19. 19 + 4 = 23. 23 * 2 = 46. The final answer is 46.",
         {"gold_answer": 18}, 0.30, tol=0.05):
    passed += 1

# === Group 6: 算式 1/3 对 + 答案对 ===
# "16 - 3 = 12" (错), "13 - 4 = 9" (对), "8 * 2 = 18" (错) -> numeric 1/3
# 等等: 13-4 是数字字面不是算式 (没有 =), 所以实际 3 算式: 16-3=12, 13-4=9 (有等号?), 8*2=18
# 重新检查: 文本 "16 - 3 = 12. 13 - 4 = 9. 8 * 2 = 18." -> 3 算式
# 16-3=12 错, 13-4=9 对, 8*2=18 错 -> numeric 1/3
expected_partial2 = 0.10 + 0.70 + 0.20 * (1/3)
total += 1
if test("v7-G-correct-with-some-wrong-arith",
         "16 - 3 = 12. 13 - 4 = 9. 8 * 2 = 18. The final answer is 18.",
         {"gold_answer": 18}, expected_partial2, tol=0.05, debug=True):
    passed += 1

# === Group 7: 不同收尾 marker ===
# #### 18 / \boxed{18} 都被 extract_answer 识别
total += 1
if test("v7-H-correct-hash-marker",
         "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. #### 18",
         {"gold_answer": 18}, 1.0):
    passed += 1

total += 1
if test("v7-I-correct-boxed-marker",
         "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. \\boxed{18}",
         {"gold_answer": 18}, 1.0):
    passed += 1

# === Group 8: conclusion 模式也能提取 (extract_answer 通用) ===
# "So she has 18 dollars." 会被 extract_answer 用 CONCLUSION_PATTERN 命中
total += 1
if test("v7-J-conclusion-marker",
         "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. So she has 18 dollars.",
         {"gold_answer": 18}, 1.0):
    passed += 1

# === Group 9: empty / unrelated ===
total += 1
if test("v7-K-empty",
         "",
         {"gold_answer": 18}, -0.5):
    passed += 1

# === Group 10: 答错 + 无算式 ===
# format 1, answer 0, numeric 0 -> total = 0.10 + 0 + 0 = 0.10
total += 1
if test("v7-L-wrong-no-arith",
         "She has 22 dollars. The final answer is 22.",
         {"gold_answer": 18}, 0.10, tol=0.05):
    passed += 1

# === Group 11: gold 缺失 ===
# format 1 (有 final), answer 0 (gold None 永远不匹配), numeric 0 -> total = 0.10
total += 1
r, m = score_response("The final answer is 18.", {"gold_answer": None}, **W)
if approx(r, 0.10, 0.01):
    print(f"[PASS] v7-M-gold-None                                     r={r:.3f} (expect 0.10)")
    passed += 1
else:
    print(f"[FAIL] v7-M-gold-None: r={r:.3f}, m={m}")

# === Group 12: metrics schema (score_response 返回的 keys) ===
# v7 score_response 应该有: format, answer, numeric_correctness_score, n_equations, reward, reason
# v7 砍掉: step_count_score, n_steps, no_contradiction_score, final_in_process,
#          equation_count_bonus, n_equations_total, reasoning_v2_score
# v7 score_response 不包含 'score' (那是 compute_reward 才会加)
expected_keys = {"format", "answer", "numeric_correctness_score", "n_equations",
                 "reward", "reason"}
v6b_excluded = {"step_count_score", "n_steps", "no_contradiction_score", "final_in_process",
                "equation_count_bonus", "n_equations_total", "reasoning_v2_score"}
test_completions = [
    ("normal", "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 18.", {"gold_answer": 18}),
    ("empty", "", {"gold_answer": 18}),
    ("wrong_final", "The final answer is 22.", {"gold_answer": 18}),
]
all_schema_ok = True
for case_name, completion, ref in test_completions:
    r, m = score_response(completion, ref, **W)
    keys = set(m.keys())
    has_v7 = expected_keys.issubset(keys)
    no_v6b_leak = v6b_excluded.isdisjoint(keys)
    ok = has_v7 and no_v6b_leak
    flag = "PASS" if ok else "FAIL"
    extra = keys - expected_keys
    missing = expected_keys - keys
    leaked = v6b_excluded & keys
    print(f"[{flag}] v7-schema ({case_name}): present_v7={has_v7}, no_v6b_leak={no_v6b_leak}")
    if not ok:
        print(f"        missing={missing} extra={extra} leaked_v6b={leaked}")
        all_schema_ok = False
    total += 1
if all_schema_ok:
    passed += len(test_completions)

# === Group 13: compute_reward entry (kwargs 模式, 包含 score 键) ===
total += 1
r = compute_reward("chaingsm_8shot_cot",
                    "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 18.",
                    {"gold_answer": 18}, None, **W)
if "score" in r and r["score"] > 0.95 and len(r) >= 7:
    print(f"[PASS] v7-compute-reward entry: score={r['score']:.3f}, keys={list(r.keys())[:5]}...")
    passed += 1
else:
    print(f"[FAIL] v7-compute-reward entry: {r}")

# === Group 14: 验证 v7 公式权重 0.10/0.70/0.20 ===
# 全对 + 全算式对: 0.10 + 0.70 + 0.20 = 1.0
# 全错: -0.5
# format 1 + answer 0 + numeric 1.0: 0.10 + 0 + 0.20 = 0.30
total += 1
r1, _ = score_response("16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 18.", {"gold_answer": 18}, **W)
r2, _ = score_response("", {"gold_answer": 18}, **W)
r3, _ = score_response("16 + 3 = 19. 19 + 4 = 23. 23 * 2 = 46. The final answer is 46.", {"gold_answer": 18}, **W)
if approx(r1, 1.0, 0.01) and approx(r2, -0.5, 0.01) and approx(r3, 0.30, 0.05):
    print(f"[PASS] v7-weights-sanity: r_perfect={r1:.3f}, r_empty={r2:.3f}, r_wrong_wrong={r3:.3f}")
    passed += 1
else:
    print(f"[FAIL] v7-weights-sanity: r1={r1}, r2={r2}, r3={r3}")

# === Group 15: 算式对答错但数字在过程出现 (no_contradiction 不再独立计分) ===
# "16 - 3 = 13. 13 - 4 = 9. The final answer is 22." -> numeric 2/2=1.0, answer 0
# total = 0.10 + 0 + 0.20 = 0.30
total += 1
if test("v7-O-contradiction-no-score",
         "16 - 3 = 13. 13 - 4 = 9. The final answer is 22.",
         {"gold_answer": 18}, 0.30, tol=0.05):
    passed += 1

# === Group 16: 分数答案 (1/2, 0.5) ===
# "16 / 4 = 4. The final answer is 4." -> 4.0 == 4
total += 1
if test("v7-P-correct-division",
         "She has 16 eggs, gives 4 away. 16 / 4 = 4 eggs left. The final answer is 4.",
         {"gold_answer": 4}, 1.0):
    passed += 1

# === Group 17: 负数 ===
# 0.5B 在 8-shot 自由推理下也可能给负数 answer
# "5 - 12 = -7. The final answer is -7." -> -7 == -7
total += 1
if test("v7-Q-correct-negative",
         "She had 5 dollars, owes 12. 5 - 12 = -7. The final answer is -7.",
         {"gold_answer": -7}, 1.0):
    passed += 1

print()
print(f"=== {passed}/{total} PASSED ===")
sys.exit(0 if passed == total else 1)
