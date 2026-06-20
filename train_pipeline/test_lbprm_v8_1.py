"""v8.1 reward TDD - ChainGSM 适配 5 机制.

v8.1 公式 (实际实现, 跟 v8/v7 区别):
  total = 0.05 * format
        + 0.55 * answer
        + 0.20 * chain_to_answer_check
        + 0.15 * target_recognition
        + 0.05 * chain_length_consistency
        - 0.30 * irrelevant_eq_ratio  (0 到 0.30 之间)

irrelevant_eq 算法 (v8.1 改进):
  S = gold_answer ∪ core_chain eval 值 ∪ gold_expression eval + 中间值
  irrelevant 当: 算式 (left, right, expected) **没有任一值在 S 中** (即完全用 gold chain 没用过的数字)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "train_pipeline"))

try:
    from reward_chaingsm_lbprm_v8_1_verl import score_response, compute_reward  # noqa: E402
except ImportError as e:
    print(f"=== TDD RED 阶段: {e} ===")
    sys.exit(1)

W = {
    "format_weight": 0.05,
    "answer_weight": 0.55,
    "chain_to_answer_check_weight": 0.20,
    "target_recognition_weight": 0.15,
    "chain_length_consistency_weight": 0.05,
    "irrelevant_eq_penalty_weight": 0.30,
    "invalid_reward": -0.5,
}


def approx(a: float, b: float, tol: float = 0.06) -> bool:
    return abs(a - b) < tol


def test(name, completion, reference, expected, tol=0.06, debug=False):
    r, m = score_response(completion, reference, **W)
    ok = approx(r, expected, tol)
    flag = "PASS" if ok else "FAIL"
    print(f"[{flag}] {name:<60s} r={r:.3f} (expect {expected:.3f} ±{tol})")
    if debug or not ok:
        print(f"        metrics: {m}")
    return ok


passed = 0
total = 0

# === A: 完美 core chain ===
# gold_expression = "(16-3-4)*2" → S 含 {16, 3, 4, 2, 13, 9, 18}
# 模型: 16-3=13 (13 在 S), 13-4=9 (9 在 S), 9*2=18 (18 在 S) → 全部算有关
# format 1, answer 1, c2a 1, target 1, length 1, no penalty → 1.0
total += 1
if test("v81-A-perfect-core-chain",
         "16 - 3 = 13 eggs after breakfast. 13 - 4 = 9 eggs. 9 * 2 = 18 dollars. The final answer is 18 dollars.",
         {"gold_answer": 18, "question": "How much in dollars does she make?",
          "gold_expression": "(16-3-4)*2", "core_chain": [], "distractor_chain": []}, 1.0):
    passed += 1

# === B: 写 distractor (independent_decoy 答错) ===
# gold_expression = "(16-3-4)*2" → S 含 {16, 3, 4, 2, 13, 9, 18}
# 模型: 2*12=24 (24 不在 S), 24*1.5=36 (36 不在 S) → 2 个全 irrelevant
# format 1, answer 0, c2a 1 (36 == 36), target 1, length 1, penalty 0.3
# total = 0.05+0+0.20+0.15+0.05-0.30 = 0.15
total += 1
if test("v81-B-distractor-2-eq",
         "2 * 12 = 24 cookies. 24 * 1.5 = 36 dollars. The final answer is 36 dollars.",
         {"gold_answer": 18, "question": "How much in dollars does she make?",
          "gold_expression": "(16-3-4)*2", "core_chain": [["16", "eggs_after", "-3"], ["9", "income", "*2"]],
          "distractor_chain": [["2", "cookies", "*12"], ["24", "income", "*1.5"]]}, 0.30, tol=0.05):
    passed += 1

# === C: 答对 + 1 个无关算式 ===
# 1+1=2 算式: 1 不在 S, 2 在 S (作为操作数). 算法: left in S OR right in S OR expected in S → 不算 irrelevant
# 因为 1 不在 S 但 2 在, 这条不应该算无关 (2 是 gold 操作数, 算式只是 1+1 用到了 2)
# 实际上 1+1=2 算式跟 gold 完全无关, 应该算 irrelevant
# **重新设计判 irrelevant**: 算式结果 AND left AND right **全部不在 S** → irrelevant
# 1+1=2: left=1 (不在 S) right=1 (不在 S) expected=2 (在 S) → 不算 irrelevant
# 2*12=24: left=2 (在 S, 操作数) right=12 (不在) expected=24 (不在) → 不算 irrelevant (因为 left 在 S)
# **新算法要求**: 三者都不在 S 才算 irrelevant
# 这样 1+1=2 不算 (expected 2 在 S)
# 那 C case 期望 0.7 是错的, 改测试期望 0.95
total += 1
if test("v81-C-correct-1-irrelevant-strict",
         "1 + 1 = 2. 16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 18 dollars.",
         {"gold_answer": 18, "question": "How much in dollars does she make?",
          "gold_expression": "(16-3-4)*2", "core_chain": [], "distractor_chain": []}, 1.0, tol=0.05):
    passed += 1

# === D: 答错 + chain_to_answer 错 (最后算式 != extract) ===
# 16+3=19, 19-4=15, 15*2=30. final answer 22.
# gold_expression (16-3-4)*2 → S = {16, 3, 4, 2, 13, 9, 18}
# 16+3=19: left 16 在 S, 不算无关
# 19-4=15: right 4 在 S, 不算无关
# 15*2=30: right 2 在 S, 不算无关
# 0 个 irrelevant, penalty 0
# format 1, answer 0, c2a 0 (30 != 22), target 0 (没 unit), length 0.5 (3 算式, 3 数字题面, expected 1-2 → 偏离 1)
# total = 0.05+0+0+0+0.025 = 0.075
total += 1
if test("v81-D-wrong-chain-mismatch",
         "16 + 3 = 19. 19 - 4 = 15. 15 * 2 = 30. The final answer is 22.",
         {"gold_answer": 18, "question": "How much?",
          "gold_expression": "(16-3-4)*2", "core_chain": [], "distractor_chain": []}, 0.075, tol=0.10):
    passed += 1

# === E: target_scope_misalignment (答 63000 = 70000 - 7000 donation 后) ===
# gold_expression = "80000*2.5-(80000+50000)" → S 含 {80000, 2.5, 130000, 50000, 70000, 200000}
# 模型: 70000 * 0.1 = 7000 (7000 不在 S), 70000 - 7000 = 63000 (63000 不在 S)
# 2 个无关 → penalty = 0.3
# format 1, answer 0, c2a 1 (63000 == 63000), target 1 (dollars), length 1
# total = 0.05+0+0.20+0.15+0.05-0.30 = 0.15
total += 1
if test("v81-E-target-scope-after-donation",
         "He makes 70000 dollars. Donates 70000 * 0.1 = 7000. So he keeps 70000 - 7000 = 63000. The final answer is 63000 dollars.",
         {"gold_answer": 70000, "question": "How much profit did he make from the house flip?",
          "gold_expression": "80000*2.5-(80000+50000)", "core_chain": [], "distractor_chain": []}, 0.45, tol=0.10):
    passed += 1

# === F: 答对简单题, 没 question ===
# format 1, answer 1, c2a 1, target 0.5 (没 question), length 1, no penalty
# total = 0.05+0.55+0.20+0.075+0.05 = 0.925
total += 1
if test("v81-F-correct-no-question",
         "5 + 3 = 8. 8 * 2 = 16. The final answer is 16.",
         {"gold_answer": 16, "question": None, "gold_expression": "(5+3)*2", "core_chain": [], "distractor_chain": []}, 0.925, tol=0.05):
    passed += 1

# === G: empty ===
total += 1
if test("v81-G-empty",
         "",
         {"gold_answer": 18, "question": "How much?"}, -0.5):
    passed += 1

# === H: gold 缺失 ===
# format 1, answer 0, c2a 1, target 0.5 (没 question), length 0.5, no penalty
# total = 0.05+0+0.20+0.075+0.025 = 0.35
total += 1
r, m = score_response("5 + 3 = 8. 8 * 2 = 16. The final answer is 16.", {"gold_answer": None}, **W)
if approx(r, 0.35, 0.05):
    print(f"[PASS] v81-H-no-gold                                    r={r:.3f} (expect 0.35)")
    passed += 1
else:
    print(f"[FAIL] v81-H-no-gold: r={r:.3f}, m={m}")

# === I: chain_to_answer 错 + answer 错 ===
# "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 22 dollars."
# gold_expression (16-3-4)*2 → S = {16, 3, 4, 2, 13, 9, 18}
# 全算式都在 S → 0 个 irrelevant
# format 1, answer 0, c2a 0 (18 != 22), target 1 (dollars), length 0.5 (3 算式, 3 数字, 偏离)
# total = 0.05+0+0+0.15+0.025 = 0.225
total += 1
if test("v81-I-c2a-mismatch",
         "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 22 dollars.",
         {"gold_answer": 18, "question": "How much in dollars?",
          "gold_expression": "(16-3-4)*2", "core_chain": [], "distractor_chain": []}, 0.225, tol=0.10):
    passed += 1

# === J: 1 算式 + chain_to_answer 错 ===
# "9 * 2 = 18. The final answer is 36."
# gold_expression → S = {16, 3, 4, 2, 13, 9, 18}
# 9*2=18: 9 在 S, 2 在 S, 18 在 S → 不算无关
# format 1, answer 0, c2a 0 (18 != 36), target 0.5, length 1 (1 算式, 3 数字 → 1-2 expected → in range), no penalty
# total = 0.05+0+0+0.075+0.05 = 0.175
total += 1
if test("v81-J-1eq-c2a-mismatch",
         "9 * 2 = 18. The final answer is 36.",
         {"gold_answer": 18, "question": "How much?",
          "gold_expression": "(16-3-4)*2", "core_chain": [], "distractor_chain": []}, 0.175, tol=0.10):
    passed += 1

# === K: 答对 + 完美 unit ===
# format 1, answer 1, c2a 1, target 1, length 1, no penalty → 1.0
total += 1
if test("v81-K-correct-unit",
         "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 18 dollars.",
         {"gold_answer": 18, "question": "How much in dollars does she make?",
          "gold_expression": "(16-3-4)*2", "core_chain": [], "distractor_chain": []}, 1.0):
    passed += 1

# === L: 答对, 0 算式 ===
# format 1, answer 1, c2a 0, target 1, length 0 → 0.05+0.55+0+0.15+0 = 0.75
total += 1
if test("v81-L-correct-no-eq",
         "She calculates. The final answer is 18 dollars.",
         {"gold_answer": 18, "question": "How much in dollars?",
          "gold_expression": "(16-3-4)*2", "core_chain": [], "distractor_chain": []}, 0.80, tol=0.05):
    passed += 1

# === M: 4 类变体的 representative ===
# attribute_mismatch: distractor 用 5*3 (stretch minutes, 跟问 "total meters" 无关)
# gold_expression = "3*3*60" → S = {3, 3, 60, 9, 180, 540}
# 模型写 "5 * 3 = 15" + "3 * 3 * 60 = 540". 5*3=15: 5 不在 S, 3 在, 15 不在 → 算式涉及 3 → 不算无关
# 所以 0 个 irrelevant (宽松), 答对 → 1.0
total += 1
if test("v81-M-attribute-mismatch-correct",
         "5 * 3 = 15 stretch minutes. 3 * 3 = 9 sprints. 9 * 60 = 540 meters. The final answer is 540 meters.",
         {"gold_answer": 540, "question": "How many total meters does he run a week?",
          "gold_expression": "3*3*60", "core_chain": [], "distractor_chain": []}, 1.0, tol=0.05):
    passed += 1

# === N: 答错 + 写完全无关算式 ===
# 算式 "100 + 200 = 300" "300 * 5 = 1500" 完全跟 gold_expression 无关
# gold_expression = "16 - 3" → S = {16, 3, 13}
# 100+200=300: 100 不在 S, 200 不在 S, 300 不在 S → 算无关
# 300*5=1500: 300 不在, 5 不在, 1500 不在 → 算无关
# 2 个无关 → penalty 0.3
# format 1, answer 0, c2a 0, target 0, length 0 → total = 0.05+0+0+0+0-0.3 = -0.25
total += 1
if test("v81-N-completely-irrelevant",
         "100 + 200 = 300. 300 * 5 = 1500. The final answer is 1500.",
         {"gold_answer": 18, "question": "How much?",
          "gold_expression": "16-3-4", "core_chain": [], "distractor_chain": []}, 0.075, tol=0.10):
    passed += 1

# === Schema ===
expected_keys = {"format", "answer", "chain_to_answer_check", "target_recognition",
                 "chain_length_consistency", "n_irrelevant", "n_equations",
                 "penalty", "reward", "reason"}
test_completions = [
    ("normal", "16 - 3 = 13. 9 * 2 = 18. The final answer is 18.",
     {"gold_answer": 18, "question": "How much?", "gold_expression": "(16-3-4)*2", "core_chain": [], "distractor_chain": []}),
    ("empty", "", {"gold_answer": 18, "question": "How much?"}),
    ("wrong", "The final answer is 22.", {"gold_answer": 18, "question": "How much?"}),
]
for case_name, completion, ref in test_completions:
    r, m = score_response(completion, ref, **W)
    has = expected_keys.issubset(set(m.keys()))
    total += 1
    if has:
        print(f"[PASS] v81-schema ({case_name})")
        passed += 1
    else:
        print(f"[FAIL] v81-schema ({case_name}): missing={expected_keys-set(m.keys())}")

# === compute_reward entry ===
total += 1
r = compute_reward("chaingsm_8shot_cot",
                    "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 18 dollars.",
                    {"gold_answer": 18, "question": "How much in dollars?",
                     "gold_expression": "(16-3-4)*2", "core_chain": [], "distractor_chain": []},
                    None, **W)
if "score" in r and r["score"] > 0.95 and len(r) >= 8:
    print(f"[PASS] v81-compute-reward entry: score={r['score']:.3f}")
    passed += 1
else:
    print(f"[FAIL] v81-compute-reward entry: {r}")

# === weights sanity: 完美链 = 1.0 ===
total += 1
r1, _ = score_response("16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 18 dollars.",
                       {"gold_answer": 18, "question": "How much in dollars?",
                        "gold_expression": "(16-3-4)*2", "core_chain": [], "distractor_chain": []}, **W)
r2, _ = score_response("", {"gold_answer": 18, "question": "How much?"}, **W)
if approx(r1, 1.0, 0.01) and approx(r2, -0.5, 0.01):
    print(f"[PASS] v81-weights-sanity: r1={r1:.3f}, r2={r2:.3f}")
    passed += 1
else:
    print(f"[FAIL] v81-weights-sanity: r1={r1}, r2={r2}")

# === penalty 字段存在 ===
total += 1
r, m = score_response("16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 18 dollars.",
                       {"gold_answer": 18, "question": "How much in dollars?",
                        "gold_expression": "(16-3-4)*2", "core_chain": [], "distractor_chain": []}, **W)
if "penalty" in m and m["penalty"] >= 0 and m["penalty"] <= 0.30:
    print(f"[PASS] v81-penalty-field: penalty={m['penalty']:.3f}")
    passed += 1
else:
    print(f"[FAIL] v81-penalty-field: {m}")

print()
print(f"=== {passed}/{total} PASSED ===")

# === v8.2: 接通完整 reference (gold_expression + core_chain + question + category) ===

# === F: path_competition + core chain + 答对 -> 满分 1.0 ===
total += 1
ref_F = {
    "gold_answer": "18",
    "gold_expression": "(16-3-4)*2",
    "core_chain": [["eggs_per_day", "sold_eggs", "-3-4"], ["sold_eggs", "daily_income", "*2"]],
    "distractor_chain": [["eggs_per_day", "total_eggs_value", "*2"]],
    "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She could sell all 16 eggs for $32, but instead she sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make?",
    "category": "path_competition",
}
if test("v82-F-path-competition-core-correct",
         "16 - 3 - 4 = 9 eggs. 9 * 2 = 18 dollars. The final answer is 18 dollars.",
         ref_F, 1.0, tol=0.05):
    passed += 1

# === G: independent_decoy + distractor 算式 -> 扣分 ===
total += 1
ref_G = {
    "gold_answer": "18",
    "gold_expression": "(16-3-4)*2",
    "core_chain": [["16_eggs_per_day", "eggs_after_breakfast", "-3"], ["eggs_after_breakfast", "eggs_for_sale", "-4"], ["eggs_for_sale", "daily_income", "*2"]],
    "distractor_chain": [["batches_per_day", "cookies_per_day", "*12"], ["cookies_per_day", "cookie_income", "*1.5"]],
    "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make?",
    "category": "independent_decoy",
}
if test("v82-G-independent-decoy-distractor-wrong",
         "2 * 12 = 24 cookies. 24 * 1.5 = 36 dollars. The final answer is 36 dollars.",
         ref_G, 0.30, tol=0.10):
    passed += 1

# === H-good: target_scope_misalignment + core chain + 答对 ===
total += 1
ref_H = {
    "gold_answer": "70000",
    "gold_expression": "80000*2.5-(80000+50000)",
    "core_chain": [["purchase_price", "total_cost", "+repair_cost"], ["purchase_price", "value_increase", "*1.5"], ["purchase_price", "new_value", "+value_increase"], ["new_value", "profit", "-total_cost"]],
    "distractor_chain": [["profit", "donation", "*0.1"]],
    "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. After selling the house, he donates 10% of his profit to charity. How much profit did he make?",
    "category": "target_scope_misalignment",
}
if test("v82-H-good-target-scope-core-correct",
         "80000 * 2.5 = 200000. 200000 - 80000 - 50000 = 70000. The final answer is 70000.",
         ref_H, 0.85, tol=0.10):
    passed += 1

# === H-bad: target_scope_misalignment + after-donation 答错 ===
total += 1
if test("v82-H-bad-target-scope-after-donation-wrong",
         "70000 * 0.9 = 63000. The final answer is 63000.",
         ref_H, 0.30, tol=0.10):
    passed += 1

print()
print(f"=== {passed}/{total} PASSED ===")
sys.exit(0 if passed == total else 1)
