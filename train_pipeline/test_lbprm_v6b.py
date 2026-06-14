"""v6-B reward 行为测试. 覆盖 reasoning_v2 4 子项 + format + answer 边界 + 权重验证.

v6-B 公式:
  total = 0.10·format + 0.55·answer + 0.35·reasoning_v2
  reasoning_v2 = 0.4·step_count + 0.4·numeric_correctness + 0.1·no_contradiction + 0.1·equation_count_bonus

  step_count: 3-7 步满分 1.0, <3 扣 0.3, >7 扣 0.2
  equation_count_bonus: 0→0.0, 1→0.3, 2→0.6, >=3→1.0

用法: cd /home/wwq416/snap/wwq/math-chain && \
      /home/wwq416/miniconda3/envs/math_chain_verl/bin/python \
      train_pipeline/test_lbprm_v6b.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "train_pipeline"))

from reward_chaingsm_lbprm_v6b_verl import score_response, compute_reward  # noqa: E402

# v6-B 权重
W = {"format_weight": 0.10, "answer_weight": 0.55, "reasoning_v2_weight": 0.35}


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
total += 1
if test("v6b-A-perfect-3steps-correct-arith",
         "Step 1: Janet eats 3 eggs.\n3 eggs.\nStep 2: She bakes 4 muffins.\n4 eggs.\nStep 3: 16 - 3 - 4 = 9 eggs remain.\n9 eggs.\nThe final answer is 18.",
         {"gold_answer": 18}, 0.84, tol=0.1, debug=True):
    passed += 1
# 解释: n_steps=7 (句子被切碎), 1 个算式 9 = 16-3-4 不直接命中 (e.q. 9 是结果),
# 算式正确率 0/1, eq_count_bonus 0.3, reasoning = 0.4*1.0 + 0.4*0.0 + 0.1*1.0 + 0.1*0.3 = 0.53
# reward = 0.1*1.0 + 0.55*1.0 + 0.35*0.53 = 0.836

total += 1
if test("v6b-B-perfect-4steps-balanced",
         "First, 16 - 3 = 13 eggs left after breakfast. Then 13 - 4 = 9 eggs after baking. She sells 9 at $2 each, so 9 * 2 = 18 dollars. The final answer is 18.",
         {"gold_answer": 18}, 1.0):
    passed += 1

# === Group 2: 答案错 ===
total += 1
if test("v6b-C-correct-arith-wrong-final",
         "Step 1: 16 - 3 = 13.\nStep 2: 13 - 4 = 9.\nStep 3: 9 * 2 = 18.\nThe final answer is 22.",
         {"gold_answer": 18}, 0.45, tol=0.1):
    passed += 1
# reward = 0.1*1.0 + 0.55*0.0 + 0.35*1.0 = 0.45

total += 1
if test("v6b-F-wrong-step-arith-correct",
         "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 99.",
         {"gold_answer": 18}, 0.45, tol=0.1):
    passed += 1
# 同 C

# === Group 3: 答案对, 算式部分错 (D/E) ===
total += 1
if test("v6b-D-correct-wrong-arithmetic",
         "Step 1: 16 - 3 = 12. Step 2: 12 - 4 = 8. Step 3: 8 * 2 = 18. The final answer is 18.",
         {"gold_answer": 18}, 0.91, tol=0.1):
    passed += 1
# 算式正确 1/3, eq_count 3, no_contradiction 1.0
# reasoning = 0.4*1.0 + 0.4*0.33 + 0.1*1.0 + 0.1*1.0 = 0.733
# reward = 0.1 + 0.55 + 0.35*0.733 = 0.907

total += 1
if test("v6b-E-correct-arith-final-in-process",
         "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 18.",
         {"gold_answer": 18}, 1.0):
    passed += 1

# === Group 4: 全错 (G/Q) ===
total += 1
if test("v6b-G-wrong-wrong-wrong",
         "16 + 3 = 19. 19 + 4 = 23. 23 * 2 = 46. The final answer is 46.",
         {"gold_answer": 18}, 0.45, tol=0.1):
    passed += 1
# format 1.0 (有 final), answer 0.0, reasoning 算式 3/3 正确 (e.q. 19=16+3 算对), eq_count 3, no_contradiction 1
# reasoning = 0.4*1 + 0.4*1 + 0.1*1 + 0.1*1 = 1.0
# reward = 0.1 + 0 + 0.35*1.0 = 0.45

total += 1
if test("v6b-Q-wrong-arith-wrong-final",
         "16 - 3 = 12. 13 - 4 = 8. 8 * 2 = 16. The final answer is 99.",
         {"gold_answer": 18}, 0.35, tol=0.1):
    passed += 1
# 算式 0/3 正确, eq_count 3, no_contradiction 0 (99 不在过程), final_in_process 0
# reasoning = 0.4*1 + 0.4*0 + 0.1*0 + 0.1*1 = 0.5
# reward = 0.1 + 0 + 0.35*0.5 = 0.275

# === Group 5: 步骤数 ===
total += 1
if test("v6b-N-only-2steps",
         "She has 18. The final answer is 18.",
         {"gold_answer": 18}, 0.78, tol=0.1):
    passed += 1
# 2 步 -> step_count 0.7, 0 算式 -> numeric 0 + eq_count_bonus 0
# reasoning = 0.4*0.7 + 0.4*0 + 0.1*1 + 0.1*0 = 0.38
# reward = 0.1 + 0.55 + 0.35*0.38 = 0.783

total += 1
if test("v6b-O-too-many-steps",
         "Step 1: 16 eggs. Step 2: 3 breakfast. Step 3: 4 baking. Step 4: 16 - 3 = 13. Step 5: 13 - 4 = 9. Step 6: 9 * 2 = 18. Step 7: 18 dollars. Step 8: 18. The final answer is 18.",
         {"gold_answer": 18}, 0.97, tol=0.15):
    passed += 1
# 8 步 (按 . 切), step_count 0.8, 算式 1/1 正确, eq_count_bonus 0.3
# reasoning = 0.4*0.8 + 0.4*1 + 0.1*1 + 0.1*0.3 = 0.85
# reward = 0.1 + 0.55 + 0.35*0.85 = 0.948

# === Group 6: 答对+无算式 (H) ===
total += 1
if test("v6b-H-correct-no-equations",
         "She has 18 dollars total after selling. The final answer is 18.",
         {"gold_answer": 18}, 0.78, tol=0.1):
    passed += 1
# 同 v6b-N (n_steps 2, 0 算式)

# === Group 7: 不同收尾 marker (I/J) ===
total += 1
if test("v6b-I-correct-hash-marker",
         "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. #### 18",
         {"gold_answer": 18}, 1.0):
    passed += 1

total += 1
if test("v6b-J-correct-boxed-marker",
         "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. \\boxed{18}",
         {"gold_answer": 18}, 1.0):
    passed += 1

# === Group 8: 算式数量奖励 (K/R/S) ===
total += 1
if test("v6b-K-correct-many-equations",
         "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. 18 + 0 = 18. 18 - 0 = 18. The final answer is 18.",
         {"gold_answer": 18}, 1.0):
    passed += 1

total += 1
if test("v6b-R-correct-only-1-equation",
         "She has 18. Just 9 * 2 = 18. The final answer is 18.",
         {"gold_answer": 18}, 1.0, tol=0.1):
    passed += 1
# 3 步, 1 算式 正确, eq_count_bonus 0.3
# reasoning = 0.4*1 + 0.4*1 + 0.1*1 + 0.1*0.3 = 0.93
# reward = 0.1 + 0.55 + 0.35*0.93 = 0.976

total += 1
if test("v6b-S-correct-2-equations",
         "16 - 3 = 13. Then 13 * 1 = 13. The final answer is 13.",
         {"gold_answer": 13}, 1.0, tol=0.1):
    passed += 1
# 3 步, 2 算式 正确, eq_count_bonus 0.6
# reasoning = 0.4*1 + 0.4*1 + 0.1*1 + 0.1*0.6 = 0.96
# reward = 0.1 + 0.55 + 0.35*0.96 = 0.986

# === Group 9: empty / unrelated ===
total += 1
if test("v6b-L-empty",
         "",
         {"gold_answer": 18}, -0.5):
    passed += 1

total += 1
if test("v6b-M-unrelated-with-correct-final",
         "The sky is blue today. Birds fly. The final answer is 18.",
         {"gold_answer": 18}, 0.83, tol=0.1):
    passed += 1
# 答对+格式对+0 算式, 跟 H/N 一样 0.783
# 实际 n_steps=3 (按 . 切) -> step_count 1.0, 算式 0, eq_bonus 0
# reasoning = 0.4*1 + 0.4*0 + 0.1*1 + 0.1*0 = 0.5
# reward = 0.1 + 0.55 + 0.35*0.5 = 0.825

# === Group 10: 矛盾 (P) ===
total += 1
if test("v6b-P-wrong-final-in-process-but-arith-ok",
         "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 18. Actually 22.",
         {"gold_answer": 18}, 1.0, tol=0.1):
    passed += 1
# extract_answer 拿到的是 18, 不是 22, 所以答对
# reward = 0.1 + 0.55 + 0.35*1.0 = 1.0

# === Group 11: 8-shot 风格 (T) ===
total += 1
if test("v6b-T-classic-8shot-style",
         "Let's think step by step. Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 for friends. So 16 - 3 - 4 = 9 eggs remain. She sells them at $2 per egg: 9 * 2 = 18. The answer is 18 dollars. The final answer is 18.",
         {"gold_answer": 18}, 0.92, tol=0.1):
    passed += 1
# 7 步, 2 算式 (16-3-4 算式无 =, 9*2=18 有 =), eq_count_bonus 0.6
# reasoning = 0.4*1 + 0.4*0.5 + 0.1*1 + 0.1*0.6 = 0.76
# reward = 0.1 + 0.55 + 0.35*0.76 = 0.916

# === Group 12: 跨测试兼容 (U) ===
total += 1
if test("v6b-U-v6-old-correct-wrong-final",
         "Step 1: 3 + 4 = 7. Step 2: 16 - 7 = 9. Step 3: 9 * 2 = 18. The final answer is 22.",
         {"gold_answer": 18}, 0.45, tol=0.1):
    passed += 1

# === Group 13: 权重 (纯权重验证) ===
total += 1
# 全部子项 = 1, 总分应该 = 0.10 + 0.55 + 0.35*1.0 = 1.0
r, m = score_response(
    "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. 18 + 0 = 18. The final answer is 18.",
    {"gold_answer": 18}, **W)
if approx(r, 1.0):
    print(f"[PASS] v6b-V-weight-all-1                    r={r:.3f} (expect 1.000 ±0.06)")
    passed += 1
else:
    print(f"[FAIL] v6b-V-weight-all-1                    r={r:.3f} (expect 1.000 ±0.06)")
    print(f"        metrics: {m}")
total += 1

# format=1, answer=0, reasoning=1
total += 1
r, m = score_response(
    "Step 1: 16 - 3 = 13. Step 2: 13 - 4 = 9. Step 3: 9 * 2 = 18. The final answer is 99.",
    {"gold_answer": 18}, **W)
# format 1 (有 final), answer 0 (99), reasoning 算式 3/3, eq_count 3, no_contradiction 0 (99 不在), final_in_process 0
# reasoning = 0.4*1 + 0.4*1 + 0.1*0 + 0.1*1 = 0.9
# reward = 0.1 + 0 + 0.35*0.9 = 0.415
if approx(r, 0.42, tol=0.1):
    print(f"[PASS] v6b-W-format1-answer0-reasoning1      r={r:.3f} (expect 0.42 ±0.1)")
    passed += 1
else:
    print(f"[FAIL] v6b-W-format1-answer0-reasoning1      r={r:.3f} (expect 0.42 ±0.1)")
    print(f"        metrics: {m}")

# === Group 14: schema 一致性 ===
expected_keys = {
    "format", "answer", "step_count_score", "n_steps",
    "numeric_correctness_score", "n_equations",
    "no_contradiction_score", "final_in_process",
    "equation_count_bonus", "n_equations_total",
    "reasoning_v2_score", "reward", "reason"
}
test_completions = [
    "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 18.",
    "",
    "The final answer is 22.",
    "16 - 3 = 13. 9 * 2 = 18. The final answer is 18.",
    "random unrelated text without final marker",
    "16 - 3 = 13. The final answer is 13.",
]
for i, comp in enumerate(test_completions):
    r, m = score_response(comp, {"gold_answer": 18}, **W)
    keys = set(m.keys())
    if keys == expected_keys:
        passed += 1
        print(f"[PASS] schema case {i}                              all {len(expected_keys)} keys present")
    else:
        print(f"[FAIL] schema case {i}: keys mismatch. got extra={keys - expected_keys}, missing={expected_keys - keys}")
total += len(test_completions)

# === Group 15: compute_reward entry ===
total += 1
r = compute_reward("chaingsm_8shot_cot",
                    "16 - 3 = 13. 13 - 4 = 9. 9 * 2 = 18. The final answer is 18.",
                    {"gold_answer": 18}, None, **W)
if "score" in r and r["score"] > 0.95 and len(r) >= 13:
    print(f"[PASS] compute_reward entry: score={r['score']:.3f}, keys={list(r.keys())[:5]}...")
    passed += 1
else:
    print(f"[FAIL] compute_reward entry: {r}")

print()
print(f"=== {passed}/{total} PASSED ===")
sys.exit(0 if passed == total else 1)
