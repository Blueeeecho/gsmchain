"""v6 reward 行为测试. 覆盖 reasoning_quality 3 子项 + format + answer 边界.

用法: cd /home/wwq416/snap/wwq/math-chain && \
      /home/wwq416/miniconda3/envs/math_chain_verl/bin/python \
      train_pipeline/test_lbprm_v6.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "train_pipeline"))

from reward_chaingsm_lbprm_v6_verl import score_response, compute_reward  # noqa: E402


# 权重跟实际训练一致
W = {"format_weight": 0.15, "answer_weight": 0.60, "reasoning_quality_weight": 0.25}
GOLD = 18  # 21 - 3 = 18


def approx(a: float, b: float, tol: float = 0.05) -> bool:
    return abs(a - b) < tol


def test(name, completion, reference, expected, tol=0.05, debug=False):
    r, m = score_response(completion, reference, **W)
    ok = abs(r - expected) < tol
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name:55s} r={r:.3f} (expect {expected:.3f} ±{tol})")
    if not ok or debug:
        print(f"        metrics={m}")
    return ok


def test_metrics_keys(name, completion, reference, expected_keys):
    """验证所有 metrics schema 都有"""
    r, m = score_response(completion, reference, **W)
    missing = [k for k in expected_keys if k not in m]
    if missing:
        print(f"[FAIL] {name}: missing keys {missing}")
        return False
    print(f"[PASS] {name:55s} (all {len(expected_keys)} keys present)")
    return True


CASES = []

# === 完美 ===
# A: 3 步 + 算式 + 答对 = 0.15 + 0.60 + 0.25*1.0 = 1.0
CASES.append((
    "v6-A-perfect-3steps-with-correct-arithmetic",
    "Let's think step by step. There are originally 21 trees. 3 are eaten. "
    "So 21 - 3 = 18. The final answer is 18.",
    {"gold_answer": "18"},
    1.0,
))

# B: 4 步 + 算式 + 答对 = 同 A
CASES.append((
    "v6-B-perfect-4steps",
    "Let's think step by step. There are originally 21 trees. "
    "3 are eaten for breakfast. So 21 - 3 = 18 remain. "
    "The final answer is 18.",
    {"gold_answer": "18"},
    1.0,
))

# === 算式对 + 答错 ===
# C: 算式对 + 答错 (no_contradiction 高因为 final 在过程算出的某值中)
# 0.15 + 0.0 + 0.25 * 1.0 = 0.4
CASES.append((
    "v6-C-correct-arithmetic-wrong-final",
    "Let's think step by step. There are 21 trees. 3 are eaten. "
    "21 - 3 = 18. But 18 + 6 = 24. The final answer is 24.",
    {"gold_answer": "18"},
    0.4,
))

# === 答对 + 算式错 ===
# D: 答对 + 算式错, final 18 在过程
# reasoning_quality = 0.5*1.0 + 0.3*0.0 + 0.2*1.0 = 0.7
# reward = 0.15 + 0.60 + 0.25*0.7 = 0.925
CASES.append((
    "v6-D-correct-wrong-arithmetic",
    "Let's think step by step. 21 - 3 = 17. But 17 + 1 = 18. The final answer is 18.",
    {"gold_answer": "18"},
    0.925,
))

# === 答对 + 算式对 + final 不在过程 ===
# E: 答对 + 算式对 + final 不在过程
# reasoning_quality = 0.5*1.0 + 0.3*1.0 + 0.2*0.0 = 0.65
# reward = 0.15 + 0.60 + 0.25*0.65 = 0.9125
CASES.append((
    "v6-E-correct-arithmetic-final-not-in-process",
    "Let's think step by step. 21 - 3 = 18. 18 + 1 = 19. The final answer is 18.",
    {"gold_answer": "18"},
    1.0,
))

# === 答错 + 算式对 + final 在过程 ===
# F: 答错 + 算式对 + final 19 = 14+5
# reasoning_quality = 0.5*1.0 + 0.3*1.0 + 0.2*1.0 = 1.0
# reward = 0.15 + 0.0 + 0.25*1.0 = 0.4
CASES.append((
    "v6-F-wrong-step-arith-correct",
    "Let's think step by step. 21 - 3 = 18. 18 - 4 = 14. 14 + 5 = 19. "
    "The final answer is 19.",
    {"gold_answer": "14"},
    0.4,
))

# === 答错 + 算式错 + final 在过程 (但 final 跟算式不匹配) ===
# G: 答错 + 算式错 + final 17 在过程
# reasoning_quality = 0.5*1.0 + 0.3*0.0 + 0.2*1.0 = 0.7
# reward = 0.15 + 0.0 + 0.25*0.7 = 0.325
CASES.append((
    "v6-G-wrong-wrong-wrong",
    "Let's think step by step. 21 - 3 = 17. The final answer is 17.",
    {"gold_answer": "18"},
    0.325,
))

# === 答对 + 没算式 (纯自然语言) ===
# H: 4 步 + 没算式 + 答对
# reasoning_quality = 0.5*1.0 + 0.3*0.0 + 0.2*1.0 = 0.7
# reward = 0.15 + 0.60 + 0.25*0.7 = 0.925
CASES.append((
    "v6-H-correct-no-equations",
    "Let's think step by step. There are originally twenty one trees. "
    "After three are eaten, eighteen remain. "
    "Then nothing else happens. The final answer is 18.",
    {"gold_answer": "18"},
    0.925,
))

# === 答对 + #### 风格 marker ===
CASES.append((
    "v6-I-correct-hash-marker",
    "Let's think step by step. 21 - 3 = 18. #### 18",
    {"gold_answer": "18"},
    1.0,
))

# === 答对 + \\boxed{} 风格 marker ===
CASES.append((
    "v6-J-correct-boxed-marker",
    "Let's think step by step. 21 - 3 = 18. \\boxed{18}",
    {"gold_answer": "18"},
    1.0,
))

# === 答对 + 多步 + 算式正确 + final 在过程 ===
CASES.append((
    "v6-K-correct-multistep-proper",
    "Let's think step by step. There are 21 trees. 3 are eaten for breakfast. "
    "So 21 - 3 = 18. Then 4 are baked for muffins. 18 - 4 = 14. "
    "The final answer is 14.",
    {"gold_answer": "14"},
    1.0,
))

# === 空 ===
CASES.append((
    "v6-L-empty",
    "",
    {"gold_answer": "18"},
    -0.5,
))

# === 完全无关 ===
# M: "I don't know..." (2 句 + 没算式 + final 100 不在过程)
# reasoning_quality = 0.5*0.7 + 0.3*0.0 + 0.2*0.0 = 0.35
# reward = 0.15 + 0.0 + 0.25*0.35 = 0.2375
# 但 extract_answer 会从 "answer is 100" 提取 100, format 1.0
CASES.append((
    "v6-M-completely-unrelated",
    "I don't know the answer. The final answer is 100.",
    {"gold_answer": "18"},
    0.2375,
))

# === 答对 + 算式 + 5 步 ===
CASES.append((
    "v6-N-perfect-5steps",
    "Let's think step by step. There are 21 trees originally. "
    "After 3 are eaten, 21 - 3 = 18 remain. "
    "Then nothing is added. So 18 + 0 = 18. "
    "The final answer is 18.",
    {"gold_answer": "18"},
    1.0,
))

# === 步骤多 (8 步) 扣 step_count ===
# O: 8 步 + 算式对 + 答对, step_count = 1.0 - 0.1*(8-7) = 0.9
# reasoning_quality = 0.5*0.9 + 0.3*1.0 + 0.2*1.0 = 0.95
# reward = 0.15 + 0.60 + 0.25*0.95 = 0.9875
CASES.append((
    "v6-O-correct-8steps",
    "Step 1. There are 21 trees. Step 2. 3 are eaten. Step 3. So 21 - 3 = 18. "
    "Step 4. Let me verify. Step 5. 18 + 0 = 18. Step 6. The answer should be 18. "
    "Step 7. Let me check. Step 8. 18 = 18. The final answer is 18.",
    {"gold_answer": "18"},
    0.938,
))

# === 答对 + 1 步 (扣 step_count) ===
# P: "21 - 3 = 18. The final answer is 18." = 2 句
# step_count = 1.0 - 0.3*(3-2) = 0.7
# reasoning_quality = 0.5*0.7 + 0.3*1.0 + 0.2*1.0 = 0.85
# reward = 0.15 + 0.60 + 0.25*0.85 = 0.9625
CASES.append((
    "v6-P-correct-only-2steps",
    "21 - 3 = 18. The final answer is 18.",
    {"gold_answer": "18"},
    0.9625,
))

# === 答错 + 算式错 + final 错 (但 final 在过程) ===
# Q: 21 - 3 = 17, final 17
# reasoning_quality = 0.5*1.0 + 0.3*0.0 + 0.2*1.0 = 0.7
# reward = 0.15 + 0.0 + 0.25*0.7 = 0.325
CASES.append((
    "v6-Q-wrong-only-final-in-process",
    "Let's think step by step. 21 - 3 = 17. The final answer is 17.",
    {"gold_answer": "18"},
    0.325,
))

# === 答错 + 算式错 + final 不在过程 ===
# R: 21 - 3 = 17, final 100 (不在过程)
# reasoning_quality = 0.5*1.0 + 0.3*0.0 + 0.2*0.0 = 0.5
# reward = 0.15 + 0.0 + 0.25*0.5 = 0.275
CASES.append((
    "v6-R-wrong-final-not-in-process",
    "Let's think step by step. 21 - 3 = 17. The final answer is 100.",
    {"gold_answer": "18"},
    0.275,
))

# === 答对 + 算式 + final 在过程 (经典 8-shot 风格) ===
CASES.append((
    "v6-S-classic-8shot-style",
    "Let's think step by step. There are 15 trees originally. "
    "Then there were 21 trees after some more were planted. "
    "So there must have been 21 - 15 = 6. The final answer is 6.",
    {"gold_answer": "6"},
    1.0,
))

# === 答错 + 算式对 + 矛盾 (final 不在过程, 算式对但 final 错) ===
# T: 21 - 3 = 18, final 19 (不在过程算式中)
# reasoning_quality = 0.5*1.0 + 0.3*1.0 + 0.2*0.0 = 0.65
# reward = 0.15 + 0.0 + 0.25*0.65 = 0.3125
CASES.append((
    "v6-T-wrong-final-not-in-process-but-arith-ok",
    "Let's think step by step. 21 - 3 = 18. The final answer is 19.",
    {"gold_answer": "18"},
    0.4,
))


# 期望的 metrics schema 验证
EXPECTED_KEYS = [
    "format", "answer", "step_count_score", "n_steps",
    "numeric_correctness_score", "n_equations",
    "no_contradiction_score", "final_in_process",
    "reasoning_quality_score", "reward", "reason",
]


def main():
    print(f"=== Running {len(CASES)} v6 cases + 5 schema checks + 1 entry check ===")
    n_pass = 0
    total = len(CASES) + 5 + 1

    for case in CASES:
        name, completion, reference, expected = case
        if test(name, completion, reference, expected):
            n_pass += 1

    # 验证 metrics schema
    print()
    for case in CASES[:5]:
        name, completion, reference, _ = case
        if test_metrics_keys(f"{name} (schema check)", completion, reference, EXPECTED_KEYS):
            n_pass += 1

    # 验证 compute_reward entry
    completion = "Let's think. 21 - 3 = 18. The final answer is 18."
    out = compute_reward("chaingsm_8shot_cot", completion, "18", None)
    if isinstance(out, dict) and "score" in out:
        print(f"[PASS] compute_reward entry: score={out['score']:.3f}, keys={list(out.keys())}")
        n_pass += 1
    else:
        print(f"[FAIL] compute_reward entry: {type(out)} {out}")

    print(f"\n=== {n_pass}/{total} PASSED ===")
    if n_pass < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
