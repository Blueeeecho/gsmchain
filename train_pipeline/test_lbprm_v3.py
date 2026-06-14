"""v3 reward 行为测试。completion 一律传 json 字符串。"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path("/home/wwq416/snap/wwq/math-chain/train_pipeline")))

from reward_chaingsm_lbprm_v3_verl import score_response

def S(d):
    """把 dict 转成 json 字符串。"""
    return json.dumps(d, ensure_ascii=False)

cases = []

# --- A. 完美 gold 链 ---
cases.append(("A-perfect-gold",
    S({"selected_steps":[
        {"variable":"out0","expression":"12/60","value":"0.2"},
        {"variable":"out1","expression":"out0 * 50","value":"10"}],
     "final_expression":"out1","answer":"10"}), 10, 0.85, 1.01))

# --- B. recompute 风格 ---
cases.append(("B-recompute",
    S({"selected_steps":[
        {"variable":"minute_rate","expression":"12 / 60","value":"0.2"},
        {"variable":"earnings","expression":"12 / 60 * 50","value":"10"}],
     "final_expression":"12 / 60 * 50","answer":"10"}), 10, 0.85, 1.01))

# --- C. 中间插死 step(原 v2 期望 0.76,v3 因 chain 仍能推出 answer 而得 0.85+) ---
cases.append(("C-dead-middle",
    S({"selected_steps":[
        {"variable":"minute_rate","expression":"12 / 60","value":"0.2"},
        {"variable":"weather","expression":"70 + 5","value":"75"},
        {"variable":"earnings","expression":"0.2 * 50","value":"10"}],
     "final_expression":"0.2 * 50","answer":"10"}), 10, 0.65, 1.01))

# --- D. 答错(7 vs gold=10),但 chain 内部自洽:v3 应得 ~0.35(答错被掐) ---
cases.append(("D-self-consistent-wrong",
    S({"selected_steps":[
        {"variable":"minute_rate","expression":"12 / 60","value":"0.2"},
        {"variable":"earnings","expression":"0.2 * 50","value":"10"},
        {"variable":"after_snack","expression":"10 - 3","value":"7"}],
     "final_expression":"10 - 3","answer":"7"}), 10, 0.15, 0.25))

# --- E. 答对 by coincidence(原 v2 0.60,v3 因 c2a=1 而得高分) ---
cases.append(("E-pollution-coincidence",
    S({"selected_steps":[
        {"variable":"minute_rate","expression":"12 / 60","value":"0.2"},
        {"variable":"earnings","expression":"0.2 * 50","value":"10"},
        {"variable":"weird","expression":"10 + 999","value":"1009"}],
     "final_expression":"10 + 999","answer":"1009"}), 1009, 0.65, 1.01))

# --- F. JSON 崩坏 ---
cases.append(("F-bad-json", "this is not json", 10, -0.5, 0.01))

# --- G. format 字段缺失 ---
cases.append(("G-missing-fields", S({"selected_steps":[]}), 10, -0.5, 0.01))

# --- H. 答对但 chain 没真引用 step variable ---
# step1 var "a" 不在 "0.2*50" 里 → step1 死
# c2a=1, step_calc=1, liveness=0.5, no_degen=1
# chain_quality = 0.5*0.5 + 0.3*1 + 0.2*1 = 0.65
# total = 0.2 + 0.55 + 0.25*0.65 = 0.91
cases.append(("H-correct-no-var-ref",
    S({"selected_steps":[
        {"variable":"a","expression":"12/60","value":"0.2"},
        {"variable":"b","expression":"0.2*50","value":"10"}],
     "final_expression":"0.2*50","answer":"10"}), 10, 0.85, 0.96))

# --- I. 经典 hacking:16-3-4 = 7,7*2=14,gold=18,模型写 14 ---
# c2a=1, step_calc=1(全部 step value 跟 expression 算出来一致), liveness 全死
# chain_quality = 0.5*0 + 0.3*1 + 0.2*1 = 0.5
# total = 0.2 + 0.55 + 0.25*0.5 = 0.875
# 答错(14 vs 18)→ answer=0,format=1 → total = 0.2 + 0 + 0.125 = 0.325
cases.append(("I-hacking-classic-wrong-ans",
    S({"selected_steps":[
        {"variable":"breakfast","expression":"16-3","value":"13"},
        {"variable":"baking muffins","expression":"4","value":"4"},
        {"variable":"remaining eggs","expression":"16-3-4","value":"7"},
        {"variable":"selling price","expression":"2","value":"2"},
        {"variable":"total earnings","expression":"7*2","value":"14"}],
     "final_expression":"14","answer":"14"}), 18, 0.10, 0.30))

# --- J. 答对,大部分 step 死,c2a=1 ---
cases.append(("J-mostly-dead-but-correct",
    S({"selected_steps":[
        {"variable":"unknown_a","expression":"100+100","value":"200"},
        {"variable":"unknown_b","expression":"999+1","value":"1000"},
        {"variable":"target_step","expression":"200-1000","value":"-800"}],
     "final_expression":"-800","answer":"-800"}), -800, 0.80, 1.01))

# --- K. 答对,step_calc 0, liveness 0, no_degen 1 ---
# c2a=1, chain_quality=0+0+0.2=0.2
# total = 0.2 + 0.55 + 0.25*0.2 = 0.80
cases.append(("K-correct-no-calc-no-ref",
    S({"selected_steps":[
        {"variable":"x","expression":"never_evaluated","value":"0"},
        {"variable":"y","expression":"also_bad","value":"0"}],
     "final_expression":"3","answer":"3"}), 3, 0.75, 0.85))

# --- L. 空 steps(format fail) ---
cases.append(("L-empty-steps",
    S({"selected_steps":[],"final_expression":"3","answer":"3"}), 3, -0.5, 0.01))

# --- M. answer 文本"three",gold=3,answer=0 ---
cases.append(("M-answer-text-not-num",
    S({"selected_steps":[{"variable":"x","expression":"3","value":"3"}],
      "final_expression":"3","answer":"three"}), 3, 0.30, 0.55))

# --- N. final 不可 eval 但 step[0] 推出 answer(c2a 路径 2) ---
cases.append(("N-final-uneval-step-ok",
    S({"selected_steps":[{"variable":"x","expression":"3","value":"3"}],
      "final_expression":"x x x","answer":"3"}), 3, 0.65, 1.01))

# --- O. 3.333 decimal rounding ---
cases.append(("O-decimal-rounding",
    S({"selected_steps":[
        {"variable":"rate","expression":"10/3","value":"3.333"},
        {"variable":"ans","expression":"3.333*3","value":"10"}],
     "final_expression":"3.333*3","answer":"10"}), 10, 0.85, 1.01))

# --- P. final 不可 eval 且 step 全不可 eval,答错,应被掐 ---
# c2a=0, chain_quality 0, total = 0.2 + 0 + 0 = 0.2(应得 -0.25 = 0.2 * invalid_reward * 0.5 = -0.25)
# 但 invalid_reward * 0.5 = -0.25
cases.append(("P-all-uneval-wrong-ans",
    S({"selected_steps":[
        {"variable":"x","expression":"abc","value":"foo"},
        {"variable":"y","expression":"def","value":"bar"}],
     "final_expression":"baz","answer":"999"}), 1, 0.10, 0.30))

# --- Q. **关键对比**:同一个 chain 答对 vs 答错,reward 差距应≥0.3 ---
# 答对版 total ≈ 0.91,答错版 total ≈ 0.325
# 答对 - 答错 ≈ 0.59(说明 v3 真的把梯度推回答对上)
cases.append(("QA-correct", S({"selected_steps":[
    {"variable":"a","expression":"12/60","value":"0.2"},
    {"variable":"b","expression":"0.2*50","value":"10"}],
    "final_expression":"0.2*50","answer":"10"}), 10, 0.85, 0.96))
cases.append(("QA-wrong", S({"selected_steps":[
    {"variable":"a","expression":"12/60","value":"0.2"},
    {"variable":"b","expression":"0.2*50","value":"10"}],
    "final_expression":"0.2*50","answer":"999"}), 10, 0.30, 0.45))

# ============= 跑 =============
print(f"=== Running {len(cases)} v3 cases ===\n")
all_pass = True
for name, completion, gold, lo, hi in cases:
    r, m = score_response(completion, {"gold_answer": gold})
    ok = lo <= r <= hi
    status = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    metrics_summary = {k: m.get(k) for k in [
        "format", "answer", "chain_to_answer_ok",
        "causal_liveness_score", "step_calc_score", "no_degenerate_score",
        "chain_quality_score", "gated_chain_quality", "reward",
    ]}
    print(f"[{status}] {name:30s}  r={r:.3f}  expected=[{lo:.2f}, {hi:.2f}]")
    print(f"         metrics: {json.dumps(metrics_summary, ensure_ascii=False)}")

print(f"\n=== {'ALL PASS' if all_pass else 'SOME FAILED'} ===")
sys.exit(0 if all_pass else 1)
