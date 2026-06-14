"""v5 reward 行为测试。

继承 v3 全部 18 case (验证 v3 行为没破坏)
+ v5 新增 8 个 case (验证 v5 改写正确):
  - liveness 松绑: (a') value 子表达式匹配
  - liveness 松绑: (a'') value 出现在 final_expression 子表达式
  - length_bonus: n_steps ∈ [3, 6] 时 +0.05
  - length_bonus: n_steps 0/1/2/7/8 时 +0
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path("/home/wwq416/snap/wwq/math-chain/train_pipeline")))

from reward_chaingsm_lbprm_v5_verl import score_response

def S(d):
    return json.dumps(d, ensure_ascii=False)

# ---- v3 全部 18 case (与 v3 期望一致) ----
v3_cases = [
    ("A-perfect-gold",
        S({"selected_steps":[
            {"variable":"out0","expression":"12/60","value":"0.2"},
            {"variable":"out1","expression":"out0 * 50","value":"10"}],
         "final_expression":"out1","answer":"10"}), 10, 0.85, 1.01),
    ("B-recompute",
        S({"selected_steps":[
            {"variable":"minute_rate","expression":"12 / 60","value":"0.2"},
            {"variable":"earnings","expression":"12 / 60 * 50","value":"10"}],
         "final_expression":"12 / 60 * 50","answer":"10"}), 10, 0.85, 1.01),
    ("C-dead-middle",
        S({"selected_steps":[
            {"variable":"minute_rate","expression":"12 / 60","value":"0.2"},
            {"variable":"weather","expression":"70 + 5","value":"75"},
            {"variable":"earnings","expression":"0.2 * 50","value":"10"}],
         "final_expression":"0.2 * 50","answer":"10"}), 10, 0.65, 1.01),
    ("D-self-consistent-wrong",
        S({"selected_steps":[
            {"variable":"minute_rate","expression":"12 / 60","value":"0.2"},
            {"variable":"earnings","expression":"0.2 * 50","value":"10"},
            {"variable":"after_snack","expression":"10 - 3","value":"7"}],
         "final_expression":"10 - 3","answer":"7"}), 10, 0.15, 0.25),
    ("E-pollution-coincidence",
        S({"selected_steps":[
            {"variable":"minute_rate","expression":"12 / 60","value":"0.2"},
            {"variable":"earnings","expression":"0.2 * 50","value":"10"},
            {"variable":"weird","expression":"10 + 999","value":"1009"}],
         "final_expression":"10 + 999","answer":"1009"}), 1009, 0.65, 1.10),
    ("F-bad-json", "this is not json", 10, -0.5, 0.01),
    ("G-missing-fields", S({"selected_steps":[]}), 10, -0.5, 0.01),
    ("H-correct-no-var-ref",
        S({"selected_steps":[
            {"variable":"a","expression":"12/60","value":"0.2"},
            {"variable":"b","expression":"0.2*50","value":"10"}],
         "final_expression":"0.2*50","answer":"10"}), 10, 0.85, 1.05),
    ("I-hacking-classic-wrong-ans",
        S({"selected_steps":[
            {"variable":"breakfast","expression":"16-3","value":"13"},
            {"variable":"baking muffins","expression":"4","value":"4"},
            {"variable":"remaining eggs","expression":"16-3-4","value":"7"},
            {"variable":"selling price","expression":"2","value":"2"},
            {"variable":"total earnings","expression":"7*2","value":"14"}],
         "final_expression":"14","answer":"14"}), 18, 0.10, 0.30),
    ("J-mostly-dead-but-correct",
        S({"selected_steps":[
            {"variable":"unknown_a","expression":"100+100","value":"200"},
            {"variable":"unknown_b","expression":"999+1","value":"1000"},
            {"variable":"target_step","expression":"200-1000","value":"-800"}],
         "final_expression":"-800","answer":"-800"}), -800, 0.80, 1.10),
    ("K-correct-no-calc-no-ref",
        S({"selected_steps":[
            {"variable":"x","expression":"never_evaluated","value":"0"},
            {"variable":"y","expression":"also_bad","value":"0"}],
         "final_expression":"3","answer":"3"}), 3, 0.75, 0.85),
    ("L-empty-steps",
        S({"selected_steps":[],"final_expression":"3","answer":"3"}), 3, -0.5, 0.01),
    ("M-answer-text-not-num",
        S({"selected_steps":[{"variable":"x","expression":"3","value":"3"}],
          "final_expression":"3","answer":"three"}), 3, 0.30, 0.55),
    ("N-final-uneval-step-ok",
        S({"selected_steps":[{"variable":"x","expression":"3","value":"3"}],
          "final_expression":"x x x","answer":"3"}), 3, 0.65, 1.01),
    ("O-decimal-rounding",
        S({"selected_steps":[
            {"variable":"rate","expression":"10/3","value":"3.333"},
            {"variable":"ans","expression":"3.333*3","value":"10"}],
         "final_expression":"3.333*3","answer":"10"}), 10, 0.85, 1.01),
    ("P-all-uneval-wrong-ans",
        S({"selected_steps":[
            {"variable":"x","expression":"abc","value":"foo"},
            {"variable":"y","expression":"def","value":"bar"}],
         "final_expression":"baz","answer":"999"}), 1, 0.10, 0.30),
    ("QA-correct", S({"selected_steps":[
        {"variable":"a","expression":"12/60","value":"0.2"},
        {"variable":"b","expression":"0.2*50","value":"10"}],
        "final_expression":"0.2*50","answer":"10"}), 10, 0.85, 1.05),
    ("QA-wrong", S({"selected_steps":[
        {"variable":"a","expression":"12/60","value":"0.2"},
        {"variable":"b","expression":"0.2*50","value":"10"}],
        "final_expression":"0.2*50","answer":"999"}), 10, 0.30, 0.45),
]

# ---- v5 新增 8 case ----
# V5-1: liveness (a') 命中 → liveness 更高
#   step[0] value=0.2 出现在 step[1] expression "0.2 * 50" 的子表达式里
#   没有 variable 引用, v3 不会让 step[0] 活, v5 会
v5_cases = [
    # V5-1: v3 case H (no var ref) + v5 (a') 命中: liveness 从 0.5 → 1.0
    #   step[0] value=0.2 出现在 step[1] expression "0.2*50" 子表达式
    #   liveness: step[0]=True (v5 a' 命中), step[1]=True (v3 a 命中)
    #   chain_quality = 0.5*1.0 + 0.3*1.0 + 0.2*1.0 = 1.0
    #   total = 0.2 + 0.55 + 0.25*1.0 = 1.0
    #   n_steps=2, length_bonus=0
    ("V5-1-subexpr-liveness",
        S({"selected_steps":[
            {"variable":"a","expression":"12/60","value":"0.2"},
            {"variable":"b","expression":"0.2*50","value":"10"}],
         "final_expression":"0.2*50","answer":"10"}), 10, 0.95, 1.01),

    # V5-2: (a'') 命中: step[0] value 出现在 final_expression 子表达式
    #   step[0] value=5 出现在 final "10 + 5"
    #   但 step[0] variable "a" 不在后续 step 也不在 final → v3 不活
    #   v5: 0.2 出现在 final "0.2*50" 子表达式 (跟 V5-1 同场景, 但 final)
    ("V5-2-final-subexpr-liveness",
        S({"selected_steps":[
            {"variable":"a","expression":"100+100","value":"200"},
            {"variable":"b","expression":"50*1","value":"50"}],
         "final_expression":"200 + 50","answer":"250"}), 250, 0.95, 1.01),

    # V5-3: length_bonus 命中: n_steps=3, 全对
    #   total = 0.2 + 0.55 + 0.25*1.0 + 0.05 = 1.05
    ("V5-3-length-bonus-3steps",
        S({"selected_steps":[
            {"variable":"a","expression":"12/60","value":"0.2"},
            {"variable":"b","expression":"0.2*50","value":"10"},
            {"variable":"final","expression":"10","value":"10"}],
         "final_expression":"10","answer":"10"}), 10, 1.00, 1.10),

    # V5-4: length_bonus 命中: n_steps=6, 全对
    #   total = 0.2 + 0.55 + 0.25*1.0 + 0.05 = 1.05
    ("V5-4-length-bonus-6steps",
        S({"selected_steps":[
            {"variable":"a","expression":"12/60","value":"0.2"},
            {"variable":"b","expression":"0.2*50","value":"10"},
            {"variable":"c","expression":"10+1","value":"11"},
            {"variable":"d","expression":"11-1","value":"10"},
            {"variable":"e","expression":"10","value":"10"},
            {"variable":"f","expression":"10","value":"10"}],
         "final_expression":"10","answer":"10"}), 10, 1.00, 1.10),

    # V5-5: length_bonus 不命中: n_steps=2, 跟 V3 case A 同分数 (0.95)
    ("V5-5-no-length-bonus-2steps",
        S({"selected_steps":[
            {"variable":"a","expression":"12/60","value":"0.2"},
            {"variable":"b","expression":"0.2*50","value":"10"}],
         "final_expression":"0.2*50","answer":"10"}), 10, 0.90, 1.00),

    # V5-6: length_bonus 不命中: n_steps=7, 跟 n_steps=6 比应低 0.05
    #   n=7 时 bonus=0, n=6 时 bonus=0.05
    #   total = 0.2 + 0.55 + 0.25*1.0 + 0 = 1.0
    ("V5-6-no-length-bonus-7steps",
        S({"selected_steps":[
            {"variable":"a","expression":"12/60","value":"0.2"},
            {"variable":"b","expression":"0.2*50","value":"10"},
            {"variable":"c","expression":"10+1","value":"11"},
            {"variable":"d","expression":"11-1","value":"10"},
            {"variable":"e","expression":"10","value":"10"},
            {"variable":"f","expression":"10","value":"10"},
            {"variable":"g","expression":"10","value":"10"}],
         "final_expression":"10","answer":"10"}), 10, 0.95, 1.05),

    # V5-7: 答对 + 链混乱 (step 越界 12), n_steps=2 (无 bonus)
    #   测 no_degenerate 扣分仍生效
    ("V5-7-degenerate-no-bonus",
        S({"selected_steps":[
            {"variable":"a","expression":"never","value":"0"},
            {"variable":"b","expression":"also_bad","value":"0"}],
         "final_expression":"3","answer":"3"}), 3, 0.65, 0.85),

    # V5-8: 答对 (999==999) + n_steps=3 (length_bonus 命中) + chain 内部 step value 不真引用
    #   bonus 不应该让答错的高分 (c2a=0, chain_quality 不计入)
    #   c2a=0 (eval(final "3")=3 ≠ 999), chain_quality=0.5, total=0.2+0.55+0+0.05=0.80
    ("V5-8-wrong-with-length-bonus",
        S({"selected_steps":[
            {"variable":"a","expression":"1","value":"1"},
            {"variable":"b","expression":"2","value":"2"},
            {"variable":"c","expression":"3","value":"3"}],
         "final_expression":"3","answer":"999"}), 999, 0.70, 0.90),
]

cases = v3_cases + v5_cases

# ============= 跑 =============
print(f"=== Running {len(cases)} v5 cases ({len(v3_cases)} v3 + {len(v5_cases)} v5 new) ===\n")
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
        "chain_quality_score", "gated_chain_quality",
        "n_live", "n_steps", "length_bonus", "length_bonus_flag", "reward",
    ]}
    print(f"[{status}] {name:35s}  r={r:.3f}  expected=[{lo:.2f}, {hi:.2f}]")
    print(f"         metrics: {json.dumps(metrics_summary, ensure_ascii=False)}")

print(f"\n=== {'ALL PASS' if all_pass else 'SOME FAILED'} ===")
sys.exit(0 if all_pass else 1)
