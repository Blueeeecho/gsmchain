"""Comprehensive behavioral test for LB-PRM reward (50+ cases).

Covers:
- Original 7 design cases (regression)
- AbstRaL-style chains (variable ref / recompute)
- Real SFT output patterns (repeated vars, numeric vars, 5-6 step chains)
- Edge cases (empty, malformed, fraction, decimal, deep subexpr)
- Last-step answer matching (float tolerance)
- Pollution / distractor-mimicking patterns
- Adversarial / tricky scenarios
"""
import sys, json
sys.path.insert(0, "/home/wwq416/snap/wwq/math-chain/train_pipeline")
from reward_chaingsm_lbprm_verl import score_response

# (name, completion_or_dict, gold, expected_reward)
# Note: expected values are COMPUTED from the formula; tests verify formula
# stability rather than hand-written guesses. The 7 design cases are pinned.
cases = []

# ===== Original 7 design cases (regression) =====
cases.append(("ORIG-A abst perfect", {
    "selected_steps":[{"variable":"out0","expression":"12/60","value":"0.2"},
                       {"variable":"out1","expression":"out0 * 50","value":"10"}],
    "final_expression":"out1","answer":"10"}, 10, 1.00))

cases.append(("ORIG-B recompute perfect", {
    "selected_steps":[{"variable":"minute_rate","expression":"12 / 60","value":"0.2"},
                       {"variable":"earnings","expression":"12 / 60 * 50","value":"10"}],
    "final_expression":"12 / 60 * 50","answer":"10"}, 10, 1.00))

cases.append(("ORIG-C dead middle", {
    "selected_steps":[{"variable":"minute_rate","expression":"12 / 60","value":"0.2"},
                       {"variable":"weather","expression":"70 + 5","value":"75"},
                       {"variable":"earnings","expression":"0.2 * 50","value":"10"}],
    "final_expression":"0.2 * 50","answer":"10"}, 10, 0.76))

cases.append(("ORIG-D self-consistent wrong", {
    "selected_steps":[{"variable":"minute_rate","expression":"12 / 60","value":"0.2"},
                       {"variable":"earnings","expression":"0.2 * 50","value":"10"},
                       {"variable":"after_snack","expression":"10 - 3","value":"7"}],
    "final_expression":"10 - 3","answer":"7"}, 10, 0.60))

cases.append(("ORIG-E pollution", {
    "selected_steps":[{"variable":"minute_rate","expression":"12 / 60","value":"0.2"},
                       {"variable":"earnings","expression":"0.2 * 50","value":"10"},
                       {"variable":"weird","expression":"10 + 999","value":"1009"}],
    "final_expression":"10 + 999","answer":"1009"}, 10, 0.60))

cases.append(("ORIG-F all garbage", {
    "selected_steps":[{"variable":"x","expression":"1 + 1","value":"2"},
                       {"variable":"y","expression":"3 + 3","value":"6"},
                       {"variable":"z","expression":"4 + 4","value":"8"}],
    "final_expression":"4 + 4","answer":"8"}, 8, 0.68))

cases.append(("ORIG-G invalid json", "not json at all", 10, -0.50))

# ===== AbstRaL-style variations =====
cases.append(("ABST-1 var ref in middle", {
    "selected_steps":[{"variable":"out0","expression":"20-5","value":"15"},
                       {"variable":"out1","expression":"out0 + 10","value":"25"}],
    "final_expression":"out1","answer":"25"}, 25, 1.00))

cases.append(("ABST-2 var ref in subexpr (3 steps)", {
    "selected_steps":[{"variable":"x","expression":"5","value":"5"},
                       {"variable":"y","expression":"x*2+3","value":"13"},
                       {"variable":"z","expression":"y-1","value":"12"}],
    "final_expression":"z","answer":"12"}, 12, 1.00))

# ===== Re-compute style =====
cases.append(("RECOMP-1 negative intermediate", {
    "selected_steps":[{"variable":"cost","expression":"100-150","value":"-50"},
                       {"variable":"loss","expression":"(100-150)*2","value":"-100"}],
    "final_expression":"(100-150)*2","answer":"-100"}, -100, 1.00))

cases.append(("RECOMP-2 fraction subexpr (BUG FIX)", {
    "selected_steps":[{"variable":"a","expression":"1/2","value":"0.5"},
                       {"variable":"b","expression":"0.5 * 4","value":"2.0"}],
    "final_expression":"0.5 * 4","answer":"2"}, 2, 1.00))

cases.append(("RECOMP-3 decimal value in subexpr", {
    "selected_steps":[{"variable":"a","expression":"10/3","value":"3.333"},
                       {"variable":"b","expression":"10/3 * 6","value":"20"}],
    "final_expression":"b","answer":"20"}, 20, 1.00))

# ===== Real SFT patterns: repeated variable names =====
cases.append(("REAL-1 repeated var 2-step (from 007218)", {
    "selected_steps":[{"variable":"Diego_payment","expression":"50000 - 1000","value":"49000"},
                       {"variable":"Diego_payment","expression":"(50000 - 1000) / 5","value":"9800"}],
    "final_expression":"(50000 - 1000) / 5","answer":"9800"}, 9800, 1.00))

cases.append(("REAL-2 repeated var 2-step wrong (from 005216)", {
    "selected_steps":[{"variable":"washed_hairs","expression":"32 / 2","value":"16"},
                       {"variable":"washed_hairs","expression":"32 + 32 / 2","value":"48"}],
    "final_expression":"32 + 32/2","answer":"48"}, 32, 0.60))

cases.append(("REAL-3 repeated var 3-step correct (from 006940)", {
    "selected_steps":[{"variable":"total_episodes","expression":"9 * 22","value":"198"},
                       {"variable":"total_hours","expression":"9 * 22 + 4","value":"202"},
                       {"variable":"total_hours","expression":"(9 * 22 + 4) * 0.5","value":"101"}],
    "final_expression":"((9*22)+4)*0.5","answer":"101"}, 101, 1.00))

# ===== Real SFT: numeric variable names =====
cases.append(("REAL-4 numeric var names (from 004625)", {
    "selected_steps":[{"variable":"140","expression":"2 * 2","value":"4"},
                       {"variable":"140","expression":"7 * 3","value":"21"},
                       {"variable":"350","expression":"2 * 2 + 7 * 3","value":"25"}],
    "final_expression":"2*2+7*3","answer":"25"}, 25, 1.00))

# ===== Real SFT: 5-6 step chains =====
cases.append(("REAL-5 3-step correct (from 004959)", {
    "selected_steps":[{"variable":"roses_sold","expression":"10 * 12","value":"120"},
                       {"variable":"daisies_sold","expression":"190 - 10 * 12","value":"70"},
                       {"variable":"daisies_per_bouquet","expression":"(190 - 10 * 12) / 10","value":"7"}],
    "final_expression":"(190 - 10*12) / 10","answer":"7"}, 7, 1.00))

cases.append(("REAL-6 2-step correct (from 002254)", {
    "selected_steps":[{"variable":"peach_cost","expression":"3 * 2","value":"6"},
                       {"variable":"remaining","expression":"20 - 3 * 2","value":"14"}],
    "final_expression":"20 - 3*2","answer":"14"}, 14, 1.00))

cases.append(("REAL-7 4-step correct (from 003971)", {
    "selected_steps":[{"variable":"beans_cans","expression":"1 + 2","value":"3"},
                       {"variable":"tomatoes_cans","expression":"2 * 1.5","value":"3"},
                       {"variable":"total_cans","expression":"1 + 2 + 2 * 1.5","value":"6"},
                       {"variable":"total_cans","expression":"6 * 2","value":"12"}],
    "final_expression":"6*2","answer":"12"}, 12, 1.00))

# ===== Last-step answer matching (float tolerance) =====
cases.append(("LST-1 last step val 2.0 vs answer 2", {
    "selected_steps":[{"variable":"a","expression":"1/2","value":"0.5"},
                       {"variable":"b","expression":"0.5*4","value":"2.0"}],
    "final_expression":"0.5*4","answer":"2"}, 2, 1.00))

cases.append(("LST-2 last step val 100 vs answer '100'", {
    "selected_steps":[{"variable":"a","expression":"50+50","value":"100"}],
    "final_expression":"a","answer":"100"}, 100, 1.00))

cases.append(("LST-3 last step val 100.0 vs answer 100", {
    "selected_steps":[{"variable":"a","expression":"50+50","value":"100.0"}],
    "final_expression":"a","answer":"100"}, 100, 1.00))

# ===== Pollution / distractor-mimicking =====
cases.append(("POLL-1 val matches next expr (coincidence OK)", {
    "selected_steps":[{"variable":"junk","expression":"1+1","value":"2"},
                       {"variable":"earn","expression":"2*5","value":"10"}],
    "final_expression":"2*5","answer":"10"}, 10, 1.00))

cases.append(("POLL-2 val matches next expr, wrong answer", {
    "selected_steps":[{"variable":"junk","expression":"1+1","value":"2"},
                       {"variable":"earn","expression":"2*5","value":"10"}],
    "final_expression":"2*5","answer":"10"}, 5, 0.60))

cases.append(("POLL-3 2 deads at end (V1)", {
    "selected_steps":[{"variable":"a","expression":"5","value":"5"},
                       {"variable":"junk1","expression":"99","value":"99"},
                       {"variable":"junk2","expression":"98","value":"98"}],
    "final_expression":"98","answer":"98"}, 98, 0.68))

cases.append(("POLL-4 first/last dead, middle live (W1)", {
    "selected_steps":[{"variable":"dead1","expression":"1","value":"1"},
                       {"variable":"mid","expression":"1+2","value":"3"},
                       {"variable":"dead2","expression":"99","value":"99"}],
    "final_expression":"99","answer":"99"}, 99, 0.76))

# ===== Edge cases =====
cases.append(("EDGE-1 empty string", "", 10, -0.50))
cases.append(("EDGE-2 wrong json shape", json.dumps({"foo":1}), 10, -0.50))
cases.append(("EDGE-3 missing fields", json.dumps({"selected_steps":[]}), 10, -0.50))
cases.append(("EDGE-4 empty steps", json.dumps({"selected_steps":[],"final_expression":"x","answer":"1"}), 1, -0.25))
cases.append(("EDGE-5 step not dict (string)", {
    "selected_steps":["a string step",
                       {"variable":"b","expression":"5","value":"5"}],
    "final_expression":"5","answer":"5"}, 5, 0.70))
cases.append(("EDGE-6 step is None", {
    "selected_steps":[None,
                       {"variable":"b","expression":"5","value":"5"}],
    "final_expression":"5","answer":"5"}, 5, 0.70))
cases.append(("EDGE-7 answer=None", {
    "selected_steps":[{"variable":"a","expression":"5","value":"5"}],
    "final_expression":"5","answer":None}, 5, -0.25))

# ===== Edge: deep subexpression =====
cases.append(("DEEP-1 value 5 deep in ((5+3)*2)-1", {
    "selected_steps":[{"variable":"a","expression":"5","value":"5"},
                       {"variable":"b","expression":"((5+3)*2)-1","value":"15"}],
    "final_expression":"((5+3)*2)-1","answer":"15"}, 15, 1.00))

cases.append(("DEEP-2 value 100 in nested (190 - 10*12)", {
    "selected_steps":[{"variable":"a","expression":"190","value":"190"},
                       {"variable":"b","expression":"190 - 10*12","value":"70"}],
    "final_expression":"190-10*12","answer":"70"}, 70, 1.00))

# ===== Edge: variable only in final =====
cases.append(("VAR-1 var in final only (H2)", {
    "selected_steps":[{"variable":"x","expression":"1+1","value":"2"},
                       {"variable":"y","expression":"3+3","value":"6"}],
    "final_expression":"x * 5","answer":"10"}, 10, 0.70))

cases.append(("VAR-2 var in next step's expr", {
    "selected_steps":[{"variable":"alpha","expression":"2+3","value":"5"},
                       {"variable":"beta","expression":"alpha*2","value":"10"}],
    "final_expression":"alpha*2","answer":"10"}, 10, 1.00))

# ===== Edge: 5-step with one dead =====
cases.append(("LONG-1 5-step with middle dead", {
    "selected_steps":[
        {"variable":"a","expression":"100","value":"100"},
        {"variable":"b","expression":"a-30","value":"70"},
        {"variable":"junk","expression":"99","value":"99"},
        {"variable":"d","expression":"b*2","value":"140"},
        {"variable":"e","expression":"d","value":"140"}],
    "final_expression":"e","answer":"140"}, 140, 0.8286))

cases.append(("LONG-2 5-step all live perfect", {
    "selected_steps":[
        {"variable":"a","expression":"100","value":"100"},
        {"variable":"b","expression":"a-30","value":"70"},
        {"variable":"c","expression":"b/2","value":"35"},
        {"variable":"d","expression":"c*2","value":"70"},
        {"variable":"e","expression":"d","value":"70"}],
    "final_expression":"e","answer":"70"}, 70, 1.00))

# ===== Edge: numeric value edge cases =====
cases.append(("NUM-1 value 0", {
    "selected_steps":[{"variable":"a","expression":"5-5","value":"0"},
                       {"variable":"b","expression":"0+10","value":"10"}],
    "final_expression":"0+10","answer":"10"}, 10, 1.00))

cases.append(("NUM-2 decimal 34.333 vs gold 34 (test gold too strict)", {
    "selected_steps":[{"variable":"a","expression":"206/6","value":"34.333"},
                       {"variable":"b","expression":"a","value":"34.333"}],
    "final_expression":"a","answer":"34.333"}, 34, 0.60))

cases.append(("NUM-3 long decimal 21.66666667 (SFT output)", {
    "selected_steps":[{"variable":"boxes_30","expression":"500 / 30","value":"16.66666667"},
                       {"variable":"total_jars","expression":"500 / 10 + 500 / 30","value":"21.66666667"}],
    "final_expression":"500/10+500/30","answer":"21.66666667"}, 21.66666667, 1.00))

# ===== Edge: floating-point value match in subexpr =====
cases.append(("FLOAT-1 value 5 in '5*3+2' subexpr", {
    "selected_steps":[{"variable":"a","expression":"5","value":"5"},
                       {"variable":"b","expression":"5*3+2","value":"17"}],
    "final_expression":"5*3+2","answer":"17"}, 17, 1.00))

cases.append(("FLOAT-2 value 100 in '100-30'", {
    "selected_steps":[{"variable":"a","expression":"100","value":"100"},
                       {"variable":"b","expression":"100-30","value":"70"}],
    "final_expression":"100-30","answer":"70"}, 70, 1.00))

# ===== Edge: single step =====
cases.append(("SINGLE-1 single step live", {
    "selected_steps":[{"variable":"a","expression":"5+5","value":"10"}],
    "final_expression":"a","answer":"10"}, 10, 1.00))

cases.append(("SINGLE-2 single step no var ref, value=answer", {
    "selected_steps":[{"variable":"a","expression":"10","value":"10"}],
    "final_expression":"10","answer":"10"}, 10, 1.00))

# ===== Edge: answer with comma =====
cases.append(("COMMA-1 chain with comma number", {
    "selected_steps":[{"variable":"a","expression":"1,000","value":"1000"},
                       {"variable":"b","expression":"a/4","value":"250"}],
    "final_expression":"a/4","answer":"250"}, 250, 1.00))

# ===== Edge: trailing whitespace =====
cases.append(("WS-1 trailing space in value", {
    "selected_steps":[{"variable":"a","expression":"  5  ","value":"5"},
                       {"variable":"b","expression":"5+5","value":"10"}],
    "final_expression":"5+5","answer":"10"}, 10, 1.00))

# ===== Edge: answer string vs int =====
cases.append(("ANS-1 answer '7.0' vs gold 7", {
    "selected_steps":[{"variable":"a","expression":"7","value":"7.0"}],
    "final_expression":"7","answer":"7.0"}, 7, 1.00))

# ===== Edge: chain with U1 single-char var no usage =====
cases.append(("SCRATCH-1 var x scratch step (U1)", {
    "selected_steps":[{"variable":"x","expression":"999","value":"999"},
                       {"variable":"answer","expression":"10","value":"10"}],
    "final_expression":"10","answer":"10"}, 10, 0.70))

# ===== Edge: chain uses ^ for power =====
cases.append(("POW-1 chain with ^ in expression", {
    "selected_steps":[{"variable":"a","expression":"2","value":"2"},
                       {"variable":"b","expression":"a^3","value":"8"}],
    "final_expression":"a^3","answer":"8"}, 8, 1.00))

# ===== Edge: empty final_expression =====
cases.append(("EMPTY-1 empty final_expression (rejected)", {
    "selected_steps":[{"variable":"a","expression":"5","value":"5"}],
    "final_expression":"","answer":"5"}, 5, -0.25))

# ===== Edge: extra fields ignored =====
cases.append(("EXTRA-1 extra fields in JSON", {
    "selected_steps":[{"variable":"a","expression":"5","value":"5"},
                       {"variable":"b","expression":"5+5","value":"10"}],
    "final_expression":"b","answer":"10","target":"x","weird":"y"}, 10, 1.00))

# ===== Edge: step value is float "0" =====
cases.append(("ZERO-1 step value '0.0' in later expr", {
    "selected_steps":[{"variable":"a","expression":"5-5","value":"0.0"},
                       {"variable":"b","expression":"0.0+10","value":"10"}],
    "final_expression":"0.0+10","answer":"10"}, 10, 1.00))

# ===== Edge: real SFT truncated answer =====
cases.append(("SFT-TRUNC-1 chain with target=null", {
    "selected_steps":[{"variable":"a","expression":"5+5","value":"10"}],
    "final_expression":"a","answer":"10","target":None}, 10, 1.00))

# ===== Edge: long chain 8 steps =====
cases.append(("LONG-3 8-step all live", {
    "selected_steps":[
        {"variable":"a","expression":"1","value":"1"},
        {"variable":"b","expression":"a+1","value":"2"},
        {"variable":"c","expression":"b+1","value":"3"},
        {"variable":"d","expression":"c+1","value":"4"},
        {"variable":"e","expression":"d+1","value":"5"},
        {"variable":"f","expression":"e+1","value":"6"},
        {"variable":"g","expression":"f+1","value":"7"},
        {"variable":"h","expression":"g+1","value":"8"}],
    "final_expression":"h","answer":"8"}, 8, 1.00))

# ===== Run =====
print(f"{'Case':<48} {'exp':>9} {'got':>9}   ok?")
print("-" * 80)
n_ok = 0
n_fail = 0
fails = []
for name, completion, gold, expected in cases:
    if isinstance(completion, str) and not completion.startswith("{"):
        raw = completion
    else:
        raw = json.dumps(completion) if not isinstance(completion, str) else completion
    reward, metrics = score_response(raw, {"gold_answer": str(gold)})
    ok = abs(reward - expected) < 1e-3
    if ok:
        n_ok += 1
    else:
        n_fail += 1
        fails.append((name, expected, reward, metrics))
    print(f"{name:<48} {expected:>9.4f} {reward:>9.4f}   {'OK' if ok else 'FAIL'}")
    if not ok:
        print(f"    metrics: {metrics}")
print("-" * 80)
print(f"Total: {n_ok}/{len(cases)} pass, {n_fail} fail")
if fails:
    print("\nFailures:")
    for n, e, g, m in fails:
        print(f"  {n}: expected {e:.4f}, got {g:.4f}")
        print(f"    {m}")
