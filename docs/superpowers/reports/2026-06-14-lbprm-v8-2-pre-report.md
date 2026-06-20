# v8.2 训练预报告 (2026-06-14 16:35-17:00)

> 关联 spec: `docs/superpowers/specs/2026-06-14-lbprm-v8-2-design.md`
> 关联 plan: `docs/superpowers/plans/2026-06-14-lbprm-v8-2-plan.md`

## 1. 本阶段完成项

- ✅ v8.2 spec/plan 落盘
- ✅ TDD 修复 `preprocess_chaingsm_8shot_cot.py:50` 拼齐 reference 5 字段 (gold_expression / core_chain / distractor_chain / question / category)
- ✅ 新 parquet 重生成 (7419 条, ground_truth 6 字段齐全)
- ✅ TDD v8.1 reward 接通信号验证 (24/24 PASS, 含 4 个新 case F/G/H-good/H-bad)
- ✅ SFT ckpt eval 完成: original 0.3374 (SFT 摧毁 8-shot, 跟 v6 一样)
- ✅ 修 v8 GRPO 脚本: 加 4 个权重 env default
- ✅ GRPO 1000 步启动: 0.5B base + v8.1 5 机制 reward + 新 parquet

## 2. 根因复盘

v8.1 reward 设计方向正确，但 `preprocess_chaingsm_8shot_cot.py:50` 写 parquet 时只塞 `gold_answer`，漏掉 5 个关键字段 (gold_expression / core_chain / distractor_chain / question / category)。原始 14946.jsonl 100% 有这些字段 (7055/7055)。

**修复前 v8.1 reward 实际表现** (空 reference `{gold_answer: 18}` 下):
- Case F (path_competition, core chain 答对): 期望 1.0, 实际 **0.70** (n_irrelevant=2, penalty=0.20)
- Case G (independent_decoy, distractor 算式): 期望 <0.5, 实际 0.40
- Case H-bad (target_scope, after-donation 错): 期望 <0.4, 实际 0.30

**修复后 v8.1 reward 实际表现** (完整 reference 下):
- Case F: 1.000 ✅ n_irrelevant=0, penalty=0
- Case G: 0.300 ✅ penalty=0.15
- Case H-good: 0.850 ✅ penalty=0
- Case H-bad: 0.300 ✅

## 3. SFT ckpt eval (失败)

| 指标 | baseline (0.5B+8-shot) | SFT ckpt 1ep 8-shot | delta |
|---|---:|---:|---:|
| overall | 0.2438 | 0.2537 | +0.99pp |
| **original** | **0.4329** | **0.3374** | **-9.55pp** ❌ |
| independent_decoy | 0.1615 | 0.2269 | +6.54pp |
| attribute_mismatch | 0.1967 | 0.2222 | +2.55pp |
| path_competition | 0.2102 | 0.2212 | +1.10pp |
| target_scope_misalignment | 0.1689 | 0.2379 | +6.90pp |

**核心发现**: SFT 摧毁了 8-shot 能力 (跟 v6 sft_2epoch 一样). SFT 数据是 free-form CoT, 跟 8-shot 评测 prompt 协议没对齐, 模型"忘了"按 8-shot few-shot 模板格式回答.

**决策**: 跳过 SFT ckpt 起点, **0.5B base 直接跑 v8.2 GRPO 1000 步**. SFT 数据存在但当前方案不兼容, 留作 v8.3 迭代.

## 4. v8.2 GRPO 启动 (进行中)

- **PID**: 2016423
- **时间**: 17:00:33 启动
- **起点**: 0.5B base (HF 原始, 无 SFT)
- **数据**: 新 parquet (7419 条, 4 类变体均衡, ground_truth 6 字段)
- **reward**: v8.1 5 机制 (format 0.05 / answer 0.55 / c2a 0.20 / target 0.15 / length 0.05 / -0.30 penalty)
- **超参**: LR=5e-7, KL=0.02, ROLLOUT_N=4, MAX_STEPS=1000, SAVE_FREQ=200
- **预期**: 1000 步 × 13s = 3.6h + 5 × 5min eval = 4h

## 5. step 1-11 训练指标 (4:33 PM-4:35 PM)

| step | reward | answer | c2a | target | n_irrelevant | n_equations | penalty | time/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.388 | 0.3125 | 0.25 | 0.5 | 0.125 | 1.375 | 0.009 | 14.2s |
| 11 | 0.291 | 0.1875 | 0.0625 | 0.5 | 0.0 | 1.0 | 0.0 | 13.2s |

**信号全对**:
- ✅ format 1.0 (0.5B 收尾 marker 100% ok)
- ✅ target_recognition 0.5 (拿到 question 后中性)
- ✅ n_irrelevant 接近 0 (0.5B base 几乎没写 distractor, 起步就符合)
- ✅ penalty 0 (0 扣分)
- ⚠️ answer 0.19-0.31 (跟 baseline 0.18 一致, 训练初期正常)
- ⚠️ c2a 0.06-0.25 (链收尾对不上 extract_answer, 模型自由推理常常不严格)

## 6. 终止与判断

| 情况 | 动作 |
|---|---|
| GRPO step 200 original >= 0.46 | 视为达成, 后续观测收敛 |
| GRPO step 200 original < 0.43 (基线 = 0.4329) | 反思 reward 起点 (v7 是 0.4428) |
| GRPO step 200-400 持平且 < 0.46 | 反思: penalty 0.30 太重 → 改 0.20 |
| GRPO step 600 original >= 0.46 | 强制 stop at 600 |
| GRPO 1000 步仍未达成 | 迭代 v8.3: 加 8-shot 模板相似度 / chain coverage 机制 |

## 7. 监控点

- 训练 30 步滑窗: `outputs/train/local/grpo_verl_lbprm_v8/Qwen2.5-0.5B-Instruct/grpo_verl_v82_base/20260614_170033/metrics/train_metrics.jsonl`
- 5 节点 eval: `.../eval/step_0200/summary_by_category.json` (start ~17:30-18:00)
- 关键看 `original` 子项准确率

---

## 8. 后续 (实际跑完)

- ✅ 2026-06-14 17:00-20:20: v8.2 GRPO 1000 步训练完成
- ✅ 2026-06-14 20:23-20:36: 5 节点 eval 跑完
- ❌ v8.2 失败: best original 0.4003 (距 0.46 目标 -5.97pp)
- ✅ 详见 final 报告: `docs/superpowers/reports/2026-06-14-lbprm-v8-2-report.md`
- ✅ 评测摘要: `outputs/train/local/grpo_verl_lbprm_v8/Qwen2.5-0.5B-Instruct/grpo_verl_v82_base/20260614_170033/eval/latest_metrics.json`
