# v7 训练结果报告 (2026-06-14)

> 关联 spec: `docs/superpowers/specs/2026-06-14-lbprm-v7-design.md`
> 关联 plan: `docs/superpowers/plans/2026-06-14-lbprm-v7-plan.md`
> Run 目录: `outputs/train/local/grpo_verl_lbprm_v7/Qwen2.5-0.5B-Instruct/grpo_verl_v7/20260614_133352/`
> 跑批时间: 2026-06-14 13:33 ~ 15:21 (训练 1:48) + 15:22-15:49 (5 节点 eval 27min)

## 1. 训练配置 (实际)

| 项 | 值 |
|---|---|
| 起点 | `Qwen2.5-0.5B-Instruct` HF 原始 (无 SFT) |
| 测试集 | `gsm8k_test_clean.jsonl` 5467 条 |
| 训练集 | `verl_grpo_train_8shot_cot.parquet` 7419 条 (8-shot CoT prompt) |
| MAX_STEPS | 500 |
| SAVE_FREQ | 100 (5 个 eval 节点: 100/200/300/400/500) |
| ROLLOUT_N | 4 |
| TEMPERATURE | 0.9 |
| ACTOR_LR | 5e-7 |
| KL_COEF | 0.02 |
| REWARD | v7 公式 (3 子项) |
| 权重 | format=0.10, answer=0.70, numeric_correctness=0.20 |
| 单步耗时 | ~12.5s/step (1300 tok/prom × 4 rollouts) |
| GPU | RTX 5090 32GB |

## 2. 测试集评测结果 (主线指标)

| 评测点 | overall 5467 | **original 1319** | indep_decoy 1102 | attr_mismatch 1017 | path_comp 999 | target_scope 1030 |
|---|---:|---:|---:|---:|---:|---:|
| **baseline (0.5B + 8-shot)** | 0.2438 | **0.4329** (571) | 0.1615 (178) | 0.1967 (200) | 0.2102 (210) | 0.1689 (174) |
| **v7 step_100 (最佳)** | 0.2425 | **0.4428** (584) | 0.1488 (164) | 0.2075 (211) | 0.1932 (193) | 0.1689 (174) |
| v7 step_200 | 0.2398 | 0.4011 (529) | 0.1434 (158) | 0.1967 (200) | 0.2012 (201) | 0.1641 (169) |
| v7 step_300 | 0.2526 | 0.4049 (534) | 0.1461 (161) | 0.2104 (214) | 0.2192 (219) | 0.1835 (189) |
| v7 step_400 | 0.2647 | 0.4094 (540) | 0.1543 (170) | 0.2232 (227) | 0.2322 (232) | 0.1961 (202) |
| v7 step_500 | 0.2627 | 0.4056 (535) | 0.1506 (166) | 0.2202 (224) | 0.2332 (233) | 0.1942 (200) |
| **目标** | — | **>= 0.46** | — | — | — | — |

**关键发现**:
- ❌ **v7 训练未达成 original >= 0.46**
- ⚠️ **最佳 ckpt = step_100: 0.4428** (+0.99pp vs baseline 0.4329), 距 0.46 还差 1.72pp
- ❌ step_200-500 全部跌 2.3-3.2pp (policy 退化, reward hacking numeric 子项)
- ✅ step_400/500 4 类变体涨 1-3pp, 但 original 仍然 -2.4pp

## 3. 训练内 reward 分布 (30 步滑窗均值)

| 窗口 | reward | answer | numeric | entropy | kl_loss |
|---|---:|---:|---:|---:|---:|
| 1-30 | 0.280 | 0.123 | 0.473 | 0.485 | 0.00089 |
| 31-60 | 0.284 | 0.113 | 0.530 | 0.497 | 0.00107 |
| 61-90 | 0.284 | 0.106 | 0.546 | 0.482 | 0.00150 |
| 91-120 | 0.312 | 0.142 | 0.563 | 0.474 | 0.00275 |
| 121-150 | 0.319 | 0.142 | 0.599 | 0.463 | 0.00598 |
| 151-180 | 0.289 | 0.100 | 0.597 | 0.460 | 0.01022 |
| 181-210 | 0.307 | 0.125 | 0.598 | 0.464 | 0.01448 |
| 211-240 | 0.291 | 0.100 | 0.604 | 0.478 | 0.01547 |
| 241-270 | 0.336 | 0.158 | 0.628 | 0.425 | 0.01519 |
| 271-300 | 0.309 | 0.119 | 0.630 | 0.436 | 0.01652 |
| 301-330 | 0.323 | 0.144 | 0.612 | 0.429 | 0.01633 |
| 331-360 | 0.345 | 0.165 | 0.640 | 0.434 | 0.01459 |
| 361-380 | 0.305 | 0.108 | 0.640 | 0.460 | 0.01580 |
| 381-500 (含) | 0.290 | 0.113 | 0.557 | 0.430 | 0.01456 |

**关键观察**:
- ✅ **numeric_correctness 大涨**: 0.473 → 0.640 (+16.7pp), 0.5B base 算式能力确实在涨
- ❌ **answer 子项不动**: 0.10-0.16 区间震荡, 没突破 baseline 0.18
- ⚠️ **kl_loss 涨 17x**: 0.0009 → 0.016, policy 严重偏离 ref
- ✅ **entropy 0.43-0.50 健康**, 不需担忧 mode collapse

## 4. v7 失败根因分析

### 4.1 Reward formula 偏 numeric
v7 公式 0.10·format + 0.70·answer + 0.20·numeric_correctness **表面** answer 权重 0.70 最高, 但**实际信号**:
- answer 是 0/1 二值, 在 0.18 baseline 下, 答对时 +0.70, 答错时 0, 梯度信号 = 0.70 × P(correct) × adv
- numeric 是 0-1 连续, 0.34 baseline 答对时 +0.07, 答错时 -0.07, 梯度信号更密
- **GRPO 优化方向**: 模型学会"算式对" + "答案错" (numeric 0.7 也能拿 0.14 reward, 不需要答对)
- step_200+ numeric 涨到 0.65, 但 answer 退到 0.10, policy "hacks" numeric 子项

### 4.2 0.5B base 容量限制
- 8-shot CoT 原生 0.4329 (baseline)
- 0.5B 在 answer=0.70 强信号下, **0.18 → 0.20 都难**
- 同样在 0.5B + 8-shot CoT 协议下, 自由推理的算式 / 推理能力是上限
- v3 GRPO 4 类变体齐涨 30% (JSON 协议), 但 original 没破 30% — 跟 v7 答不出 original 是同一原因

### 4.3 KL 锚定失效
- KL 0.02 起步, kl_loss 0.001 起步 (policy 几乎不动 ref)
- 500 步后 kl_loss 0.017 (17x 涨), policy 严重偏离
- **但 eval 表现跌** — KL 没用, policy overfit 到 numeric 子项

## 5. v7 反思 (vs v6 prereflex 预测)

v6 prereflex 预测: "0.5B + 8-shot CoT + GRPO 500 步后, 现实上限: original 50-55% / overall 35-40%"

**实测**: best 0.4428 / 0.2425 — **远低于预测**

**为什么预测错**:
- v6 prereflex 假设 0.5B 在 8-shot CoT 上有 "30-35% overall 上限" 空间
- 实测: 0.5B 自由推理在 8-shot 协议下 0.18 answer 已经接近上限
- **0.5B 容量 vs 8-shot CoT 推理深度 不匹配**: 模型能模仿 8-shot 的算式结构, 但 reasoning depth 不够

## 6. v8 方向 (基于 v7 真实数据)

### 候选 A: SFT 1 epoch 8-shot CoT 起点 (最可能突破)
- **思路**: 用 `sft_train_v2.jsonl` (14528 条) 给 0.5B base 做 1 epoch SFT
- **理由**: SFT 让模型先"学会" 8-shot CoT 协议, GRPO 在此基础上优化
- **数据已就绪**: `sft_train_v2.jsonl` (free-form CoT, 不带 8-shot examples)
- **风险**: SFT 1 epoch 可能破坏 8-shot 能力 (v6 prereflex 已证 sft_2epoch 摧毁 8-shot, 但 sft_v2 是 1 epoch + CoT 协议, 不同)
- **时间**: 30 min SFT 准备 + 1.5h SFT 训练 + 1.5h GRPO 1000 步 = ~3.5h

### 候选 B: 改 reward 公式 (砍 numeric, 提 answer)
- **思路**: reward = 0.05·format + 0.90·answer + 0.05·length (鼓励答对)
- **理由**: v7 失败根因是 numeric 抢了 answer 梯度
- **风险**: answer 0/1 二值信号稀疏, GRPO 训练不稳定
- **时间**: 1.5h GRPO 1000 步

### 候选 C: 加 step_count reasoning 0.10 (v6 砍掉的拿回来)
- **思路**: reward = 0.10·format + 0.60·answer + 0.20·numeric + 0.10·step_count (>=2 算式给分)
- **理由**: step_count 鼓励 "展开算式" 而非 "凑算式"
- **风险**: 0.5B 在 step_count 0.74 接近满分, 区分度低
- **时间**: 1.5h GRPO 1000 步

### 候选 D: 换 LLM-as-judge reward (大改)
- **思路**: 用 vLLM judge 答对率, 取代规则 reward
- **风险**: 0.5B 自己 judge 自己, 信号噪声大; 工程复杂
- **时间**: 4h+ 实现

## 7. 决策 (v7 -> v8)

**采用 v8 候选 A (SFT 1 epoch 8-shot CoT 起点) + 候选 B reward (砍 numeric 提 answer)**:
- **Step 1**: SFT 1 epoch with `sft_train_v2.jsonl` (14528 条), 起点 = 0.5B base
- **Step 2**: SFT ckpt → GRPO 1000 步, reward = 0.05·format + 0.85·answer + 0.10·numeric (砍 numeric 抢梯度)
- **Step 3**: 5 个 eval 节点 step_200/400/600/800/1000
- **Step 4**: original >= 0.46 达成 / 不达成就续训 v8.1

**时间预算**: SFT 1.5h + GRPO 1.5h + eval 25min = 3.5h

## 8. 决策记录 (2026-06-14)

- ✅ 2026-06-14 13:33-15:21: v7 500 步训练完成 (1:48)
- ✅ 2026-06-14 15:22-15:49: v7 5 节点 eval 完成, best = step_100 original 0.4428
- ❌ v7 失败: 距 0.46 目标还差 1.72pp
- ⏳ 2026-06-14 15:50: v8 spec/plan 落盘 (SFT 1 epoch + 砍 numeric)
- ⏳ 2026-06-14 16:00-17:30: v8 SFT 1 epoch 训练
- ⏳ 2026-06-14 17:30-19:00: v8 GRPO 1000 步
- ⏳ 2026-06-14 19:00-19:30: v8 eval 5 节点 + 报告
