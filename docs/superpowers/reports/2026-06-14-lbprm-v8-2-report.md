# v8.2 训练结果报告 (2026-06-14)

> 关联 spec: `docs/superpowers/specs/2026-06-14-lbprm-v8-2-design.md`
> 关联 plan: `docs/superpowers/plans/2026-06-14-lbprm-v8-2-plan.md`
> 关联 pre-report: `docs/superpowers/reports/2026-06-14-lbprm-v8-2-pre-report.md`
> Run 目录: `outputs/train/local/grpo_verl_lbprm_v8/Qwen2.5-0.5B-Instruct/grpo_verl_v82_base/20260614_170033/`
> 跑批时间: 2026-06-14 17:00:33 ~ 20:20:13 (训练 3:19:40) + 20:23:26 ~ 20:36:03 (5 节点 eval 12:37)

## 1. 训练配置 (实际)

| 项 | 值 |
|---|---|
| 起点 | `Qwen2.5-0.5B-Instruct` HF 原始 (无 SFT) |
| 测试集 | `gsm8k_test_clean.jsonl` 5467 条 |
| 训练集 | `verl_grpo_train_8shot_cot.parquet` 7419 条 (8-shot CoT prompt) |
| Parquet schema | 6 字段: gold_answer / gold_expression / core_chain / distractor_chain / question / category |
| MAX_STEPS | 1000 |
| SAVE_FREQ | 200 (5 个 eval 节点: 200/400/600/800/1000) |
| ROLLOUT_N | 4 |
| TEMPERATURE | 0.9 |
| ACTOR_LR | 5e-7 |
| KL_COEF | 0.02 |
| REWARD | v8.1 5 机制 + irrelevant_eq 扣分 (接通信号版) |
| 权重 | format=0.05, answer=0.55, chain_to_answer_check=0.20, target_recognition=0.15, chain_length_consistency=0.05, -0.30×irrelevant_eq_ratio |
| 单步耗时 | ~12-14s/step (1010 tok/prom × 4 rollouts) |
| 总训练时长 | 3.3h |
| GPU | RTX 5090 32GB |

## 2. 测试集评测结果 (主线指标)

### 2.1 5 节点详细数据

| 评测点 | overall 5467 | **original 1319** | indep_decoy 1102 | attr_mismatch 1017 | path_comp 999 | target_scope 1030 |
|---|---:|---:|---:|---:|---:|---:|
| **baseline (0.5B + 8-shot)** | 0.2438 | **0.4329** (571) | 0.1615 (178) | 0.1967 (200) | 0.2102 (210) | 0.1689 (174) |
| **v7 step_100 (历史 best)** | 0.2425 | **0.4428** (584) | 0.1488 (164) | 0.2075 (211) | 0.1932 (193) | 0.1689 (174) |
| v8.2 step_200 | 0.2308 (1262) | **0.4003** (528) | 0.1434 (158) | 0.2094 (213) | 0.1942 (194) | 0.1641 (169) |
| v8.2 step_400 | 0.2634 (1440) | 0.3616 (477) | 0.2114 (233) | 0.2458 (250) | 0.2503 (250) | 0.2233 (230) |
| v8.2 step_600 | 0.3020 (1651) | 0.3480 (459) | 0.2668 (294) | 0.3009 (306) | 0.2973 (297) | 0.2864 (295) |
| v8.2 step_800 | **0.3170** (1733) | 0.3480 (459) | 0.2967 (327) | 0.3078 (313) | 0.3133 (313) | 0.3117 (321) |
| v8.2 step_1000 | 0.3163 (1729) | 0.3548 (468) | 0.2913 (321) | 0.3058 (311) | 0.3113 (311) | 0.3087 (318) |
| **目标** | — | **>= 0.46** (607/1319) | — | — | — | — |

### 2.2 关键发现

- ❌ **v8.2 训练未达成 original >= 0.46**
- ⚠️ **best by original = step_200: 0.4003** (-3.26pp vs baseline 0.4329)
- ⚠️ **best by overall = step_800: 0.3170** (+7.32pp vs baseline 0.2438)
- ✅ **4 类变体齐涨**: indep_decoy 16.15% → 29.13% (+12.98pp), attr_mismatch 19.67% → 30.58% (+10.91pp), path_comp 21.02% → 31.13% (+10.11pp), target_scope 16.89% → 30.87% (+13.98pp)
- ❌ **original 子集跌**: 43.29% → 35.48% (-7.81pp), 全部 5 节点都低于 baseline
- ⚠️ **核心 trade-off 现象**: 4 类变体涨 13pp 同时 original 跌 8pp, 模型在"抗干扰"和"无干扰推理"之间出现明显的能力转移

### 2.3 Best 节点选择 (按主线指标)

| 指标 | best 节点 | 数值 | 距 0.46 目标 |
|---|---|---:|---:|
| original (主线) | step_200 | 0.4003 | -6.0pp |
| overall (辅助) | step_800 | 0.3170 | — |
| 4 类变体均值 | step_800 | 0.3074 | — |
| answer 子项 reward | step_970-999 | 0.319 | — |

**结论**: 没有任何节点达成 original >= 0.46, v8.2 失败。

## 3. 训练内 reward 分布 (30 步滑窗均值)

| 窗口 | reward | answer | c2a | target | n_irrelevant | n_equations | penalty | kl_loss | entropy | pg_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-30 | 0.286 | 0.121 | 0.342 | 0.548 | 0.27 | 1.82 | 0.030 | 0.00091 | 0.4956 | 0.0319 |
| 30-59 | 0.278 | 0.100 | 0.369 | 0.542 | 0.34 | 1.89 | 0.032 | 0.00097 | 0.4783 | 0.0348 |
| 100-129 | 0.309 | 0.150 | 0.431 | 0.548 | 0.44 | 2.21 | 0.042 | 0.00450 | 0.5025 | 0.0534 |
| 200-229 | 0.314 | 0.113 | 0.558 | 0.537 | 0.43 | 2.42 | 0.040 | 0.01942 | 0.4298 | 0.0391 |
| 300-329 | 0.331 | 0.108 | 0.604 | 0.550 | 0.38 | 2.51 | 0.032 | 0.03340 | 0.4436 | 0.0692 |
| 400-429 | 0.419 | 0.212 | 0.721 | 0.523 | 0.23 | 2.23 | 0.021 | 0.05189 | 0.3937 | 0.0095 |
| 500-529 | 0.432 | 0.221 | 0.794 | 0.540 | 0.34 | 2.86 | 0.029 | 0.05526 | 0.3488 | 0.0230 |
| 600-629 | 0.455 | 0.231 | 0.856 | 0.529 | 0.27 | 2.91 | 0.023 | 0.06709 | 0.3510 | 0.0298 |
| 700-729 | 0.466 | 0.267 | 0.860 | 0.517 | 0.31 | 2.82 | 0.030 | 0.06317 | 0.3475 | 0.0245 |
| 800-829 | 0.487 | 0.300 | 0.856 | 0.521 | 0.30 | 2.69 | 0.028 | 0.07532 | 0.3321 | 0.0486 |
| 900-929 | 0.482 | 0.267 | 0.896 | 0.523 | 0.25 | 2.94 | 0.022 | 0.06704 | 0.3273 | 0.0381 |
| 970-999 | 0.504 | 0.319 | 0.890 | 0.537 | 0.33 | 2.88 | 0.030 | 0.07260 | 0.3279 | 0.0295 |

### 3.1 reward 子项变化 (1-30 → 970-999)

| 子项 | 起步 | 末段 | delta | 解读 |
|---|---:|---:|---:|---|
| total reward | 0.286 | 0.504 | +0.218 | ✅ 大涨 |
| answer (0.55 权重) | 0.121 | 0.319 | +0.198 | ⚠️ 涨但 base 0.18 都未破 |
| chain_to_answer_check (0.20) | 0.342 | 0.890 | +0.548 | ✅ **接通信号, 数据修复直接证据** |
| target_recognition (0.15) | 0.548 | 0.537 | -0.011 | ❌ **没接通信号** (详见 §4.2) |
| chain_length_consistency (0.05) | 1.0 | 1.0 | 0 | ⚠️ 全程 1.0 中性 (v8.1 留的中性) |
| n_irrelevant (扣分基础) | 0.27 | 0.33 | +0.06 | ⚠️ 模型开始写"看似相关"算式 |
| n_equations | 1.82 | 2.88 | +1.06 | ✅ 算式数涨 (跟 c2a 涨相关) |
| penalty (扣分) | 0.030 | 0.030 | 0 | ❌ **几乎没扣分** (详见 §4.3) |
| kl_loss | 0.00091 | 0.07260 | +0.0717 | ⚠️ **80x 涨, policy 严重偏离 ref** |
| entropy | 0.4956 | 0.3279 | -0.168 | ⚠️ policy 收窄 34% |

### 3.2 response_length 变化

| 窗口 | resp_len | clip_ratio | 解读 |
|---|---:|---:|---|
| 1-30 | 232.0 | 0.029 | 起步, 部分撞 512 限制 |
| 200-229 | 148.9 | 0.002 | 学短 |
| 400-429 | 114.2 | 0.000 | 稳定 100-120 tokens |
| 600-629 | 120.4 | 0.000 | 稳定 |
| 800-829 | 117.4 | 0.002 | 稳定 |
| 970-999 | 119.7 | 0.002 | 稳定 |

**信号**: 模型从 232 tokens 收到 114-120 tokens, **学会了"短推理解答"**。c2a 涨到 0.89 直接验证: 短推理 + 链收尾 marker → extract_answer 命中率高。

### 3.3 accuracy (训练 batch 内 answer 正确率, 30 步滑窗)

| 窗口 | reward/accuracy | 解读 |
|---|---:|---|
| 1-30 | 0.121 | 起点 baseline 0.18 之下 |
| 200-229 | 0.113 | 平台 |
| 400-429 | 0.212 | 突破 0.18 |
| 600-629 | 0.231 | 持续涨 |
| 800-829 | 0.300 | +12pp |
| 970-999 | 0.319 | +20pp |

**信号**: 训练 batch 内的 answer 正确率从 12% 涨到 32%, 涨 20pp. **但 eval 上 original 只从 40% 跌到 35%**, 说明训练 reward 跟 eval 上 original 出现 **decoupling**。

## 4. v8.2 失败根因分析 (新发现)

### 4.1 核心 Trade-off 现象 (新发现)

**v7 跟 v8.2 都失败的根因是同一个, 但表现不同**:

| | v7 (3 子项) | v8.2 (5 机制) |
|---|---|---|
| 4 类变体涨 | 0-3pp (无) | **+10-14pp** (显著) |
| original 涨/跌 | -2-3pp (跌) | **-7.81pp** (大幅跌) |
| numeric 子项 reward | +16.7pp (抢梯度) | (v8.2 已砍) |
| answer 子项 reward | -0.01pp (不动) | +19.8pp (涨) |
| c2a 子项 reward | n/a | +54.8pp (接通) |

**v8.2 真正"接通信号"了**: 5 机制全部涨, 4 类变体涨 10-14pp. **但这正是问题**:
- 训练 reward 视角下, 模型"学会了"避开 distractor 算式 + 链收尾 + 答短
- eval 上 original (无干扰) 跌 8pp, 说明模型**为了避开"假干扰"而过度怀疑"真推理"**
- **0.5B 容量在 8-shot 协议下, 抗干扰能力跟无干扰推理能力是零和博弈**

### 4.2 target_recognition 信号没接上

训练内 target_recognition 30 步滑窗 0.537-0.550, **几乎没有变化** (1-30 = 0.548, 970-999 = 0.537). 原因:
- v8.1 实现里 target_recognition 对 number-only 答案 (如 "70000") 总是返回 0.0, 因为 "profit" 关键词不在 text
- TDD case H-good (target_scope + core chain + 答对, 期望 >0.8) 实测 0.85, 但其中 target 子项 0.0 (number-only bug)
- v8.2 沿用 v8.1 实现, 没修这个 bug, target_recognition 在 number-only 答案上"全错"
- 5 节点 eval 上 target_scope 涨 14pp 是 c2a + answer 的贡献, **target_recognition 几乎没贡献**

### 4.3 penalty 几乎没扣分

`reward_components/penalty/mean` 在 970-999 滑窗 = 0.030, **跟 1-30 起步的 0.030 几乎一样**:
- 起步 0.5B base 几乎不写 distractor 算式 (n_irrelevant 0.27), penalty 自然小
- 训练后 n_irrelevant 0.33 (几乎不变), penalty 0.030
- **预期**: GRPO 应该让模型写更"对齐 gold chain"的算式, penalty 上升
- **实际**: 模型学会了"少写算式" (n_equations 1.82→2.88 是 8-shot CoT 协议 baseline, 不算多写)
- **irrelevant_eq 机制没起到 push 模型主动对齐 gold chain 的作用**

### 4.4 KL 锚定失效 (跟 v7 一致)

- kl_loss 0.00091 → 0.07260, **80x 涨**
- KL_COEF=0.02 起步, 跟 v7 一样
- **但 eval 表现**: 4 类变体涨, original 跌, KL 没把 policy 锚到 ref
- policy 严重偏离 (entropy 0.50→0.33), 但**不是 mode collapse** (entropy 还健康)

### 4.5 response_length 缩短副作用

模型 response 从 232 tokens 收到 114-120 tokens:
- 表面看是好事 (短推理, c2a 涨)
- 副作用: **原 original 题需要 2-3 步推理的, 短回答容易跳步**
- sample 证据: step_1000 original 答错的样本 "Janet's ducks" (gold 18), 模型输出 "She eats 3 * 7 = 21 ... 4 * 7 = 28 ... 16 - 21 - 28 = -33 ... since she can't sell negative, sells all 16 = 16 * 2 = 32"
  - 模型把"每天" 错算成"每周" (乘 7), 短回答里没有 self-check 机制

## 5. 跟 v7/v8.1 对比

| 维度 | v7 (3 子项) | v8.1 (数据 schema 缺) | v8.2 (数据 schema 修) |
|---|---|---|---|
| 起点 | 0.5B base | 0.5B base | 0.5B base |
| 数据 schema 字段 | gold_answer only | gold_answer only | **6 字段齐全** |
| c2a 接通信号 | n/a | n/a | **0.34→0.89 ✅** |
| target 接通信号 | n/a | n/a | **0.55→0.54 ❌** |
| irrelevant 接通信号 | n/a | n/a | **0.27→0.33 ⚠️** |
| best original | 0.4428 (step_100) | 0.3374 (SFT ckpt, 失败) | **0.4003 (step_200)** |
| best overall | 0.2647 (step_400) | n/a | **0.3170 (step_800)** |
| 4 类变体涨 | 0-3pp | n/a | **+10-14pp** |
| 距 0.46 目标 | -1.72pp | n/a | **-5.97pp** |

**v8.2 进步 vs v7**:
- 4 类变体齐涨 +10-14pp (v7 是 0-3pp)
- 砍 numeric 抢梯度 (v8.2 整体 reward 涨 0.22, v7 答错区间是 0.20-0.40 平台)
- 接通 c2a 信号 (chain 收尾对齐 extract_answer)
- overall 涨 +7.32pp (baseline 0.2438 → step_800 0.3170)

**v8.2 退步 vs v7**:
- original 跌 7.81pp (v7 -2-3pp)
- best ckpt 是 step_200 (跟 v7 一样, 都是训练早期)
- 训练步数 1000 没带来收敛

## 6. v8.2 反思 (预测 vs 实测)

### 6.1 v8.2 预报告预测

> 预期: GRPO 1000 步 × 13s = 3.6h + 5 × 5min eval = 4h
> 预期: step 200 eval original >= 0.46 → 视为达成

**实测**: 训练 3.3h (跟预测 3.6h 接近), eval 13min (比预期 25min 短, 因为 0.5B 小). **best 0.4003, 距 0.46 -5.97pp**, 预测失败.

### 6.2 预测错的原因

1. **5 机制奖励"信号接上"不等于"涨 original"**: v8.2 把 c2a/n_equations/answer/4 类变体全部接上, 但 original 反而跌
2. **4 类变体涨跟 original 是 trade-off**: 0.5B 容量限制下, 模型在抗干扰和无干扰推理之间被迫二选一
3. **target_recognition bug 没修**: v8.2 沿用 v8.1 的 number-only 答案 bug, target_scope 涨 14pp 但 target 子项几乎没贡献
4. **0.5B + 8-shot 协议天花板确认 (第三次)**: v3 JSON 协议 30%, v7 8-shot 协议 44%, v8.2 8-shot 协议 35% (跌). 不同协议都是同一根因: **0.5B 容量在自由推理 + 干扰识别双任务下被榨干**

## 7. v9 方向 (基于 v8.2 真实数据)

### 候选 A: 修 target_recognition bug + 加 8-shot 模板相似度 (最小改动)
- **思路**: 修 v8.1 target_recognition 的 number-only 答案 bug, 加 "匹配任意 unit 关键词 OR answer 数字本身 → 0.5 中性" 规则
- **附加**: 加 8-shot 模板相似度子项, 鼓励模型输出"跟 8-shot 协议对齐"的算式结构
- **理由**: target_recognition 信号要真接通; 8-shot 协议对齐是关键
- **风险**: 加相似度子项需要 8-shot example 的 reference 编码, 工程中等
- **时间**: 1h 实现 + 3.3h 训练 + 0.5h eval = ~5h

### 候选 B: SFT + 8-shot 模板相似度 reward (回到 SFT 路线)
- **思路**: v8.2 验证 0.5B base + 8-shot 协议 GRPO 是 trade-off, 改 SFT 1 epoch 8-shot 协议对齐 + GRPO 优化
- **理由**: SFT 摧毁 8-shot 是因为 SFT 数据跟评测 prompt 协议不匹配, 如果 SFT 数据本身就是 8-shot CoT 协议, 配合 GRPO 应该能涨
- **风险**: SFT 数据要重新做, 工程大
- **时间**: 半天数据准备 + 1.5h SFT + 3.3h GRPO = ~5.5h

### 候选 C: 承认 0.5B 容量上限, 转向 1.5B (用户授权外)
- **思路**: 1.5B + 8-shot CoT 原生 original = 0.7225, 远超 0.46 目标
- **理由**: v3/v7/v8.2 三次证明 0.5B 在 ChainGSM 干扰任务下是天花板
- **风险**: 1.5B 训练时间 / 显存 / 远程依赖 都不在当前本地硬件
- **时间**: 取决于训练资源, 可能 1-2 天

### 候选 D: 接受 0.5B 0.40 真实上限, 转做多 checkpoint 集成
- **思路**: 集成 v7 (0.4428) + v8.2 step_200 (0.4003) + SFT ckpt (0.3374), 投票 / 加权
- **理由**: 单 ckpt 都不达, 集成可能突破
- **风险**: 集成方案需要推理时间 3x, 工程复杂
- **时间**: 4-6h 实现 + 评测

## 8. 决策 (v8.2 -> v9)

**采用候选 A (修 target_recognition bug + 8-shot 模板相似度)**:
- 理由: v8.2 把 c2a 接入信号是 v8.2 唯一成功, 修 target bug 后 5 机制能全接入
- v8.2 的 4 类变体涨 +10-14pp 已经验证 reward 设计方向, 增量改进比推倒重来成本低

**或者候选 D (ckpt 集成)**: 如果用户优先拿结果, 集成 v7 + v8.2 step_200 是成本最低的 "best of both worlds"。

**v9 时间预算** (候选 A):
- target_recognition bug fix + 8-shot 模板相似度子项: 1h
- TDD case 补充 (target_scope + number-only 答案 + 8-shot 协议相似度): 0.5h
- GRPO 1000 步重训: 3.3h
- 5 节点 eval: 0.5h
- v9 报告: 0.5h
- **总: ~5.5h**

## 9. 决策记录 (2026-06-14)

- ✅ 2026-06-14 13:33-15:21: v7 500 步训练 (失败, best 0.4428)
- ✅ 2026-06-14 15:22-15:49: v7 5 节点 eval
- ✅ 2026-06-14 15:50-15:55: v7 报告 + v8 spec/plan/reward
- ❌ v7 失败 (numeric 抢 answer 梯度, 0.5B 0.44 上限)
- ✅ 2026-06-14 15:55-16:05: v8 TDD/reward (4 子项)
- ✅ 2026-06-14 16:08-16:10: v8.1 5 机制 reward (TDD 20/20 PASS)
- ✅ 2026-06-14 16:10-16:32: SFT 1 epoch 训完 (eval 0.3374 摧毁 8-shot)
- ✅ 2026-06-14 16:35-16:44: v8.2 spec/plan/pre-report 落盘
- ✅ 2026-06-14 16:41: preprocess_chaingsm_8shot_cot.py:50 修数据 schema (5 字段齐全)
- ✅ 2026-06-14 16:42: 重生 verl_grpo_train_8shot_cot.parquet (7419 条, 6 字段)
- ✅ 2026-06-14 16:44-16:50: SFT ckpt eval 跑完 (original 0.3374 失败)
- ✅ 2026-06-14 17:00-20:20: v8.2 GRPO 1000 步训练 (3.3h)
- ✅ 2026-06-14 20:23-20:36: v8.2 5 节点 eval 跑完 (13min)
- ❌ v8.2 失败: best original 0.4003 (距 0.46 -5.97pp)
- ⏳ 2026-06-14 20:36+: v8.2 报告 (本文件) + 等待 v9 决策
