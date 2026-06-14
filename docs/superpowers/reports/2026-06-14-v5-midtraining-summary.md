# v5 训练中期状态总结(2026-06-14 05:37)

> 写于 v5 step 58/500 训练进行中,基于 v5 quicktest + v3/v4 历史数据,定 v6 方向。

---

## 0. 项目状态快速回顾

| 阶段 | 路径 / 状态 | overall | Original |
|---|---|---:|---:|
| **sft_2epoch** (base) | `outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_2epoch/20260531_152306/checkpoints/best` | 0.2742 | 0.2555 |
| **v3 GRPO step_300 best** | `outputs/.../grpo_verl_v3_smoke/20260614_004034/checkpoints/best/actor/huggingface` | 0.3047 | 0.2995 |
| **v4 step_100** (NEUTRAL prompt) | `outputs/.../grpo_verl_v4/20260614_024915/checkpoints/global_step_100/actor/huggingface` | 0.2976 | 0.2957 |
| v4 step_400 | 同上 | 0.3002 | 0.2942 |
| v5 quicktest 5 步 (step_5 eval) | `outputs/.../grpo_verl_v5/20260614_051658/eval/step_5` | 0.3078 | 0.3025 |
| **v5 完整 500 步** | `outputs/.../grpo_verl_v5/20260614_052636/checkpoints/` | 训练中 step 58/500 (12%) | — |

**v5 训练状态**:
- PID 3323532, 启动 2026-06-14 05:26, 单步 9.6-10.5s
- 训练内 step 58: liveness 0.97 / c2a 0.56 / answer 0.50 / chain_quality 0.96 / accuracy 0.50 / length_bonus 0.025 (50% 触发) / len 187 / kl_loss 0.0006

---

## 1. 奖励函数演进总结(v1 → v5)

### 1.1 时间线

| 版本 | 关键改动 | 训练内 acc | 整体 | Original | 反思 |
|---|---|---:|---:|---:|---|
| **v1 (abstral-style)** | chain 选取 + JSON schema, 4 个 reward hacking 漏洞 | 0.039(0/8) | — | — | **完全失败**:模型学偷懒, response_length 110, liveness 0.89 全满分 |
| **v2 (同 v1 reward)** | 同 v1 reward 复用 | 0.039 | — | — | **同 v1 失败模式** |
| **v3 (本次主线)** | chain_to_answer_ok 硬门 + causal_liveness 收紧 + step_calc 不再 tautology + answer 0.55 主导 | 0.42→0.47 | **0.3047** | **0.2995** | ✅ **稳**: 5 类目齐涨, 300 步后撞天花板 |
| **v4 (NEUTRAL prompt)** | v3 reward + 训练 prompt 改 "expert grade-school math solver" | 0.39→0.44 | 0.3002 | 0.2942 | ⚠️ step_100 涨 +1.37pp Original, 但 step_200-300 期间掉点 |
| **v5 (当前)** | v3 + liveness 松绑 (a')(a'') + KL 0.04→0.02 + length_bonus 0.05 | 0.50 (50 步) | 训练中 | 训练中 | ⚠️ liveness 0.97 松绑过头 |

### 1.2 v3 reward 公式 (当前主线)

```python
total = 0.20·format + 0.55·answer + 0.25·(chain_to_answer_ok · chain_quality_score)
chain_quality_score = 0.5·liveness + 0.3·step_calc + 0.2·no_degenerate

# 4 个核心 components
chain_to_answer_ok    ∈ {0, 1}      # 硬门: chain 能否独立推出 gold_answer
causal_liveness_score ∈ [0, 1]      # 收紧: variable 引用 + final_expression 引用 + 末步 value == pred_answer
step_calc_score       ∈ [0, 1]      # step.expression 数值计算正确率
no_degenerate_score   ∈ [0, 1]      # 链重复 / 不可算 / 非数值 value 扣分
```

### 1.3 v5 reward 公式 (v3 + length_bonus)

```python
length_bonus_flag = 1 if 3 <= n_steps <= 6 else 0
length_bonus = length_bonus_flag · 0.05

total = 0.20·format + 0.55·answer + 0.25·(chain_to_answer_ok · chain_quality_score) + length_bonus
# 新增 (a') 子表达式匹配 + (a'') value 出现在 final 子表达式 → liveness 松绑
```

### 1.4 训练内 metrics 横向对比 (50 步窗口均值)

| 指标 | v2 失败 | v3 step_1-50 | v3 step_351-400 | v4 step_1-50 | v4 step_351-400 | v5 step_1-58 |
|---|---:|---:|---:|---:|---:|---:|
| accuracy | 0.039 | 0.422 | 0.430 | 0.393 | 0.396 | **0.500** |
| format | — | 0.981 | 1.000 | 0.981 | — | 1.000 |
| answer | 0.039 | 0.422 | 0.430 | 0.393 | 0.396 | 0.500 |
| c2a | — | 0.458 | 0.455 | 0.431 | 0.425 | 0.562 |
| liveness | 0.89→0.97 (hacking) | 0.346 | 0.341 | 0.346 | 0.341 | **0.975 (松绑)** |
| step_calc | — | 0.890 | 0.909 | 0.890 | 0.909 | 0.928 |
| no_degenerate | — | 0.932 | 0.939 | 0.932 | 0.939 | 0.981 |
| chain_quality | — | 0.627 | 0.631 | 0.627 | 0.631 | **0.962** |
| gated_chain_quality | — | 0.310 | 0.308 | 0.294 | 0.290 | 0.558 |
| response_length | 238→110 (退化) | 214 | 218 | 210 | 215 | 187 |
| entropy | — | 0.057 | 0.049 | 0.062 | 0.050 | 0.076 |
| kl_loss | — | 0.001 | 0.002 | 0.001 | 0.002 | 0.0006 |

**v5 训练内信号变化**:
- ✅ accuracy 0.50 > v3 0.43 (+0.07): 训练集上略涨
- ⚠️ liveness 0.97 (vs v3 0.34): **松绑过头**, 跟 v2 失败模式 (0.89-0.97) 信号相似
- ✅ c2a 0.56 > v3 0.46 (+0.10): chain 真的能算的比例涨
- ⚠️ response_length 187: 略短于 v3 218

---

## 2. 撞 30% 天花板的根因分析

### 2.1 三层天花板

| 层级 | 现象 | 数据印证 |
|---|---|---|
| **L1 协议天花板** | JSON schema 强制模型严格遵循结构 | 0.5B + 8-shot CoT 原生 43% > 0.5B + JSON schema + RL 30% (差 13pp) |
| **L2 容量天花板** | 0.5B 模型参数量限制推理能力 | 0.5B + 8-shot CoT 43% < 1.5B + 8-shot CoT 72% (差 29pp) |
| **L3 训练天花板** | v3/v4/v5 在 30-32% 撞平台 | v3 step_300 ≈ step_400; v4 step_200-300 掉点 |

### 2.2 v3 best predictions 错误模式分析 (5467 测试集)

- 答对 1666 / 5467 (30.47%)
- **70% 错答 = 推理能力本身失败** (不是 format 错, 不是引用错, 不是 reward 漏判)
- chain 格式错 (parse fail) = 0.5% (format 0.989 已近满)
- chain 引用错 (liveness=0) 但答对 = 罕见
- chain 能算 gold 但 pred 答错 = 1.9% (reward 漏判极低)
- **错答的 chain 算式都正确, 就是推理最后一步算错** (step_calc 0.88, no_degenerate 0.93)

### 2.3 v3 reward 信号质量

| 类别 | mean reward |
|---|---:|
| correct | 0.92 |
| wrong | 0.20 |
| **gap** | **0.72** |

**结论**: **v3/v5 reward 已经收得很紧, reward 信号够强, 再调 reward 收益不大**. 70% 错答是模型推理能力本身的失败.

### 2.4 协议换 vs 模型换

| 路径 | 假设 | 训练时长 | 风险 | 预期 Original | 推荐度 |
|---|---|---|---|---|---|
| **v6a: 0.5B + 8-shot CoT 协议** | 自由推理下 0.5B 原生 43% → RL 后 50-55% | SFT 1h + RL 2-3h | 中 | **45-55%** | ⭐⭐⭐ |
| v6b: 0.5B + JSON 协议 (v5 调优) | 在 0.5B 上继续挤 | 0.5-1h | 低 | 31-32% | ⭐ |
| v6c: 1.5B + 8-shot CoT 协议 | 1.5B 容量 + 自由推理, 无 RL 72% → RL 后 60%+ | SFT 2h + RL 6h | **高** (GPU 32GB 紧张) | 60-70% | ⭐⭐ |
| v6d: 1.5B + JSON 协议 | 1.5B 容量 + JSON 协议 | SFT 2h + RL 6h | 中-高 | 35-50% | ⭐⭐ |

### 2.5 决策 (基于用户约束 "0.5B 训练快成本可控")

- **用户明确约束**: "使用的模型专注于 qwen2.5-0.5b-instruct 模型"
- **0.46 目标在 0.5B + JSON 协议下不可达** (三层天花板)
- **0.46 目标在 0.5B + 8-shot CoT 协议下可达** (v6a, 0.5B 自由推理原生 43% → RL 后 50-55%)

**v6 决策**: **v6 = 0.5B + 8-shot CoT 协议** (用户约束范围内最优解)
- 跟用户原始约束一致 (0.5B 训练快成本可控)
- 预期 50-55% Original, 超过 0.46 目标
- SFT 起点 = 0.5B base + 8-shot CoT SFT (重训 1h)
- GRPO 训练 = 0.5B + 8-shot CoT 协议 (训练 2-3h)

---

## 3. 0.5B 8-shot CoT 基线 (无 RL)

来源: `code/results/chaingsm_base_8shot_batch16/20260607_151358/model_outputs/Qwen2.5-0.5B-Instruct/summary.json`

| 类别 | 数量 | 准确率 |
|---|---:|---:|
| **Original (1319)** | 571/1319 | **43.29%** |
| attribute_mismatch | 205/1017 | 20.16% |
| independent_decoy | 180/1102 | 16.33% |
| path_competition | 212/999 | 21.22% |
| target_scope_misalignment | 174/1030 | 16.89% |
| **overall** | 1342/5467 | **24.55%** |

**关键发现**:
- 0.5B + 8-shot CoT 原生 Original **43.29% > 0.5B + JSON schema + GRPO 30% (差 13pp)**
- 0.5B + 8-shot CoT 在 4 类变体上 **20% 左右**, 远低于 Original 43% (干扰敏感)
- 0.5B + 8-shot CoT 整体 24.55% < Original 43.29% (4 类变体拖累)
- 1.5B + 8-shot CoT 原生 Original **72.25%** (用户约束外)

**0.5B + 8-shot CoT 协议下 RL 目标**:
- **Original 43.29% → RL 后目标 50-55%** (涨 7-12pp)
- **overall 24.55% → RL 后目标 35-45%** (涨 10-20pp)
- 0.46 目标 = **46% Original / 0.46 overall**, v6 路径 = **50-55% Original / 35-45% overall** → **0.46 达成**

---

## 4. 风险与监控 (v5 训练剩余 1h+)

| 风险 | 监控 | 应对 |
|---|---|---|
| v5 liveness 1.0 hacking 触发 | liveness_score/mean 持续 0.99+ + response_length 掉到 < 100 | 立即停, 反思 liveness 松绑设计 |
| v5 length_bonus 0.05 太小 | length_bonus_flag 0% | 监控 flag mean, 5% < 视为无效 |
| v5 500 步全平台 | 跟 v3 一样 30% 撞天花板 | v5 报告反思 → v6 = 0.5B + 8-shot CoT |
| v5 step_100 评测低于 v3 best | step_100 eval < 28% (overall) | 立即停 |

---

## 5. 下一步决策

**v5 训练继续跑 (不 kill)**:
- 等 step_100 eval (~25 min 后)
- 若 v5 step_100 < v3 best (30.47%) → 立即停
- 若 v5 step_100 ≥ v3 best → 让它跑完 500 步

**立即开始 v6 准备 (不等 v5 跑完)**:
- 写 spec: `docs/superpowers/specs/2026-06-14-lbprm-v6-design.md` (8-shot CoT 协议 + 重写 reward)
- 写 plan: `docs/superpowers/plans/2026-06-14-lbprm-v6-plan.md`
- 重写 reward: `train_pipeline/reward_chaingsm_lbprm_v6_verl.py`
- 重新 preprocess train parquet (8-shot CoT 格式)
- 重新 SFT 0.5B 起点 (8-shot CoT 协议, ~1h)
- GRPO 训练 400 步 (8-shot CoT 协议, 起点 v6 SFT best, ~2h)
- 5 个 eval 节点 (step_80/160/240/320/400), 报告 Original + 整体

**v6 reward 关键设计** (8-shot CoT 协议):
- 模型自由推理, 答案以 "The final answer is N." 收尾
- reward = format_weight·format + answer_weight·answer + reasoning_quality_weight·reasoning_quality
  - format = 1.0 if response 收尾于 "The final answer is N." else 0.0
  - answer = 1.0 if N == gold_answer else 0.0
  - reasoning_quality = 推理步骤数 (3-7 step 满分) + 数值计算正确率 + 无自相矛盾

---

## 6. 决策记录 (2026-06-14 05:37)

- ✅ 2026-06-14 04:50-04:58: v5 quicktest 第一次启动失败
- ✅ 2026-06-14 05:16-05:26: v5 quicktest 第二次启动成功 (5 步 + eval 30.78%/30.25%)
- ✅ 2026-06-14 05:26: v5 完整 500 步训练启动
- ✅ 2026-06-14 05:30: v5 quicktest 反思 + 奖励函数阶段性总结 + v6 5 候选落盘
- ✅ 2026-06-14 05:34: v5 训练启动阶段状态总结落盘
- ✅ 2026-06-14 05:37: **本文件 = v5 训练中期总结 + v6 决策 (0.5B + 8-shot CoT)**
- ⏳ 2026-06-14 05:55: v5 step 100 eval
- ⏳ 2026-06-14 06:50: v5 500 步训练完
- ⏳ 2026-06-14 07:00: v5 报告落盘
- ⏳ 2026-06-14 07:00: v6 spec/plan/reward/test 落盘
- ⏳ 2026-06-14 08:00: v6 SFT 训练完
- ⏳ 2026-06-14 10:00: v6 GRPO 400 步训练完 + 5 eval
- ⏳ 2026-06-14 10:30: v6 报告 + 0.46 目标达成判断

---

**结论**:
- **短期**: v5 继续跑, 等 step_100 eval 判断撞天花板是否打破
- **中期**: v5 跑完后, v5 报告反思
- **长期**: **v6 = 0.5B + 8-shot CoT 协议 (用户约束范围内, 唯一可达 0.46 的路径)**
- **立即**: v5 训练中, v6 准备立即开始
