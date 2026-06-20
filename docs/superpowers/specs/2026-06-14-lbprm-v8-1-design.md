# LB-PRM v8.1 设计与训练计划 (2026-06-14)

> 接 v7 失败 + v8 reward "不配 ChainGSM 干扰式样" 反馈.
> v7 best: step_100 original 0.4428 (距 0.46 -1.72pp).
> v8 (4 子项 0.05/0.85/0.05/0.05) 修 v7 砍 numeric 抢梯度, 但**没适配 ChainGSM 干扰**。
> v8.1 重新设计: **5 机制 reward, 真正惩罚 distractor_chain 算式**.

---

## 1. 目标与差距

| 指标 | 数值 |
|---|---:|
| 0.5B base + 8-shot CoT 原生 | 0.4329 (571/1319) |
| v7 best (step_100) | 0.4428 (+0.99pp, 失败) |
| **目标** | **>= 0.46 (607/1319)** |
| v8 reward 砍 numeric | 数值不涨 answer, 仍可能撞 30% 天花板 |

**用户反馈**: v7/v8 reward "一点心意都没有, 也没适配数据风格问题类型 (为问题添加分心连)".

## 2. ChainGSM 数据特点 (v8.1 设计动机)

每条 ChainGSM 题目有:
- `core_chain`: 真正能推出 gold 的算式链, e.g. `[[16, eggs_after, -3], [9, income, *2]]`
- `distractor_chain`: **干扰**算式链, 用无关 entity (eg cookies 代替 eggs), e.g. `[[2, cookies, *12], [24, income, *1.5]]`
- `gold_expression`: 完整 gold 算式, e.g. `(16-3-4)*2 = 18`
- `distractor_expression`: 干扰算式完整表达式, e.g. `2*12*1.5 = 36`

**核心问题**: 模型必须写 core_chain 算式, **不能**写 distractor_chain 算式.
但 0.5B 在 4 类变体上答错率是 original 的 3-4 倍 (16-21% vs 43%).

**v7/v8 reward 完全没有捕捉**"模型是否在写 distractor 算式"这个核心信号.

## 3. v8.1 reward 设计 (5 机制)

```
total = 0.05 * format                          # 收尾 marker
      + 0.55 * answer                          # 答对 gold
      + 0.20 * chain_to_answer_check           # 最后算式 Z == extract_answer (反抄错)
      + 0.15 * target_recognition              # answer unit 匹配题面 (反 target_scope_misalignment)
      + 0.05 * chain_length_consistency        # 算式数 vs 题面复杂度 (轻量, 避免凑算式)
      - 0.30 * irrelevant_eq_ratio             # 写 distractor 算式, 扣分 0-0.30
```

总: 0.05+0.55+0.20+0.15+0.05 = 1.00, penalty 0-0.30 扣分.

### 3.1 `chain_to_answer_check` 0.20
抓最后算式 `X op Y = Z`, 检查 `Z == extract_answer` 文本中的数.
**反 "chain 写对但抄错 final answer"**.

### 3.2 `target_recognition` 0.15
抓题面问什么 (e.g. "How much in dollars", "How many total meters"), 抓 answer 的 unit 形式.
unit 匹配 → 1.0, 不匹配 → 0.0.
**反 target_scope_misalignment** (问 profit 答 after-donation 错).

### 3.3 `irrelevant_eq_penalty` 0-0.30 扣分 (**v8.1 核心创新**)
**核心反干扰机制**:
1. 构建"应该出现"中间值集合 S:
   - 从 `gold_expression` 解析 + 模拟链式执行, 收集所有操作数和中间值
   - 例: `(16-3-4)*2` → S = {16, 3, 4, 2, 13, 9, 18}
2. 解析模型算式 chain
3. 算式 (left, op, right, expected) **任一值在 S** → 算跟 gold chain 有关
4. **全部不在 S** → 算无关 (irrelevant), 扣分
5. penalty = (n_irrelevant / n_eq) * 0.30

**核心反 distractor_chain 干扰**:
- 写 `2*12=24, 24*1.5=36` (cookies 干扰), 24/1.5/36 都不在 S → 2 个 irrelevant → 扣 0.30
- 写 `16-3=13, 13-4=9, 9*2=18` (core), 13/9/18 都在 S → 0 个 irrelevant

### 3.4 `format` 0.05 + `answer` 0.55
跟 v7 一样, 但 answer 0.55 不是 v8 的 0.85 — 因为新增 3 个机制分担信号, 不需要 answer 单项过强.

### 3.5 `chain_length_consistency` 0.05
**v8.1 简化**: 默认 1.0 (避免误判). 未来可以加题面数字数启发式.

## 4. 训练计划 (3 步)

### 步骤 1: SFT 1 epoch 8-shot CoT 起点 (16:10-16:30)
- 起点: 0.5B base
- 数据: `sft_train_v2.jsonl` (14528 条)
- 训练步: 908 effective (batch_size 2, grad_accum 8)
- LR: 2e-5
- MAX_LENGTH: 1024
- 预计 15-18 min (OOM fix 后 batch_size 2)

### 步骤 2: SFT ckpt eval (~5 min)
- 跑 `code/eval_chaingsm_base_8shot.py` 看 original 数字
- 预期: 0.45-0.50 (vs base 0.4329)
- 如果 original 已 >= 0.46 → 取 SFT ckpt, 不跑 GRPO

### 步骤 3: GRPO 1000 步 + 5 eval 节点 (v8.1 reward)
- 起点: SFT ckpt
- reward: v8.1 5 机制 (上)
- MAX_STEPS: 1000
- SAVE_FREQ: 200
- 5 个 eval 节点 200/400/600/800/1000
- 预计 2.5h

## 5. 终止与判断

- 1000 step 自动停
- step 200 eval original >= 0.46 → 视为达成
- step 200 eval original < SFT ckpt → 反思 SFT 是否破坏
- step 400 eval 持平且 < 0.46 → 反思 reward
- step 600 eval >= 0.46 → 强制 stop at 600

## 6. 风险与回退

| 风险 | 应对 |
|---|---|
| SFT 摧毁 8-shot (跟 v6 一样) | step_200 eval < 0.40 → 改 0.5 epoch SFT (v8.1.1) |
| irrelevant_eq_penalty 误判 (S 集合不全) | 调权重或 fallback 关闭 penalty |
| GRPO 1000 步撞 0.45 | answer 0.55→0.65, penalty 0.30→0.40 |
| 时间超 5h | 接受, 优先 SFT ckpt 数字 |

## 7. 落盘物

| 类型 | 路径 |
|---|---|
| spec | `docs/superpowers/specs/2026-06-14-lbprm-v8-1-design.md` (本文件) |
| plan | `docs/superpowers/plans/2026-06-14-lbprm-v8-1-plan.md` |
| v8.1 reward | `train_pipeline/reward_chaingsm_lbprm_v8_1_verl.py` |
| v8.1 TDD | `train_pipeline/test_lbprm_v8_1.py` (20/20 PASS) |
| SFT 入口 | `train_scripts/local/run_sft_8shot_cot_1ep.sh` |
| GRPO 入口 (v8.1) | `train_scripts/local/run_grpo_verl_lbprm_v8.sh` (改用 v8.1 reward) |
| 报告 | `docs/superpowers/reports/2026-06-14-lbprm-v8-1-report.md` |

## 8. 决策记录 (2026-06-14)

- ✅ 2026-06-14 13:33-15:21: v7 500 步训练 (失败, best 0.4428)
- ✅ 2026-06-14 15:22-15:49: v7 5 节点 eval
- ✅ 2026-06-14 15:50-15:55: v7 报告 + v8 spec/plan/reward
- ❌ v7 失败 (numeric 抢 answer 梯度, policy 退步)
- ✅ 2026-06-14 15:55-16:05: v8 TDD/reward (4 子项)
- ⚠️ 2026-06-14 16:05-16:08: v8 reward 被批评"不配 ChainGSM 干扰"
- ✅ 2026-06-14 16:08-16:10: v8.1 5 机制 reward 设计 + TDD RED
- ✅ 2026-06-14 16:10-16:10: v8.1 reward 实现 (TDD GREEN, 20/20 PASS)
- ⏳ 2026-06-14 16:10-16:30: SFT 1 epoch 训练 (batch_size 2, grad_accum 8)
- ⏳ 2026-06-14 16:30-16:35: SFT ckpt eval
- ⏳ 2026-06-14 16:35-16:50: GRPO 启动 v8.1 reward
- ⏳ 2026-06-14 16:50-19:30: GRPO 1000 步 + 5 eval 节点
- ⏳ 2026-06-14 19:30-20:00: v8.1 报告
