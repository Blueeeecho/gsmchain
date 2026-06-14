# LB-PRM v8 设计与训练计划 (2026-06-14)

> 接 v7 失败报告 (`docs/superpowers/reports/2026-06-14-lbprm-v7-report.md`).
> 用户硬指标: `original >= 0.46` (1319 条干净集 original 子集, 0.5B + 8-shot CoT 协议).
> v7 best: step_100 original 0.4428, 距目标 -1.72pp.
> 用户授权: 可以完全推翻 reward, 训练步数可提高 1000 步.

---

## 1. 目标与差距

| 指标 | 数值 | 状态 |
|---|---:|---|
| 0.5B base + 8-shot CoT 原生 | 0.4329 (571/1319) | baseline |
| v7 best (step_100) | 0.4428 (584/1319) | +0.99pp, **失败** |
| **目标** | **>= 0.46 (607/1319)** | 差 23 个 correct |
| v7 step_200-500 | 0.40-0.41 | **policy 退化** |

**v7 失败根因**:
- reward 公式 0.10/0.70/0.20 中 numeric 子项 (0-1 连续信号) 抢了 answer (0/1 稀疏信号) 梯度
- 训练 200 步后 numeric 涨到 0.65, answer 退到 0.10, policy "reward hack" numeric
- kl_loss 0.001 → 0.017 涨 17x, policy 严重偏离 ref

## 2. v8 核心创新: 改 reward 公式 + 加 SFT 起点

**v8 双管齐下**:
- (A) **改 reward 公式**: 砍 numeric 抢梯度, 提 answer 主导
- (B) **加 SFT 1 epoch 起点**: 让模型先学会 8-shot CoT 协议结构

### 2.1 v8 reward 公式 (4 子项, 总和=1.0)

```
v8_total = 0.05 * format
         + 0.85 * answer
         + 0.05 * numeric_correctness
         + 0.05 * step_count (鼓励展开算式, >= 2 算式给 1.0)
```

**对比 v7 (3 子项 0.10/0.70/0.20)**:
- format 0.10 → 0.05 (base 已 0.95+, 砍半)
- answer 0.70 → **0.85** (大幅提, 主线信号)
- numeric 0.20 → **0.05** (大幅砍, 抢 answer 梯度的元凶)
- 新增 step_count 0.05 (鼓励展开算式)

**step_count 评分**:
- 0 算式: 0
- 1 算式: 0.5
- 2+ 算式: 1.0

**为什么这样改**:
- answer 0.85: 直接对齐 "original 数字对" 目标, 二值 0/1 信号
- numeric 0.05: 仍然奖励算式对, 但权重小到不能抢主线
- step_count 0.05: 鼓励展开算式, 跟 numeric 配合
- format 0.05: 保留 hard constraint

**风险**:
- answer 0/1 二值, GRPO 优势函数方差可能大, 训练初期不稳定
- **对策**: LR 5e-7 不变, KL 0.02 不变, ROLLOUT_N=4 不变, baseline 已稳

### 2.2 SFT 1 epoch 8-shot CoT 起点

**数据**: `chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/sft_train_v2.jsonl` (14528 条)

**数据特征**:
- 格式: `messages = [system, user, assistant]`, system="You are an expert math problem solver. Solve step by step.", user=题目, assistant=CoT+答案
- prompt 均长 216 tokens (max 509), 远短于 8-shot 1029 — SFT 快
- 100% 有 "The final answer is N." 收尾
- 类别: original 7473 + 4 类变体 1700+ (平衡)

**SFT 超参**:
- 起点: 0.5B base
- 训练步: 14528 / batch_size 16 = ~908 步 (1 epoch)
- LR: 2e-5 (常规 SFT LR)
- MAX_LENGTH: 1024 (prompt 216 + response 800 留余量)
- 数据集: 用 `train_pipeline/train_sft_trl.py` (已存在)
- **时间预算**: ~30 min (TRL + RTX 5090, batch_size=16, 14528 条)

**风险**:
- v6 prereflex 已证: sft_2epoch + 8-shot 协议 original 0.13 (SFT 摧毁 8-shot)
- **sft_v2 的差异**: sft_v2 是 free-form CoT (无 8-shot examples), 不是 JSON schema
- **v6 失败原因**: sft_2epoch 训练数据是 JSON schema 协议, 模型学 "必须 JSON"
- **v8 不重复该错误**: sft_v2 是 CoT 协议, 跟 8-shot CoT 协议同向

## 3. v8 训练计划 (两步)

### 步骤 1: SFT 1 epoch (15:50 - 16:30)
```bash
DATA=/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/sft_train_v2.jsonl
MODEL=/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct
OUTPUT_DIR=/home/wwq416/snap/wwq/math-chain/outputs/sft/sft_8shot_cot_1ep
MAX_STEPS=908  # 1 epoch
BATCH_SIZE=16
LR=2e-5

/home/wwq416/miniconda3/envs/math_chain_verl/bin/python -m train_pipeline.train_sft_trl \
  --config train_configs/local/sft.yaml \
  --set training.output_dir=$OUTPUT_DIR \
  --set training.learning_rate=$LR \
  --set training.per_device_train_batch_size=16 \
  --set training.max_steps=908 \
  --set data.train_files=$DATA
```

**验证 SFT ckpt (eval_chaingsm_base_8shot.py)**:
- 预期: original 0.45-0.50 (vs base 0.4329, +1-7pp)

### 步骤 2: GRPO 1000 步 (16:30 - 19:00)
- 起点: SFT ckpt (路径在 `outputs/sft/sft_8shot_cot_1ep/checkpoints/best/`)
- 训练 prompt: `verl_grpo_train_8shot_cot.parquet` (7419 条, 跟 v7 一样)
- 测试 prompt: 8-shot CoT (跟 v7 一样)
- reward: v8 公式 (0.05/0.85/0.05/0.05)
- MAX_STEPS: 1000
- SAVE_FREQ: 200 (5 个 eval 节点: 200/400/600/800/1000)
- 其他超参: 跟 v7 一致 (LR=5e-7, KL=0.02, ROLLOUT_N=4, MAX_PROMPT=1280, MAX_RESPONSE=512)

### 步骤 3: 5 节点 eval + 报告 (19:00 - 19:30)
- entry 脚本自动跑 5 个 vLLM eval
- 报告: best ckpt 对比 baseline 0.4329, 目标 >= 0.46

## 4. 终止条件

1. 达到 1000 step 自动停
2. step 200 eval original >= 0.46 → 视为达成, 后续 eval 观测收敛
3. step 200 eval original < SFT ckpt original → 反思 SFT 是否破坏
4. step 400 eval original 持平且 < 0.46 → 反思 reward 公式
5. step 600 eval original >= 0.46 → 强制 stop at 600

## 5. 监控

- SFT 训练: `outputs/sft/sft_8shot_cot_1ep/metrics/`, 关注 train_loss 下降
- SFT ckpt eval: `outputs/baselines/sft_8shot_cot_1ep_*/summary_by_category.json`
- GRPO 训练: `outputs/train/local/grpo_verl_lbprm_v8/.../metrics/train_metrics.jsonl`
  - 关注 `answer` 子项 0.18 → ? 突破
  - 关注 `numeric` 子项 0.05 是否不再抢梯度
  - 关注 `kl_loss` < 0.01 (健康)
- GRPO eval: `outputs/.../eval/step_200/...` 5 个节点

## 6. 风险与回退

| 风险 | 应对 |
|---|---|
| SFT 1 epoch 摧毁 8-shot (跟 v6 sft_2epoch 类似) | step_200 eval original < 0.40 → 反思 SFT, 跑 v8.1 (0.5 epoch SFT) |
| answer 0.85 训练不稳定, GRPO 散 | LR 5e-7 → 3e-7, KL 0.02 → 0.04 |
| SFT ckpt 已经 0.46, GRPO 反而退步 | 取 SFT ckpt 报目标达成, 不再跑 GRPO |
| GRPO 1000 步撞 0.45 | 反思 answer 0.85→0.90, 或加 step_count 0.10 |
| 时间超预算 (3.5h → 5h) | 接受, 优先保证 SFT ckpt 数字 |

## 7. 落盘物

| 类型 | 路径 |
|---|---|
| spec | `docs/superpowers/specs/2026-06-14-lbprm-v8-design.md` (本文件) |
| plan | `docs/superpowers/plans/2026-06-14-lbprm-v8-plan.md` |
| v8 reward | `train_pipeline/reward_chaingsm_lbprm_v8_verl.py` |
| v8 reward 测试 (TDD) | `train_pipeline/test_lbprm_v8.py` |
| v8 SFT 入口 | `train_scripts/local/run_sft_8shot_cot_1ep.sh` |
| v8 GRPO 入口 | `train_scripts/local/run_grpo_verl_lbprm_v8.sh` |
| 报告 | `docs/superpowers/reports/2026-06-14-lbprm-v8-report.md` |

## 8. 决策记录 (2026-06-14)

- ✅ 2026-06-14 13:33-15:21: v7 500 步训练完成
- ✅ 2026-06-14 15:22-15:49: v7 5 节点 eval, best 0.4428
- ❌ v7 失败 (numeric 抢 answer 梯度)
- ⏳ 2026-06-14 15:50: v8 spec/plan 落盘
- ⏳ 2026-06-14 15:55-16:00: v8 reward TDD
- ⏳ 2026-06-14 16:00-16:30: SFT 1 epoch 训练
- ⏳ 2026-06-14 16:30-16:40: SFT ckpt eval
- ⏳ 2026-06-14 16:40-17:00: v8 GRPO 入口 + 启动
- ⏳ 2026-06-14 17:00-19:00: v8 GRPO 1000 步 + 5 eval
- ⏳ 2026-06-14 19:00-19:30: v8 报告
