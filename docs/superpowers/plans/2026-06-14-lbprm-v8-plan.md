# LB-PRM v8 实施计划 (2026-06-14)

> 关联 spec: `docs/superpowers/specs/2026-06-14-lbprm-v8-design.md`
> 起点: SFT 1 epoch 8-shot CoT ckpt (0.5B base 训 1 epoch with `sft_train_v2.jsonl`)
> 目标: original 子集 >= 0.46 (v7 best 0.4428, 需 +1.72pp)

---

## 1. 落地物

| 类型 | 路径 | 状态 |
|---|---|---|
| spec | `docs/superpowers/specs/2026-06-14-lbprm-v8-design.md` | ✅ 已落盘 |
| plan | `docs/superpowers/plans/2026-06-14-lbprm-v8-plan.md` | ✅ 本文件 |
| v8 reward 测试 (TDD RED) | `train_pipeline/test_lbprm_v8.py` | ⏳ 步骤 1 |
| v8 reward 实现 (TDD GREEN) | `train_pipeline/reward_chaingsm_lbprm_v8_verl.py` | ⏳ 步骤 2 |
| SFT 入口 | `train_scripts/local/run_sft_8shot_cot_1ep.sh` | ⏳ 步骤 3 |
| SFT 1 epoch 训练 | `outputs/sft/sft_8shot_cot_1ep/...` | ⏳ 步骤 4 |
| SFT ckpt 评测 | `outputs/baselines/sft_8shot_cot_1ep_*/summary_by_category.json` | ⏳ 步骤 5 |
| v8 GRPO 入口 | `train_scripts/local/run_grpo_verl_lbprm_v8.sh` | ⏳ 步骤 6 |
| v8 GRPO 训练 (1000 步) | `outputs/train/local/grpo_verl_lbprm_v8/...` | ⏳ 步骤 7 |
| 5 节点 eval | step_200/400/600/800/1000 | ⏳ 步骤 8 |
| 报告 | `docs/superpowers/reports/2026-06-14-lbprm-v8-report.md` | ⏳ 步骤 9 |

## 2. v8 reward 公式 (vs v7)

| 子项 | v7 | v8 |
|---|---:|---:|
| format | 0.10 | 0.05 |
| answer | 0.70 | **0.85** |
| numeric_correctness | 0.20 | **0.05** |
| step_count | (无) | **0.05** (新) |
| 总和 | 1.00 | 1.00 |

**v8 step_count 子项**:
- 0 算式: 0.0
- 1 算式: 0.5
- 2+ 算式: 1.0

## 3. 训练参数

### SFT 1 epoch
- 起点: `Qwen2.5-0.5B-Instruct`
- 数据: `sft_train_v2.jsonl` (14528 条)
- LR: 2e-5
- batch_size: 16
- max_steps: 908
- MAX_LENGTH: 1024
- 输出: `outputs/sft/sft_8shot_cot_1ep/.../best/`

### GRPO 1000 步
- 起点: SFT ckpt
- 训练 prompt: `verl_grpo_train_8shot_cot.parquet` (7419 条)
- LR: 5e-7
- KL: 0.02
- ROLLOUT_N: 4
- MAX_PROMPT: 1280
- MAX_RESPONSE: 512
- MAX_STEPS: 1000
- SAVE_FREQ: 200

## 4. 终止与判断

- 1000 step 自动停
- step 200 eval original >= 0.46 → 视为达成
- step 200 eval original < SFT ckpt original → 反思 SFT
- step 400 eval original 持平且 < 0.46 → 反思 reward
- step 600 eval original >= 0.46 → 强制 stop at 600

## 5. 监控要点

- SFT: train_loss 1-2 → < 0.5 下降
- SFT ckpt eval: original >= 0.45 (预期 SFT 后能突破 0.43 baseline)
- GRPO: answer 子项 0.18 → 0.25+ (关键涨点信号)
- GRPO: kl_loss < 0.01 (健康)
- GRPO: entropy 0.4-0.55 (健康)

## 6. 风险与回退

| 风险 | 应对 |
|---|---|
| SFT 摧毁 8-shot (跟 v6 一样) | step_200 eval < 0.40 → 改 0.5 epoch SFT (v8.1) |
| answer 0.85 训练不稳定 | LR 5e-7→3e-7, KL 0.02→0.04 |
| SFT ckpt 已 0.46 | 取 SFT ckpt, 不跑 GRPO |
| GRPO 1000 步撞 0.45 | answer 0.85→0.90, 或加 step_count 0.10 |
| 时间超 5h | 接受, 优先 SFT ckpt 数字 |

## 7. 时间预算

- v8 reward TDD + 实现: 10 min
- SFT 1 epoch: 30 min
- SFT ckpt eval: 5 min
- GRPO 1000 步: 2.2h (12.5s/step × 1000)
- 5 节点 eval: 25 min
- 总: 3.5h

## 8. 执行步骤 (TDD 风格)

### 步骤 1: TDD RED — 写 v8 reward 测试
- `train_pipeline/test_lbprm_v8.py`
- 覆盖 4 子项 + 权重验证 + schema + 边界

### 步骤 2: TDD GREEN — 写 v8 reward
- `train_pipeline/reward_chaingsm_lbprm_v8_verl.py`
- 公式: 0.05·format + 0.85·answer + 0.05·numeric + 0.05·step_count

### 步骤 3: 写 SFT 入口
- `train_scripts/local/run_sft_8shot_cot_1ep.sh`
- 用 sft_train_v2.jsonl 训 1 epoch

### 步骤 4: 跑 SFT 1 epoch
- 14528 条, batch_size 16, LR 2e-5, 908 步
- ~30 min

### 步骤 5: 评 SFT ckpt
- `code/eval_chaingsm_base_8shot.py` 跑 baseline

### 步骤 6: 写 v8 GRPO 入口
- 拷 v7 改: MODEL 起点, REWARD_PATH, 权重

### 步骤 7: 跑 v8 GRPO 1000 步
- 5 个 eval 节点 200/400/600/800/1000

### 步骤 8: 监控 5 节点 eval
- 等 best ckpt original >= 0.46

### 步骤 9: 落盘 v8 报告

## 9. 决策记录

- 2026-06-14 15:50: v8 spec/plan 落盘
- 2026-06-14 15:55: v8 reward TDD 21 cases
- 2026-06-14 16:00: SFT 1 epoch 启动
- 2026-06-14 16:30: SFT 1 epoch 完成 + ckpt eval
- 2026-06-14 16:40: v8 GRPO 启动
- 2026-06-14 19:00: v8 GRPO 1000 步 + 5 eval 完成
- 2026-06-14 19:30: v8 报告
