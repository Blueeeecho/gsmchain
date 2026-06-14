# LB-PRM v7 实施计划 (2026-06-14)

> 关联 spec: `docs/superpowers/specs/2026-06-14-lbprm-v7-design.md`
> 起点: `/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct` (HF 原始, 无 SFT)
> 目标: original 子集 >= 0.46 (实测 0.5B base 8-shot 原生 = 0.4329, 涨 2.71pp)

---

## 1. 落地物

| 类型 | 路径 | 状态 |
|---|---|---|
| spec | `docs/superpowers/specs/2026-06-14-lbprm-v7-design.md` | ✅ 已落盘 |
| plan | `docs/superpowers/plans/2026-06-14-lbprm-v7-plan.md` | ✅ 本文件 |
| reward 测试 (TDD RED) | `train_pipeline/test_lbprm_v7.py` | ⏳ 步骤 1 |
| reward 实现 (TDD GREEN) | `train_pipeline/reward_chaingsm_lbprm_v7_verl.py` | ⏳ 步骤 2 |
| 训练入口 | `train_scripts/local/run_grpo_verl_lbprm_v7.sh` | ⏳ 步骤 3 |
| smoke test | 5 步 + step_5 eval | ⏳ 步骤 4 |
| 完整训练 | 1000 步 + 5 次 eval | ⏳ 步骤 5 |
| 报告 | `docs/superpowers/reports/2026-06-14-lbprm-v7-report.md` | ⏳ 步骤 6 |

## 2. 训练参数 (硬编码, 可在脚本顶部覆盖)

- 起点: `/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct`
- 训练集: `verl_grpo_train_8shot_cot.parquet` (7419 条, 已落盘)
- 测试集: `gsm8k_test_clean.jsonl` (5467 条, original 1319)
- MAX_STEPS=1000, SAVE_FREQ=200 (eval 节点 200/400/600/800/1000)
- ROLLOUT_N=4, TRAIN_BATCH_SIZE=4
- MAX_PROMPT_LENGTH=1280, MAX_RESPONSE_LENGTH=512
- ACTOR_LR=5e-7, KL_LOSS_COEF=0.02, ROLLOUT_TEMPERATURE=0.9
- v7 reward 权重: format=0.10, answer=0.70, numeric_correctness=0.20

## 3. 与 v6 唯一区别

- 起点: sft_2epoch/best → 0.5B base (无 SFT) — 跳过 SFT 摧毁 8-shot 能力的陷阱
- reward: 简化为 3 子项 (0.70/0.10/0.20), 砍掉 step_count/no_contradiction/equation_count_bonus
- 训练步数: 400 → 1000
- 其他沿用 v6 训练协议

## 4. 终止与判断

- 跑满 1000 step → 自然停
- step 200 eval original >= 0.46 → 视为达成, 后续 eval 观测收敛
- step 200 eval original < 0.43 (= 0.5B base 原生) → 立即停, 反思起点
- step 200-400 连续 2 次 original 持平且 < 0.46 → 停, 反思 reward 权重
- step 600 eval original >= 0.46 → 强制 stop at 600

## 5. 监控要点

- 每 200 步看 `eval/step_NNNN/summary_by_category.json` 的 original 准确率
- 每 50 步看 `metrics/train_metrics.jsonl` 的 reward_components/answer/mean, numeric_correctness_score/mean
- 监控 entropy 0.05-0.12 (健康), > 0.15 反思
- 监控 kl_loss 0.0001-0.001 (跟 v5/v6 持平)

## 6. 风险与回退

- 风险 1: 0.5B base 在 8-shot 自由推理下 entropy 高, 训练波动大
  - 应对: 监控 entropy, > 0.15 → 反思 LR (5e-7→3e-7)
- 风险 2: GRPO 1000 步撞 0.45 (差 1pp)
  - 应对: 反思 answer 权重 (0.70→0.80), 或 max_response_length 512→768
- 风险 3: 4 类变体反向退化
  - 应对: 监控 by_category.json, 任何一类降 5pp → 反思 reward

## 7. 执行时间预算

- 数据 preprocess: 0 (8-shot CoT 数据已落盘, 7419 条)
- baseline eval (0.5B base + 8-shot CoT): ~5 min
- 100 step 训练: ~3 min (1.5s/step)
- 每次 vLLM eval (5467): ~5 min
- 总计: 5 + 10×(3+5) = 95 min ≈ 1.6 h (5 个 eval 节点 200/400/600/800/1000)
- 1000 步: 1000 × 1.5s = 25 min 训练 + 5 × 5 min eval = 50 min ≈ 0.85h
- 总: 5 + 50 ≈ 55 min ≈ 1h
- 缓冲 + 反思: 2h 整

## 8. v8 候选 (等 v7 报告再选)

按 v7 报告结果定 v8 方向:
1. **加 answer 权重**: v8 = answer 0.70→0.80, numeric 0.20→0.10
2. **加 max_response_length**: 512→768, 给推理更多 token
3. **加 8-shot SFT 起点**: 0.5B base + 8-shot CoT SFT 1 epoch (自生成 trace 过滤) + GRPO
4. **加 reasoning 子项**: 把 step_count 加回来, 但权重 0.10

## 9. v7 实施步骤 (按时间顺序, TDD)

### 步骤 1: TDD RED — 写 v7 reward 测试 (`train_pipeline/test_lbprm_v7.py`)

≥ 20 case, 覆盖:
- format 子项 (extract_answer 命中 / 不命中)
- answer 子项 (数值匹配 / 不匹配 / gold 为 None)
- numeric_correctness 子项 (算式全对 / 全错 / 部分对 / 0 算式)
- 总公式权重 (0.10/0.70/0.20)
- 边界 (空字符串, 超长回答)
- 跟 v6b 的差异 (v7 没有 step_count_score, no_contradiction_score, equation_count_bonus)

预期: 测试运行报 ImportError (reward module 还没写), 算 RED.

### 步骤 2: TDD GREEN — 写 v7 reward (`train_pipeline/reward_chaingsm_lbprm_v7_verl.py`)

最小实现, 复用 v6b 的:
- `_to_float`, `_floats_close`
- `_numeric_correctness_score` (从 v6b 复制)
- `extract_answer`, `is_correct` (从 code/gsm_answer_extractor 导入)
- `score_response` (改公式为 0.70/0.10/0.20)
- `compute_reward` (同 v6b 接口契约)

预期: 步骤 1 的测试全 PASS, 算 GREEN.

### 步骤 3: 写 v7 训练入口 (`train_scripts/local/run_grpo_verl_lbprm_v7.sh`)

拷 v6 改:
- MODEL 起点: sft_2epoch/best → 0.5B base
- REWARD_PATH: v6b → v7
- FORMAT_WEIGHT/ANSWER_WEIGHT/REASONING_QUALITY_WEIGHT → 0.10/0.70/0.20
- MAX_STEPS: 400 → 1000
- SAVE_FREQ: 80 → 200
- ROLLOUT_N/TRAIN_BATCH_SIZE: 4/4 (不变)
- RUN_NAME 默认: grpo_verl_v7

### 步骤 4: smoke test (5 步 + 1 次 eval, ~15 min)

```bash
EVAL_BASELINE=1 \
MAX_STEPS=5 SAVE_FREQ=5 \
RUN_NAME=grpo_verl_v7_smoke \
bash train_scripts/local/run_grpo_verl_lbprm_v7.sh
```

预期:
- baseline eval 跑出 original ≈ 0.43 (跟 8-shot 原生一致)
- 5 步训练内 accuracy 0.3-0.4 (起点 base 直接 8-shot)
- step_5 eval 数字合理 (允许比 baseline 略低, 训练刚开始)

### 步骤 5: 完整 1000 步训练 + 5 次 eval (~1h)

```bash
EVAL_BASELINE=1 \
MAX_STEPS=1000 SAVE_FREQ=200 \
RUN_NAME=grpo_verl_v7 \
bash train_scripts/local/run_grpo_verl_lbprm_v7.sh
```

监控:
- step 200 eval original → 决定是否继续
- step 400/600/800/1000 持续观测
- 任一节点 original >= 0.46 → 视为达成

### 步骤 6: 落盘 v7 报告 (`docs/superpowers/reports/2026-06-14-lbprm-v7-report.md`)

报告内容:
- baseline 0.5B base + 8-shot CoT 真实数字 (original)
- step 200/400/600/800/1000 每次的 original + overall + 4 类变体
- 跟 v3 best (30.47% / 29.95% original) 对比
- 跟 0.5B base 原生 (24.55% / 43.29%) 对比
- 目标达成判断 (original >= 0.46 是 / 否)
- v8 方向 (基于真实数据)

## 10. 决策记录

- 2026-06-14: v7 spec/plan 落盘
- 2026-06-14: v7 reward 测试 (RED) + reward 实现 (GREEN) + entry 落盘
- 2026-06-14: v7 smoke (5 步 + step_5 eval) 真实数字
- 2026-06-14: v7 完整 1000 步 + 5 次 eval
- 2026-06-14: v7 报告 + original >= 0.46 验证 + v8 方向
