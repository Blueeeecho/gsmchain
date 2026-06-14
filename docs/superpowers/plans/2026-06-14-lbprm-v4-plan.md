# LB-PRM v4 实施计划(2026-06-14)

> 关联 spec: `docs/superpowers/specs/2026-06-14-lbprm-v4-design.md`
> 关联报告: `docs/superpowers/reports/2026-06-14-lbprm-v4-report.md` (待落盘)

## 1. 落地物

| 类型 | 路径 | 状态 |
|---|---|---|
| spec | `docs/superpowers/specs/2026-06-14-lbprm-v4-design.md` | ✅ 已落盘 |
| plan | `docs/superpowers/plans/2026-06-14-lbprm-v4-plan.md` | ✅ 本文件 |
| reward 函数 | `train_pipeline/reward_chaingsm_lbprm_v3_verl.py` (沿用 v3) | ✅ 18 个行为测试全过 |
| 训练入口 | `train_scripts/local/run_grpo_verl_lbprm_v4.sh` | ⏳ 准备落盘 |
| 反思报告 | `docs/superpowers/reports/2026-06-14-lbprm-v4-report.md` | ⏳ 跑完落盘 |

## 2. 训练参数(全部硬编码, 可在脚本顶部覆盖)

- SFT 起点: `outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_2epoch/20260531_152306/checkpoints/best`
- 训练集: `chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/verl_grpo_train_neutral.parquet` (7055, **NEUTRAL prompt**)
- 测试集: `chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl` (5467)
- MAX_STEPS: 400
- SAVE_FREQ: 100 (每 100 step 评测, 共 4 个 eval 节点)
- ROLLOUT_N: 4, ROLLOUT_GPU_MEM_UTIL: 0.3, TRAIN_BATCH: 4
- v3 reward 权重: format=0.20, answer=0.55, chain_quality=0.25
- ACTOR_LR: 5e-7, KL_LOSS_COEF: 0.04 (沿用 v3)

## 3. 与 v3 唯一区别

- DATA 路径: `verl_grpo_train.parquet` → `verl_grpo_train_neutral.parquet`
- REWARD_PATH 不变
- 其它全部沿用 v3

## 4. 终止与判断

- 跑满 400 step → 自然停
- 200 step 评估后 Original 连续 2 次 ≤ 上次 AND overall 连续 2 次 ≤ 上次 → 停
- 没收益的 checkpoint 训练后**删除** (按用户要求, 只删 ckpt, 留代码+报告+eval_result.json)
- **v4 起步 baseline = sft_2epoch 起点, 26.59% on 5467 / 25.5% on original**
- **v4 比较点 = v3 best (30.47% / 29.95%)** (按用户原话"持续改进", v4 必须能维持或涨过 v3 best, 否则 v5 改方向)

## 5. 监控要点

- `metrics/train_metrics.jsonl` 每 step 的 reward_components 系列 (v3 已验证 verl 会自动写)
- `eval/step_NNNN/eval_result.json` 的 overall_accuracy + 5 类目
- 关键比较点: baseline 26.59% → step100 → 200 → 300 → 400 → v3 best 30.47%

## 6. 风险与回退

- 如果 v4 训练 step100 评测比 baseline 还掉 5pp+, 立即停
- 如果 reward components 在 step1 全部≈0, policy 无信号, 在 step50 主动停并改 v5
- 如果 GPU OOM, 降到 ROLLOUT_GPU_MEM_UTIL=0.2 + MAX_RESPONSE_LENGTH=768

## 7. 执行时间预算

- baseline eval: ~10 min (用 0.5B 模型在 5467 条)
- 100 step 训练: ~15-25 min (v2/v3 节奏)
- 每次 vLLM eval (5467): ~8-12 min
- 总计: ~10 + 4×(20+10) = 130 min ≈ 2.2 h
- 缓冲 + 反思: 3 h 整

## 8. v5 候选(等 v4 报告出来再选)

按 v3 报告 §5.2 候选方向, 假设 v4 prompt 对齐后有提升空间:

1. **causal_liveness 再松绑** (v3.5 → v5): 重新打开 v3 删的 "value 子串匹配" (理由: v3 收紧后 liveness 0.34 偏低, 可能挡了真引用信号)
2. **放宽 KL**: 0.04 → 0.02
3. **加 anti-degenerate 长度 bonus**: n_steps ∈ [3, 6] 时 +0.05
4. **重写 reward** (大改): 加入"答对 + chain 真的引用前文 step"作为额外硬门
5. **数据增强**: 把 "SFT 答错的样本" 加进训练当负样本 (重做 parquet)
6. **换 1.5B 模型**: 8-shot CoT 上限 72.25%, 即使 RL 掉 10pp 仍 > 0.6
