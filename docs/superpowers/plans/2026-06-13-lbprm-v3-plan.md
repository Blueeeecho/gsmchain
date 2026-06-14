# LB-PRM v3 实施计划(2026-06-13)

> 关联 spec: `docs/superpowers/specs/2026-06-13-lbprm-v3-design.md`
> 关联报告: `docs/superpowers/reports/2026-06-13-lbprm-v3-report.md`(待落盘)

## 1. 落地物

| 类型 | 路径 | 状态 |
|---|---|---|
| reward 函数 | `train_pipeline/reward_chaingsm_lbprm_v3_verl.py` | ✅ 已写,18 个行为测试全过 |
| 行为测试 | `train_pipeline/test_lbprm_v3.py` | ✅ 全过 |
| 训练入口 | `train_scripts/local/run_grpo_verl_lbprm_v3.sh` | ✅ syntax OK |
| spec | `docs/superpowers/specs/2026-06-13-lbprm-v3-design.md` | ✅ 已落盘 |
| 计划 | `docs/superpowers/plans/2026-06-13-lbprm-v3-plan.md` | ✅ 本文件 |
| 反思报告 | `docs/superpowers/reports/2026-06-13-lbprm-v3-report.md` | ⏳ 跑完落盘 |

## 2. 训练参数(全部硬编码,可在脚本顶部覆盖)

- SFT 起点: `outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_2epoch/20260531_152306/checkpoints/best`
- 训练集: `chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/verl_grpo_train.parquet` (7055)
- 测试集: `chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl` (5467)
- MAX_STEPS: 400
- SAVE_FREQ: 100(每 100 step 保存+评测,共 4 个 eval 节点)
- ROLLOUT_N: 4, ROLLOUT_GPU_MEM_UTIL: 0.3, TRAIN_BATCH: 4
- v3 reward 权重: format=0.20, answer=0.55, chain_quality=0.25
- ACTOR_LR: 5e-7, KL_LOSS_COEF: 0.04(沿用 v2 烟测)

## 3. 终止与判断

- 跑满 400 step → 自然停
- 任一档 eval "Original ≤ 上次 AND overall ≤ 上次" 记为"无提升",200 step 后连续 2 次无提升则停
- 没收益的 checkpoint 训练后**删除**(按用户要求,只删 ckpt,留代码+报告+eval_result.json)

## 4. 监控要点

- `metrics/train_metrics.jsonl` 每 step 的 reward_components 系列(verl 会自动写,因为我们新加的 metric 字段在 score_response 的 metrics dict 里)
- `eval/step_NNNN/eval_result.json` 的 overall_accuracy + 5 类目
- 关键比较点: baseline 26.59% → step100 → 200 → 300 → 400

## 5. 风险与回退

- 如果 v3 训练 step100 评测比 baseline 还掉 5pp+,立即停
- 如果 reward components 在 step1 全部≈0,policy 无信号,在 step50 主动停并改 v4
- 如果 GPU OOM,降到 ROLLOUT_GPU_MEM_UTIL=0.2 + MAX_RESPONSE_LENGTH=768

## 6. 执行时间预算

- baseline eval: ~10 min(用 0.5B 模型在 5467 条)
- 100 step 训练: ~15-25 min(v2 烟测节奏)
- 每次 vLLM eval(5467): ~8-12 min
- 总计: ~10 + 4×(20+10) = 130 min ≈ 2.2 h
- 缓冲 + 反思: 3 h 整
