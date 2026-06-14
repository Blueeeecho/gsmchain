# LB-PRM v5 实施计划(2026-06-14)

> 关联 spec: `docs/superpowers/specs/2026-06-14-lbprm-v5-design.md`
> 关联报告: `docs/superpowers/reports/2026-06-14-lbprm-v5-report.md` (待落盘)

## 1. 落地物

| 类型 | 路径 | 状态 |
|---|---|---|
| spec | `docs/superpowers/specs/2026-06-14-lbprm-v5-design.md` | ✅ 已落盘 |
| plan | `docs/superpowers/plans/2026-06-14-lbprm-v5-plan.md` | ✅ 本文件 |
| reward 函数 | `train_pipeline/reward_chaingsm_lbprm_v5_verl.py` | ✅ 26 个行为测试全过(18 v3 + 8 v5 new) |
| 行为测试 | `train_pipeline/test_lbprm_v5.py` | ✅ |
| 训练入口 | `train_scripts/local/run_grpo_verl_lbprm_v5.sh` | ⏳ 准备落盘 |
| 反思报告 | `docs/superpowers/reports/2026-06-14-lbprm-v5-report.md` | ⏳ 跑完落盘 |

## 2. 训练参数(全部硬编码, 可在脚本顶部覆盖)

- **起点**: `outputs/train/local/grpo_verl_lbprm_v3/Qwen2.5-0.5B-Instruct/grpo_verl_v3_smoke/20260614_004034/checkpoints/best` (= v3 step_300, 30.47% / 29.95%)
- **训练集**: `chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/verl_grpo_train_neutral.parquet` (7055, NEUTRAL prompt)
- **测试集**: `chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl` (5467)
- **MAX_STEPS**: 500 (v3 在 300 步撞平台, v5 多给 200 步)
- **SAVE_FREQ**: 100 (每 100 step 评测, 共 5 个 eval 节点: step_100/200/300/400/500)
- **ROLLOUT_N**: 4, **ROLLOUT_GPU_MEM_UTIL**: 0.3, **TRAIN_BATCH**: 4
- **v5 reward 权重**: format=0.20, answer=0.55, chain_quality=0.25, length_bonus=0.05 (n_steps ∈ [3, 6])
- **ACTOR_LR**: 5e-7, **KL_LOSS_COEF**: 0.02 (v3/v4 的 0.04 → 0.02, 放宽 2x)

## 3. 与 v3/v4 唯一区别

- **起点**: v3 best (sft_2epoch → grpo_v3 step_300), 不用 v4 任何 ckpt
- **数据**: NEUTRAL prompt 训练集 (跟 v4 一致)
- **KL 系数**: 0.04 → 0.02 (放宽 2x)
- **reward**: v5 reward (基于 v3 改写, liveness 松绑 + length_bonus 0.05)
- **MAX_STEPS**: 400 → 500
- 其它全部沿用 v3/v4

## 4. 终止与判断

- 跑满 500 step → 自然停
- 200 step 评估后 Original 连续 2 次 ≤ 上次 AND overall 连续 2 次 ≤ 上次 → 停
- step 100 eval 比 v3 best baseline 掉 > 2pp → 立即停, 切 v4 step_100 起点重跑 (备选)
- 没收益的 checkpoint 训练后**删除** (按用户要求, 只删 ckpt, 留代码+报告+eval_result.json)
- **v5 起步 baseline = v3 best**, **v5 比较点 = v3 best** (30.47% / 29.95%)
- **v5 成功标准**: v5 best > v3 best overall (30.47%) OR v5 best > v3 best original (29.95%)

## 5. 监控要点

- `metrics/train_metrics.jsonl` 每 step 的 reward_components 系列 (v3/v4 已验证 verl 会自动写)
- `eval/step_NNNN/eval_result.json` 的 overall_accuracy + 5 类目
- **关键监控**:
  - liveness_score 涨点 (v5 设计目标: 0.33-0.37 → 0.45-0.55)
  - n_steps 分布 (v5 length_bonus 应让 n_steps 分布向 [3, 6] 集中)
  - accuracy 涨点 (v5 期望比 v3 同期快)

## 6. 风险与回退

- **风险 1: liveness 松绑触发了新的 hacking**
  - 表现: 训练内 accuracy 早期不涨 / chain 越写越短
  - 应对: 监控 response_length 和 accuracy, step 50 内 len < 100 或 准确率不涨 → 立即停, 改 v5.1 把 (a') 改回严格
- **风险 2: KL 放宽 0.02 仍然不够**
  - 表现: step_100-300 跟 v3 持平
  - 应对: step_300 eval 后定 v6 = KL 0.01
- **风险 3: length_bonus 0.05 太小没意义**
  - 表现: 跟 v3 没差别
  - 应对: step_100 调试时改 0.10 试
- **风险 4: v3 best 起点 + NEUTRAL 训练集迁移失败**
  - 表现: step_100 掉 > 2pp vs v3 best
  - 应对: 切 v4 step_100 起点重跑 v5
- **风险 5: v5 训练 500 步全平台**
  - 表现: 跟 v3 一样 30% 撞天花板
  - 应对: v5 报告反思 → v6 换 1.5B

## 7. 执行时间预算

- baseline eval: ~10 min (用 v3 best 0.5B 模型在 5467 条)
- 100 step 训练: ~20 min (12s/step)
- 每次 vLLM eval (5467): ~10 min
- 总计: ~10 + 5×(20+10) = 160 min ≈ 2.7 h
- 缓冲 + 反思: 3 h 整

## 8. v6 候选(等 v5 报告出来再选)

按 v3 报告 §5.2 + v4 报告 §5 候选方向:

1. **换 Qwen2.5-1.5B 模型 + 8-shot CoT 协议**(0.5B + JSON 协议 + RL 天花板 30-32% 已撞)
2. **重写 reward (D)**: 加入"答对 + chain 真的引用前文 step"作为额外硬门
3. **数据增强 (E)**: 把 SFT 答错的样本加进训练当负样本
4. **v5.1 微调**: liveness 松绑幅度调整 (回到严格版本) / length_bonus 改成 0.10

