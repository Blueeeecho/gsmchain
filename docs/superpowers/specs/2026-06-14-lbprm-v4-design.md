# LB-PRM v4 设计与训练计划(2026-06-14)

> **作者**: 主线接力(接 LB-PRM v3)
> **核心目的**: 修复 v3 的 **"训练-评测 prompt 错位"** 缺陷, 看 v3 reward 在 prompt 对齐后还能不能再涨。
> **本计划范围**: v4 = v3 reward + 训练 parquet 换为 NEUTRAL prompt 版 + 400 step GRPO + 每 100 step 评测 + 反思。

---

## 1. 起点与基线

| 项 | 值 | 路径 |
|---|---|---|
| SFT 起点 | `sft_2epoch/20260531_152306` best(26.59% on 5467) | `outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_2epoch/20260531_152306/checkpoints/best` |
| v3 best | GRPO step_300 overall 30.47% / original 29.95% | `outputs/train/local/grpo_verl_lbprm_v3/.../checkpoints/best` |
| 训练集 | **NEW** `verl_grpo_train_neutral.parquet` 7055 条 4 类变体 (**NEUTRAL prompt**) | `chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/verl_grpo_train_neutral.parquet` |
| 测试集 | `gsm8k_test_clean.jsonl` 5467 条 5 类 | `chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl` |
| 上次 GRPO | v3 step_300 overall 30.47% / original 29.95% | v3 report |

**v3 prompt 错位现状**:
- v3 训练 parquet `verl_grpo_train.parquet` 的 system prompt 是: `You are a careful mathematical reasoning assistant. Select only the computation chain that answers the question, ignore distractor chains, and return JSON only.`
- v3 评测脚本 `eval_vllm_chaingsm` `method=train_json_prompt` 用的 prompt 是: `NEUTRAL_SYSTEM_PROMPT` + `NEUTRAL_USER_TEMPLATE` (没有 "ignore distractor" 字样, 措辞是 "expert grade-school math solver" + "Identify the quantity being asked")
- 两套 prompt 在 "任务定义 / 输出格式 / 措辞" 上均有差异

**v4 修复方案**: 训练 parquet 换为 `verl_grpo_train_neutral.parquet` (与评测 prompt 同一套 NEUTRAL), **其它 v3 reward 函数 / 权重 / KL / LR 全不动**。

---

## 2. v4 设计目标

- **首要**: 让训练时模型见到的 prompt 与评测时一致, 避免 policy 学到 "ignore distractor" 这类训练专属词。
- **其次**: 沿用 v3 reward 函数 (含 chain_to_answer_ok 硬门、causal_liveness 收紧、answer 0.55 主导), 验证在 prompt 对齐下 v3 reward 仍能涨。
- **再次**: 不重写 reward, 不动 KL/LR 调度, 不动 prompt schema, 不动 chain 结构。
- **不追求**: 这一版就达到 > 0.46; 即使涨 1-2pp 算 v4 成功, 后续 v5/v6 继续试其他方向。

---

## 3. v4 公式(沿用 v3)

```
chain_to_answer_ok       ∈ {0, 1}
causal_liveness_score   ∈ [0, 1]
step_calc_score         ∈ [0, 1]
no_degenerate_score     ∈ [0, 1]

total = 0.20·format + 0.55·answer + 0.25·(chain_to_answer_ok · chain_quality_score)
chain_quality_score = 0.5·liveness + 0.3·step_calc + 0.2·no_degenerate
```

reward 函数文件: `train_pipeline/reward_chaingsm_lbprm_v3_verl.py` (沿用, v4 不重写)
入口脚本: `train_scripts/local/run_grpo_verl_lbprm_v4.sh` (NEW, 拷 v3 改 DATA 路径与 RUN_NAME)

---

## 4. v4 训练计划

- **起点**: `sft_2epoch/20260531_152306/checkpoints/best` (baseline 26.59% on 5467, Original 25.5%)
- **模型**: Qwen2.5-0.5B-Instruct
- **入口**: `train_scripts/local/run_grpo_verl_lbprm_v4.sh`
- **REWARD_PATH**: `train_pipeline/reward_chaingsm_lbprm_v3_verl.py` (v3 原文件, 行为测试已过)
- **DATA**: `chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/verl_grpo_train_neutral.parquet` (**NEW**: NEUTRAL prompt)
- **GRPO 配置**(沿用 v3):
  - `actor_rollout_ref.rollout.n = 4`
  - `actor_rollout_ref.rollout.temperature = 0.9`
  - `actor_rollout_ref.rollout.gpu_memory_utilization = 0.3`
  - `actor_rollout_ref.rollout.max_response_length = 1024`
  - `trainer.total_epochs = 2`
  - `trainer.save_freq = 100` (每 100 step 评测)
  - `trainer.test_freq = -1` (verl 内部 val 关闭)
- **步数上限**: `MAX_STEPS=400`
- **评测**: 每个 save_freq 触发 `eval_vllm_chaingsm`, method=train_json_prompt, 5467 条, 报告 Original(1319) + 整体(5467) + 4 类变体

### 4.1 终止条件

1. 到达 400 step → 自然停
2. 200 step 评估后, Original 连续 2 次 ≤ 上次 AND overall 连续 2 次 ≤ 上次 → 停, 按用户原话"没提升就删 ckpt"

### 4.2 verl 接口契约(沿用 v3 已踩坑)

- `compute_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs)` kwargs 模式
- 返回 `{"score": r, **metrics}` 不返回 tuple / 裸 float
- 所有 early-return 路径返回同样 metrics schema (沿用 v3 `_METRICS_SCHEMA`)

---

## 5. 评测与对比

每次评测会写:
- `eval/step_NNNN/eval_result.json` (over 5467)
- `eval/step_NNNN/predictions.jsonl` (模型全量输出)
- `eval/baseline/eval_result.json` (sft_2epoch 起点 baseline)

需要落地的新文件:
1. `train_scripts/local/run_grpo_verl_lbprm_v4.sh` — v4 训练入口 (NEW)
2. `docs/superpowers/plans/2026-06-14-lbprm-v4-plan.md` — 实施 plan (NEW)
3. `docs/superpowers/reports/2026-06-14-lbprm-v4-report.md` — 跑完反思 (NEW)

需要新做的:
- baseline eval (用 `sft_2epoch` 起点 0.5B)
- 4 个 ckpt eval (step_100/200/300/400)

---

## 6. 风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| prompt 改后模型早期不能输出合法 JSON | format 命中率 0, reward 全 -0.5 | 监控 `format/mean` 与 `reward/mean`; step 50 内 format < 0.5 主动停 |
| 训练集 prompt 改后, gold_trace / distractor_trace 等 reward_model 字段不变, 不影响 reward 计算 | 风险低 | 抽 10 条训练样本看 reward 是否在 [0, 1] 合理 |
| vLLM 评测和训练 prompt 完全一致, 但 max_tokens 2048 仍是 v3 设定 | 风险低 | 监控 response_length 分布 |
| 400 步内模型学不到 "在 NEUTRAL prompt 下答对" | step_300 overall < step_200 | v4 终止, 进入 v5 (liveness 松绑) |

---

## 7. 决策记录

- 2026-06-14: v3 ckpt 清理完成, 保留 best (= step_300, overall 30.47%), 删 step_100/200/400
- 2026-06-14: v4 = 单变量改动 prompt 对齐, 其它 v3 reward 不动
- 2026-06-14: 400 step 封顶, 没涨就停 + 删 ckpt
- 2026-06-14: 跑 v4 期间, 准备 v5 候选 (liveness 松绑 / KL 放宽 / 长度 bonus / prompt 重写 / 数据增强) 等 v4 报告出来后排序
