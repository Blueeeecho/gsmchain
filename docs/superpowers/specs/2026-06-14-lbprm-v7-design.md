# LB-PRM v7 设计与训练计划 (2026-06-14)

> 接 v6 反思 (`docs/superpowers/reports/2026-06-14-v6-prereflex.md`).
> 用户硬指标: `original >= 0.46` (1319 条干净集 original 子集, 0.5B + 8-shot CoT 协议).
> 实测锚点: 0.5B base + 8-shot CoT 原生 (无 RL) = original 0.4329.

---

## 1. 目标与可达成性

- 主线目标: original 子集准确率 >= 0.46 (涨 >= 2.71pp, 相对 0.5B base 原生 0.4329).
- 次要观测: overall >= 0.25 (从 0.2455), 4 类变体不倒退.
- 训练步数: 1000 步 (用户授权, 5 次 eval 节点: 200/400/600/800/1000).
- 时间预算: 0.5B + 单卡 RTX 5090 ~ 1.0-1.5s/step + 5 次 5467 条 vLLM 评测各 ~5min ≈ 3.0-3.5h.

## 2. 起点与理由 (推翻 v3-v6 的所有起点)

| 候选 | original baseline | 备注 |
|---|---:|---|
| 0.5B + sft_2epoch/best + 8-shot CoT | 0.1296 (实测, 1319) | SFT 摧毁 8-shot 能力, 排除 |
| 0.5B + v3 best + 8-shot CoT | (未测, 估计 5-15%) | JSON 协议训练残留, 排除 |
| **0.5B base + 8-shot CoT 原生** | **0.4329 (实测, 1319)** | **采用 — 干净起点, 无协议污染** |
| 0.5B base + 1 epoch 8-shot SFT (候选 E) | 未实测, 估计 0.50-0.55 | 留作 v7.1 备选 |

**采用**: 直接用 `/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct` 作为 v7 起点.

## 3. 训练 prompt 协议 (与评测 100% 对齐)

- 训练 prompt: `chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/verl_grpo_train_8shot_cot.parquet`
- 训练 prompt 内容: 完整 8-shot CoT 模板 (system + 8 轮 user/assistant 示例 + 当前题)
- 评测 prompt: `code/eval_chaingsm_base_8shot.py: build_qwen_messages` 同模板
- 答案提取: `code/gsm_answer_extractor.py: extract_answer, is_correct` (仓库唯一标准)
- 测试集: `chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl` (5467 条)

## 4. v7 reward (推翻 v6, v6b, 简化结构)

v6 / v6b 的 4 子项结构在 0.5B 自由推理下问题:
- 0.5B base 8-shot 原生 format=0.95+, step_count=0.74, no_contradiction=0.875 — 这三项几乎满分, 区分度低
- numeric_correctness=0.22 — 真正有信号的空间在算式
- 0.5B 自由推理下撞 max_tokens=512 占 32.8% — 起点本身能写出算式

**v7 简化设计 (3 子项, 总权重 = 1.0)**:
```
total = 0.70 * answer
      + 0.20 * numeric_correctness
      + 0.10 * format
```

- `answer = 1.0` 当 extract_answer(pred) 与 gold 数值匹配, 否则 0.0
- `numeric_correctness` = 算式正确率 (0-1, 等同 v6b 算法)
- `format` = 收尾于 "The final answer is N." (通过 extract_answer 能否取到值判定)

**理由**:
- answer 0.70 是最大信号, 直接对齐目标 "original 数字对"
- 砍 step_count (区分度低, 0.74 已接近满分) 和 no_contradiction (0.875 满分, 噪声)
- 砍 equation_count_bonus (reward 噪音, 鼓励 "凑算式" 反而不利于推理质量)
- 保留 numeric_correctness (0.22 → 是 0.5B 真正的薄弱点)
- format 0.10 (起 baseline 已 0.95+, 但仍是必要 hard constraint)

**对比 v3/v5/v6/v6b**:
| 维度 | v3 | v5 | v6 | v6b | **v7** |
|---|---|---|---|---|---|
| answer 权重 | 0.55 | 0.55 | 0.60 | 0.55 | **0.70** |
| format 权重 | 0.20 | 0.20 | 0.15 | 0.10 | **0.10** |
| reasoning 权重 | 0.25 | 0.25 | 0.25 | 0.35 | **0.20** (只 numeric) |
| 子项数 | 3 | 3 | 3 | 4 | **2 (只有 numeric)** |
| 协议 | JSON | JSON | 8-shot | 8-shot | **8-shot** |

## 5. 训练参数 (相对 v6 微调)

- **起点**: `/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct`
- **训练集**: `verl_grpo_train_8shot_cot.parquet` (7419 条, 跟 v6 一致)
- **测试集**: `gsm8k_test_clean.jsonl` (5467 条, original 1319 条)
- **MAX_STEPS**: 1000
- **SAVE_FREQ**: 200 (5 个 eval 节点: 200/400/600/800/1000)
- **ROLLOUT_N**: 4
- **TRAIN_BATCH_SIZE**: 4
- **MAX_PROMPT_LENGTH**: 1280 (8-shot prompt max=1029, 留 25% 余量)
- **MAX_RESPONSE_LENGTH**: 512 (CoT 通常 < 300)
- **ACTOR_LR**: 5e-7 (跟 v5/v6 一致, 已验证稳定)
- **KL_LOSS_COEF**: 0.02 (跟 v5/v6 一致)
- **ROLLOUT_TEMPERATURE**: 0.9
- **EVAL_BASELINE**: 1 (跑一次 base + 8-shot 评测作为锚点)

## 6. 终止条件 (按 prereflex 报告框架)

1. 达到 1000 step 自动停
2. step 200 eval original >= 0.46 → 视为目标达成, 后续 eval 用来观察收敛
3. step 200 eval original < 0.43 (= base 8-shot 原生) → 立即停, 反思起点或 prompt
4. step 200-400 连续 2 次 original 持平且 < 0.46 → 停, 反思 reward 权重
5. step 600 eval original >= 0.46 → 强制 stop at 600 (节省 800/1000)

## 7. 监控

- `metrics/train_metrics.jsonl` 每 step:
  - `reward/accuracy`, `critic/rewards/mean`, `critic/score/mean`
  - `reward_components/format/mean`, `reward_components/answer/mean`, `reward_components/numeric_correctness_score/mean`
  - `actor/entropy`, `actor/pg_loss`, `timing_s/step`
- `eval/step_NNNN/summary_overall.json` + `summary_by_category.json`:
  - 关注 original 类目准确率 (主目标)
  - 4 类变体不退化

## 8. 风险与回退

| 风险 | 应对 |
|---|---|
| 0.5B base 起点在 8-shot 协议下自由推理 entropy 高, 训练波动大 | 监控 entropy, > 0.15 → 反思 LR / KL |
| GRPO 1000 步撞 0.45 (差 1pp) | 反思 answer 权重 (0.70→0.80), 或 max_response_length 512→768 |
| 4 类变体反向退化 | 监控 by_category.json, 任何一类降 5pp → 停, 反思 reasoning |
| 时间超预算 (3.5h → 5h) | 接受, 优先保证 step 200 真实数字, 后续 200/400 可评估后决定是否继续 |

## 9. 落盘物

| 类型 | 路径 |
|---|---|
| spec | `docs/superpowers/specs/2026-06-14-lbprm-v7-design.md` (本文件) |
| plan | `docs/superpowers/plans/2026-06-14-lbprm-v7-plan.md` |
| reward | `train_pipeline/reward_chaingsm_lbprm_v7_verl.py` |
| reward 测试 (TDD) | `train_pipeline/test_lbprm_v7.py` |
| 训练入口 | `train_scripts/local/run_grpo_verl_lbprm_v7.sh` |
| 报告 | `docs/superpowers/reports/2026-06-14-lbprm-v7-report.md` |

## 10. 决策记录

- 2026-06-14: v7 spec/plan 落盘
- 2026-06-14: v7 reward + test + entry 落盘
- 2026-06-14: v7 smoke (5 步 + step_5 eval)
- 2026-06-14: v7 完整 1000 步训练 + 5 次 eval
- 2026-06-14: v7 报告 + original >= 0.46 验证
