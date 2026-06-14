# LB-PRM v6 实施计划 (2026-06-14)

> 关联 spec: `docs/superpowers/specs/2026-06-14-lbprm-v6-design.md`
> 关联报告: `docs/superpowers/reports/2026-06-14-lbprm-v6-report.md` (待落盘)
> 起点: `sft_2epoch/20260531_152306/checkpoints/best` (27.42% / 25.55% Original)
> 目标: 0.46 overall (主线 Original 50%+)

---

## 1. 落地物

| 类型 | 路径 | 状态 |
|---|---|---|
| spec | `docs/superpowers/specs/2026-06-14-lbprm-v6-design.md` | ✅ 已落盘 |
| plan | `docs/superpowers/plans/2026-06-14-lbprm-v6-plan.md` | ✅ 本文件 |
| reward 函数 | `train_pipeline/reward_chaingsm_lbprm_v6_verl.py` | ⏳ 准备落盘 |
| 行为测试 | `train_pipeline/test_lbprm_v6.py` | ⏳ 准备落盘 |
| 数据 preprocess | `train_pipeline/preprocess_chaingsm_8shot_cot.py` | ⏳ 准备落盘 |
| 训练入口 | `train_scripts/local/run_grpo_verl_lbprm_v6.sh` | ⏳ 准备落盘 |
| 反思报告 | `docs/superpowers/reports/2026-06-14-lbprm-v6-report.md` | ⏳ 跑完落盘 |

---

## 2. 训练参数 (全部硬编码, 可在脚本顶部覆盖)

- **起点**: `outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_2epoch/20260531_152306/checkpoints/best` (= 27.42% / 25.55% Original, JSON schema 协议起点)
- **训练集**: `chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/verl_grpo_train_8shot_cot.parquet` (7055, 8-shot CoT prompt, **新建**)
- **测试集**: `chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl` (5467, 8-shot CoT 评测)
- **MAX_STEPS**: 400 (用户约束: 每类奖励函数 400 步)
- **SAVE_FREQ**: 80 (5 个 eval 节点: step_80/160/240/320/400)
- **ROLLOUT_N**: 4, **TRAIN_BATCH_SIZE**: 4
- **MAX_PROMPT_LENGTH**: 1024 (8-shot prompt 估算 ~800-900 token)
- **MAX_RESPONSE_LENGTH**: 512 (CoT 推理通常 < 300 token, 但留余量)
- **ACTOR_LR**: 5e-7, **KL_LOSS_COEF**: 0.02 (跟 v5 一致, 已验证稳定)
- **v6 reward 权重**: format=0.15, answer=0.60, reasoning_quality=0.25

---

## 3. 与 v3/v4/v5 唯一区别

- **起点**: sft_2epoch/best (用户指定), 不在 v3 best / v4 / v5 接力
- **训练 prompt**: 8-shot CoT 完整模板 (同 EIGHT_SHOT_EXAMPLES), 评测 prompt = 训练 prompt (100% 对齐)
- **reward**: 完全重写, 适配 8-shot CoT 协议 (无 JSON schema 字段)
  - format = 1.0 if 收尾于 "The final answer is N." else 0.0
  - answer = 1.0 if N 数值 == gold_answer else 0.0
  - reasoning_quality = 0.5·step_count + 0.3·numeric_correctness + 0.2·no_contradiction
- **MAX_STEPS**: 400 (用户约束)
- **MAX_PROMPT_LENGTH**: 768 → 1024 (8-shot prompt 较长)
- **MAX_RESPONSE_LENGTH**: 1024 → 512 (CoT 推理通常 < 300 token)
- 其它全部沿用 v3/v4/v5

---

## 4. 终止与判断

- 跑满 400 step → 自然停
- 200 step 评估后 Original 连续 2 次 ≤ 上次 AND overall 连续 2 次 ≤ 上次 → 停
- step 80 eval < 0.5B + 8-shot CoT 原生 (43.29% Original) → 立即停, 反思 prompt 模板或 reward 设计
- 没收益的 checkpoint 训练后**删除** (按用户要求, 只删 ckpt, 留代码+报告+eval_result.json)
- **v6 起步 baseline = sft_2epoch/best 用 8-shot CoT 评测** (27.42% JSON → 预期 35-40% 8-shot, 因为 prompt 对齐)
- **v6 成功标准**: v6 best > 0.5B + 8-shot CoT 原生 (43.29% Original, 24.55% overall)

---

## 5. 监控要点

- `metrics/train_metrics.jsonl` 每 step 的 reward_components 系列 (v3/v4/v5 已验证 verl 会自动写)
- `eval/step_NNNN/eval_result.json` 的 overall_accuracy + 5 类目
- **关键监控**:
  - format 触发率 (期望 95%+)
  - answer 准确率 (期望 50%+)
  - reasoning_quality 涨点 (step_count / numeric_correctness / no_contradiction 3 子项分布)
  - KL 锚定 0.02 (跟 v5 一致)
  - entropy 0.05-0.08 (健康)

---

## 6. 风险与回退

- **风险 1: 8-shot prompt 过长导致 truncate**
  - 表现: rollout 输出跟评测 prompt 不一致
  - 应对: 调到 MAX_PROMPT_LENGTH=1536 或 2048, 验证 GPU 内存够
- **风险 2: sft_2epoch/best 起点没学过 8-shot CoT 协议, 早期 rollout 答错率高**
  - 表现: 训练内 accuracy 1-50 步 < 0.2
  - 应对: 监控 accuracy, step 50 内 < 0.2 → 反思 prompt 模板
- **风险 3: v6 GRPO reasoning_quality 涨点不转化到 eval**
  - 表现: step_count 涨但 Original 不涨
  - 应对: 反思 reward 子项权重, 调 reasoning_quality 0.25→0.30
- **风险 4: 0.5B + 8-shot CoT 上限 50-55% Original, 不达 0.46 overall**
  - 表现: v6 best Original 50% 但 overall 35% < 46%
  - 应对: 反思 4 类变体处理, 加 SFT 起点 (v6.1)
- **风险 5: 训练/评测 prompt 漂移**
  - 表现: 评测时模型不按 8-shot 格式输出
  - 应对: 评测脚本完全复用 code/eval_chaingsm_base_8shot.py

---

## 7. 执行时间预算

- 数据 preprocess: ~10 min (7055 条 prompt 拼接 + tokenize)
- baseline eval (sft_2epoch/best 用 8-shot CoT): ~5 min
- 100 step 训练: ~20 min (12s/step, 0.5B)
- 每次 vLLM eval (5467): ~10 min
- 总计: ~10 + 5 + 5×(20+10) = 165 min ≈ 2.75 h
- 缓冲 + 反思: 3 h 整

---

## 8. v7 候选 (等 v6 报告出来再选)

按 v6 报告结果定 v7 方向:
1. **加 SFT 起点**: v6.1 = 0.5B base + 8-shot CoT SFT 2 epoch + GRPO 400 步 (跟 v6 比, SFT 让模型先学会 8-shot 协议)
2. **加更多训练数据**: 加 gsm8k_train.jsonl (14946 条) 进 RL 训练集
3. **加 answer weight**: v6.1 = answer 0.60→0.70, 让模型专注算对答案
4. **加思考模式触发**: 在 prompt 加 "Think carefully step by step" 鼓励长 chain

---

## 9. v6 实施步骤 (按时间顺序)

### 步骤 1: 写 v6 reward (`train_pipeline/reward_chaingsm_lbprm_v6_verl.py`)
- 复用 `code/gsm_answer_extractor.py: extract_answer, is_correct`
- 8-shot CoT 适配: format (收尾 "The final answer is N."), answer (N 匹配), reasoning_quality (3 子项)
- 沿用 verl 接口契约 (kwargs 模式 + `{"score": r, **metrics}` + `_METRICS_SCHEMA`)

### 步骤 2: 写 v6 行为测试 (`train_pipeline/test_lbprm_v6.py`)
- ≥ 20 case, 覆盖 reasoning_quality 3 子项 + format + answer 边界
- 复用 v3/v5 测试结构

### 步骤 3: 写 v6 数据 preprocess (`train_pipeline/preprocess_chaingsm_8shot_cot.py`)
- 复用 `code/eval_chaingsm_base_8shot.py: build_qwen_messages` + EIGHT_SHOT_EXAMPLES
- 输出 `verl_grpo_train_8shot_cot.parquet`

### 步骤 4: 写 v6 GRPO 训练入口 (`train_scripts/local/run_grpo_verl_lbprm_v6.sh`)
- 拷 v5 改 reward/起点/RUN_NAME/MAX_PROMPT_LENGTH/MAX_RESPONSE_LENGTH/MAX_STEPS/SAVE_FREQ

### 步骤 5: smoke test (5 步 + 1 次 eval, ~15 min)
- 验证 v6 reward + 8-shot CoT prompt + preprocess 拼装正确
- 5 步内 accuracy 预期 ~0.2-0.3 (sft 起点没学过 8-shot, 早期 rollout 答错率高, 正常)

### 步骤 6: 完整 400 步训练 + 5 次 eval (~3h)
- 监控 training progress + 训练内 metrics + 5 次 eval

### 步骤 7: 落盘 v6 报告 + 反思 (`docs/superpowers/reports/2026-06-14-lbprm-v6-report.md`)
- 报告 Original (1319) + 整体 (5467) + 4 类变体
- 反思 v6 vs v3/v5 涨点是否转化, 0.46 目标是否达成
- 决定 v7 方向

---

## 10. 决策记录

- ✅ 2026-06-14: v6 spec/plan 落盘
- ⏳ 2026-06-14: v6 reward + test + preprocess + entry 落盘
- ⏳ 2026-06-14: v6 smoke test (5 步)
- ⏳ 2026-06-14: v6 完整 400 步训练 + 5 次 eval
- ⏳ 2026-06-14: v6 报告 + 反思 + 0.46 目标达成判断
