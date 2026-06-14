# LB-PRM v6 设计与训练计划 (2026-06-14)

> **作者**: 主线接力(接 LB-PRM v5)
> **核心目的**: 在 0.5B + JSON schema 协议下撞 30% 天花板的情况下,**彻底换协议为 8-shot CoT**,在用户约束 (0.5B 模型, sft_2epoch 后) 范围内达成 0.46 目标
> **本计划范围**: v6 reward 函数 (重写, 适配 8-shot CoT 协议) + 8-shot CoT 训练数据 preprocess + 400 step GRPO 训练 + 5 次评测 + 反思
> **SFT 决策**: **不做 SFT 重训**, 起点直接用 sft_2epoch/best, 训练 prompt 改成 8-shot CoT, 让 GRPO 自适应新协议 (类似 v4 当时做的事, 但改成完整的 8-shot CoT)

---

## 1. 起点与基线

| 项 | 值 |
|---|---|
| **v6 GRPO 起点** | `sft_2epoch/20260531_152306/checkpoints/best` (27.42% / 25.55% Original, 用户指定起点) |
| v5 baseline (对比点) | v3 best (30.47% / 29.95% Original) — 撞天花板 |
| 0.5B + 8-shot CoT 原生 (无 RL) | 24.55% overall / **43.29% Original** (无 RL, 自由推理基线) |
| 0.46 目标 | overall ≥ 46% (主要看 Original, 但 4 类变体也要涨) |

### 1.1 为什么 v6 不用 v3 best / v5 接力

- v3 best / v5 都在 **JSON schema 协议**上 GRPO 训练 400-500 步, 训练 prompt 是 JSON schema (含 "ignore distractor" 旧 prompt)
- 让它们在 8-shot CoT 评测 prompt 上跑, 等于"起点学错协议", GRPO reward 信号会偏向"chain 选取"而不是"自由推理"
- v6 起点 = sft_2epoch/best (用户指定) + 训练 prompt = 8-shot CoT, **GRPO 自适应新协议**, 跟 v3/v5 完全分离

### 1.2 v6 vs v4 关系

- v4: sft_2epoch/best 起点 + NEUTRAL prompt (系统 prompt 改 "expert grade-school math solver") + v3 reward
  - 结果: step_100 Original 29.57% (涨 +1.37pp), step_200-300 掉点, **跟 v3 比没破天花板**
- v6: sft_2epoch/best 起点 + **8-shot CoT 完整 prompt** (8 examples + system) + **v6 reward (8-shot CoT 适配)**
  - 关键差异: v4 只改了 system prompt, v6 改的是**完整 8-shot CoT prompt 模板** (跟评测完全一致)
  - 关键差异: v4 reward 还是 v3 的 (chain 选取), v6 reward 完全重写 (自由推理 + 答案匹配)

### 1.3 0.5B 自由推理基线 (无 RL, 关键数字)

| 模型 + 协议 | Original | overall | 4 类变体均值 |
|---|---:|---:|---:|
| 0.5B + 8-shot CoT 原生 (无 RL) | **43.29%** | 24.55% | 18.61% |
| 0.5B + JSON schema + RL (v3 best) | 29.95% | 30.47% | 30.59% |
| 1.5B + 8-shot CoT 原生 (无 RL) | 72.25% | 49.11% | 38.71% |

**关键洞察**:
- 0.5B + 8-shot CoT 自由推理下 Original **43.29%** (无 RL) — 是**撞天花板上界的关键锚点**
- 0.5B + 8-shot CoT 在 4 类变体上 **18-20%** (干扰敏感, 因为自由推理下模型易被 distractor 干扰)
- 0.5B + 8-shot CoT 整体 24.55% < Original 43.29% (4 类变体拖累)
- **0.46 目标 (overall ≥ 46%) 需要 4 类变体也涨到 30%+** (从 20% 涨 10pp)

---

## 2. v6 设计目标

### 2.1 三个核心方向

| 编号 | 方向 | 改动 | 风险 |
|---|---|---|---|
| **A. 协议换为 8-shot CoT** | 训练 prompt + 评测 prompt + 推理输出全部统一为 8-shot CoT 协议 | 模型自由推理, 答案以 "The final answer is N." 收尾 | **中** — 0.5B 自由推理 43% 原生, 涨点空间大 |
| **B. 重写 reward (8-shot CoT 适配)** | 完全抛弃 selected_steps / final_expression / JSON 字段 | 改用: format (以 N. 收尾) + answer (N 数字匹配) + reasoning_quality (推理步骤质量) | **中** — 新协议没现成 reward, 需新设计 + 测试 |
| **C. 不做 SFT 重训** | sft_2epoch/best 已经 SFT 2 epoch, 直接 GRPO 接力 | 训练 prompt 改成 8-shot CoT, GRPO 自适应 | **低** — 0.5B sft 后已经能跟 8-shot CoT 协议 (43% 原生) |

### 2.2 v6 reward 公式

8-shot CoT 协议下,模型输出是自然语言 + 数字答案,不再有 JSON 结构. Reward 必须重写:

```python
# 公式
total = format_weight · format
      + answer_weight · answer
      + reasoning_quality_weight · reasoning_quality

# 默认权重: format=0.15, answer=0.60, reasoning_quality=0.25
# (相对 v3 的 0.20/0.55/0.25: format 降低, answer 升高, reasoning_quality 保留)

# format = 1.0 if response 收尾于 "The final answer is N." else 0.0
# answer = 1.0 if N 数值 == gold_answer else 0.0
# reasoning_quality = 0.5·step_count + 0.3·numeric_correctness + 0.2·no_contradiction
```

### 2.3 v6 reasoning_quality 详细设计

8-shot CoT 协议下,模型输出是自然语言推理 + "The final answer is N.". 推理质量可拆为:

| 子项 | 描述 | 范围 |
|---|---|---|
| **step_count** | 推理步骤数 (基于句号 + 数字计算行数), 3-7 步满分 1.0, <3 步扣 0.3, >7 步扣 0.1 | [0, 1] |
| **numeric_correctness** | 推理过程中出现的所有算式 (基于 "X op Y = Z" 或 "X op Y" 模式), 数值计算正确率 | [0, 1] |
| **no_contradiction** | 推理过程中所有出现的数值, 是否最终答案 N 与推理过程一致 (N 是不是推理过程中的某个中间结果或最终值) | [0, 1] |

公式: `reasoning_quality = 0.5·step_count + 0.3·numeric_correctness + 0.2·no_contradiction`

**重要**: v6 reasoning_quality **不**是 v3 chain_quality 的同义词:
- v3 chain_quality = 0.5·liveness + 0.3·step_calc + 0.2·no_degenerate (针对 JSON schema chain 结构)
- v6 reasoning_quality = 0.5·step_count + 0.3·numeric_correctness + 0.2·no_contradiction (针对自然语言推理)

### 2.4 v6 训练 prompt 模板 (8-shot CoT)

```python
# 完全复用 code/eval_chaingsm_base_8shot.py: build_qwen_messages
messages = [{"role": "system", "content": EIGHT_SHOT_SYSTEM_PROMPT}]
for ex_q, ex_a in EIGHT_SHOT_EXAMPLES:  # 8 个 (Q, A) 示例
    messages.append({"role": "user", "content": f"Q: {ex_q}\nA:"})
    messages.append({"role": "assistant", "content": ex_a})
messages.append({"role": "user", "content": f"Q: {question}\nA: Let's think step by step."})
```

**注意**: 训练时 8 个 few-shot examples 来自 `EIGHT_SHOT_EXAMPLES` (与评测完全一致),保证训练-评测 prompt 对齐。

### 2.5 v6 评测 prompt 模板

```python
# 完全复用 code/eval_chaingsm_base_8shot.py: build_qwen_messages
# 跟训练 prompt 完全一致 (同 EIGHT_SHOT_EXAMPLES, 同 system, 同 user 格式)
```

评测时使用 `code/eval_chaingsm_base_8shot.py` (5467 条 5 类目, 报告 Original + 整体 + 4 类变体), 跑 `model_outputs/<model_name>/`。

**训练-评测 prompt 100% 对齐** 是 v6 跟 v3/v4/v5 最大的差异, 也是预期能突破 30% 天花板的关键。

---

## 3. v6 训练计划

### 3.1 数据 preprocess (前置, ~10 min)

- **输入**: `chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/` 原始训练数据 (7055 条)
- **输出**: `verl_grpo_train_8shot_cot.parquet` (7055 条, 8-shot CoT 风格 prompt 拼接后 tokenize)
- **prompt 模板**: `build_qwen_messages(question)` (跟评测 prompt 完全一致)
- **verl 输入格式**: `{"prompt": "<rendered_chat_str>", "data_source": "chaingsm_8shot_cot", "extra_info": {"gold_answer": "..."}, "reward_model": {"ground_truth": "..."}}`

### 3.2 GRPO 阶段 (400 步, ~2h)

- **起点**: sft_2epoch/best (`outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_2epoch/20260531_152306/checkpoints/best`)
- **训练集**: `chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/verl_grpo_train_8shot_cot.parquet` (7055 条, 8-shot CoT prompt)
- **测试集**: `chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl` 5467 条 5 类
- **MAX_STEPS**: 400 (用户约束: 每类奖励函数 400 步)
- **SAVE_FREQ**: 80 (5 个 eval 节点: step_80/160/240/320/400)
- **ROLLOUT_N**: 4
- **TRAIN_BATCH_SIZE**: 4
- **MAX_PROMPT_LENGTH**: 1024 (8-shot prompt 较长, base 0.5B + 8 examples + system 估算 ~800-900 token)
- **MAX_RESPONSE_LENGTH**: 512 (CoT 推理通常 < 300 token, 但留余量)
- **ACTOR_LR**: 5e-7
- **KL_LOSS_COEF**: 0.02 (跟 v5 一致, 已验证稳定)
- **v6 reward 权重**: format=0.15, answer=0.60, reasoning_quality=0.25
- **GPU**: RTX 5090 32GB (0.5B 全参 ~2GB + vLLM 7GB + optim 6GB ≈ 15GB/32GB, 充足)

### 3.3 终止条件

1. 到达 400 step → 自然停
2. 200 step 评估后 Original 连续 2 次 ≤ 上次 AND overall 连续 2 次 ≤ 上次 → 停
3. step 80 eval < 0.5B + 8-shot CoT 原生 (43.29% Original) → 立即停, 反思 prompt 模板或 reward 设计
4. step 80 eval < v3 best (30.47% overall) → 立即停, 反思

### 3.4 verl 接口契约 (沿用 v3/v5 已踩坑)

- `compute_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs)` kwargs 模式
- 返回 `{"score": r, **metrics}` 不返回 tuple / 裸 float
- 所有 early-return 路径返回同样 metrics schema (用 `_METRICS_SCHEMA` 全局常量)

---

## 4. 评测与对比

每次评测 (save_freq 触发):
- `eval/step_NNNN/eval_result.json` (over 5467, 5 类目 + overall)
- `eval/step_NNNN/predictions.jsonl` (模型全量输出)
- `eval/baseline/eval_result.json` (sft_2epoch/best 起点 baseline, 用 8-shot CoT 评测)

需要落地的新文件:
1. `train_pipeline/reward_chaingsm_lbprm_v6_verl.py` — v6 reward (新建, 8-shot CoT 协议)
2. `train_pipeline/test_lbprm_v6.py` — 行为测试 (≥ 20 case, 覆盖 reasoning_quality 3 子项)
3. `train_pipeline/preprocess_chaingsm_8shot_cot.py` — 8-shot CoT prompt 训练数据 preprocess (新建)
4. `train_scripts/local/run_grpo_verl_lbprm_v6.sh` — 训练入口 (拷 v5 改 reward/起点/RUN_NAME)
5. `docs/superpowers/plans/2026-06-14-lbprm-v6-plan.md` — 实施 plan
6. `docs/superpowers/reports/2026-06-14-lbprm-v6-report.md` — 跑完后落盘反思

---

## 5. 风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| (A) 8-shot CoT 训练 prompt 过长 (8 examples + system + 当前问题) | max_prompt_length 1024 不够, truncate 丢失 few-shot 信号 | 调到 1536 或 2048, 验证 GPU 内存够 |
| (B) v6 GRPO 起点 sft_2epoch/best 没学过 8-shot CoT 协议, 早期 rollout 答错率高 | 训练内 accuracy 1-30 步 < 0.2 | 监控 accuracy, step 50 内 < 0.2 → 反思 SFT 起点或 prompt 模板 |
| (C) v6 GRPO 训练内 reasoning_quality 涨点不转化到 eval | step_count / numeric_correctness 涨但 Original 不涨 | 反思 reward 子项权重, 调 reasoning_quality 0.25→0.30 |
| (D) 0.5B 在 8-shot CoT 协议下 4 类变体干扰敏感 (20% vs Original 43%) | 4 类变体涨点慢 | 加大 answer weight 0.60→0.70, 让模型专注算对答案 (不论干扰) |
| (E) 8-shot CoT 训练数据 preprocess 拼 prompt 出错 | 训练后模型输出乱 | preprocess 后做小规模 GRPO smoke test (5 step), 抽检输出 |
| (F) 训练/评测 prompt 漂移 (训练时 8-shot, 评测时不是) | 评测时模型不按 8-shot 格式输出 | 评测脚本完全复用 code/eval_chaingsm_base_8shot.py, 训练-评测 prompt 100% 一致 |
| (G) 0.5B + 8-shot CoT 协议上限就是 50-55% Original, 不达 0.46 overall | v6 best Original 50% 但 overall 35% < 46% | 反思 4 类变体处理, 加 SFT 起点 (v6.1) |

---

## 6. 决策记录

- 2026-06-14: v5 训练仍在跑 (step 58/500), v5 设计 = v3 reward 微调 (liveness 松绑 + length_bonus), 撞天花板风险大
- 2026-06-14: v5 quicktest 5 步 + 1 次 eval (30.78%/30.25%) 跟 v3 best (30.47%/29.95%) 持平, 5 步 noise 范围内
- 2026-06-14: 0.5B + 8-shot CoT 原生 (无 RL) Original 43.29%, 远高于 0.5B + JSON schema + RL 30% (差 13pp)
- 2026-06-14: 0.5B + 8-shot CoT 协议是用户约束 (0.5B) 范围内唯一可达 0.46 的路径
- 2026-06-14: v6 = 0.5B + 8-shot CoT 协议 (sft_2epoch/best 起点 + GRPO 400 步, 不做 SFT 重训)
- 2026-06-14: v6 reward 公式 = 0.15·format + 0.60·answer + 0.25·reasoning_quality (8-shot CoT 协议适配)
- 2026-06-14: v6 评测脚本 = 复用 code/eval_chaingsm_base_8shot.py (5467 条 5 类目)
- 2026-06-14: v6 训练 prompt = 8-shot CoT 完整模板 (同 EIGHT_SHOT_EXAMPLES), 评测 prompt = 同 (训练-评测 100% 对齐)
- 2026-06-14: v6 起点 = sft_2epoch/best (用户指定), 不在 v3 best 接力, 避免协议冲突
