# LB-PRM v5 设计与训练计划(2026-06-14)

> **作者**: 主线接力(接 LB-PRM v4)
> **核心目的**: 在 v3 (30.47% / 29.95% Original) 撞到的 30% 天花板上,**通过多变量组合**(liveness 松绑 + KL 放宽 + 长度 bonus)**找 0-2pp 突破**。
> **本计划范围**: v5 reward 函数(基于 v3 改写) + 500 step GRPO 训练 + 每 100 step 评测 + 反思。
> **链路关系**: v5 接力 v3 best 起点,**不用 v4 任何 ckpt** (v4 step_200 掉点, v4 step_100 整体 29.76% < v3 best 30.47%)。

---

## 1. 起点与基线

| 项 | 值 | 路径 |
|---|---|---|
| **v5 起点** | **v3 best** (GRPO step_300, overall 30.47% / original 29.95%) | `outputs/train/local/grpo_verl_lbprm_v3/Qwen2.5-0.5B-Instruct/grpo_verl_v3_smoke/20260614_004034/checkpoints/best` |
| v3 best backup | sft_2epoch/best(回滚点, baseline 27.42% / 25.55%)| `outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_2epoch/20260531_152306/checkpoints/best` |
| 训练集 | `verl_grpo_train_neutral.parquet` 7055 条 NEUTRAL prompt | `chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/verl_grpo_train_neutral.parquet` |
| 测试集 | `gsm8k_test_clean.jsonl` 5467 条 5 类 | `chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl` |
| 上次 GRPO | v4 step_100 = 29.76% / 29.57% (best) | v4 report §2 |
| v3 报告 §5.2 候选 6 个 | 见下文 §2.2 | |

### 1.1 为什么 v5 接力 v3 best 不接力 v4 step_100

| 候选起点 | overall | original | 风险 |
|---|---:|---:|---|
| **v3 step_300 best** | **30.47%** | **29.95%** | 起点 prompt 旧 (v3 SYSTEM 含 "ignore distractor"),但 4 步 ckpt 都 30%+,稳 |
| v4 step_100 | 29.76% | 29.57% | 起点 prompt 新 (NEUTRAL),但 step_200 之后掉点,接力 v5 风险高 |
| v4 step_400 | 30.02% | 29.42% | 起点 prompt 新 (NEUTRAL),整体好但 original 最低 |
| sft_2epoch/best | 27.42% | 25.55% | 回滚到 baseline,放弃 v3/v4 全部训练 |

**决策**: **v5 起点 = v3 step_300 best** (整体最高 + original 最高 + 4 步全稳)。

但 v5 用 NEUTRAL prompt 训练集 (跟 v4 一致),**v3 起点 + NEUTRAL 训练 = "起点用旧 prompt 训练过, 进入新 prompt 训练" 的迁移场景**。这是有意为之:
- 让 v3 best 那个 30% 水平的 chain 选取能力**适应 NEUTRAL prompt** (跟评测 prompt 对齐)
- 同时叠加 v5 改写后的 liveness reward

### 1.2 v4 step_100 留作 v5 备选起点

如果 v5 from v3 best 跑 100 步后掉点 > 2pp,**立即**切到 v4 step_100 重跑 v5。

---

## 2. v5 设计目标

### 2.1 三个核心方向(按 v3 报告 §5.2 排序组合 2+3+5)

| 编号 | 方向 | v3 当前 | v5 改动 | 风险 |
|---|---|---|---|---|
| **A. liveness 松绑** | 重新打开 v3 删的 "value 子串匹配" | v3 causal_liveness 仅认 variable 引用 | v5 causal_liveness 同时认 variable 引用 + value 子串匹配(用 ast.walk 1e-3 容差)| **中** — v2 失败原因不是子串匹配本身, 而是其他 reward hacking 漏洞, v3 已修 |
| **B. KL 放宽** | 0.04 (v3/v4) | 0.02 | **低** — 减半 KL 锚定, 允许更大学习步幅 |
| **C. 长度 bonus** | 无 | n_steps ∈ [3, 6] 时 +0.05 | **低** — 鼓励"格式规整的 chain" |

### 2.2 不做的方向(v3 报告 §5.2 候选 1, 4, 6)

- **候选 1 prompt 迁移**: v4 已做完(单变量改动), v5 接力 v3 best 起点 + NEUTRAL 训练集 = 隐式完成 prompt 迁移
- **候选 4 数据增强**: 优先级低, 留 v6/v7
- **候选 6 换 1.5B**: 留 v6 (如果 v5 也撞天花板)

### 2.3 v5 reward 公式

```
v3 components:
  chain_to_answer_ok       ∈ {0, 1}                    # 保留
  causal_liveness_score   ∈ [0, 1]                    # 松绑 (新增 value 子串匹配)
  step_calc_score         ∈ [0, 1]                    # 保留
  no_degenerate_score     ∈ [0, 1]                    # 保留

v5 新增:
  length_bonus            ∈ {0.0, 0.05}                # n_steps ∈ [3, 6] 时 +0.05

v3 权重:
  format_weight = 0.20
  answer_weight = 0.55
  chain_quality_weight = 0.25
  chain_quality_score = 0.5·liveness + 0.3·step_calc + 0.2·no_degenerate

v5 公式 (v3 基础上 + length_bonus):
  total = format_weight·format
        + answer_weight·answer
        + chain_quality_weight·(chain_to_answer_ok · chain_quality_score)
        + length_bonus
```

### 2.4 v5 liveness 松绑实现细节

v3 `_causal_liveness` 实现(reward_chaingsm_lbprm_v3_verl.py:185-227):
```python
# (a) variable 作为单词出现在后续 step 的 expression 里
# (b) variable 出现在 final_expression 里
# (c) 末步:eval(step.expression) == eval(final_expression)
# (d) 末步:value 严格等于 pred_answer
```

**v5 新增**:
```python
# (a'') value 字符串子串匹配 (v2 失败的子表达式扫描, v5 改用 ast.walk 1e-3 容差)
#   - 遍历后续 step 的 expression
#   - ast.walk 取所有子表达式节点
#   - _safe_eval(sub_expr) 得到数值
#   - abs(eval_val - step.value) < 1e-3 视为活
# (a''') step.value 出现在 final_expression 的子表达式里(同样容差)
```

**为什么 v2 失败, v5 应该能成功**:
- v2 失败原因 = (a) 字符串子串 + (a') 子表达式扫描 + (b) step_consistency tautology + (c) final_consistency 重复计 answer 4 个漏洞并存
- v5 只重启 v2 的 (a') 子表达式扫描, 同时保留 v3 的所有修复:
  - chain_to_answer_ok 硬门 ✅
  - step_calc_score(不再是 tautology, 因 v3 chain_to_answer_ok 已掐住答错)✅
  - final_consistency 已被 chain_to_answer_ok 替代 ✅
  - answer 0.55 主导 ✅
- 所以 v5 (a') 重启**不会**触发 v2 同样的 hacking

---

## 3. v5 训练计划

- **起点**: v3 best (`grpo_verl_lbprm_v3/.../20260614_004034/checkpoints/best`)
- **模型**: Qwen2.5-0.5B-Instruct
- **入口**: `train_scripts/local/run_grpo_verl_lbprm_v5.sh`
- **REWARD_PATH**: `train_pipeline/reward_chaingsm_lbprm_v5_verl.py` (新建)
- **GRPO 配置** (v3 基础上 + KL 放宽):
  - `actor_rollout_ref.rollout.n = 4`
  - `actor_rollout_ref.rollout.temperature = 0.9`
  - `actor_rollout_ref.rollout.gpu_memory_utilization = 0.3`
  - `actor_rollout_ref.rollout.max_response_length = 1024`
  - `trainer.total_epochs = 1`
  - `trainer.save_freq = 100` (每 100 step 保存, 触发评测)
  - `trainer.test_freq = -1` (verl 内部 val 关闭)
- **KL 系数**: `KL_LOSS_COEF=0.02` (v3/v4 的 0.04 → 0.02)
- **步数上限**: `MAX_STEPS=500` (v3 在 300 步撞平台, v5 多给 200 步)
- **评测**: 每个 save_freq 触发 `eval_vllm_chaingsm`, method=train_json_prompt, 5467 条, 报告 Original(1319) + 整体(5467) + 4 类变体

### 3.1 终止条件

1. 到达 500 step → 自然停
2. 200 step 评估后, Original 连续 2 次 ≤ 上次 AND overall 连续 2 次 ≤ 上次 → 停
3. step 100 eval 比 v3 best baseline 掉 > 2pp → 立即停, 切 v4 step_100 起点重跑

### 3.2 verl 接口契约(沿用 v3 已踩坑)

- `compute_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs)` kwargs 模式
- 返回 `{"score": r, **metrics}` 不返回 tuple / 裸 float
- 所有 early-return 路径返回同样 metrics schema (沿用 v3 `_METRICS_SCHEMA` + 新增 length_bonus / n_steps)

---

## 4. 评测与对比

每次评测会写:
- `eval/step_NNNN/eval_result.json` (over 5467)
- `eval/step_NNNN/predictions.jsonl` (模型全量输出)
- `eval/baseline/eval_result.json` (v3 best 起点 baseline)

需要落地的新文件:
1. `train_pipeline/reward_chaingsm_lbprm_v5_verl.py` — v5 reward (新建, 基于 v3 改写)
2. `train_pipeline/test_lbprm_v5.py` — 行为测试 (≥ 20 case, 覆盖 liveness 松绑新规则)
3. `train_scripts/local/run_grpo_verl_lbprm_v5.sh` — 训练入口 (拷 v4 改 reward/起点/RUN_NAME/KL_COEF)
4. `docs/superpowers/plans/2026-06-14-lbprm-v5-plan.md` — 实施 plan
5. `docs/superpowers/reports/2026-06-14-lbprm-v5-report.md` — 跑完后落盘反思

---

## 5. 风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| (a') value 子串匹配触发了新的 hacking | 训练内 accuracy 早期不涨 / chain 越写越短 | 监控 response_length 和 accuracy, step 50 内 < 100 / 准确率不涨 → 立即停 |
| KL 放宽 0.02 仍然不够 | step_100-300 跟 v3 持平 | step_300 eval 后定 v6 = KL 0.01 或换 KL loss type |
| 长度 bonus 0.05 太小没意义 | 跟 v3 没差别 | 调试时改 0.10 试 |
| v3 best 起点 + NEUTRAL 训练集迁移失败 | step_100 掉 > 2pp | 切 v4 step_100 起点重跑 v5 |
| v5 训练 500 步全平台 | 跟 v3 一样 30% 撞天花板 | v5 报告反思 → v6 换 1.5B |

---

## 6. 决策记录

- 2026-06-14: v4 ckpt 处置完成, 保留 step_100/400 (29.76%/30.02% overall), 删 step_200/300
- 2026-06-14: v4 报告反思 = prompt 对齐单变量不够, 进入 v5 多变量组合
- 2026-06-14: v5 = (2) liveness 松绑 + (3) KL 0.04→0.02 + (5) 长度 bonus, 起点 v3 best
- 2026-06-14: v5 步数 500 步封顶(比 v3/v4 的 400 步多 100 步, 给 v5 改动更多收敛时间)
- 2026-06-14: v5 失败则 v6 = 换 Qwen2.5-1.5B + 8-shot CoT 协议(0.5B + JSON 协议 + RL 天花板 ~30-32% 已撞)
- 2026-06-14: 0.46 目标在 0.5B + JSON 协议下不可达, 须 1.5B 才能突破

