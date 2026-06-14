# v4 GRPO 训练进度交接(2026-06-14 04:00)

> **本文件目的**:在 v4 训练仍在跑期间,把"为什么 v4 这么设计 / 跑前/中观察 / 候选 v5 方向"全部集中落盘,
> 作为下一次主线接力(等 v4 跑完后定 v5)时的 single source of truth。
> **关联**:v3 报告 `2026-06-13-lbprm-v3-report.md` / v4 spec `2026-06-14-lbprm-v4-design.md` / v4 plan `2026-06-14-lbprm-v4-plan.md`

---

## 0. 项目当前阶段定位

- **不再复现 AbstRaL**(已废弃,见 `_deprecated_2026-06-13-abstral-fail-reproduction-spec.md`)
- **专注自己的 PRM 训练**:SFT → GRPO 路线
- **当前 SFT 起点**:`sft_2epoch/20260531_152306/checkpoints/best`(27.42% / 25.55% Original on 5467)
- **当前 GRPO 起点**:v3 best (= step_300, 30.47% / 29.95% Original)
- **核心目标**:在 Original 1319 子集上持续涨准确率
- **协议**:JSON schema 协议(`train_json_prompt`,greedy decode, max_tokens 2048)

---

## 1. v4 设计回顾(单变量改动:prompt 对齐)

v3 报告 §4.3 指出关键问题:**训练 prompt ≠ 评测 prompt**

| 协议 | 旧 SYSTEM(训练 v3) | 新 SYSTEM(训练 v4 = 评测) |
|---|---|---|
| 训练 | "You are a careful mathematical reasoning assistant. Select only the computation chain that answers the question, **ignore distractor chains**, and return JSON only." | "You are an expert grade-school math solver. **Identify the quantity being asked, solve it carefully step by step**, and follow the requested output format." |
| 评测 | — | NEUTRAL_SYSTEM_PROMPT(同上 v4)|

**v3 的训练 prompt 用了 "ignore distractor chains"**,但 v3 训练集 `verl_grpo_train.parquet` 实际上**只有"真链"**,**根本没有 distractor 链**!
(v3 spec §1 第 2 行:`verl_grpo_train.parquet 7055 条 4 类变体`)

这导致 v3 训练时模型学"ignore distractor"是**没意义**的,反而占用了 rollout 容量,导致 step_300 后进入平台。

**v4 修复**:把训练 parquet 换成 `verl_grpo_train_neutral.parquet`(NEUTRAL prompt,与评测 prompt 一致),其它 v3 reward 函数、权重、KL、LR 全不动。

---

## 2. v4 训练实时观察(step 248/400, ~3h)

### 2.1 训练内 100 步窗口均值

| 窗口 | acc | reward | fmt | answer | c2a | chain_quality | len | entropy | pg_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-100 | 0.400 | 0.486 | 0.989 | 0.400 | 0.433 | 0.634 | 203.7 | 0.057 | — |
| 101-200 | 0.417 | 0.503 | 0.998 | 0.417 | 0.444 | 0.627 | 215.2 | 0.051 | — |
| 201-248 | 0.431 | 0.513 | 0.999 | 0.431 | 0.449 | 0.638 | 205.3 | 0.050 | +0.006 |

**观察**:
- ✅ 训练内 accuracy 单调上升:+3.1pp over step 1-248(0.400→0.431)
- ✅ format 接近 1.0,健康
- ✅ response_length 稳定 203-215(没出现 v2 那种降到 110 的退化)
- ✅ KL loss 0.001-0.002(在 0.04 系数下稳定,policy 跟 ref 接近)
- ✅ entropy 0.050,没崩
- ⚠️ causal_liveness 0.353 偏低(v3 同位置 0.34-0.37,**这是 v3 收紧的副作用**)
- ⚠️ step_calc 0.91,no_degenerate 0.93(链质量子项基本在高位,但 liveness 拖累)

### 2.2 v4 vs v3 baseline 完全一致

- v4 baseline eval: **overall 27.42% / original 25.55%**(同 sft_2epoch 起点)
- 说明 sft_2epoch/best → eval/train_json_prompt 协议是稳定的对照点

### 2.3 距离完成还需多久

- step 248/400 还差 152 步 × ~12s = ~30 min
- 训练完后自动评测 step_100/200/300/400,4 次 × ~10 min = ~40 min
- 预计 v4 全部完成 ~**05:15**

---

## 3. v3 报告里的 6 个 v5 候选方向(原文 §5.2)

按"v3 哪些限制最难突破"排序:

1. **prompt 迁移** — **v4 已做**(改为 NEUTRAL prompt 对齐评测 prompt)
2. **causal_liveness 再松绑** — v3 收紧后 liveness 0.34 偏低,可能挡了真引用信号
3. **加 LR 调度 / 退火** — 保持 5e-7 不变,但 KL 系数 0.04 → 0.02
4. **扩大训练数据多样性** — 加"verl 0.5B 模型在测试集上的错误样本作为负样本"
5. **anti_degenerate 给"格式规整的 chain"额外奖励** — n_steps ∈ [3, 6] 时 +0.05
6. **换 1.5B 模型** — 8-shot CoT 上限 72.25%

**v3 报告原文 v4 计划**:组合 (1)+(2)+(3)+(5),不做 (4)

---

## 4. v4 跑完后,主线 v5 决策逻辑

### 4.1 v4 预期

- v4 baseline = 27.42% / 25.55% Original
- v4 best 预期 = 30-32% / 30-32% Original(跟 v3 best 持平或略涨)
- 4 个 ckpt 都需在 v4 baseline 上比较

### 4.2 v4 跑完后的关键判断

| v4 表现 | v5 决策 |
|---|---|
| v4 best > v3 best (>30.47% / >29.95%) | ✅ 验证 prompt 对齐有效,继续按 v3 报告 §5.2 选 v5 方向(优先 liveness 松绑 + KL 放宽)|
| v4 best ≈ v3 best(±1pp) | ⚠️ prompt 对齐已做完但撞天花板,需要 v5 大改 reward 或换协议 |
| v4 best < v3 best | ❌ prompt 改坏了,需要回滚到 v3 + 重新定 v5 |

### 4.3 v5 候选(本轮主线)

- **A. liveness 松绑**(风险中):重新打开 v3 删的 "value 子串匹配"(v2 失败原因不是子串匹配本身,而是其他 reward hacking 漏洞并存)
- **B. KL 放宽 0.04 → 0.02**(风险低):放 anchor,允许更大学习步幅
- **C. 长度 bonus n_steps ∈ [3, 6] 时 +0.05**(风险低):鼓励"格式规整的 chain"
- **D. 重写 reward**:加入"答对 + chain 真的引用前文 step"作为额外硬门
- **E. 数据增强**:把 SFT 答错的样本加进训练当负样本(重做 parquet)
- **F. 换 1.5B 模型**:8-shot CoT 上限 72.25%,即使 RL 掉 10pp 仍 > 0.6

### 4.4 0.46 目标分析

- 0.5B base + 8-shot CoT 上限 ~43-45%
- 0.5B + JSON schema 协议下 v3 best 30.47% / 29.95%
- 要达 0.46 → 需 +16pp → 0.5B 几乎不可能
- **如果 v5/v6 都涨不动 → 必须换 1.5B** 或换 8-shot CoT 协议(无 JSON schema)

---

## 5. 已踩坑清单(给 v5 参考)

### 5.1 verl 0.8.0 接口契约(v3 spec §4.5)

- `compute_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs)` kwargs 模式
- 返回 `{"score": r, **metrics}`(不是 tuple / 裸 float)
- 所有 early-return 路径返回同样 metrics schema(用 `_METRICS_SCHEMA` 全局常量)

### 5.2 v2 的 4 个失败模式(已避免)

1. liveness 字符串子串匹配 → 巧合拿满分
2. step_consistency tautology → 自圆其说
3. final_consistency 三子项 2/3 重复计 answer
4. chain_quality 67% 占比 + answer 仅 7% → policy 优化稠密信号

### 5.3 v3 的 4 个修好(已验证)

1. chain_to_answer_ok 硬门 ✅
2. causal_liveness 收紧 ✅(可能过头,见 v5 候选 A)
3. format 计分变化 ✅
4. answer 0.55 主导 ✅

### 5.4 JSON 输出约束(给 v5 reward 重写参考)

- 训练 prompt 要求输出 JSON: `{target, selected_steps:[{variable, description, expression, value}], final_expression, answer}`
- 模型必须**严格**按这个 schema 输出
- v3 reward 用 `_parse_response` 容错(支持裸 JSON / ```json``` 包裹 / 嵌在文本里)
- v3 reward 假设字段名 `selected_steps / final_expression / answer`(v3 spec §3.1.1 假设 v3 训练 prompt 用了这套 schema,**v4 NEUTRAL prompt 同样用这套**)

---

## 6. 关键文件清单

| 类型 | 路径 | 状态 |
|---|---|---|
| v3 报告 | `docs/superpowers/reports/2026-06-13-lbprm-v3-report.md` | ✅ 已落盘 |
| v3 spec | `docs/superpowers/specs/2026-06-13-lbprm-v3-design.md` | ✅ |
| v3 plan | `docs/superpowers/plans/2026-06-13-lbprm-v3-plan.md` | ✅ |
| v3 reward | `train_pipeline/reward_chaingsm_lbprm_v3_verl.py` | ✅ 18 个行为测试全过 |
| v3 测试 | `train_pipeline/test_lbprm_v3.py` | ✅ |
| v3 入口 | `train_scripts/local/run_grpo_verl_lbprm_v3.sh` | ✅ |
| v3 best | `outputs/train/local/grpo_verl_lbprm_v3/.../20260614_004034/checkpoints/best` | ✅ 30.47% / 29.95% |
| v4 spec | `docs/superpowers/specs/2026-06-14-lbprm-v4-design.md` | ✅ |
| v4 plan | `docs/superpowers/plans/2026-06-14-lbprm-v4-plan.md` | ✅ |
| v4 入口 | `train_scripts/local/run_grpo_verl_lbprm_v4.sh` | ✅ |
| v4 run 目录 | `outputs/train/local/grpo_verl_lbprm_v4/.../20260614_024915/` | 🏃 训练中 step 248/400 |
| v4 报告 | `docs/superpowers/reports/2026-06-14-lbprm-v4-report.md` | ⏳ 跑完填 |
| **本交接文件** | `docs/superpowers/reports/2026-06-14-v4-progress-handoff.md` | ✅ 本文件 |

---

## 7. 主线下一步动作

1. **等 v4 训练跑完**(~25 min)+ 4 个 ckpt 评测(~40 min)→ 预计 ~05:15
2. **落盘 v4 报告**(`2026-06-14-lbprm-v4-report.md`):填实测数据,对比 v3,定 v5 方向
3. **处置 v4 ckpt**:保留 best,删 step_100/200/400(节省盘)
4. **写 v5 spec** + plan + reward + test + 入口
5. **v5 训练 400 步**,每 100 步评测
6. **v5 跑完落盘报告**,再决定 v6

