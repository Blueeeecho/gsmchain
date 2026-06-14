# LB-PRM v3 设计与训练计划(2026-06-13)

> **作者**: 主线接力(接 LB-PRM v2)
> **核心目的**: 修复 v2 在 `grpo_verl_lbprm_base_2ep_200step` 上观察到的 **reward hacking** —— chain_quality 单调上升、answer 不动、response 越来越短、vLLM 评测从 7.24% 掉到 5.87%。
> **本计划范围**: v3 reward 函数 + 一次 400 step GRPO 烟测 + 反思。每版 reward 都是单 run、400 step 封顶、按需续命或换版。

---

## 1. 起点与基线

| 项 | 值 | 路径 |
|---|---|---|
| SFT 起点 | `sft_2epoch/20260531_152306` best(26.59% on 5467) | `outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_2epoch/20260531_152306/checkpoints/best` |
| 训练集 | `verl_grpo_train.parquet` 7055 条 4 类变体 | `chaingsm_data/data/final/rl_preprocessed/gsm8k_train_balanced_one_variant_14946/verl_grpo_train.parquet` |
| 测试集 | `gsm8k_test_clean.jsonl` 5467 条 5 类 | `chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl` |
| 上次 GRPO 烟测 | `grpo_verl_lbprm_base_2ep_200step/20260609_153629` | baseline 7.24% → step200 7.81% → step400 5.87% |
| 上次 v2 reward 训练内分布 | answer 4% 全程不动,format 78→99,liveness 0.55→0.89,step_cons 0.19→0.71,response 238→110 | 见前一轮反思 |

**v2 reward 公式回顾**:

```
total = 0.2·format + 0.4·answer + 0.4·chain_quality_score
chain_quality_score = (0.4·liveness + 0.3·step_consistency + 0.3·final_consistency) · anti_degenerate
```

**v2 的 4 个失败模式**:
1. `liveness` 字符串子串匹配(0.2 出现在 0.2*50 里算 liveness)→ 巧合拿满分
2. `step_consistency` 是 tautology(eval(expr)==value,跟答案对错无关)→ 自圆其说
3. `final_consistency` 三子项里有 2/3 在重复计 answer 命中
4. 整体:chain_quality 67% 占比 + answer 仅 7% → policy 优先优化稠密信号

---

## 2. v3 设计目标

- **首要**:把 policy 梯度推回"答对"上。answer 信号必须占据总 reward 的主导地位。
- **其次**:chain_quality 作为"答对之后的锦上添花",不能在答错时给高奖励。
- **再次**:liveness 收紧,避免"巧合匹配"被识别为 liveness。
- **不追求**:56-case 行为测试 100% 复现 v2 的具体分数。v3 行为测试要重新校核。

---

## 3. v3 公式

```
chain_to_answer_ok       ∈ {0, 1}                  # 硬门:chain 真的能推出 answer
causal_liveness_score   ∈ [0, 1]                  # 收紧:每 step value 被后续 step 真引用
step_calc_score         ∈ [0, 1]                  # eval(step.expression) == step.value(保留)
no_degenerate_score     ∈ [0, 1]                  # 抗退化(保留)

total = 0.20·format
      + 0.55·answer
      + 0.25·(chain_to_answer_ok · (0.5·causal_liveness_score + 0.3·step_calc_score + 0.2·no_degenerate_score))
```

### 3.1 三个新定义

#### 3.1.1 `chain_to_answer_ok`(硬门)

```
chain_to_answer_ok = 1 iff (eval(final_expression) == pred_answer)
                       OR (任一 step expression 求值 == pred_answer)
                  else 0
```

**含义**:这一项是 0 还是 1,直接决定 chain_quality 这一坨奖励是否生效。

#### 3.1.2 `causal_liveness_score`(收紧版 liveness)

v2 的 (a) 字符串子串 + (a') 子表达式扫描两个口子都放水。v3 改为:
- 删除子表达式扫描(0.2 巧合不再被识别)
- 删除 value 字符串子串匹配
- `variable` 用 `\b<var>\b` 单词边界匹配,避免 "a" 出现在 "apple" 里被算引用
- 末步:eval(step.expression) == eval(final_expression) 视为活
- 末步:value_k 与 pred_answer 字符串严格相等 视为活

#### 3.1.3 `step_calc_score` 与 `no_degenerate_score`

跟 v2 保持一致;`no_degenerate_score` 不再做"final_expression == 字面 answer"的额外扣分(因为 `chain_to_answer_ok` 已经把这条收紧)。

---

## 4. v3 训练计划

- **起点**:`sft_2epoch/20260531_152306/checkpoints/best`
- **模型**:Qwen2.5-0.5B-Instruct(沿用)
- **入口**:`train_scripts/local/run_grpo_verl.sh`
- **REWARD_PATH**:`train_pipeline/reward_chaingsm_lbprm_v3_verl.py`(新建,基于 v2 改写)
- **GRPO 配置**(沿用 grpo_verl.yaml,只改 reward_path):
  - `actor_rollout_ref.rollout.n = 4`
  - `actor_rollout_ref.rollout.temperature = 0.9`
  - `actor_rollout_ref.rollout.gpu_memory_utilization = 0.3`
  - `actor_rollout_ref.rollout.max_response_length = 1024`
  - `trainer.total_epochs = 2`(让 max_steps 生效)
  - **`trainer.save_freq = 100`**(每 100 step 保存,触发评测)
  - `trainer.test_freq = -1`(verl 内部 val 关闭)
- **步数上限**:`MAX_STEPS=400`
- **评测**:每个 save_freq 触发 `eval_vllm_chaingsm`,读 `eval_result.json` 的 `overall_accuracy` 和 `by_category[0..5]`,报告 Original(1,319)准确率 + 整体(5,467)准确率
- **终止条件**:
  - 到达 400 step → 停止
  - 任一档评测相比上次**没提升**(Original ≤ 上次 AND overall ≤ 上次)→ 记为"无提升",在 200 step 评估一次,200~400 期间连续两次"无提升"才停(避免偶发抖动误杀)
- **保留**:
  - 代码、reward 函数、spec、总结、metrics 永久保留
  - checkpoint 视结果:**有收益**保留 best,没收益按用户要求**删除 checkpoint**(但保留 eval_result.json 作为证据)

---



## 4.5 已踩坑的 verl 0.8.0 接口契约(实测 2026-06-14)

- `compute_reward` 签名必须支持 **kwargs 模式**,因为 verl `naive` reward manager 按以下方式调:
  ```python
  compute_score(
      data_source=...,
      solution_str=...,
      ground_truth=...,
      extra_info=...,
      **extra_reward_kwargs,
  )
  ```
  **不能**写 `def compute_reward(data_sources, solution_strs, ground_truths, ...)`(复数 + 位置参数,会被 `_call_with_kwargs` 当成 args,报 missing 3 required positional arguments)。
- 返回值:verl 期望 `{"score": r, **metrics}` ——verl 取 `result["score"]` 作 `reward_score`,其他 key 写到 `reward_extra_info`,trainer 会自动聚合成 `reward_components/<key>/mean` 写到 `train_metrics.jsonl`。
  - **不能**返回 `tuple(r, m)`,verl 会把 tuple 当 score 取第一个元素,metrics 丢失。
  - **不能**返回裸 `r`,否则 `reward_components/*` 一律变成 `acc=r`,所有分量都看不见。
- 入口脚本里不要用 `exec > >(tee) 2> >(tee)`,SIGHUP 传播会让后台 stdout 缓冲丢失。改用每段命令自己 `> file 2>&1` 重定向,parent 脚本输出由 nohup / setsid 接住。



### 4.5.2 KeyError 陷阱(实测 2026-06-14 第二次启动)

`verl/experimental/agent_loop/agent_loop.py:949` 在 `_postprocess` 里:
```python
non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])
```
它会收集**所有** sample 的 `reward_extra_infos` 字典里出现的 key,然后逐个 sample 取值。如果某些 sample 走 `invalid_json` 早期 return,metrics dict 缺 `chain_to_answer_ok` 等后续 key → KeyError → 整个 step 崩溃。

**修复**:v3 reward 在所有 early-return 路径(`invalid_json` / `missing_fields` / `empty_steps_or_fields`)都返回**同样的 metrics schema**(所有 key 都在,值默认 0,reason 字段标记原因)。`score_response` 内部用 `_METRICS_SCHEMA` 全局常量保证一致性。

## 5. 评测与对比

每次评测会写:
- `eval/step_NNNN/eval_result.json`(over 5,467)
- `eval/step_NNNN/predictions.jsonl`(模型全量输出)
- `eval/latest_metrics.json`
- `eval/epoch_summary.jsonl`(append)

需要落地的新文件:
1. `train_pipeline/reward_chaingsm_lbprm_v3_verl.py` — v3 reward
2. `train_pipeline/test_lbprm_v3.py` — 行为测试(60+ case,重新校核)
3. `train_scripts/local/run_grpo_verl_lbprm_v3.sh` — 训练入口(明确写 400 step 封顶 + 100 step 评测)
4. `docs/superpowers/plans/2026-06-13-lbprm-v3-plan.md` — 实施 plan
5. `docs/superpowers/reports/2026-06-13-lbprm-v3-report.md` — 跑完后落盘反思

---

## 6. 风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| answer 信号太稀疏(0.5B 命中率低) | 早期所有 step reward ≈ 0.2·format 0.16,policy 收不到梯度 | 调高 `n=8` 增加组内多样性;后续 v4 引入 KL 锚定 |
| 收紧 liveness 后所有 step 都死 | causal_liveness_score 全 0 | 先跑 200 条 SFT 输出,看新 liveness 分布;再调 |
| format 行为改变(模型开始用更少 steps) | response_length 继续下降 | 监控 response_length;若 < 80,说明在写垃圾 |
| 整体仍掉 1pp+ | 跟 v2 同样 reward hacking 模式 | 反思 spec,设计 v4(可能加 KL 系数 / 改 prompt) |

---

## 7. 决策记录

- 2026-06-13:主线明确"不要 AbstRaL 复现,专注自己训练 + 自己 PRM"
- 2026-06-13:基于 `sft_2epoch` best checkpoint(26.59%) 接力
- 2026-06-13:v3 = "answer-as-gate" + causal_liveness 收紧,不重写 prompt、不动 prompt schema
- 2026-06-13:每版 reward 单独 spec,跑 400 step,无提升就停
