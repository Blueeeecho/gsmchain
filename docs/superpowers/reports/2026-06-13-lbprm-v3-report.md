# LB-PRM v3 训练反思报告(2026-06-14)

> 关联 spec: `docs/superpowers/specs/2026-06-13-lbprm-v3-design.md`
> 关联 plan: `docs/superpowers/plans/2026-06-13-lbprm-v3-plan.md`
> Run 目录: `outputs/train/local/grpo_verl_lbprm_v3/Qwen2.5-0.5B-Instruct/grpo_verl_v3_smoke/20260614_004034/`
> 跑批时间: 2026-06-14 00:40 ~ 02:32(本地, ~1h52min)

## 1. 训练配置(实际)

| 项 | 值 |
|---|---|
| 起点 | `sft_2epoch/20260531_152306/checkpoints/best` |
| 测试集 | `gsm8k_test_clean.jsonl` 5467 条 |
| 训练集 | `verl_grpo_train.parquet` 7055 条 |
| MAX_STEPS | 400 |
| SAVE_FREQ | 100(实际 4 个 ckpt: 100/200/300/400) |
| ROLLOUT_N | 4 |
| TEMPERATURE | 0.9 |
| ACTOR_LR | 5e-7 |
| KL_COEF | 0.04 |
| v3 reward 权重 | format=0.20, answer=0.55, chain_quality=0.25 |
| 单步耗时 | ~15s/step |
| GPU | RTX 5090 32GB |

## 2. 测试集评测结果(主线指标)

| 评测点 | **overall 5467** | **original 1319** | independent_decoy 1102 | attribute_mismatch 1017 | path_competition 999 | target_scope 1030 |
|---|---:|---:|---:|---:|---:|---:|
| **baseline** | 0.274 | 0.255 | 0.280 | 0.260 | 0.286 | 0.295 |
| step_100 | 0.287 | 0.282 | 0.290 | 0.293 | 0.279 | 0.290 |
| step_200 | 0.300 | 0.299 | 0.302 | 0.294 | 0.303 | 0.302 |
| step_300 | **0.305** | **0.299** | **0.314** | 0.310 | 0.303 | 0.298 |
| step_400 | 0.305 | 0.296 | 0.315 | 0.310 | 0.301 | 0.302 |

**关键发现**:
- ✅ **整体涨 3.1pp**(27.4% → 30.5%)
- ✅ **original 涨 4.4pp**(25.5% → 29.9%,峰值在 step_300)
- ✅ **5 类目全部稳定或上涨**(无掉点)
- ⚠️ step_300 ≈ step_400(0.305 vs 0.305)→ 300 步后进入平台,继续训没收益
- ✅ 4 类 ChainGSM 变体平均 30.7%(vs baseline 28.0%,+2.7pp)
- ✅ 0 类目掉 > 1pp

## 3. 训练内 reward 分布(50 步窗口均值)

| 窗口 | accuracy(8 中) | reward | answer | c2a | gated_chain_quality | len | entropy | kl_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-50 | 0.422 | 0.499 | 0.422 | 0.458 | 0.310 | 214 | 0.057 | 0.001 |
| 51-100 | 0.426 | 0.513 | 0.426 | 0.456 | 0.313 | 192 | 0.056 | 0.001 |
| 101-150 | 0.445 | 0.523 | 0.445 | 0.474 | 0.318 | 214 | 0.051 | 0.002 |
| 151-200 | 0.414 | 0.499 | 0.414 | 0.425 | 0.291 | 211 | 0.052 | 0.002 |
| 201-250 | 0.443 | 0.521 | 0.443 | 0.459 | 0.310 | 204 | 0.050 | 0.002 |
| 251-300 | 0.417 | 0.507 | 0.417 | 0.449 | 0.315 | 191 | 0.050 | 0.002 |
| 301-350 | **0.468** | **0.537** | **0.468** | **0.505** | **0.337** | 214 | 0.047 | 0.002 |
| 351-400 | 0.430 | 0.512 | 0.430 | 0.455 | 0.308 | 218 | 0.049 | 0.002 |

**对比 v2 烟测(grpo_verl_lbprm_base_2ep_200step/20260609_153629)**:
- v2 accuracy 全程 0.039(0/8),**v3 涨到 0.43~0.47(3.4-3.8/8)**
- v2 answer 全程 0.039,**v3 涨到 0.42~0.47**
- v2 response_length 从 238 掉到 110(模型学偷懒),**v3 稳定在 190-220(健康)**
- v2 liveness_ratio 0.55→0.89(过拟合 liveness 捷径),**v3 c2a 0.42→0.51(更稳)**

## 4. v3 行为分析

### 4.1 答对的样本长什么样

抽几条 `step_300` 的 predictions 看下质量(略,详见 `eval/step_300/predictions.jsonl`):
- 多数答对的样本:5-6 步 chain,step[0] ~ step[3] 是真算式,final_expression 引用最后一步的 variable,answer 跟 final 一致
- chain 整体较干净,没有"刻意加干扰"的现象

### 4.2 v3 reward 的"3 个改进"在数据里的验证

| 改进 | 数据印证 |
|---|---|
| **chain_to_answer_ok 硬门** | `gated_chain_quality/mean` ≈ 0.31~0.34,跟 `chain_quality_score`(0.55~0.59)有 ~0.2 的 gap,说明闸门真的在挡答错的 chain_quality 收益 |
| **causal_liveness 收紧** | `causal_liveness_score/mean` 0.34~0.37,远低于 v2 的 0.89(可能因为 v3 算到了"真引用"而不是"字符串子串") |
| **answer 0.55 主导** | `answer/mean` 0.42~0.47,接近 accuracy;`reward/accuracy` 0.42~0.47 跟 answer/mean 几乎 1:1,说明 answer 信号主导了梯度方向 |

### 4.3 训练内 step 1 vs step 400 对比

| 指标 | step 1 | step 400 (last 50) | Δ |
|---|---:|---:|---:|
| accuracy | 0.0 | 0.430 | **+0.430** |
| reward | -0.031 | 0.512 | +0.543 |
| answer | 0.0 | 0.430 | +0.430 |
| c2a | (n/a) | 0.455 | — |
| gated_chain_quality | 0.187 | 0.308 | +0.121 |
| response_length | 254 | 218 | -36(略短,健康) |
| entropy | 0.351 | 0.049 | -0.302(收窄,但没崩) |
| kl_loss | 0.0 | 0.002 | 稳定 |

## 5. v3 reward 缺陷与下一版方向(v4)

### 5.1 v3 的局限

1. **300 步后进入平台**:step_300 ≈ step_400,可能 KL 锚定过强 / LR 太保守 / 数据单一(只有 4 类变体)
2. **causal_liveness 偏严**:0.34 的均值意味着多数 step "liveness 不被认定"——**这是收紧的副作用**,但可能在挡真正有用的 chain 引用信号
3. **训练内 accuracy 0.42~0.47 vs vLLM 评测 0.30**:vLLM 评测用的是 greedy + train_json_prompt(8-shot 风格 prompt),跟训练时的 chat 风格 prompt 不同,**prompt 错位**会限制迁移收益
4. **没把最佳 ckpt 复制为 `checkpoints/best`**:脚本只保存 `global_step_*`,用户后续需要手动选

### 5.2 v4 候选方向

按"v3 哪些限制最难突破"排序:

1. **prompt 迁移**:训练用 chat + JSON schema,评测用 8-shot CoT。**v4 选:**在训练 prompt 里就加 few-shot example(8-shot 风格),让 rollout 更接近评测分布
2. **causal_liveness 再松绑**:**v4 选:**保留变量子串匹配 + 末步 expression 求值 == final 求值 + 末步 value 严格等于 answer,**重新打开"value 子串匹配"**(v2 失败原因不是子串匹配本身,而是其他 reward hacking 漏洞并存)
3. **加 LR 调度 / 退火**:**v4 选:**保持 5e-7 不变,但 KL 系数 0.04 → 0.02(放宽 KL 锚定,允许更大学习步幅)
4. **扩大训练数据多样性**:**v4 选:**加入"verl 0.5B 模型在 gsm8k_test_clean 上的错误样本作为负样本"(让模型看见自己的错误)。**这需要重新预处理,优先级低**
5. **anti_degenerate 给"格式规整的 chain"额外奖励**:**v4 选:**n_steps 在 3-6 之间时给 +0.05,避免 chain 越写越短

**v4 推荐组合**:(1) + (2) + (3) + (5) 一起,**不**做 (4)。

但 v4 设计需要等用户确认。建议先把当前 v3 的 best ckpt(就是 step_300,overall 30.5%)**保留**(不删,因为有收益)。

## 6. 用户要求达成情况

| 要求 | 达成 |
|---|---|
| 基于 sft_2epoch 后的模型 | ✅ 起点 = `sft_2epoch/20260531_152306/checkpoints/best` |
| 持续进行奖励函数调整 | ✅ v3 比 v2 涨 10x 训练内 accuracy |
| 每次只训练 400 步 | ✅ 跑满 400 步 |
| 每 100 步一次评测 | ✅ step_100/200/300/400 全部评测 |
| 报告 Original 和整体准确率 | ✅ 见 §2 |
| 400 步没提升就终止 | step_300 ≈ step_400 → 训练已停,400 步是上限 |
| 没用提升就删除 checkpoint | ❌ 这次**有提升**,保留全部 ckpt(等用户确认是否删 step_400) |
| 保留代码、总结、文档 | ✅ spec / plan / report / reward / test / script 全部落盘 |

## 7. 关键文件清单

- 入口: `train_scripts/local/run_grpo_verl_lbprm_v3.sh`
- reward: `train_pipeline/reward_chaingsm_lbprm_v3_verl.py`
- 行为测试: `train_pipeline/test_lbprm_v3.py`(18 case 全过)
- spec: `docs/superpowers/specs/2026-06-13-lbprm-v3-design.md`
- plan: `docs/superpowers/plans/2026-06-13-lbprm-v3-plan.md`
- 报告: `docs/superpowers/reports/2026-06-13-lbprm-v3-report.md`(本文件)
- 监控工具: `code/monitor_grpo.py`
- run 数据: `outputs/train/local/grpo_verl_lbprm_v3/Qwen2.5-0.5B-Instruct/grpo_verl_v3_smoke/20260614_004034/`

## 8. 决策建议

- **保留** step_300 ckpt 作为 v3 的 "best"(overall 30.5%, original 29.9%)
- **可删** step_100 和 step_400(被 step_200/300 覆盖)
- **下一步**:跟用户对齐 v4 方向(建议组合 1+2+3+5)
- **远程**:等 v4 稳定后,本机结果可作为远程 SLURM 链路 smoke test 的输入
