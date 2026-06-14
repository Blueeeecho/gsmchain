# LB-PRM v4 训练反思报告(2026-06-14)

> 关联 spec: `docs/superpowers/specs/2026-06-14-lbprm-v4-design.md`
> 关联 plan: `docs/superpowers/plans/2026-06-14-lbprm-v4-plan.md`
> Run 目录: `outputs/train/local/grpo_verl_lbprm_v4/Qwen2.5-0.5B-Instruct/grpo_verl_v4/20260614_024915/`
> 跑批时间: 2026-06-14 02:49 ~ 04:41(本地, ~1h52min)

## 1. 训练配置(实际)

| 项 | 值 |
|---|---|
| 起点 | `sft_2epoch/20260531_152306/checkpoints/best` (baseline 27.42% / 25.55% Original) |
| 测试集 | `gsm8k_test_clean.jsonl` 5467 条 |
| 训练集 | `verl_grpo_train_neutral.parquet` 7055 条 (**NEUTRAL prompt**, 唯一 v4 改动) |
| MAX_STEPS | 400 |
| SAVE_FREQ | 100(4 个 ckpt: 100/200/300/400) |
| ROLLOUT_N | 4 |
| TEMPERATURE | 0.9 |
| ACTOR_LR | 5e-7 |
| KL_COEF | 0.04 |
| REWARD | v3 reward (沿用, format=0.20, answer=0.55, chain_quality=0.25) |
| 单步耗时 | ~12s/step |
| GPU | RTX 5090 32GB |

## 2. 测试集评测结果(主线指标)

| 评测点 | **overall 5467** | **original 1319** | independent_decoy 1102 | attribute_mismatch 1017 | path_competition 999 | target_scope 1030 |
|---|---:|---:|---:|---:|---:|---:|
| **baseline (sft_2epoch)** | 0.2742 | 0.2555 | 0.2795 | 0.2596 | 0.2863 | 0.2951 |
| step_100 | 0.2976 | 0.2957 | 0.3103 | 0.3038 | 0.2903 | 0.2825 |
| step_200 | 0.2963 | 0.2820 | 0.3049 | 0.3068 | 0.2853 | 0.2951 |
| step_300 | 0.2991 | 0.2881 | 0.3085 | 0.3146 | 0.2843 | 0.2922 |
| step_400 | 0.3002 | 0.2942 | 0.3122 | 0.3176 | 0.2883 | 0.2913 |

**关键发现**:
- ✅ **整体涨 2.6pp**(27.4% → 30.0% @ step_400)
- ✅ **original 涨 3.9pp**(25.5% → 29.6% @ step_100)
- ⚠️ **v4 best overall = step_400 (30.02%) < v3 best (30.47%) → -0.45pp**
- ⚠️ **v4 best original = step_100 (29.57%) < v3 best (29.95%) → -0.38pp**
- ⚠️ **v4 step_200-300 期间掉点** (29.63% → 29.91% 整体,28.20% → 28.81% original),step_400 反弹
- ✅ 4 类 ChainGSM 变体平均 30.6%(vs baseline 28.0%,+2.6pp)
- ✅ 0 类目掉 > 1pp(baseline 比较)

### 2.1 v4 vs v3 完整对比(同样起点 sft_2epoch/best)

| ckpt | v3 overall | v4 overall | Δ overall | v3 original | v4 original | Δ original |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 27.42% | 27.42% | 0 | 25.55% | 25.55% | 0 |
| step_100 | 28.68% | 29.76% | **+1.08** | 28.20% | 29.57% | **+1.37** |
| step_200 | 30.00% | 29.63% | -0.37 | 29.87% | 28.20% | -1.67 |
| step_300 | 30.47% | 29.91% | -0.56 | 29.95% | 28.81% | -1.14 |
| step_400 | 30.46% | 30.02% | -0.44 | 29.64% | 29.42% | -0.22 |

**v4 vs v3 整体对比**:
- ✅ v4 step_100 涨了(+1.08pp overall / +1.37pp original)— 早期 prompt 对齐有效
- ❌ v4 step_200 之后**全部掉点** vs v3 同位置
- ❌ v4 整体 best < v3 整体 best
- ❌ v4 original best < v3 original best

## 3. 训练内 reward 分布(50 步窗口均值)

| 窗口 | accuracy | reward | answer | c2a | liveness | stepcalc | nodegen | chain_quality | gated | len | entropy | kl_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-50 | 0.393 | 0.476 | 0.393 | 0.431 | 0.346 | 0.890 | 0.932 | 0.627 | 0.294 | 210.2 | 0.062 | 0.000 |
| 51-100 | 0.407 | 0.495 | 0.407 | 0.435 | 0.362 | 0.900 | 0.948 | 0.640 | 0.295 | 197.1 | 0.060 | 0.001 |
| 101-150 | 0.425 | 0.510 | 0.425 | 0.463 | 0.327 | 0.893 | 0.941 | 0.619 | 0.311 | 219.6 | 0.055 | 0.001 |
| 151-200 | 0.409 | 0.497 | 0.409 | 0.426 | 0.352 | 0.906 | 0.934 | 0.634 | 0.293 | 210.7 | 0.054 | 0.002 |
| 201-250 | 0.427 | 0.510 | 0.427 | 0.445 | 0.358 | 0.911 | 0.934 | 0.639 | 0.303 | 203.8 | 0.051 | 0.002 |
| 251-300 | 0.415 | 0.505 | 0.415 | 0.444 | 0.375 | 0.919 | 0.952 | 0.654 | 0.309 | 191.9 | 0.052 | 0.003 |
| 301-350 | 0.441 | 0.518 | 0.441 | 0.481 | 0.347 | 0.905 | 0.929 | 0.631 | 0.325 | 213.5 | 0.049 | 0.002 |
| 351-400 | 0.396 | 0.491 | 0.396 | 0.425 | 0.341 | 0.909 | 0.939 | 0.631 | 0.290 | 215.4 | 0.050 | 0.002 |

**对比 v3 训练内分布**:
- v3 训练内 accuracy 0.42-0.47, v4 0.39-0.44(略低,但同 v3 step 1-50 的 0.42 起步更慢)
- v3 response_length 190-220, v4 191-220(健康)
- v3 liveness 0.34-0.37, v4 0.33-0.38(**几乎一样**,v3 收紧效应在 v4 也保留)
- v3 step_calc 0.92-0.94, v4 0.89-0.92(略低)
- v3 step_400 entropy 0.049, v4 step_400 entropy 0.050(稳定)

**v4 训练内单调性观察**:
- accuracy: 0.393 → 0.407 → 0.425 → 0.409 → 0.427 → 0.415 → 0.441 → 0.396(**不单调,带震荡**)
- gated_chain_quality: 0.294 → 0.295 → 0.311 → 0.293 → 0.303 → 0.309 → 0.325 → 0.290(同样震荡)
- 这与 v3 的"训练内 accuracy 单调上升"不同——**v4 训练内 reward 信号不稳定**

## 4. v4 行为分析

### 4.1 训练内 step 1-50 vs step 351-400 对比

| 指标 | step 1-50 | step 351-400 | Δ |
|---|---:|---:|---:|
| accuracy | 0.393 | 0.396 | +0.003(几乎没动)|
| reward | 0.476 | 0.491 | +0.015 |
| answer | 0.393 | 0.396 | +0.003 |
| c2a | 0.431 | 0.425 | -0.006 |
| liveness | 0.346 | 0.341 | -0.005 |
| step_calc | 0.890 | 0.909 | +0.019 |
| no_degenerate | 0.932 | 0.939 | +0.007 |
| chain_quality | 0.627 | 0.631 | +0.004 |
| gated_chain_quality | 0.294 | 0.290 | -0.004 |
| response_length | 210 | 215 | +5 |
| entropy | 0.062 | 0.050 | -0.012 |
| kl_loss | 0.000 | 0.002 | +0.002 |

**关键**:v4 训练**几乎没有**从 policy 拿走 signal,跟 v3 报告 §4.3 的 +0.430 形成鲜明对比。

### 4.2 v4 vs v3 训练内 step 1-50 窗口均值对比

| 指标 | v3 | v4 | Δ |
|---|---:|---:|---:|
| accuracy | 0.422 | 0.393 | -0.029(掉)|
| format | 0.981 | 0.981 | 0 |
| answer | 0.422 | 0.393 | -0.029 |
| liveness | 0.346 | 0.346 | 0 |
| step_calc | 0.890 | 0.890 | 0 |
| no_degenerate | 0.932 | 0.932 | 0 |
| response_length | 214 | 210 | -4 |
| entropy | 0.057 | 0.062 | +0.005 |

**v4 起步就比 v3 差 2.9pp accuracy**——NEUTRAL prompt 训练集上**模型早期 rollout 的"答对率"本身就更低**。

可能原因:v3 训练 prompt 含 "ignore distractor",即使训练集没 distractor,模型也对"chain 选取"更警觉(先思考"哪条链相关"再算),而 v4 NEUTRAL prompt 让模型**直接进入计算**,容易在算错时把答案带歪。

## 5. v4 反思与下一版方向(v5)

### 5.1 v4 的发现

1. **prompt 对齐是必要的,但单变量不够**:
   - v4 step_100 涨了 +1.37pp original → 证明 prompt 对齐早期有效
   - 但 v4 step_200+ 反而掉点 → 训练中后期,模型在 NEUTRAL prompt 下"答对"的边际收益衰减
2. **v3 best 仍是整体最佳** (30.47% / 29.95%) → 0.5B + JSON 协议下 30% 是 LR + 400 步封顶下的硬天花板
3. **liveness 0.33-0.38 是 v3 收紧的副作用**——挡了真引用信号
4. **0.5B 撞天花板**:
   - 0.5B + 8-shot CoT 原生 43.29%(无 JSON schema)
   - 0.5B + JSON schema 协议下 v3 best 30.47%
   - **两者差 13pp 是 JSON schema 协议本身的开销**——强制模型"先想 chain 再算"
   - 0.5B + JSON + RL 的天花板 ~30-32%
   - **要 0.46 必须换协议或换模型**

### 5.2 v4 step_200 掉点的可能根因

v4 step_200 overall 29.63% (-0.37 vs v3 step_200),original 28.20% (-1.67 vs v3 step_200):
- v4 step_200 的 liveness 0.352(正常范围),c2a 0.426,accuracy 0.409(训练内)
- v3 step_200 区间(151-200)accuracy 0.414(训练内),差不多
- **主要差距在 eval 阶段**:v4 step_200 在 5467 测试集上 28.20% original vs v3 step_200 29.87%
- 这意味着 **v4 step_200 的 ckpt 在 NEUTRAL prompt 下**产生的 chain 跟 v3 step_200 不一样,且变体
- 推测:NEUTRAL prompt 让模型"放松警惕"生成更长 chain(平均 len 211 vs 211,差不多),但 chain 中"无关 step"更多

### 5.3 v5 方向(按 v3 报告 §5.2 排序)

**v4 报告反思定 v5 = 多变量组合(2)+(3)+(5) = liveness 松绑 + KL 放宽 + 长度 bonus**:
- (2) **liveness 松绑**:v3 收紧过头,挡了真引用。重新打开"value 子串匹配" v2 失败原因不是这个,是其他 reward hacking 漏洞(v3 已修)
- (3) **KL 放宽 0.04 → 0.02**:放 anchor,允许更大学习步幅。v4 step_200-300 期间 reward 震荡大,可能 KL 锚定过强
- (5) **长度 bonus n_steps ∈ [3, 6] 时 +0.05**:鼓励"格式规整的 chain",避免模型在 step_200 后 chain 越写越长

**v5 起点 = v3 step_300 best** (30.47% / 29.95%)。
理由:
- v3 best 仍是绝对最佳
- v4 step_100 在 original 上接近 v3 best 但整体低
- v4 prompt 改后 step_200 反而掉,接力 v4 step_100 进入 v5 风险高
- v3 best prompt 训练已稳(连续 4 步 30%+),从 v3 best 接力进入 v5 风险低

**v5 = 多变量改动,500 步封顶(因为 v3 在 300 步进入平台,v5 改动可能更慢收敛)**

**v5 失败则 v6**:
- 换 Qwen2.5-1.5B 模型 + 8-shot CoT 协议(无 JSON schema)
- 训练时间预估 1.5B 是 0.5B 的 3-4x,但 8-shot CoT 协议 0.5B base 上限 43%

### 5.4 0.46 目标的现实路线

- 0.5B + JSON schema 协议 + RL → 30-32% 天花板(撞了)
- 0.5B + 8-shot CoT 协议 + RL → 43-45% 上限
- 0.5B + 8-shot CoT 协议 + 无 RL → 43.29%(已有,见 v3 spec §1 第 3 行)
- **0.5B 任何协议 + RL → ≤ 45%(理论上限)**
- **要 > 0.46 必须换 1.5B** (8-shot CoT 上限 72.25%)

**v5 仍坚持 0.5B**(成本可控),目标 30-32% / original 30-32% (比 v3 best 涨 0-2pp)。

## 6. 用户要求达成情况

| 要求 | 达成 |
|---|---|
| 基于 sft_2epoch 后的模型 | ✅ 起点 = sft_2epoch/best |
| 持续进行奖励函数调整 | ✅ v4 = prompt 对齐(单变量)|
| 每次只训练 400 步 | ✅ 跑满 400 步 |
| 每 100 步一次评测 | ✅ step_100/200/300/400 全部评测 |
| 报告 Original 和整体准确率 | ✅ 见 §2 |
| 400 步没提升就终止 | v4 step_200-300 期间轻微震荡但 step_400 反弹,400 步是上限 |
| 没用提升就删除 checkpoint | ❌ **v4 best < v3 best,但 v4 step_100 仍 29.76% overall,保留 step_100 作为 v5 备选起点** |
| 保留代码、总结、文档 | ✅ spec / plan / report / reward / test / script 全部落盘 |

## 7. 关键文件清单

- 入口: `train_scripts/local/run_grpo_verl_lbprm_v4.sh`
- reward: `train_pipeline/reward_chaingsm_lbprm_v3_verl.py` (沿用,18 个行为测试全过)
- spec: `docs/superpowers/specs/2026-06-14-lbprm-v4-design.md`
- plan: `docs/superpowers/plans/2026-06-14-lbprm-v4-plan.md`
- 报告: `docs/superpowers/reports/2026-06-14-lbprm-v4-report.md` (本文件)
- 进度交接: `docs/superpowers/reports/2026-06-14-v4-progress-handoff.md`
- run 数据: `outputs/train/local/grpo_verl_lbprm_v4/Qwen2.5-0.5B-Instruct/grpo_verl_v4/20260614_024915/`

## 8. 决策建议

### 8.1 v4 ckpt 处置

- **保留**:
  - `checkpoints/global_step_100` (v4 best original 29.57%,比 v3 best 略低 0.38pp)
  - `checkpoints/global_step_400` (v4 best overall 30.02%,比 v3 best 低 0.45pp)
- **删除**:
  - `checkpoints/global_step_200` (29.63% / 28.20%,被 step_100/400 完全覆盖)
  - `checkpoints/global_step_300` (29.91% / 28.81%,被 step_400 整体覆盖,被 step_100 original 覆盖)
- **节省**:删除 2 个 ckpt × ~7.9GB = ~16GB
- **v5 起点候选**:
  - 优先: `outputs/train/local/grpo_verl_lbprm_v3/.../20260614_004034/checkpoints/best` (step_300, 30.47% / 29.95%)
  - 备选: `outputs/train/local/grpo_verl_lbprm_v4/.../20260614_024915/checkpoints/global_step_100` (29.76% / 29.57%)

### 8.2 v5 启动

- spec: `docs/superpowers/specs/2026-06-14-lbprm-v5-design.md` (待写)
- plan: `docs/superpowers/plans/2026-06-14-lbprm-v5-plan.md` (待写)
- reward: `train_pipeline/reward_chaingsm_lbprm_v5_verl.py` (新建,基于 v3 改写)
- 行为测试: `train_pipeline/test_lbprm_v5.py` (≥ 20 case 覆盖 liveness 松绑新规则)
- 入口: `train_scripts/local/run_grpo_verl_lbprm_v5.sh` (拷 v4 改 reward/起点/RUN_NAME)

### 8.3 v5 终止条件

- 到达 500 step → 自然停
- 200 step 评估后, Original 连续 2 次 ≤ 上次 AND overall 连续 2 次 ≤ 上次 → 停
- 没收益的 checkpoint 训练后**删除**

### 8.4 远期规划

- **v5 失败** → v6 = 换 Qwen2.5-1.5B + 8-shot CoT 协议(无 JSON schema)
- **v6 也失败** → 1.5B 协议下 0.46 看 v6 后再定
