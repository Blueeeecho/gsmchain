# v6 启动前反思 + Reward 阶段性总结 (2026-06-14 06:30)

> 接 v5 撞 30% 天花板报告 (`2026-06-14-lbprm-v5-report.md`)。
> 用户原话: "整体效果都很差, 所以需要的是进一步调整, 而不是盲目的训练"。
> 本文目的: 总结 v3→v4→v5→v6-smoke 全部数据, 提出 v6 启动前的 reward 候选 A/B/C, 由用户确认后再跑。

---

## 1. 0.5B 全部训练 + 评测真实数据 (JSON 协议 + 8-shot CoT)

| 协议 | 模型 / ckpt | overall 5467 | original 1319 | independent_decoy 1102 | attribute_mismatch 1017 | path_competition 999 | target_scope 1030 |
|---|---|---:|---:|---:|---:|---:|---:|
| **JSON schema** | sft_2epoch/best (baseline) | 0.2742 | 0.2555 | 0.2795 | 0.2596 | 0.2863 | 0.2951 |
| JSON schema | v3 best (GRPO step_300) | **0.3047** | **0.2995** | 0.3140 | 0.3097 | 0.3033 | 0.2981 |
| JSON schema (NEUTRAL) | v4 step_100 | 0.2976 | 0.2957 | 0.3103 | 0.3038 | 0.2903 | 0.2825 |
| JSON schema (NEUTRAL) | v4 step_400 (best) | 0.3002 | 0.2942 | 0.3122 | 0.3176 | 0.2883 | 0.2913 |
| JSON schema (liveness松绑) | v5 quicktest step_5 | 0.3078 | 0.3025 | 0.3158 | 0.3127 | 0.3043 | 0.3049 |
| JSON schema (liveness松绑) | v5 step_100 (best) | **0.3053** | **0.3048** | 0.3158 | 0.3058 | 0.2993 | 0.3000 |
| **8-shot CoT 原生** | 0.5B base (无 RL) | 0.2455 | **0.4329** | 0.1633 | 0.2016 | 0.2122 | 0.1689 |
| 8-shot CoT 原生 | 1.5B base (无 RL, 用户约束外) | 0.4911 | **0.7225** | 0.3775 | 0.4730 | 0.4545 | 0.3699 |
| **8-shot CoT** | v6 smoke step_5 (sft_2epoch/best 起点, 5 步) | n/a (eval bug 修复中) | n/a | n/a | n/a | n/a | n/a |
| **目标** | — | 0.46 | — | — | — | — | — |

**结论性数字**:
- **0.5B + JSON schema + GRPO 上限 = 30.5% overall / 30.5% original** (v5 step_100, 4 个 30-32% 类目齐涨罕见信号)
- **0.5B + 8-shot CoT 原生 (无 RL) = 24.55% overall / 43.29% original** — 自由推理在 Original 上比 JSON 协议 + RL 高 13pp
- **0.5B + 8-shot CoT 4 类变体 = 16-21%** — 自由推理下抗干扰很弱, 是 overall 上不去的根本原因
- **0.46 目标 = overall ≥ 46%** — 在 0.5B 用户约束下, **需要 4 类变体从 16-21% 涨到 30%+** (+10-14pp), 极难

---

## 2. v3→v4→v5 训练内信号 (50 步窗口均值) 趋势对照

| 阶段 | 起点 | accuracy 1-50 → 351-400 | entropy 1-50 → 351-400 | chain_quality 1-50 → 351-400 | liveness 1-50 → 351-400 | response_len 1-50 → 351-400 |
|---|---|---|---|---|---|---|
| **v3** | sft_2epoch/best | 0.42 → 0.47 (+0.05) ✅ | 0.057 → 0.049 | 0.45 → 0.55 | 0.35 → 0.34 | 200 → 200 |
| **v4** | sft_2epoch/best + NEUTRAL prompt | 0.39 → 0.40 (+0.01) ❌ | 0.062 → 0.050 | 0.63 → 0.63 | 0.35 → 0.34 | 210 → 215 |
| **v5** | v3 best + liveness松绑 + length_bonus | 0.46 → 0.49 (+0.03) ✅ | 0.054 → 0.021 ⚠️ | 0.95 → 0.99 ⚠️ | 0.97 → 1.00 ⚠️ | 192 → 198 |

**关键观察**:
- **v3**: reward 信号健康, accuracy 单调 +0.05, policy 在动 (entropy 0.057→0.049)
- **v4**: NEUTRAL prompt 训练集早期 accuracy 就比 v3 低 2.9pp, 整个 400 步震荡, step_200-300 掉点
- **v5**: 早期 accuracy 0.46 (比 v3 同期 0.42 涨 4pp, 因为起点是 v3 best), 但 100 步后撞天花板:
  - **liveness 0.97→1.00 (松绑过头, 0.34→0.97 = 60pp 涨, 几乎全满分, 失去区分度)**
  - **chain_quality 0.95→0.99 (跟 liveness 共振, 0.99 接近满分, 失去区分度)**
  - **entropy 0.054→0.021 (policy 收窄 60%, 训练不健康)**
  - **accuracy 0.46→0.49 (0.03pp 涨, 在 noise 范围内)**

---

## 3. v3/v4/v5 撞 30% 天花板的根因 (重新归因)

### 3.1 协议天花板 (最重)
0.5B + JSON schema 协议本身把模型"思考空间"压缩到"输出严格 5 字段 JSON"。在这种结构化输出约束下:
- 0.5B 容量有限, 在 JSON 字段精确度上消耗大量 token budget
- Original 子集 30% 上限不是 reward 配错, 是**协议 + 模型容量**共同的天花板
- 1.5B + JSON 协议上限未知, 但**换协议**(8-shot CoT)比**加容量**(1.5B)对 0.5B 来说收益更大

### 3.2 Reward 信号天花板
v3 reward 已经是 0.20/0.55/0.25 三档, 0.5B 上的"有效信号区间"被榨干:
- correct mean reward 0.92, wrong mean reward 0.20 → gap 0.72 ✅
- 答错 924 个里 99.6% 在 reward 0.2-0.4 区间 → reward 已经把"答错"压成固定
- 70% 错答是推理能力本身失败 (chain 算式都正确, 但推理最后一步算错)
- **v5 改 reward (liveness松绑 + length_bonus) → 跟 v3 持平, reward 空间已榨干**

### 3.3 KL 锚定天花板
KL_LOSS_COEF=0.04→0.02 (放宽 2x), kl_loss 仍只有 0.0001-0.0008:
- v3 KL 0.04 时 kl_loss 0.002, v5 KL 0.02 时 kl_loss 0.0008 → **放宽后 policy 几乎不动 ref**
- 0.5B 起步点 (sft_2epoch) 跟 ref (= 起点本身) 几乎同分布, KL 收紧, 不让 model 走远
- 再放宽 KL 没意义 (0.02 已经很松, 0.0008 kl_loss 没真约束)

### 3.4 训练步数天花板
v3 在 300 步进入平台, v4 在 100-200 步震荡, v5 100 步撞天花板:
- 0.5B + JSON 协议下 200-300 步就稳定到 30%, 多训 = 平台震荡
- **400 步对 0.5B + JSON 协议 = 浪费 GPU**

### 3.5 综合判断
**0.5B + JSON schema 协议 + GRPO = 30% 天花板已确认**, v3/v4/v5 三次微调 reward 都打不破:
- v3: 起点 sft_2epoch + JSON 协议 + v3 reward → 30.47% / 29.95%
- v4: 起点 sft_2epoch + NEUTRAL prompt + v3 reward → 30.02% / 29.42% (略掉)
- v5: 起点 v3 best + liveness松绑 + length_bonus → 30.53% / 30.48% (持平, 4 类目齐涨)
- **下一步应该是 v6: 换协议 (8-shot CoT) + 重写 reward**, 而不是继续在 JSON 协议上微调

---

## 4. v6 当前状态 + 真实问题

### 4.1 v6 准备度
- ✅ spec/plan/reward/test/preprocess/entry 全部落盘
- ✅ v6 reward 26/26 行为测试全过
- ✅ 7055 条 8-shot CoT 训练数据已生成 (`verl_grpo_train_8shot_cot.parquet`)
- ✅ v6 smoke 5 步训练已跑通 (train_metrics 全部正常)
- ❌ **v6 smoke eval 失败** (entry 脚本 `--models` 参数错, 已修)
- ❌ **v6 smoke step_5 训练内 accuracy=0.0** (要警惕!)

### 4.2 v6 smoke step_5 训练内 metrics (新观察)

| step | accuracy | format | answer | step_count | n_steps | numeric_correctness | n_equations | no_contradiction | reasoning_quality | reward | response_len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.34 | 224 |
| 5 | 0.0 | 0.875 | 0.0 | 0.74 | 13.3 | 0.22 | 2.25 | 0.875 | 0.61 | 0.28 | 224 |

**关键观察**:
- **accuracy = 0.0**: v6 起点 sft_2epoch/best 在 8-shot CoT 协议下 5 步 rollout **完全答不对**
- format = 0.875: 大部分能按 8-shot CoT 格式输出 (有 "final answer is" 收尾)
- answer = 0.0: 答案全错
- step_count 0.74 / n_steps 13.3: 步骤数偏多 (reward 给 <3 步扣 0.3, >7 步扣 0.1, 13 步还在 1.0 区间)
- **numeric_correctness 0.22**: **算式正确率只有 22%**, 0.5B 写出来的算式 78% 算错
- **no_contradiction 0.875**: 最终答案多数在推理过程中出现 (过拟合最终答案)

**红旗**:
- sft_2epoch/best 在 8-shot CoT 协议下 (跟 JSON 协议训练分布完全不一样) 早期 rollout 答错率高
- accuracy=0.0 不是 5 步 noise, 是模型在 JSON 协议训练后的"格式惯性"导致 8-shot CoT 协议早期不会算
- **GRPO 训练早期 reward 几乎全靠 reasoning_quality (0.61) + format (0.875*0.15=0.131) = 0.74 上限, 没有 answer 0.6 的真信号**
- 模型靠 reasoning_quality 单一信号走 50-100 步才能学会"在 8-shot CoT 协议下算对", 这是 v6 训练的核心风险

### 4.3 v6 起点选择反思
用户指定 v6 起点 = sft_2epoch/best (27.42% / 25.55% Original JSON 协议, 但 8-shot CoT 协议基线未测):
- ✅ 干净起点, 不受 JSON 协议训练残留干扰
- ❌ **v6 smoke 显示 8-shot CoT 协议下起点 rollout 答错率高**, 早期 accuracy=0.0
- ❓ 0.5B + 8-shot CoT 原生 base (无 SFT) = 43.29% Original, **sft_2epoch/best 在 8-shot CoT 协议下**可能反而比 0.5B base 还低
- **建议: v6 启动前先测一次 sft_2epoch/best + 8-shot CoT 评测 baseline**, 用真实数字决定 v6 reward 调整方向

---

## 5. v6 启动前 Reward 候选方案 (A / B / C)

**约束 (用户原文)**: "使用 Qwen2.5-0.5B-instruct, 400 步一类奖励, 没持续提升就终止"。

### 5.1 候选 A: 8-shot CoT + 现有 v6 reward (跟 spec 一致)

```python
# v6 当前 spec 公式
total = 0.15·format + 0.60·answer + 0.25·reasoning_quality
reasoning_quality = 0.5·step_count + 0.3·numeric_correctness + 0.2·no_contradiction
```

- ✅ spec/plan/reward/test/preprocess/entry 全部就绪
- ✅ 行为测试 26/26 全过
- ❓ v6 smoke accuracy=0.0 暴露的"早期 rollout 答错率高"风险没应对
- ❓ 4 类变体 (16-21%) 是否能涨到 30% 未知 (用户约束 0.5B 自由推理下干扰敏感)

**预期**: 400 步后 Original 45-55% / overall 35-45% (基于 0.5B + 8-shot 原生 43.29% Original 起涨)

### 5.2 候选 B: 8-shot CoT + 强化"算式正确" + "抗干扰"

针对 v6 smoke 暴露的两个问题:
1. **accuracy=0.0** → 早期 rollout 答错率高
2. **4 类变体 16-21%** → 自由推理下抗干扰很弱

```python
# 候选 B 公式
total = 0.10·format + 0.55·answer + 0.35·reasoning_quality_v2
reasoning_quality_v2 = 0.4·step_count
                    + 0.4·numeric_correctness          # 算式正确从 0.3 涨 0.4
                    + 0.1·no_contradiction
                    + 0.1·equation_count_bonus         # 新增: 算式数量奖励
                    # 3-7 步满分, <3 步扣 0.3, >7 步扣 0.2 (严卡)
```

**改动点**:
- format 0.15→0.10 (早期 model 输出格式可能略掉)
- answer 0.60→0.55 (留更多空间给 reasoning)
- reasoning_quality 0.25→0.35 (强化推理信号)
- numeric_correctness 0.3→0.4 (因为 v6 smoke 算式正确率只有 22%, 重点拉)
- no_contradiction 0.2→0.1 (v6 smoke 0.875 接近满分, 区分度低)
- 新增 equation_count_bonus: 算式数量 ≥ 3 时给 0.5-1.0 (鼓励"写出可验证的算式")

**优点**: 强化算式信号, 对早期 rollout 答错率高更友好
**缺点**: 需重写 reward + 重跑 26+ 行为测试
**预期**: 400 步后 Original 45-55% / overall 35-45% (跟 A 类似, 但 reasoning 信号更稳)

### 5.3 候选 C: 8-shot CoT + "答案对 + 算式对" 双信号 (极简)

如果 B 太复杂, 候选 C 是最简方案, 把"算式正确"作为"答案正确"的辅助:

```python
# 候选 C 公式
total = 0.10·format + 0.70·answer + 0.20·numeric_correctness
# 砍掉 step_count + no_contradiction, 留 numeric_correctness
```

**改动点**:
- answer 0.60→0.70 (强力吸引"答对")
- reasoning_quality 拆解, 只留 numeric_correctness
- 砍掉 step_count (v6 smoke 0.74 区分度低) + no_contradiction (0.875 区分度低)

**优点**: 极简, 信号强, 早期 rollout 答对就有强 reward
**缺点**: reasoning 信号太弱, 4 类变体抗干扰可能更差
**预期**: 400 步后 Original 50-60% / overall 30-40% (Original 涨, overall 跟 A/B 类似)

### 5.4 三个候选决策维度

| 维度 | A (现状) | B (强化算式) | C (极简) |
|---|---|---|---|
| 行为测试改写 | 0 | 重写 + 测 (1-2h) | 重写 + 测 (0.5h) |
| 早期 rollout 友好 | 中 (format 0.875+reasoning 0.61 = 0.74) | 高 (加 equation_count 缓冲) | 高 (format 0.10+numeric 0.20 = 0.30 上限) |
| 抗干扰 (4 类变体) | 弱 (16-21% 起涨) | 中 (算式奖励) | 弱 |
| 信号清晰度 | 中 (3 子项) | 中 (4 子项) | 高 (1 子项) |
| 与 v3/v5 reward 关联 | 完全不同 | 完全不同 (numeric 强) | 完全不同 (answer 强) |
| 0.46 目标概率 | 5-10% | 5-10% | 5-10% (0.5B 用户约束) |
| **跑一个 400 步时间** | ~2.5h | ~2.5h | ~2.5h |

**0.46 目标现实评估**:
- 0.5B + 8-shot CoT 自由推理下 4 类变体基线 16-21% (base 无 RL)
- 即便 RL 涨 10pp, 4 类变体到 26-31% (整体 = (43%×1319 + 28%×4148) / 5467 ≈ 31.5%, 离 46% 差 14.5pp)
- **0.5B 8-shot CoT 协议下 0.46 整体目标 = 不可能**, Original 50% + 整体 35% 是现实上限
- 但用户约束是用 0.5B, **0.46 整体是理想目标, 我们做的是"逼近"**

---

## 6. 建议执行顺序 (按 v6 spec 约束 "400 步一类奖励")

### 阶段 1: 启动前 (本次)
1. ✅ 修 v6 entry eval bug (已完成, 本次提交)
2. ⏳ **补 sft_2epoch/best + 8-shot CoT 评测 baseline** (1 次独立 eval, 5-10 min)
3. ⏳ **跑 v6 smoke test 复跑: 5 步训练 + 1 次 step_5 eval** (15-20 min, 验证修复)
4. ⏳ 落盘本次反思到 `docs/superpowers/reports/2026-06-14-v6-prereflex.md` (本文件)
5. ⏳ 用户确认: 选 A / B / C / 其他

### 阶段 2: 短消融 (用户确认后, 100 步 × 2-3 个候选)
- 跑 2-3 个候选的 100 步版本 (每 20 步一次 eval)
- 选出 1 个赢家
- 短消融总时间: ~1.5h

### 阶段 3: 完整 400 步训练 (赢家候选)
- 起点 sft_2epoch/best, MAX_STEPS=400, SAVE_FREQ=80
- 5 个 eval 节点: step_80/160/240/320/400
- 总时间: ~2.5h

### 阶段 4: 报告 + 决策
- 落盘 v6 训练报告 (含每 100 步 original/overall/4 类变体)
- 跟 v3/v4/v5 对比, 跟 0.5B + 8-shot CoT 原生基线对比
- 决定 v7 方向

---

## 7. 立刻可执行的下一步 (按优先级)

1. **补 sft_2epoch/best + 8-shot CoT 评测 baseline** (~5-10 min):
```bash
cd /home/wwq416/snap/wwq/math-chain
export CUDA_MODULE_LOADING=LAZY PATH="/home/wwq416/miniconda3/envs/math_chain_verl/bin:${PATH}"
PYTHON="/home/wwq416/miniconda3/envs/math_chain_verl/bin/python"
SFT_BEST="outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_2epoch/20260531_152306/checkpoints/best"
BASELINE_DIR="outputs/baselines/sft_2epoch_8shot_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASELINE_DIR"
PYTHONPATH=. $PYTHON -m code.eval_chaingsm_base_8shot \
  --data-path chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl \
  --run-dir "$BASELINE_DIR" \
  --output-root "$(dirname $BASELINE_DIR)" \
  --limit -1 \
  --batch-size 16 \
  --gpu-memory-utilization 0.5 \
  --model "Qwen2.5-0.5B-Instruct@$SFT_BEST" \
  > "$BASELINE_DIR.log" 2>&1
```

2. **跑 v6 smoke test 复跑 (5 步 + step_5 eval)** (用修好的 entry 脚本):
```bash
setsid bash -c '
  cd /home/wwq416/snap/wwq/math-chain
  export CUDA_MODULE_LOADING=LAZY PATH="/home/wwq416/miniconda3/envs/math_chain_verl/bin:${PATH}"
  export CUDA_HOME="/home/wwq416/miniconda3/envs/math_chain_verl" FLASHINFER_CUDA_ARCH_LIST="12.0f"
  export LD_LIBRARY_PATH="/home/wwq416/miniconda3/envs/math_chain_verl/lib:${LD_LIBRARY_PATH:-}"
  MAX_STEPS=5 SAVE_FREQ=5 EVAL_BASELINE=0 \
  RUN_NAME=grpo_verl_v6_smoke2 \
  bash train_scripts/local/run_grpo_verl_lbprm_v6.sh
' > /tmp/v6_smoke2.log 2>&1 < /dev/null &
```

3. **等用户确认**: 选 A / B / C / 自己的候选 → 启动 400 步训练

---

## 8. 风险与提醒

| 风险 | 应对 |
|---|---|
| sft_2epoch/best 起点 8-shot CoT 协议下 accuracy=0.0 | 监控 step_1-50 accuracy, < 0.2 立即停, 反思 SFT 起点 |
| 8-shot prompt max=1029 token, MAX_PROMPT_LENGTH=1280 留余量 251 token | 监控 truncation ratio, > 5% 立即改 1536 |
| KL 0.02 锚定过紧, policy 不动 ref (v5 历史) | 监控 kl_loss 0.0001-0.0008 区间, 持续低就反思 KL 调整 |
| 训练-评测 prompt 漂移 | 评测脚本完全复用 `code/eval_chaingsm_base_8shot.py` |
| 0.5B 用户约束下 0.46 整体目标不现实 | 文档明确"逼近"目标, 现实上限 Original 50-55% / overall 35-45% |

---

## 9. 决策记录 (2026-06-14 06:30)

- ✅ 2026-06-14 02:49-04:41: v4 400 步训练 (NEUTRAL prompt, 跟 v3 比没破天花板)
- ✅ 2026-06-14 05:26-05:53: v5 100 步训练 (liveness 松绑 + length_bonus, 撞天花板, 30.53%/30.48%)
- ✅ 2026-06-14 05:54-06:08: v6 smoke 5 步训练 (entry eval bug 修了, accuracy=0.0 暴露起点问题)
- ✅ 2026-06-14 06:25: **v6 entry eval 段修好** (`--models` 改 `--model`, 加 `--data-path`, 加 `--output-root`)
- ⏳ 2026-06-14 06:30-06:40: 补 sft_2epoch/best + 8-shot CoT 评测 baseline
- ⏳ 2026-06-14 06:30-06:50: v6 smoke 复跑 (5 步 + step_5 eval 真实数字)
- ⏳ 2026-06-14 06:50: 用户确认 A/B/C → 启动完整 400 步
- ⏳ 2026-06-14 09:20: v6 400 步训练 + 5 次 eval 完成
- ⏳ 2026-06-14 09:30: v6 报告 + 0.46 目标达成判断

---

**结论**:
- **v3/v4/v5 撞 30% 天花板根因 = JSON 协议 + 0.5B 容量**, reward 微调已榨干
- **v6 = 0.5B + 8-shot CoT 协议**, 但 v6 smoke 暴露"起点 sft_2epoch/best 在 8-shot 协议下 rollout 答错率高"风险
- **v6 启动前必做**: 补 sft_2epoch/best + 8-shot CoT 评测 baseline (5 min), 用真实数字定 reward 候选
- **0.46 目标在 0.5B 用户约束下不现实**, 现实上限 Original 50-55% / overall 35-45%
- **下一步**: 用户在 A/B/C 选一个 → 启动 400 步训练
