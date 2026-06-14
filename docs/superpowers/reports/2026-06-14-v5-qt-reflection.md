# v5 Quicktest 反思 + 奖励函数阶段性总结 (2026-06-14 05:30)

> 写于 v5 完整 500 步训练进行中, 基于 v5 quicktest (5 步) + 历年 v2/v3/v4 数据
> 用于支持 v5 跑完后 v6 方向决策

---

## 1. v5 Quicktest 5 步实际数据 (2026-06-14 05:16-05:26)

### 1.1 配置
- 起点: v3 best/actor/huggingface (30.47% / 29.95% Original)
- 训练: 5 step × ~15.95s/step = 1:19 总训练时间
- Eval: 1 次 (step_5, vLLM 5467 条, ~5 min)

### 1.2 训练内 metrics 趋势

| step | acc | liv | stepcalc | gated | reward | len | bonus | kl_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5625 | 1.0 | 0.943 | 0.4375 | 0.653 | 150 | 0.034 | 0.0 |
| 2 | 0.375 | 0.913 | 0.786 | 0.375 | 0.541 | 184 | 0.041 | 0.0003 |
| 3 | 0.5625 | 1.0 | 0.979 | 0.551 | 0.682 | 198 | 0.034 | 0.0001 |
| 4 | 0.8125 | 1.0 | 0.969 | 0.808 | 0.874 | 154 | 0.025 | 0.0003 |
| 5 | 0.4375 | 0.984 | 0.944 | 0.483 | 0.596 | 198 | 0.034 | 0.0002 |

均值: acc=0.55, gated=0.53, length_bonus_flag=0.66, liveness=0.98

### 1.3 Eval (step_5 on 5467)

| 指标 | v3 best | v4 step_100 | v5 quicktest step_5 | Δ vs v3 best |
|---|---:|---:|---:|---:|
| **Overall** | 0.3047 | 0.2976 | **0.3078** | **+0.31pp** ✅ |
| **Original** | 0.2995 | 0.2957 | **0.3025** | **+0.30pp** ✅ |
| attribute_mismatch | 0.3097 | 0.2930 | 0.3127 | +0.30pp ✅ |
| independent_decoy | 0.3140 | 0.3022 | 0.3158 | +0.18pp ✅ |
| path_competition | 0.3033 | 0.2993 | 0.3043 | +0.10pp ✅ |
| target_scope_misalignment | 0.2981 | 0.2981 | 0.3049 | +0.68pp ✅ |

**5 类目全部 +0.1~0.7pp** (quicktest 5 步后, 且 noise 大).

---

## 2. 关键观察与反思

### 2.1 v5 quicktest 告诉我们的

1. **环境恢复确认**: 之前 v5 启动失败是 Ray+vLLM 状态污染, 本次重启后 v5 entry 跑通了 (filter prompts → trainer init → 5 步训 → 1 次 eval → 出 predictions.jsonl 全流程)
2. **liveness 松绑过头信号**: 5 步内 liveness 0.98-1.0 持续接近满分, **reward 几乎不再区分真假引用**. 这跟 v2 失败原因(子串匹配导致 chain_quality 0.89 0.9+ → 全部满分)有相似风险
3. **length_bonus 0.05 效果不明**: 5 步内 bonus_flag 0.5-0.6875 部分触发, 但 **配合 liveness 松绑后, 1.0+ 答对的 reward = 0.55×1.0 + 0.25×(c2a×0.975) + 0.05 = 0.55+0.244+0.05 = 0.844**, 跟 v3 同位置 (0.55+0.25×0.31+0 = 0.628) 比, **reward 上限提高了 0.21** — 给模型更宽松的探索空间
4. **response_length 150-198**: v3 训练内 ~200, v4 训练内 ~200, v5 quicktest 150-198 — **没变短, 没退化** (v2 failure 表现是 len 掉到 110)
5. **KL 0.02 系数下 kl_loss 0.0001-0.0003**: 比 v3 0.04 系数下 (~0.002) 还低 10x — **KL 放宽太多, policy 几乎不动 ref**. 需要 500 步数据看是否这个趋势持续

### 2.2 5 步数据不能下结论

- 5 步 4 中答对, 0.55 accuracy mean 是 5 个数据点 noise
- 1.0 liveness 持续但只 5 步, 500 步后是否真触发 hacking 不可知
- Eval 30.78% overall 比 v3 best 涨 0.31pp **在 noise 范围内** (v3 训练内 0.42-0.47, 5 步内随机波动可达 ±5pp)

**需要 500 步完整数据 + 至少 3 次 eval 才能下结论**.

### 2.3 v3 撞天花板的根因 (基于 v3 best predictions 错误模式分析)

**v3 best on 5467 测试集**:
- 总数 5467, 答对 1666 (30.47%)
- Original 1319, 答对 395 (29.95%)
- **答错 70% 链/答案本身就是推理失败** (不是 format 错, 不是引用错)
- **chain 能算 gold 但 pred 答错 = 1.9%** (reward 漏判极低)
- **chain 格式错 (parse fail) = 0.5%** (format 0.989 已近满)
- **chain 引用错 (liveness=0) 但答对 = 罕见**

**v3 reward 信号质量**:
- correct mean reward 0.92, wrong mean reward 0.20 → **gap 0.72, reward 信号够强** ✅
- 答错 924 个里 920 个 (99.6%) 在 reward 0.2-0.4 区间 → reward 成功把"答错"压成固定 0.2-0.4
- 答错里 step_calc 0.88, no_degenerate 0.93 → 多数错答 chain 算式都正确, **就是推理最后一步算错**

**结论**: **v3 reward 已经收得很紧, 没什么 reward hacking 空间; 70% 错答是模型推理能力本身的失败**. 再调 reward 收益不大.

### 2.4 0.5B + JSON 协议的天花板

| 模型 | 协议 | 协议性质 | 测试集 | Original acc |
|---|---|---|---|---:|
| 0.5B sft_2epoch | JSON schema | 强结构 | 5467 | 25.55% |
| 0.5B v3 best | JSON schema | 强结构 | 5467 | 29.95% |
| 0.5B v5 step_5 (5 步) | JSON schema | 强结构 | 5467 | 30.25% (+0.30) |
| 0.5B base | 8-shot CoT | 自由推理 | 1319 | 43.29% (无 RL) |
| 1.5B base | 8-shot CoT | 自由推理 | 1319 | 72.25% (无 RL) |

**0.5B 撞 30% 是 JSON 协议本身的约束**:
- JSON schema 要求模型严格遵循 `{target, selected_steps: [{variable, description, expression, value}], final_expression, answer}` 格式
- 0.5B 模型在严格格式约束下, 思考空间被压缩
- 自由推理 (8-shot CoT) 下, 0.5B 原生就能到 43% (无 RL)

**v5 reward 调整天花板 ~32-35%** (v3 30.5% + liveness 松绑 0~2pp + KL 放宽 0~1pp + length_bonus 0~1pp)

**要破 0.46 必须换协议 / 换模型**, 0.5B + JSON 协议无解.

---

## 3. 奖励函数阶段性总结

### 3.1 我们调过什么

| 版本 | 关键改动 | 训练内 acc | 测试集 overall | Original |
|---|---|---:|---:|---:|
| **v1 起点** (sft_2epoch) | base | n/a | 0.2742 | 0.2555 |
| **v2 烟测** (reward hacking) | 4 个漏洞并存 | 0.039 | 0.0724 | 0.0587 ❌ |
| **v3** | chain_to_answer_ok 硬门 + causal_liveness 收紧 + answer 0.55 | 0.42-0.47 | **0.3047** | **0.2995** ✅ |
| **v4** | prompt 迁移 NEUTRAL (v3 reward 不变) | 0.39-0.44 | 0.3002 | 0.2942 |
| **v5 quicktest 5 步** | v3 + liveness 松绑 (a')(a'') + KL 0.02 + length_bonus 0.05 | 0.37-0.81 | 0.3078 | 0.3025 (5 步 noise) |

### 3.2 我们的奖励函数学到了什么

1. **answer 0.55 主导是对的** — v3/v4/v5 都验证 answer 信号是主梯度, 不能改
2. **chain_to_answer_ok 硬门是必要的** — 掐住"chain 算错但 reward 满"的 hacking
3. **causal_liveness 收紧 (v3) → 松绑 (v5) 的循环**:
   - v3 0.34 太严, 挡真引用 (liveness 0.34 → reward 少 0.04)
   - v5 0.98 太松, 几乎不区分真假引用 (reward 多 0.04 但失去信号)
   - **理想值 0.5-0.7**: 既不过严也不过松
4. **length_bonus 0.05 太弱**: 在 chain quality 0.25 系数下, 0.05 占比 20% (实际只影响 5-7% sample, 训练信号稀)
5. **KL 0.04 → 0.02**: 在 quicktest 5 步内 KL 降到 0.0001-0.0003, 实际效果是 **让 policy 几乎不动 ref, 训练速度变慢**. 500 步后看趋势

### 3.3 我们没调过但可考虑

| 方向 | 假设 | 风险 | 实施成本 |
|---|---|---|---|
| **answer 0.55 → 0.65** | 更强主梯度, 但 0.55 已主导, 可能没空间 | 低 | 改 1 个数, 5 行代码 |
| **gated_chain_quality 0.25 → 0.35** | 增加 chain 质量信号 | 中 | 改 1 个数 |
| **chain_to_answer_ok 容差 1e-3 → 1e-2** | 给模型算式容错 | 中 | 改 1 个数 |
| **length_bonus 0.05 → 0.10** | 加强格式规整奖励 | 低 | 改 1 个数 |
| **新增 chain_diversity_bonus** | 鼓励非零散 step | 中 | 新增 ~30 行 |
| **完全重写 reward** | 抛弃 v3 框架 | 高 | 重写 ~440 行 |

---

## 4. v6 方向预研

### 4.1 5 个候选方向

| 方向 | 假设 | 风险 | 成本 | 预期收益 |
|---|---|---|---|---|
| **A. 换 Qwen2.5-1.5B + 8-shot CoT 协议** | 1.5B 容量 + 自由推理 突破 0.5B JSON 协议 30% 天花板 | 高 (1.5B 全参 6GB + vLLM 7GB + optim 12GB ≈ 25GB/32GB 勉强) | 训练时间 3-4x | 50-65% (无 0.46 目标压力下) |
| **B. 换 Qwen2.5-1.5B + JSON 协议 (v5 reward)** | 1.5B 容量 + 同 reward | 中 (JSON 协议下 1.5B 上限未知) | 训练时间 3-4x | 35-50% |
| **C. v5.1 = v5 reward 微调** (answer 0.55→0.65 或 length_bonus 0.05→0.10) | 在 0.5B 上再挤 1-2pp | 低 | 跟 v5 一样, ~3h | 31-32% |
| **D. 0.5B + 8-shot CoT 协议** | 自由推理 0.5B 原生 43%, RL 后可能 50%+ | 中 (8-shot CoT 需重写 SFT 起点 + 重新跑 SFT) | SFT 1h + RL 3h | 45-55% |
| **E. 数据增强** (加 SFT 答错的 5000 样本) | 数据多样性 | 中 | 1-2h 数据 prep | 32-35% |

### 4.2 决策建议

按 "对 0.46 目标的贡献" 排序:

1. **A. 换 1.5B + 8-shot CoT (首选)**: 0.46 目标在 0.5B + JSON 协议下不可达. 1.5B + 8-shot CoT 上限 72.25% (无 RL), 即使 RL 退化 10pp 仍 60%+, 远超 0.46
2. **D. 0.5B + 8-shot CoT (备选)**: 自由推理下 0.5B 上限 43% (无 RL), RL 后 45-50% 仍差 0.46 一线
3. **C. v5.1 微调**: 0.5B + JSON 协议继续挤, 30-32% 封顶, 离 0.46 远
4. **B. 换 1.5B + JSON**: 1.5B + JSON 上限未知, 风险大
5. **E. 数据增强**: 0.5B + JSON 协议下, 32-35% 封顶

**推荐路径**:
1. **等 v5 500 步跑完** (5 ckpt 评测, ~3h)
2. **v5 报告** 落盘: 若 v5 涨到 32-33%, 验证多变量组合有效, 可考虑 v5.1 (C 方向) 再挤
3. **若 v5 跟 v3 一样撞 30%**: 直接 v6 = 换 Qwen2.5-1.5B + 8-shot CoT (A 方向)

### 4.3 A 方向实施成本预估

- **1.5B 训练 (4x 时长)**: 0.5B 500 步 ~1.5h, 1.5B ~6h
- **GPU 32GB 跑 1.5B 风险**: 全参 6GB + vLLM 7GB + optim 12GB ≈ 25GB, 留 7GB buffer
  - **需开 gradient checkpointing + 减小 vLLM gpu_mem_util**
  - 1.5B 8-shot CoT 协议 prompt 更长 (1024+ tokens), 需调 `max_prompt_length`
- **8-shot CoT 协议 ≠ JSON schema**: 训练 prompt 完全重写, 评测 prompt 也要重写
  - 需要重新 preprocess train parquet (1-2h)
  - 重新写 SFT 模型起点 (1.5B 起点需重新 SFT, ~2h)
  - 重新写 reward (8-shot CoT 协议无 selected_steps, 改用 answer 字符串匹配 + 推理步骤特征)
- **总成本**: SFT 重训 2h + RL 6h + 评测 2h = **~10 小时**

---

## 5. v5 完整 500 步训练的预期

基于 quicktest 5 步数据, 我预期:

- **100 步 eval**: 0.30-0.31 (跟 v3/v4 step_100 持平, +0~0.5pp)
- **200 步 eval**: 0.30-0.32 (liveness 松绑效应开始显现)
- **300 步 eval**: **0.31-0.33 (关键观测点)**
- **400 步 eval**: 0.30-0.33 (v3 在 300 步进入平台, v5 KL 放宽 0.02 应该慢收敛)
- **500 步 eval**: 0.30-0.33 (上限点)

**终止条件**:
- 200 步后 Original 连续 2 次 ≤ 上次 AND overall 连续 2 次 ≤ 上次 → 停
- step 100 eval 比 v3 best baseline 掉 > 2pp → 立即停, 切 v4 step_100 起点重跑

**ckpt 处置**:
- 保留 5 ckpt 中 best overall + best original 各 1 个
- 删除中间 ckpt (节省 ~40GB 盘)

---

## 6. 风险与监控

### 6.1 v5 500 步训练风险

| 风险 | 监控指标 | 应对 |
|---|---|---|
| 训练挂 (Ray 状态污染) | GPU 0% util, 无 step 进度 | 重启 v5 entry |
| vLLM 静默崩 | eval 跑到一半崩 | 跳过 baseline eval (EVAL_BASELINE=0 已设) |
| liveness 1.0 触发 hacking | response_length 掉到 < 100 | 立即停, 反思 liveness 松绑设计 |
| length_bonus 0.05 没效果 | bonus_flag 0% 触发 | 监控 bonus_flag mean, 5% < 视为无效 |

### 6.2 训练内关键监控点

- step 50 末: accuracy 应该 ≥ 0.40 (v3 训练内 1-50 = 0.42, 0.5B 起步水位)
- step 100 末: accuracy 应该 ≥ 0.42
- step 200 末: accuracy 应该 ≥ 0.44 (如果 KL 放宽有效, 应该比 v3 涨)
- step 300 末: 验证 v3 撞天花板的位置 v5 是否能突破

### 6.3 当前监控脚本

`/tmp/monitor_v5.sh` 每 30 分钟跑一次, 输出:
- 进程状态
- GPU util/memory
- 训练进度 (Training Progress X/500)
- 已保存 ckpt
- 已有 eval 结果
- 训练内 reward 趋势 (last 5 steps)

---

## 7. 决策记录 (2026-06-14 05:30)

- **2026-06-14 04:50-04:58**: v5 quicktest 第一次启动失败 (tokenizer + vLLM)
- **2026-06-14 05:16-05:26**: v5 quicktest 第二次启动成功, 5 步训练 + 1 次 eval 完成
  - step_5 eval: 30.78% overall / 30.25% original (vs v3 best 30.47%/29.95%)
  - **5 类目全部 +0.1~0.7pp** (5 步 noise, 但 5 类目齐涨罕见)
- **2026-06-14 05:26**: 启动 v5 完整 500 步训练 (run_id=20260614_052636, KL 0.02, 起点 v3 best)
- **2026-06-14 05:30**: 本反思文档落盘, 支持 v5 跑完后 v6 方向决策

---

**结论**: 
- **短期**: 等 v5 500 步跑完 (估计 1.5h 训练 + 5×5min eval = 2h)
- **中期**: v5 报告反思后定 v5.1 或 v6
- **长期**: **v6 = 换 1.5B + 8-shot CoT 协议是 0.46 目标的唯一路径**
- **立即**: 监控 v5 训练, 准备 v6 spec/plan/reward (A 方向)
