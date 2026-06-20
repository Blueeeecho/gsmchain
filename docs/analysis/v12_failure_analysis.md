# V12 GRPO 训练评测深度分析报告

> **生成时间**: 2026-06-19
> **分析对象**: v12 GRPO 训练 (Qwen2.5-0.5B-Instruct base, 1524 steps, 0/2 epoch)
> **评测对象**: 5 ckpt (300/600/900/1200/1500), gsm8k_test_clean 5467 条
> **目标**: 46.0% (paper 报告值, 1.5B+ 模型)
> **现实**: peak 21.46% (step 1500), **差距 24.5 pp**

---

## 一、核心数字 (一句话结论)

| 指标 | 值 |
|---|---|
| 训练最终 reward 中位 | 0.6-0.7 |
| 训练最终 acc 峰值 | 0.44 (单 batch) |
| **评测 peak accuracy** | **21.46% (step 1500)** |
| v11 同期 (step 1000) | 29.98% |
| v12 vs v11 同期 | **-8.5 pp (v12 反而更差)** |
| 目标 46% | 差距 24.5 pp |

---

## 二、评测曲线 (5 ckpt)

| ckpt | accuracy | 趋势 |
|---|---|---|
| 300 | 17.27% | baseline |
| 600 | 18.15% | +0.88 pp |
| 900 | 21.40% | +3.25 pp ⬆ |
| 1200 | 20.60% | -0.80 pp (回撤) |
| 1500 | 21.46% | +0.86 pp ⭐ peak |

**形态**: 0→900 抬升 4.1 pp, 900→1500 几乎不动. **训练 1200 步之后基本停滞**.

---

## 三、按 decoy 类别细分

| 类别 | n | acc | lift vs overall |
|---|---|---|---|
| **original** (无 decoy) | 1319 | **28.35%** | **+6.9 pp** |
| path_competition | 999 | 20.12% | -1.3 pp |
| independent_decoy | 1102 | 19.33% | -2.1 pp |
| attribute_mismatch | 1017 | 18.88% | -2.6 pp |
| target_scope_misalignment | 1030 | 18.74% | -2.7 pp |

**关键现象**: original 比 4 类 decoy **高 8-10 pp** —— 0.5B 模型在 decoy 干扰下损失巨大.

---

## 四、错误模式深度剖析 (step 1500, 4294 答错样本)

### 4.1 错误根因分布

| 错误类型 | 数量 | 占比 | 性质 |
|---|---|---|---|
| **final_expression 算错 (推理失败)** | 2737 | **63.7%** | **主导** |
| JSON 解析失败 | 1459 | 34.0% | 格式 |
| off-by-factor 算错 | 679 | 15.8% | 推理 |
| answer 字段与 final_expression 不一致 (R11) | 902 | 21.0% | 规则违反 |
| 答对但被判错 (gold 提取) | 97 | 2.3% | 评测 |
| 答错但 fe 算对 (gold 提取) | (≈ 50) | ~1% | 评测 |

**核心结论: 答错样本里 64% 是 final_expression 算错, 不是格式问题. 模型在"算"上失败, 不是"写 JSON"上失败.**

### 4.2 算错样本的 n_steps 分布

| n_steps | acc |
|---|---|
| 2 | **42.65%** ⭐ |
| 3 | 27.23% |
| 4 | 21.28% |
| 5 | 15.75% |
| 6 | 10.96% |
| ≥7 | ~5% |

**关键现象**: 模型步骤越多, 错误率越高. **n_steps=2 时 43% 正确, n_steps≥5 时正确率掉到 15% 以下**.
这是"长链推理崩塌" —— 0.5B 模型在多步 reasoning 上能力不足.

### 4.3 R4 规则违反: exclude_facts 含 use_facts

- **1001/4294 答错样本 (23.3%)** 中, exclude_facts 列表里出现了 use_facts 重复的事实
- **典型**: "She eats 3 eggs" 同时出现在 use_facts 和 exclude_facts 里
- 说明: 模型对 "use" vs "exclude" 的语义边界**不理解**

### 4.4 答错样本 4 类 decoy 典型 trace

#### Case A: attribute_mismatch
```
Q: Janet's ducks lay 16 eggs. She eats 3 for breakfast, bakes with 4.
   "But instead" thinks about the price of eggs she eats and uses.
   Sells the rest at $2 each.
gold: 18, pred: 26

target: dollars made at the farmers' market per day
use_facts (4): [correct facts]   ← 抄对了
exclude_facts (1): "Janet's eggs are worth $14 at the market price; this is
  an alternative scenario replaced by 'but instead'."  ← 错误理解了 but instead
steps (2):
  1. muffins: 16 - 3 = 13  ← 算式里没扣 muffins 4
  2. dollars: 13 * 2 = 26
final_expression: 16 - 3 - 13 * 2  ← 自抵消 (R1 违反)
```

**分析**: 模型理解了 but instead 是路径切换, 但漏算了 muffins 用了 4 个, 把 use_facts 抄对了但 steps 算错.

#### Case B: independent_decoy
```
Q: Wendi feeds 20 chickens, 3 cups each/day, 3 separate meals.
   Morning 15, afternoon 25, final meal = total?
gold: 20 (因为 final meal 给 20 只, 每只 1 cup = 20 cups)
pred: 120 (算成 15*3 + 25*3)

use_facts (1): "Wendi feeds each chicken 3 cups of feed per day"
exclude_facts (2): [正确排除了 20 鸡数和 3 meals]
steps: 全错 —— 把 morning 15 鸡/afternoon 25 鸡当成 part 1+2, 算成 15*3+25*3
```

**分析**: 模型正确排除了 decoy (20 鸡总数), 但推理时**又重新引入 15/25 鸡数** —— 排除→引入循环.

#### Case C: path_competition
```
Q: Josh buys $80K house, $50K repairs, "increased by 150%".
   But Mike thought it applied to total cost.
gold: 70,000 (利润)
pred: 30,000

use_facts (3): 80000, 50000, increased by 150%
exclude_facts (2): Mike 的误解  ← 正确
steps: 把 increased by 150% 完全丢掉, 直接 80000 - 50000 = 30000
```

**分析**: 模型**完全忽略了 R5 (increased by 150% = × 2.5)** 的规则, 走了 Mike 的错误路径.

#### Case D: target_scope_misalignment
```
Q: 200 GB file, 2 GB/min, 40% 时 Windows 强制重启 20 min.
   Restart download from beginning. "After that" (path switch) some rate change.
   Time to download in minutes?
gold: 160
pred: 10

use_facts (3): 2 GB/min, 20 min restart, 40% 处 20 min  ← 抄对
exclude_facts (2): "40% 不相关"  ← 错误
steps: 把 40% 当成 1 min 量级, 推出 10
```

**分析**: **R8 违反** —— "after that" 是 path switch, 但模型把 40% 当成不相关信息, 反而把路径 A 的事实弄丢了.

---

## 五、Reward 信号分析 (训练侧)

### 5.1 训练 reward 成分 (步 300 metrics)

| 成分 | mean |
|---|---|
| score | 0.556 |
| answer | 0.000 ❌ (300 步时全错) |
| step_value | 0.25 (满分 1.0) |
| core_trace_sim | 0.496 |
| core_final_sim | 0.292 |
| core (加权) | 0.455 |
| distractor_sim | 0.426 |
| distractor (1-sim) | 0.093 (低 = 排除 decoy 好) |
| n_steps | 3.75 vs gold 3.5 |
| n_gold_steps | 3.5 |

**v11 公式**: `3.0*r_answer + 1.5*r_step_value + 0.5*r_core - 0.5*r_dist`

### 5.2 训练末段 (1518-1524) 准确率 / 奖励

| step | acc | rwd | 解读 |
|---|---|---|---|
| 1518 | 0.375 | 1.82 | |
| **1519** | **0.4375** | **2.34** | 训练峰值 |
| 1520 | 0.0 | 0.59 | 急速回撤 |
| 1521 | 0.0 | 0.43 | |
| 1522 | 0.3125 | 1.45 | |
| 1524 | 0.0625 | 0.62 | |

**训练 acc 0.44 / 评测 acc 0.21 — train-eval gap 23 pp**:
- 训练集 (gsm8k_train 3051) 和测试集 (gsm8k_test 5467) 分布有差
- 训练 reward 成分以 `core_trace_sim` 为主 (软匹配), **core_final_sim 一直 < 0.3 (意味着模型最后一步数字答案跟 gold 数字差很多)**
- reward 中 `r_answer` 权重 3.0 但实际 answer 字段长期 0 → **奖励信号对"答对"不敏感**

### 5.3 Reward 设计的盲点

| 公式项 | 权重 | 训练末段均值 | 实际贡献 |
|---|---|---|---|
| r_answer | 3.0 | 0.0 (步 300) | **几乎无信号** |
| r_step_value | 1.5 | 0.25 | 弱信号 |
| r_core | 0.5 | 0.46 | 主信号 (soft match) |
| r_dist | -0.5 | 0.09 | 弱信号 |

**问题: r_answer (答对就给满分) 长期为 0, 因为模型在没学完之前极少答对. 这意味着 GRPO 的优势函数 (R - baseline) 大部分来自 r_core (soft match 答案链相似度), 而不是真正的"答对"信号.**

---

## 六、V12 vs V11 关键差异 (v12 为什么更差)

| 维度 | v11 (COT_BRACKETS) | v12 (COT_BRACKETS_V12_JSON) |
|---|---|---|
| 输出格式 | 自然语言 + `<<expr = val>>` | 纯 JSON 6 字段 |
| System prompt tokens | 118 | 795 (+677, ×6.7) |
| User prompt tokens | 1099 | 1461 (+362) |
| 总 prompt tokens | ~1217 | ~2256 (+85%) |
| max_prompt_length | 1280 | 2048 (+60%) |
| Few-shot 示范 | 1 个 (8-shot COT) | 3 个 (Ex 1/2/3) |
| Rules 数量 | 5 条 | 12 条 (R1-R12) |
| Reward 公式 | (相同 v11 公式) | (相同) |
| 0.5B 同期 step 1000 | **29.98%** | (v12 同期 step 1500 = 21.46%) |

**核心 trade-off**:
- v12 prompt 信息密度高 (含 JSON schema + 12 rules + 3 示范 + 详细字段说明) → **2256 tokens 占用 0.5B 大量 context**
- 0.5B 模型在长 prompt 下 reasoning 容量被压缩
- 0.5B 模型对 JSON 6 字段格式的格式遵循率 ~95%, 但 reasoning 能力 (final_expression 算对率) 只有 ~36%

**v12 失败的真正原因**:
- 不是 JSON 格式问题 (JSON parse fail 仅 34%)
- 不是 reward 公式问题 (v11 公式还能给到 2.34 峰值)
- **是 0.5B 模型在 2256 tokens prompt 下的 reasoning 容量被压缩** —— 训练 metric (acc 0.44 in-batch) 远高于 eval (acc 0.21 out-of-distribution), 说明训练时模型 "记住" 了 prompt 模式但**没真正学会 reasoning**

---

## 七、深层结论与改进方向

### 7.1 三大失败根源 (按影响排序)

1. **【首要】0.5B base + 长 prompt 的容量瓶颈**
   - 0.5B 模型在 2256 tokens prompt 下能"学会"格式, 但 reasoning 容量被 JSON 6 字段填满
   - 解决: 缩短 prompt (砍 few-shot 至 1 个, 砍 6 个冗余 rules, 砍 schema 注释), 让出 ~500 tokens 给 reasoning

2. **【次要】Reward 信号稀疏 + 软匹配**
   - r_answer 长期 0, GRPO 优势来自 r_core (soft match 链相似度)
   - 模型被鼓励"生成相似的推理链"而非"答对"
   - 解决: 增加 dense reward (per-step answer match, 单元测试正确性)

3. **【再次】R4 规则违反率 23% (exclude 含 use)**
   - 模型对 use_facts vs exclude_facts 的语义边界模糊
   - 解决: 简化字段 (例如合并成 "facts" + "decoys" 两个 list, 减少歧义)

### 7.2 训练-评测 gap 23 pp

- 训练集: gsm8k_train_balanced_one_variant (3051)
- 测试集: gsm8k_test_clean (5467)
- 训练 acc 0.44 (in-batch) vs 评测 acc 0.21 → **train-eval gap 23 pp**
- 原因: 训练用 GRPO, 4 sample per prompt, 模型在 4 sample 内偶有完整正确; 评测是单 sample top-k=1
- 解决: 评测用 n=4 + majority vote / pass@k

### 7.3 47% 离 46% 差多少?

| 当前 | 目标 | 差距 |
|---|---|---|
| 21.46% | 46.00% | 24.5 pp |

- 即使把所有算式错 (63.7%) 修到 80% 正确, 总 acc 也只到 30% 左右
- 要达到 46%, 需要**换 base 模型 (≥1.5B)** + **缩短 prompt** + **denser reward**

---

## 八、文档引用

- v12 prompt: `docs/prompts/v12_long_prompt.md` (4736 tokens)
- v12 reward: `train_pipeline/reward_chaingsm_v12_json_verl.py` (v11 公式 + JSON 解析)
- 评测数据: `outputs/v12_eval/step_1500/predictions.jsonl` (5467 条)
- 训练 metric: `outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/qwen2.5-0.5b-grpo-verl-v12-json/20260619_002804/metrics/train_metrics.jsonl`

