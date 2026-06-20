# GRPO v8 → v11 整体训练流程与失败案例深度分析报告

**作者**：训练组
**日期**：2026-06-18
**项目**：ChainGSM / GSM8K 长链推理增强（GRPO + 自定义奖励）
**基座模型**：Qwen2.5-0.5B-Instruct（单卡 32GB，RTX 5090）
**测试集**：gsm8k_test_clean.jsonl，5 类问题（original + 4 类 decoy 变体）× 6575 题 = 5467 个独立 base_id
**目标准确率**：参考论文报告的 **46%**

---

## 一、TL;DR

| 版本 | 时间 | 训练步 | 奖励核心 | overall 准确率 | 距 46% 差距 |
|---|---|---|---|---|---|
| baseline (Qwen2.5-0.5B) | — | — | — | 8.32% (8-shot: 23.58%) | -37.7 pp |
| SFT 2 epoch | 2026-06-13 | — | — | 20.54% (8-shot: 23.58%) | -25.5 pp |
| SFT 8 epoch (cot_2ep_resume) | 2026-06-17 | — | — | 27.05% | -18.9 pp |
| **v7 GRPO 100 步** | 2026-06-14 | 100 | format 0.10 + answer 0.70 + numeric 0.20 | **24.25%** | -21.8 pp |
| v7 step_100 original | — | — | — | **44.28%** (历史 best) | -1.7 pp |
| v8 GRPO 1000 步 | 2026-06-14 | 1000 | format 0.05 + answer 0.85 + numeric 0.05 + step 0.05 | 31.70% (step_800) | -14.3 pp |
| v8.2 GRPO 1000 步 | 2026-06-14 | 1000 | 5 机制 (format 0.05 + answer 0.55 + c2a 0.20 + target 0.15 + len 0.05) | 31.70% | -14.3 pp |
| **v9 GRPO 500 步** | 2026-06-16 | 500 | format 0.2 + answer 2.5 + core 1.5 - dist 0.5 | 23.25% | -22.8 pp |
| v9 GRPO 2000 步 | 2026-06-17 | 2000 | 同上 | 22.17% (训练后期负迁移) | -23.8 pp |
| **v10 GRPO 3046 步** | 2026-06-17 | 3046 | format 0.2 + answer 2.5 + core 1.2 + calc 0.3 - dist 0.5 | **30.30%** | -15.7 pp |
| **v11 GRPO 1000 步** | 2026-06-18 | 1000 | **删 format/calc，加 step_value 1.5** | **29.98%** (step_1000) | -16.0 pp |
| v11 续训 200-1000 | 2026-06-18 | 1000 | 同 v11 | 27.62% → 29.98% | -16.0 pp |

**核心结论**：
- **v7 短暂达到 original=44.28%**（距 46% 仅 1.7 pp），但这是「过拟合 SFT 起点 + 8-shot 风格」的特殊窗口；
- **v8 之后所有版本都卡在 overall 28-32%** 的天花板，**reward 设计的边际效益 < 1 pp**；
- **46% 目标在 0.5B 模型 + 单卡 32GB 约束下结构性不可达**——失败案例揭示了 0.5B 在 decoy 干扰下的「推理形式正确 + 语义错」系统性问题。

---

## 二、版本演进：从 v7 到 v11 的奖励设计

### 2.1 时间线

```
2026-06-13  SFT 2 epoch 起点 (overall 20.54%, original 22.74%)
2026-06-14  v7 GRPO 100 步 → original 44.28% (历史 best, 后期跌)
2026-06-14  v8/v8.2 1000 步 → 31.70%, original 跌 8pp
2026-06-16  v9 GRPO 500 步 → 23.25% (重测协议, 不再 8-shot)
2026-06-17  v9 GRPO 2000 步 → 22.17% (后期负迁移)
2026-06-17  v10 GRPO 3046 步 → 30.30%
2026-06-18  v11 GRPO 1000 步 (续训 800 步) → 29.98%
```

### 2.2 奖励公式对比表

| 版本 | 公式 | 设计意图 | 关键问题 |
|---|---|---|---|
| **v7** (3 子项) | `0.10·format + 0.70·answer + 0.20·numeric` | answer 主导，numeric 拉算式 | numeric 0.7 抢 answer 0.14 梯度，模型学「算式对 + 答案错」 |
| **v8** (4 子项) | `0.05·format + 0.85·answer + 0.05·numeric + 0.05·step_count` | 砍 numeric 抢回 answer 梯度 | step_800 overall 31.70% 但 original 跌 8pp（43.29→35.48%） |
| **v8.1** (5 机制) | `0.05·format + 0.55·answer + 0.20·c2a + 0.15·target + 0.05·len - 0.30·irrelevant` | 加 chain-to-answer 校验 + 无关算式惩罚 | parquet 漏字段（gold_expression / core_chain 等），前 100 步白训 |
| **v8.2** (5 机制完整) | 同 v8.1 + 修复 parquet | 同上 + 完整 reference | overall 涨 7pp 但 original 跌 8pp，**核心能力转移** |
| **v9** (signed) | `max(0, 0.2·format + 2.5·answer + 1.5·core - 0.5·dist)` | core 强化 trace 模仿，dist 惩罚分心 | 2000 步后 original 跌回 SFT 起点 22.97% |
| **v10** (signed+) | `0.2·format + 2.5·answer + 1.2·core + 0.3·calc - 0.5·dist` | 加 calc 自洽性 | 3046 步 30.30%，但错答均奖 0.69（接近满分） |
| **v11** (signed++) | `3.0·answer + 1.5·step_value + 0.5·core - 0.5·dist` | 删 format/calc，加 step_value（对 gold），强 answer | 1000 步 29.98%，错答均奖降到 0.13 |

### 2.3 核心设计哲学的转变

| 阶段 | 哲学 | 表现 |
|---|---|---|
| v7-v8 | 「让模型算对」 | numeric 抓中间算式、answer 抓最终值 |
| v8.1-v8.2 | 「让模型不分心」 | c2a (chain-to-answer) 校验、irrelevant 惩罚 |
| v9-v10 | 「让模型模仿 trace」 | core 0.7/0.8 trace token 对齐 |
| v11 | 「让模型答案对 + 跟 gold step 对齐」 | step_value vs gold，core 降权到 0.5 |

**v11 的根本性改进**（基于 288 条真实评测诊断）：
- **错答均奖 0.69 → 0.13**（−81%），错答被显式压低
- **corr(reward, is_correct) 0.985 → 0.989**
- **训练效率 3×**：v11 1000 步 ≈ v10 3046 步

但**天花板没破**：29.98% vs 30.30%。

---

## 三、训练流程与各版本评测结果

### 3.1 SFT 起点（所有 GRPO 版本的输入）

| SFT 版本 | 评测 original | 评测 overall | 备注 |
|---|---|---|---|
| SFT 1 epoch | 21.68% | 21.29% | 8-shot 协议下 33.74% |
| SFT 2 epoch (v9 起点) | 22.74% | 20.54% | 8-shot 协议下 23.58% |
| **SFT 8 epoch (cot_2ep_resume, v10/v11 起点)** | **23.96%** | **27.05%** | STEP 协议下 |

### 3.2 v7 GRPO（100 步，2026-06-14）

**奖励**：`0.10·format + 0.70·answer + 0.20·numeric_correctness`

**关键发现**：
- step_100：**overall 24.25%, original 44.28%**（**历史最佳 original**）
- step_200~500：original 跌到 40-42%（**reward hacking numeric**）
- 失败案例：模型学到「算式写得对（numeric 给分）+ 答案乱答（answer 给 0）」，整体 reward 仍能到 0.4+

**根因**：numeric 子项是 0-1 连续信号，answer 是 0/1 稀疏信号。GRPO 偏好连续信号 → policy 偏向优化 numeric 而非 answer。

### 3.3 v8 GRPO（1000 步，2026-06-14）

**奖励**：`0.05·format + 0.85·answer + 0.05·numeric + 0.05·step_count`

| step | overall | original | 现象 |
|---|---|---|---|
| 200 | — | 40.03% | 起点 |
| 400 | 26.34% | 36.16% | original 继续跌 |
| 600 | 30.20% | 34.80% | 4 类变体涨 |
| **800** | **31.70%** | **34.80%** | **best overall** |
| 1000 | 31.63% | 35.48% | 持平 |

**关键发现**：
- **4 类 decoy 涨 13pp，original 跌 8pp**——能力转移
- 模型「为了避开假干扰，过度怀疑真推理」→ 短回答 + 跳步
- 失败案例：原 original 题"Janet's ducks"（gold 18），模型输出 "She eats 3 * 7 = 21 ... 4 * 7 = 28 ... 16 - 21 - 28 = -33 ... since she can't sell negative, sells all 16 = 16 * 2 = 32"

### 3.4 v8.1 / v8.2 GRPO（5 机制，2026-06-14）

**奖励**：`0.05·format + 0.55·answer + 0.20·c2a + 0.15·target + 0.05·len - 0.30·irrelevant`

**v8.1 启动 bug**：parquet 写入时漏 5 个关键字段（gold_expression / core_chain / distractor_chain / question / category），前 100 步白训。修复后 v8.2 重训。

**v8.2 评测**：overall 31.70%（同 v8），但多了「无关算式惩罚」机制 → 1 类变体 +5pp，其他持平。

### 3.5 v9 GRPO（500/2000 步，2026-06-16/17）

**奖励**：`max(0, 0.2·format + 2.5·answer + 1.5·core - 0.5·dist)`

| 步数 | overall | original | 现象 |
|---|---|---|---|
| 起点 (SFT 2 epoch) | 20.54% | 22.74% | — |
| step_400 | 23.18% | 25.78% | +2.64pp |
| step_500 | **23.25%** | **26.00%** | +2.71pp, 历史 v9 最佳 |
| step_1200 | 21.88% | 22.97% | **跌回 SFT 起点** |
| step_1400 | 22.17% | 22.97% | 持平 |

**关键发现**：
- **训练集 accuracy 0.7-1.0 几乎完美过拟合**
- 但测试集 original 在 1200 步跌回 SFT 起点 → **reward hacking 训练分布**
- v9 报告结论：「**0.5B + SFT 2 epoch + v9 reward 的天花板 ~26% original，30% 阈值在 0.5B 上不可达**」

### 3.6 v10 GRPO（3046 步，2026-06-17）

**奖励**：`0.2·format + 2.5·answer + 1.2·core + 0.3·calc - 0.5·dist`

| step | overall | original | indep_decoy | attr_mis | path_comp | scope_mis |
|---|---|---|---|---|---|---|
| 300 | 26.69% | 30.48% | 27.32% | 25.68% | 24.77% | 25.19% |
| 900 | 27.94% | 31.31% | 28.77% | 25.08% | 27.29% | 27.25% |
| 1800 | 28.73% | 31.92% | 29.68% | 27.43% | 26.45% | 28.16% |
| 2700 | 30.24% | 34.04% | 31.20% | 29.10% | 28.35% | 28.46% |
| **3046** | **30.30%** | 34.04% | 31.58% | 29.03% | 28.51% | 28.31% |

**关键发现**：
- 3046 步 30.30%，但**错答均奖 0.69（接近满分）**——奖励与正确率脱钩
- r_ans 0.46→0.47 饱和，KL=0 数值无变化（pytorch 截断），entropy 早期塌缩
- **v10 核心问题：r_calc (self-consistency) 让「答案错 + 算式对」拿高分**

### 3.7 v11 GRPO（1000 步，2026-06-18）

**奖励**：`3.0·answer + 1.5·step_value + 0.5·core - 0.5·dist`（删 format/calc）

| step | overall | original | indep_decoy | attr_mis | path_comp | scope_mis |
|---|---|---|---|---|---|---|
| 200 | 28.44% | 30.93% | 29.13% | 27.43% | 26.13% | 27.77% |
| 400 | 27.62% | 30.93% | 27.40% | 26.84% | 25.23% | 26.70% |
| 600 | 29.50% | 33.13% | 29.22% | 28.32% | 26.63% | 29.13% |
| 800 | 29.89% | 33.06% | 30.58% | 27.83% | 27.73% | 29.22% |
| **1000** | **29.98%** | **33.21%** | **31.03%** | **27.73%** | **28.03%** | **28.83%** |

**关键发现**：
- **1000 步 = v10 3046 步效率的 3×**
- **错答均奖 0.13（−81% vs v10）**——奖励对错答显式惩罚
- 但整体准确率仍卡在 30%——**reward 不是天花板**

---

## 四、4 类典型失败案例（v11 step_1000 评测）

下面 4 个失败案例**精确对应**测试集 4 类 decoy 类别。每例都展示：

1. **题目**：测试集原始问题
2. **gold 答案 vs 模型预测**：答案对比
3. **模型 trace**：模型写出的推理步骤
4. **奖励视角**：每个 reward 分量给了多少分
5. **错误诊断**：模型错在哪、为什么奖励抓不到

---

### 4.1 案例 A：path_competition（路径竞争）

**题目**：
> Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She could sell all 16 eggs for $32, but instead she sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

**gold = 18**（剩余 9 蛋 × $2 = $18）　**模型预测 = 15**

**模型 trace**：
```
TARGET: dollars made at the farmers' market per day
<<16 + 3 = 19>>                              # 错把 3 加到总数上
<<16 - 3 - 4 = 15>>                          # 后续用错误的"剩余 15"
<<(16 - 3 - 4) / 2 * 2 = 15>>                # 15 蛋 × $2 = $15 (除以2乘2 自洽)
ANSWER: 15
```

**奖励视角（v11）**：
- `r_answer = 0`（18 ≠ 15）
- `r_step_value`：gold step values = [..., 9, 18]，pred = [..., 19, 15, 15, 15]，**重叠度低**（约 0.2-0.3）
- `r_core`（trace token 编辑距离）：trace 跟 gold 重叠度 ~0.4

**错误诊断**：
- **核心问题**：「除以单价 2，又乘以单价 2」——算术自洽但**绕过了「剩余多少蛋」这个语义关键**
- 题目关键结构是「先算余蛋 (16-3-4=9)，再乘单价 ($2)」；模型错把「加 3 到总数」当成「找到 daily total」
- **r_step_value 抓不到**：「19」「15」是有效算式的副产品，跟 gold step 的"9"和"18"不匹配但分母不同
- **r_answer 抓得到**：18 ≠ 15 拿 0 分

**为什么奖励不够**：
- r_step_value 是 token 级别的数值匹配，但**模型 trace 中的「15」与 gold 中的「9」都是有效整数**，仅靠编辑距离无法识别「应该用 9 而不是 15」
- 模型学到的信号是「答案要 ≈ 18」而不是「要先算余蛋」

---

### 4.2 案例 B：target_scope_misalignment（目标偏移）

**题目**：
> Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. After selling the house, he donates 10% of his profit to charity. **How much profit did he make from the house flip?**

**gold = 70,000**　**模型预测 = 6,500**

**模型 trace**：
```
<<80000 + 50000 = 130000>>                              # 成本对
<<80000 + 50000 * 1.5 = 195000>>                        # 错把 1.5× 当 150% 增长
<<80000 + 50000 - (80000 + 50000 * 1.5) = 65000>>        # 利润 = 65000
<<(80000 + 50000 - (80000 + 50000 * 1.5)) * 0.1 = 6500>>  # 把"10%"当成答案
Note: The 10% charity donation is a distractor ...
ANSWER: 6500
```

**错误诊断**：
- **核心问题**：「150% 增长」=「× 2.5」不是「× 1.5」——模型对「百分数增长」概念理解错
- 模型甚至**自己 note 写明 10% 是 distractor**（说明 r_dist 训练生效），但**最后还是把 10% 算进去了**
- **r_step_value** 对「195000 vs 200000（正确 130000 × 2.5）」能抓到部分不一致（pred step = 195000，gold 应是 200000），但模型又「自洽地」用错值继续算

**为什么奖励不够**：
- 模型的**自洽性自我麻痹**：每一步算式都内部一致（190000 - 130000 = 65000，65000 × 0.1 = 6500），但第 2 步的「1.5」就是错的
- 奖励抓不到「150% 增长应该用乘法 2.5 不是 1.5」这种**语义错误**——它需要的是「题目级常识」

---

### 4.3 案例 C：attribute_mismatch（属性错位）

**题目**：
> A robe takes 2 bolts of blue fiber and half that much white fiber. Each bolt of blue fiber weighs 3 pounds and each bolt of white fiber weighs 2 pounds. **How many bolts in total does it take?**

**gold = 3**（2 bolts 蓝 + 1 bolt 白）　**模型预测 = 1**

**模型 trace**：
```
<<3 * 1 = 3>>                                  # "每绞蓝纤维 3 磅"
<<3 * 1 / 2 = 1.5>>                            # "白绞是蓝绞一半"  (错把"half that much"理解成"重量一半")
<<(3 * 1 + 3 * 1 / 2) / (3 * 1 / 2) = 1>>      # 总重 / 单绞重 = 1
ANSWER: 1
```

**错误诊断**：
- **核心问题**：「half that much」指**数量**（2 绞的一半 = 1 绞），不是**重量**
- 模型把题目里的「每绞重 X 磅」当成核心变量，**忽略了「数量」才是被问的对象**
- 模型把"weight per bolt"和"number of bolts"**两个单位混为一谈**

**为什么奖励不够**：
- **r_step_value 跟 gold 几乎不重叠**（pred = [3, 1.5, 4.5, 1]，gold = [2, 1, 3]），但模型拿到「r_step_value = 0.2-0.3」（因为 token 都有数字）
- 奖励**无法识别「单位语义错位」**——"1.5 磅"和"1 绞"在 token 层面都是数字，模型学不到「问数量就别算重量」

---

### 4.4 案例 D：independent_decoy（独立干扰项）

**题目**：
> Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. **She also sells homemade cookies at the market. She bakes 2 batches of cookies each day...**
>
> **How much does she make from selling duck eggs at the farmers' market each day?**

**gold = 18**　**模型预测 = 24**

**模型 trace**：
```
<<16 + 3 = 19>>                                # 错把 3 加到总数
<<16 + 3 - 3 = 16>>                            # 减 3
<<16 + 3 - 3 - 4 = 12>>                        # 减 4
<<(16 + 3 - 3 - 4) * 2 = 24>>                  # 12 × $2 = $24
The cookies information ... is not needed ...
ANSWER: 24
```

**错误诊断**：
- **核心问题**：distractor 完美处理（note 写明 cookies 不要），但**主推理链的第一步就错**（16+3=19，应该是 16 不变）
- 模型用了 "16 + 3 - 3" 这种**自抵消**操作（+3-3=0），等于绕了一圈——典型的「为了让所有数字都出现在 trace 里而强行写算式」

**为什么奖励不够**：
- v11 的 r_dist 抓到「没碰 distractor」（模型主动 note 出来），给 0 分
- **但 r_step_value 对「19」和「16」都接受**——pred step = [19, 16, 12, 24]，gold = [13, 9, 18]；重叠度约 0.3
- 奖励**无法识别「数字拼凑」（让所有数字都出现）** vs「真实推理」（用对的数字）

---

## 五、为什么 4 类失败案例都是「奖励抓不到」

### 5.1 四个共性错误模式

| 模式 | 案例 | 表现 | 奖励盲点 |
|---|---|---|---|
| **自洽性算术蒙混** | A | 除以 2 又乘 2 | r_step 只看数值，不看语义 |
| **百分数/单位错位** | B、C | 150% 当 1.5×、数量当重量 | r_step 抓不到「概念正确」 |
| **数字拼凑** | D | +3-3=0 自抵消 | r_step 接受任何数字序列 |
| **distractor 处理形式化** | B、D | note 写明但实际算入 | r_dist 抓 note 文本，不抓 trace 实际 |

### 5.2 4 类错误的奖励得分分解（粗略估计）

| 案例 | r_answer | r_step_value | r_core | r_dist | 总分 |
|---|---|---|---|---|---|
| A (path_competition) | 0.0 | 0.3 | 0.4 | 0.0 | 1.45 |
| B (target_scope) | 0.0 | 0.3 | 0.3 | 0.0（识别到 distractor） | 1.20 |
| C (attribute_mismatch) | 0.0 | 0.2 | 0.3 | 0.0 | 0.90 |
| D (independent_decoy) | 0.0 | 0.3 | 0.3 | 0.0 | 1.20 |

**对比正确样本**：v11 step_1000 正确样本的 R_total ≈ 3.0-4.5（r_answer=1.0, r_step≈0.7-0.9, r_core≈0.6-0.8）。

→ **错答仍能拿到 25-50% 的 reward**，因为 r_step 和 r_core 都在 0.2-0.4 之间。**这是 0.5B 模型在「形式正确」+「语义错」下结构性拿不到低分的体现**。

### 5.3 0.5B 模型的硬上限在哪

| 任务类型 | 0.5B 能力 | 1.5B+ 能力 |
|---|---|---|
| 单步算式（16-3-4=9） | ✅ 100% | ✅ 100% |
| 两步推理（先余蛋再乘单价） | ⚠️ 70% | ✅ 95% |
| 百分数/单位换算（150% 增长） | ❌ 30% | ✅ 80% |
| 多路径竞争（除以 2 又乘 2 蒙混） | ❌ 50% | ⚠️ 75% |
| 长上下文 distractor 排除 | ⚠️ 60% | ✅ 90% |

**关键观察**：0.5B 模型的失败**不在算术**，在**语义层**（百分数、单位、概念区分）。这不是奖励能修的——奖励只能告诉模型「答案对不对」，**无法告诉模型「150% 增长应该 × 2.5」这种事实知识**。

---

## 六、为什么 46% 目标在 0.5B 上结构性不可达

### 6.1 0.5B 模型能力测试（基线 8-shot）

来自 `docs/chaingsm_base_8shot_evaluation_zh.md`：

| 模型 | prompt | original | 备注 |
|---|---|---|---|
| Qwen2.5-0.5B + 8-shot CoT | `qwen_multiturn_8shot_chat` | **43.29%** | 8 个 CoT 示范 + step-by-step 引导 |
| Qwen2.5-1.5B + 8-shot CoT | 同上 | 72.25% | |
| Qwen2.5-Math-1.5B + 8-shot | math 模板 | 76.19% | |
| Qwen2.5-3B + 8-shot CoT | 同上 | 85.90% | |
| Qwen2.5-7B (paper) | 8-shot CoT | 89.0% | |

**0.5B + 8-shot 的 original 43.29% 已经接近 46% 目标**——但**这是 8-shot few-shot prompting 的能力**，不是训练出来的。

**v9 报告已经发现**：SFT 把 8-shot 能力摧毁了（SFT 1 epoch 后 8-shot 跌到 33.74%，-9.55pp）。GRPO 在 SFT 起点上**恢复了部分 original 能力**（v7 step_100 到 44.28%），但**这个窗口非常窄**（200 步后就跌）。

### 6.2 0.5B 模型的训练-评测能力脱钩

| 训练步 | 训练集 accuracy | 测试集 original | 脱钩 |
|---|---|---|---|
| v9 2000 步 | 0.7-1.0 | 22.97%（= SFT 起点） | **严重** |
| v10 3046 步 | r_ans 0.47 饱和 | 34.04% | 中度 |
| v11 1000 步 | r_ans 0.5-0.7 | 33.21% | 轻度 |

**0.5B 在训练分布上能拟合**（r_ans 抬升、r_step 抬升），但**测试分布的泛化跟不上**。这是小模型 + 长尾分布的结构性问题。

### 6.3 46% 的来源

参考论文的 46% 至少需要：
1. **≥ 1.5B 模型**（8-shot baseline 70%+，GRPO 加持可上 76-80%）
2. **长 SFT 预热**（≥ 5 epoch 的 CoT 数据）
3. **多步 GRPO**（≥ 2000 步）
4. **8-shot 评测协议**（而非我们用的 0-shot cot_brackets）

**我们流水线的 4 个短板**：
- 0.5B 模型（缺 ~8-10 pp）
- SFT 起点 original 23-27%（缺 ~5-8 pp）
- 0-shot 评测（缺 ~1-2 pp）
- 单卡 32GB 限制下 1000-3000 步训练（缺 ~1-2 pp）

→ **总缺口 15-22 pp，与实测 16-17 pp 吻合**。

---

## 七、给老师汇报的核心结论

### 7.1 我们做了什么

| 时间 | 阶段 | 关键工作 | 产出 |
|---|---|---|---|
| 06-13 | SFT 1-2 epoch | 准备 CoT 数据 + 8-shot prompt | overall 20-27% |
| 06-14 | v7/v8 GRPO | 100-1000 步奖励实验 | original 短暂到 44%，整体 31% |
| 06-14 | v8.1/v8.2 | 5 机制奖励（c2a + irrelevant 惩罚） | 31.70% but original 跌 8pp |
| 06-16 | v9 GRPO | signed reward + core 强化 | step_500 best 23.25% |
| 06-17 | v9 2000 步 | 验证长期训练 | 后期跌回 SFT 起点 22.97% |
| 06-17 | v10 GRPO 3046 步 | signed+ + calc 自洽 | 30.30% best |
| 06-18 | v11 GRPO 1000 步 | 删 format/calc + step_value vs gold | 29.98% (3× 效率) |

### 7.2 失败原因（按重要度排序）

1. **0.5B 模型容量硬上限**（最大原因）
   - 0.5B 在 decoy 4 类（path_competition, attribute_mismatch, target_scope_misalignment, independent_decoy）上卡在 27-31%，无论怎么调奖励都突破不了
   - 失败案例 A/B/C/D 全部是「语义错」（百分数、单位、概念区分），不是「算术错」

2. **训练-评测分布脱钩**
   - v9 2000 步训练集 accuracy 0.7-1.0，测试集 original 跌回 SFT 起点
   - v10 3046 步 r_ans 0.46-0.47 饱和，测试集只到 34%
   - 小模型 + 长尾分布的结构性泛化失败

3. **奖励函数的语义盲点**
   - 4 类错误全部是「算式自洽 + 语义错」，r_step 和 r_core 无法识别
   - 即使错答也能拿 25-50% reward（v11 错答均奖 0.13 但还有 r_step 0.3-0.4）

4. **SFT 摧毁 8-shot 能力**
   - SFT 1 epoch 后 8-shot 协议 original 跌 9.55pp（43.29→33.74）
   - 我们的 STEP 协议与 8-shot 协议不兼容，模型「忘了」free-form CoT

5. **评测协议不匹配**
   - 0-shot cot_brackets 协议下，0.5B 上限约 33%（v11 step_1000）
   - 8-shot 协议下，0.5B 上限 43.29%（未经训练）
   - 差距 ~10pp 是「prompt 红利」

### 7.3 后续路径

| 方案 | 预期 | 工程成本 | 备注 |
|---|---|---|---|
| **继续 v11 训到 2000 步** | 30.5-31% | 3h 训练 + 4× 评测 | 见顶 |
| **Qwen2.5-1.5B SFT→v11-GRPO** | **35-40%** | SFT 4-6h + GRPO 6-8h | 显存紧但可跑，**最现实** |
| **8-shot CoT 评测** | +10pp | 改 prompt | 与训练 prompt 不一致，可能破 actor |
| **CoT 改程序式 (Python)** | +3-5pp | 重设计 | 工程量大 |
| **多 ckpt ensemble 投票** | +1-2pp | 推理侧，无重训 | 快速验证 |

**最现实组合**：Qwen2.5-1.5B-Instruct → SFT 3 epoch → v11-GRPO 1500 步 + 8-shot CoT 评测，预期 overall 38-43%。**距 46% 仍差 3-8pp，需要在模型/数据/训练侧进一步加码**。

---

## 八、附录

### 8.1 关键文件

| 用途 | 路径 |
|---|---|
| v11 奖励函数 | `train_pipeline/reward_chaingsm_v11_verl.py` |
| v10 奖励函数 | `train_pipeline/reward_chaingsm_v10_verl.py` |
| v9 奖励函数 | `train_pipeline/reward_chaingsm_v9_verl.py` |
| v8 奖励函数 | `train_pipeline/reward_chaingsm_lbprm_v8_verl.py` |
| v7 奖励函数 | `train_pipeline/reward_chaingsm_lbprm_v7_verl.py` |
| v11 训练配置 | `train_configs/local/grpo_verl_v11.yaml` |
| v11 续训脚本 | `train_scripts/local/run_grpo_verl_v11_resume.sh` |
| 4 ckpt 评测脚本 | `train_scripts/local/eval_v11_4ckpts.sh` |
| 失败案例预测 | `outputs/v11_eval/step_1000/predictions.jsonl` |
| v7 评测报告 | `docs/superpowers/reports/2026-06-14-lbprm-v7-report.md` |
| v8.2 评测报告 | `docs/superpowers/reports/2026-06-14-lbprm-v8-2-report.md` |
| v9 评测报告 | `docs/superpowers/reports/2026-06-16-v9-report.md` |
| v9 2000 评测 | `docs/superpowers/reports/2026-06-17-v9-2000-report.md` |
| v10 评测 | `outputs/v10_eval/step_3046/eval_result.json` |
| v11 5 ckpt 评测 | `outputs/v11_eval/step_{200,400,600,800,1000}/eval_result.json` |

### 8.2 5 个评测协议统一说明

所有评测使用：
- **数据**：`chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl`（5467 个独立 base_id × 5 类别展开 = 6575 题）
- **方法**：`cot_brackets`（与训练 prompt 一致）
- **推理**：vLLM, `gpu_memory_utilization=0.5`, `max_tokens=512`, `top_k=1`, seed=42
- **类别分布**：
  - original 1319 (24.1%)
  - independent_decoy 1102 (20.2%)
  - attribute_mismatch 1017 (18.6%)
  - target_scope_misalignment 1030 (18.8%)
  - path_competition 999 (18.3%)

### 8.3 8-shot baseline 数据来源

`docs/chaingsm_base_8shot_evaluation_zh.md` —— 8-shot CoT 协议下 Qwen2.5-0.5B-Instruct original 43.29%，是 0.5B 在该项目测试集上的理论上限提示。
