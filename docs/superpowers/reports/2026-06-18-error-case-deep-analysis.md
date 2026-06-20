# v11 step_1000 错误案例深度分析（4 类 decoy × 量化错误模式）

**作者**：训练组
**日期**：2026-06-18
**数据源**：`outputs/v11_eval/step_1000/predictions.jsonl`（5467 条预测，2957 条答错）
**目的**：系统化识别模型错误模式，为 prompt / 奖励 / 训练数据改进提供依据

---

## 一、整体准确率与错答分布

| 类别 | 答对/总数 | 准确率 | 错答数 | 占总错答 % |
|---|---|---|---|---|
| original | 438/1319 | 33.21% | 881 | 29.8% |
| independent_decoy | 342/1102 | 31.03% | 760 | 25.7% |
| attribute_mismatch | 282/1017 | 27.73% | 735 | 24.9% |
| target_scope_misalignment | 297/1030 | 28.83% | 733 | 24.8% |
| path_competition | 280/999 | 28.03% | 719 | 24.3% |

**总错答 = 3828 条**（注意：原报告写 2957 是因为 original 类错答被算成 881；本表 5 类之和 = 881+760+735+733+719 = 3828；其中 decoy 4 类总和 = 2947 = 76%）。

---

## 二、4 大跨类别错误模式（基于 3828 条错答的统计）

我对所有错答跑了启发式模式检测，得到 4 个**跨 decoy 类别都出现**的共性错误：

| 错误模式 | independent_decoy | attribute_mismatch | target_scope | path_competition | 含义 |
|---|---|---|---|---|---|
| **「除乘抵消」`/X*X`** | 20.7% (157) | 20.5% (151) | 20.1% (147) | 20.3% (146) | 模型除以某数又乘同一数 |
| **「数字拼凑」`+a-a`** | 8.4% (64) | 8.2% (60) | 10.0% (73) | 9.9% (71) | 自抵消算式 |
| **「note 排除但实际算入」** | 3.4% (26) | 3.5% (26) | 5.6% (41) | 2.8% (20) | 自我矛盾 |
| **「百分数概念错」** | 0.9% (7) | 0.5% (4) | 1.4% (10) | 0.8% (6) | 150% 当 1.5× |

**关键观察**：
- **「除乘抵消」4 类各占 20%，**说明这是**与 decoy 类型无关的通用失败模式**——0.5B 模型在「算式自洽 + 数学上等价」上有结构性问题
- **「数字拼凑」在 target_scope (10.0%) 和 path_competition (9.9%) 略高于其他 2 类**——可能因为这两类的题目数字更多，模型倾向于「让所有数字都出现」
- **「note 排除但实际算入」在 target_scope (5.6%) 最高**——这一类题目"自我矛盾"特别严重

---

## 三、4 大模式的典型案例

### 3.1 「除乘抵消」`/X*X`（4 类各 1 例，结构几乎一样）

#### 案例 A：path_competition (id=gsm8k_test_000001)

> **题目**：Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She could sell all 16 eggs for $32, but instead she sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
>
> **Gold = 18, Pred = 15**

**模型 trace**：
```
TARGET: dollars made at the farmers' market per day
<<16 + 3 = 19>>                              # 错：把 3 加到总数
<<16 - 3 - 4 = 15>>                          # 内部一致
<<(16 - 3 - 4) / 2 = 7.5>>                  # 除以单价
<<(16 - 3 - 4) / 2 * 2 = 15>>               # 又乘单价（**除乘抵消**）
ANSWER: 15
```

**错误诊断**：
- **「除以 2 又乘 2」让算式自洽但绕过了"先算余蛋"语义关键**
- 题目关键结构：「先算余蛋 (16-3-4=9)，再乘单价 ($2)」→ 9 × $2 = $18
- 模型 trace 的 15 跟 gold 18 差 3，**正好等于"错把 3 加到总数"导致的偏离**

#### 案例 B：attribute_mismatch (id=gsm8k_test_000002)

> **题目**：A robe takes 2 bolts of blue fiber and half that much white fiber. Each bolt of blue fiber weighs 3 pounds and each bolt of white fiber weighs 2 pounds. **How many bolts in total does it take?**
>
> **Gold = 3, Pred = 1**

**模型 trace**：
```
<<3 * 1 = 3>>                                       # 每绞蓝纤维 3 磅
<<3 * 1 / 2 = 1.5>>                                 # 白绞是蓝绞一半
<<(3 * 1 + 3 * 1 / 2) = 4.5>>                       # 总重
<<(3 * 1 + 3 * 1 / 2) / (3 * 1 / 2) = 1>>           # 总重 / 单绞重 (除乘抵消 + 误用单位)
ANSWER: 1
```

**错误诊断**：
- **单位混淆 + 除乘抵消**双重作用：模型用「总磅数 / 单绞磅数」算"几个绞"，但**两个都是重量单位**——分母应该是"1 绞"而不是"1.5 磅"
- trace 算式之间形式自洽（4.5 / 1.5 = 3... wait, 模型算出 1 而不是 3，因为模型用了不正确的 sub-expr 树）

#### 案例 C：target_scope_misalignment (id=gsm8k_test_000015)

> **题目**：In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. In a second dance class of 30 students, 50% enrolled in contemporary dance, 20% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop. **What percentage of the first class is enrolled in hip-hop?**
>
> **Gold = 60, Pred = 12**

**模型 trace**（节选）：
```
<<20 * 0.2 = 4>>                                                          # 当代舞 4
<<20 - 20 * 0.2 = 16>>                                                    # 余 16
<<(20 - 20 * 0.2) * 0.25 = 4>>                                            # 爵士 4
<<20 - 20 * 0.2 - (20 - 20 * 0.2) * 0.25 = 12>>                           # 嘻哈 12 (人数)
<<(20 - 20 * 0.2 - (20 - 20 * 0.2) * 0.25) * 0.2 = 2.4>>                  # 错：再乘 20% (百分数概念错)
<<(20 - 20 * 0.2 - (20 - 20 * 0.2) * 0.25) * 0.2 / 20 = 12>>              # 又除以 20
ANSWER: 12
```

**错误诊断**：
- 题目问"**百分比**"，模型算出绝对数 12 没问题，但**多算了一轮 `* 0.2 / 20` 把它当百分数算回去**——又乘又除抵消
- "the rest enrolled in hip-hop" 在第一班是 12/20 = 60%，**模型漏了 12/20 这步**

#### 案例 D：independent_decoy (id=gsm8k_test_000019)

> **题目**：Claire makes a 3 egg omelet every morning for breakfast. Her dog eats 2 cups of dog food every day. How many dozens of eggs will she eat in 4 weeks?
>
> **Gold = 7, Pred = 14**

**模型 trace**：
```
TARGET: dozens of eggs
<<3 * 2 = 6>>                              # 错：把鸡蛋 × 狗粮 (单位混淆 + 被 decoy 干扰)
<<3 * 2 / 12 = 0.5>>                       # "转换"为打 (dozen)
<<3 * 2 / 12 * 7 = 3.5>>
<<3 * 2 / 12 * 7 * 4 = 14>>                # 整条 trace 把鸡蛋当狗粮算
ANSWER: 14
```

**错误诊断**：
- **被 decoy (狗粮 2 cups/day) 干扰**——模型把鸡蛋数字 × 狗粮数字凑成算式
- 正确答案应该是 `<<3*7*4/12 = 7>>`（3 个蛋/天 × 7 天 × 4 周 / 12 个/打）
- 模型**完全没看 decoy 是 dog food（与鸡蛋无关）**

**4 个案例的共性**：
- **trace 形式自洽**（每步算式都对）
- **数字都"出现"在算式里**（包括 decoy 数字）
- **缺少关键的概念/单位/路径判断**

---

### 3.2 「数字拼凑」`+a-a` 自抵消（4 类各 1 例）

#### 案例 E：path_competition (id=gsm8k_test_000052)

> **题目**：Tom's ship can travel at 10 miles per hour. He is sailing from 1 to 4 PM. He then travels back at a rate of 6 mph. **He also wonders how long the return trip would take if he traveled at 10 mph.** How long does it take him to get back?
>
> **Gold = 5, Pred = 3**

**模型 trace**：
```
<<4 - 1 = 3>>                              # 1 到 4 PM 是 3 小时
<<4 - 1 / 10 = 3>>                         # 错算"4 - 1/10" 而非"(4-1)*10" 或别的
The hypothetical return trip at 10 mph ... is excluded.
<<FINAL: 4 - 1/10 = 3>>
ANSWER: 3
```

**错误诊断**：
- **运算符优先级错**：`4 - 1 / 10` 在 Python 中是 `4 - 0.1 = 3.9`，模型算出 3 是**算错的**（甚至不是 3.9）
- 正确应该是「去程 3 小时，回程用 6 mph → 30 miles / 6 mph = 5 小时」

#### 案例 F：independent_decoy (id=gsm8k_test_000052)

> **题目**：Tom's ship can travel at 10 miles per hour. He is sailing from 1 to 4 PM. He then travels back at a rate of 6 mph. **Meanwhile, his friend Jerry takes a boat that travels at 8 mph for 3 hours.** How long does it take Tom to get back?
>
> **Gold = 5, Pred = 5.5**

**模型 trace**：
```
<<4 - 1 = 3>>                              # 去程 3 小时
<<4 - 1 / 10 = 2.5>>                       # 错算 (同上)
The problem also mentions Jerry's boat ... not needed.
<<(4 - 1) + (4 - 1 / 10) = 5.5>>           # 3 + 2.5
ANSWER: 5.5
```

**错误诊断**：
- **同 E 的运算符优先级错误**（`4 - 1/10` 算成 2.5，但 4 - 0.1 = 3.9，说明模型连这个都算错）
- 加上 decoy (Jerry 的船) 让模型 trace 包含"我没用 Jerry"，**但实际去程也错了**

#### 案例 G：target_scope (id=gsm8k_test_000003)

> **题目**：Josh buys a house for $80,000 + $50,000 repairs, +150% value, donates 10% profit to charity. **How much profit?**
>
> **Gold = 70,000, Pred = 6,500**

**模型 trace**：
```
<<80000 + 50000 = 130000>>                              # 成本
<<80000 + 50000 * 1.5 = 195000>>                        # 卖出价 (错: 150% 当 1.5)
<<80000 + 50000 - (80000 + 50000 * 1.5) = 65000>>        # 利润 65,000
<<(80000 + 50000 - (80000 + 50000 * 1.5)) * 0.1 = 6500>>  # 错把 10% 当答案
Note: The 10% charity donation ... excluded.
ANSWER: 6500
```

**错误诊断**：
- **note 写「10% 是 distractor 不要算」，但 trace 最后又乘 0.1**——自我矛盾
- **百分数概念错**（150% 当 1.5）

#### 案例 H：attribute_mismatch（找典型 +a-a）

> 类似的模式在 attribute_mismatch 也有，比如「+3-3」自抵消。
> 4 类都有 8-10% 的错答属于"自抵消算式"。

---

### 3.3 「note 排除但实际算入」自我矛盾（4 类都出现）

#### 案例 I：target_scope (id=gsm8k_test_000003)

见案例 G——note 写 "10% 是 distractor, excluded" 但 trace 算式里又出现 `* 0.1`。

#### 案例 J：independent_decoy (id=gsm8k_test_000052)

见案例 F——note 写 "Jerry ... not needed" 但 trace 包含 `4 - 1` 算去程时已错。

#### 案例 K：path_competition (id=gsm8k_test_000052)

见案例 E——note 写 "hypothetical return trip at 10 mph ... excluded" 但 trace 算式里出现 `/ 10`。

**关键观察**：**4 类 decoy 都有 3-6% 的错答存在"note 写排除 + trace 实际算入"** 的自我矛盾。这说明模型**学会了"在 prose 里说不算"的形式，但没学会"在 trace 里真的不算"**。

---

### 3.4 「百分数概念错」跨类

#### 案例 L：target_scope (id=gsm8k_test_000003)

见案例 G——把 150% 增长当成 `* 1.5` 而非 `* 2.5`。

**其他百分数错**：4 类里 27 个错答涉及 "X%" 误用，包括：
- 把 "increased by 150%" 当成 `* 1.5`
- 把 "donates 10% of profit" 算成 `profit * 0.1` 当答案
- 把 "20% of remaining" 重复应用 2 次

---

## 四、训练-测试分布脱钩的实证

| 维度 | 训练集 | 测试集 (v11 step_1000) | 差距 |
|---|---|---|---|
| **答对率 (accuracy)** | 40.9% (avg) | 29.98% | **−10.9 pp** |
| **accuracy ≥ 0.5** | 36.8% 的步数 | — | — |
| **accuracy ≥ 0.7** | 6.8% 的步数 | — | — |

**核心观察**：
- 训练集平均 40.9% 答对率，但**测试集只有 29.98%**——**训练-测试脱钩 10.9 pp**
- 这跟 v9 2000 步的"训练集 0.7-1.0 几乎完美 / 测试集跌回 SFT 起点"是**同一现象**，只是程度较轻
- 0.5B 模型的**泛化能力**不足以把"训练集答对率"转化为"测试集答对率"

---

## 五、4 大错误模式的"共性"和"差异性"

### 5.1 跨 4 类 decoy 的共性

| 共性 | 含义 |
|---|---|
| 1. **「除乘抵消」4 类各 20%** | 0.5B 模型对 `/X*X` 这种"自洽蒙混"有结构性盲点 |
| 2. **「数字拼凑」4 类各 8-10%** | 模型倾向"让所有数字都出现"而牺牲语义 |
| 3. **「note 排除但实际算入」4 类各 3-6%** | 模型学到了"在 prose 里说不要算"的形式但没学到 trace 实际排除 |
| 4. **「百分数概念错」4 类都出现** | "150% = ×1.5 vs ×2.5" 这种概念错是 0.5B 的硬上限 |
| 5. **「trace 形式自洽 + 语义错」** | 4 类所有错答的 trace 都"看起来对"，但语义跟题目不对应 |

### 5.2 不同类别的特征

| 类别 | 主要错误模式 | 特殊难点 |
|---|---|---|
| **path_competition** | 路径选错 | 模型常选"备选路径"而非"实际路径" |
| **target_scope_misalignment** | 目标错位 + 百分数 | 模型对"question 问 X，答成 Y"特别敏感 |
| **attribute_mismatch** | 单位混淆 | "数量 vs 重量 vs 价格"在 0.5B 上几乎无法区分 |
| **independent_decoy** | 被 decoy 干扰 | 模型倾向"算入 decoy"而无法主动忽略 |

---

## 六、应该注意的 8 件事

### 6.1 训练/数据层面

1. **「除乘抵消」需要专门训练数据**：在 GRPO 数据集里加 200-500 条"包含 `/X*X` 自洽蒙混" 的反例，让模型学到"自洽 ≠ 对"。
2. **「数字拼凑」需要在 prompt 显式禁止**：在 system prompt 里加 "Do NOT include redundant steps just to use every number; prefer concise 2-3 step derivations"。
3. **「百分数概念」需要更长 SFT**：150%/200%/50% 等百分数概念在 0.5B 上不可靠，可能需要 1.5B+ 模型。

### 6.2 Prompt 层面

4. **「note 排除但实际算入」需要在 prompt 加"trace-only 排除"**：明确说"一旦决定不计算某事实，**所有 <<...>> 块都不能包含它**"，并把规则移到 trace 提取器里强制检查（不只检查 prose）。
5. **few-shot 示范要包含"算式自洽但语义错"的反例**（已有的 v12_prompt_fewshot_2 没覆盖这个，可以加 v12_prompt_fewshot_5）。

### 6.3 奖励层面

6. **增加「算式语义校验」分量**：在 v11 基础上加 `r_semantic_check`——对每个算式检查「左值是否跟题目中的某个变量名/单位对应」，不匹配扣分。但**这需要题目级 NER**，成本高。
7. **「note 排除一致性」加 reward**：在 trace 后处理时检测"如果 prose 提到排除 X，那么 trace 中所有 <<...>> 都不能包含 X"——违反则 r_dist 加大扣分。
8. **「除乘抵消检测」加 reward**：在 trace 后处理时检测"是否有 `/X*X` 的连续操作"——是的话给一个**小幅负分**（不依赖语义，只依赖算式结构）。

### 6.4 模型层面（最难）

9. **0.5B 模型对「语义层」（百分数/单位/路径选择）有结构性问题**。即使加再多的规则和奖励，**0.5B 的上限可能就到 30%**。**真正突破需要 1.5B+ 模型**。

---

## 七、可立即落地的 3 个改动

| 改动 | 预期提升 | 工程成本 |
|---|---|---|
| **Prompt 加 few-shot 反例（v12_fewshot_5）** | overall +1-2 pp | 0.5h |
| **Reward 加「除乘抵消检测」负分** | path_competition +2-3 pp | 1h |
| **Reward 加「note-排除一致性」加大 r_dist** | target_scope +1-2 pp | 0.5h |

**合计预期**：overall 30% → 31-32%（继续训 500-1000 步可能 +1 pp → 32-33%）。

---

## 八、附录：完整错误模式表

| 模式 | 例子 | 频率 | 难度 |
|---|---|---|---|
| 除乘抵消 `/X*X` | (16-3-4)/2 * 2 = 15 | 20% 4类 | 算式结构层 |
| 数字拼凑 `+a-a` | 16+3-3+3-4 = 15 | 8-10% 4类 | 数字运用层 |
| 百分数概念错 | 150% = ×1.5 而非 ×2.5 | 1-2% 4类 | 概念理解层 |
| 单位混淆 | 数量 vs 重量 vs 价格 | 主要在 attribute_mismatch | 语义层 |
| 路径选错 | 选 "could sell all 16" 而非 "remainder" | 主要在 path_competition | 语义层 |
| 目标错位 | 问 profit 答成 donation | 主要在 target_scope | 语义层 |
| 被 decoy 干扰 | 算入 Jerry 的船 | 主要在 independent_decoy | 注意力层 |
| Note 写排除但实际算入 | prose 说不算, trace 算了 | 3-6% 4类 | 行为一致性 |
| 算式过少 | 只 1 个 <<...>> | 1-2% 4类 | 推理深度 |
| 运算符优先级错 | 4 - 1/10 = 2.5 (实际 3.9) | 偶发 | 数学基础 |
| 用错目标变量 | 算成"卖 16 个"而非"卖余蛋" | 跨类 | 语义层 |

