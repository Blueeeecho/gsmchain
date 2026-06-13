可以。基于你新增的要求，应该调整为：

> **模型输出中保留 `variable` 字段，用于增强语言逻辑和中间变量意识；
> 但 reward 不对变量名做语义匹配，只把变量名作为格式完整性的一部分。**

这样既能避免模型退化成纯数字计算，也不会因为 `total_min` / `total_minimum` / `monthly_minimum_total` 这种命名差异误伤模型。

---

# 1. 当前数据格式是否支持？

支持。

你们当前固定数据格式已有：

```json
{
  "question_distracted": "...",
  "answer": "18000",
  "solution_original": "... <<300+200+500=1000>> ...",
  "core_chain": [
    ["student_loans_min", "temp1", "+200"],
    ["temp1", "total_min", "+500"],
    ["total_min", "monthly_payment", "*1.5"],
    ["monthly_payment", "annual_payment", "*12"]
  ],
  "distractor_chain": [
    ["entertainment_monthly", "entertainment_annual", "*12"]
  ],
  "gold_expression": "((300+200+500)*1.5)*12",
  "distractor_expression": "100*12"
}
```

我们可以本地构造：

```json
"gold_trace": [...]
"distractor_trace": [...]
"gold_response": {...}
"rejected_response": {...}
```

不需要额外调用大模型。

---

# 2. 统一模型输出格式

模型输出 JSON，保留 `variable` 字段。

推荐格式：

```json
{
  "target": "Jessica's annual debt payment",
  "selected_steps": [
    {
      "variable": "temp1",
      "description": "Compute the sum of student loan and credit card minimum payments.",
      "expression": "300+200",
      "value": "500"
    },
    {
      "variable": "total_min",
      "description": "Compute the total minimum monthly debt payment.",
      "expression": "300+200+500",
      "value": "1000"
    },
    {
      "variable": "monthly_payment",
      "description": "Compute the monthly payment after paying 50% more than the minimum.",
      "expression": "(300+200+500)*1.5",
      "value": "1500"
    },
    {
      "variable": "annual_payment",
      "description": "Compute the annual debt payment.",
      "expression": "((300+200+500)*1.5)*12",
      "value": "18000"
    }
  ],
  "final_expression": "((300+200+500)*1.5)*12",
  "answer": "18000"
}
```

这里的 `variable` 可以参考 `core_chain` 的 target 变量，但 reward 不要求模型完全输出同名变量。

---

# 3. 本地数据如何生成训练数据

## 3.1 总体思路

本地处理流程：

```text
原始 JSONL
→ 读取 core_chain / distractor_chain / gold_expression / distractor_expression
→ 解析 gold_expression 生成 gold_trace
→ 解析 distractor_expression 生成 distractor_trace
→ 用 trace 构造 SFT gold_response
→ 用 distractor_trace 构造 DPO rejected_response
→ GRPO 阶段直接使用 gold_trace / distractor_trace 计算 reward
```

不需要调用大模型。

---

## 3.2 生成 gold_trace

使用 `gold_expression` 的 AST 解析，按计算顺序抽取子表达式。

例如：

```text
gold_expression = ((300+200+500)*1.5)*12
```

本地解析为：

```json
"gold_trace": [
  {
    "variable": "temp1",
    "description": "Compute an intermediate sum for the minimum monthly debt payment.",
    "expression": "300+200",
    "value": "500"
  },
  {
    "variable": "total_min",
    "description": "Compute the total minimum monthly debt payment.",
    "expression": "300+200+500",
    "value": "1000"
  },
  {
    "variable": "monthly_payment",
    "description": "Compute the monthly payment after paying 50% more than the minimum.",
    "expression": "(300+200+500)*1.5",
    "value": "1500"
  },
  {
    "variable": "annual_payment",
    "description": "Compute the annual debt payment.",
    "expression": "((300+200+500)*1.5)*12",
    "value": "18000"
  }
]
```

其中 `variable` 优先来自：

```text
core_chain 中每条边的 target 字段
```

例如：

```text
temp1
total_min
monthly_payment
annual_payment
```

如果 AST 解析出来的 trace 数量和 `core_chain` target 数量一致，就一一对应。

---

## 3.3 如果 trace 数量和 core_chain 数量不一致怎么办？

本地处理仍然可以完成，不需要大模型。

### 情况 A：trace 数量少于 core_chain

例如 gold_expression 被写成：

```text
(300+200+500)*18
```

AST 可能只得到：

```text
300+200+500
(300+200+500)*18
```

这时可以使用 fallback：

```text
只给现有 trace 分配后几个关键变量；
缺失的变量不强行生成；
reward 以表达式 trace 为准，不以变量数量为准。
```

### 情况 B：trace 数量多于 core_chain

例如 expression 中有额外中间步骤。

处理：

```text
前面的多余中间变量命名为 auto_step_0, auto_step_1；
core_chain target 优先分配给最后能够通向 final_expression 的步骤。
```

### 情况 C：表达式无法解析

例如 expression 包含自然语言或格式错误。

处理：

```text
该样本标记为 trace_build_error；
训练时可以丢弃，或只用于 answer-level SFT；
不调用大模型修复。
```

---

## 3.4 生成 distractor_trace

使用 `distractor_expression` 同样本地解析。

例如：

```text
distractor_expression = 100*12
```

生成：

```json
"distractor_trace": [
  {
    "variable": "entertainment_annual",
    "description": "Compute the annual entertainment spending.",
    "expression": "100*12",
    "value": "1200"
  }
]
```

其中 `variable` 优先来自：

```text
distractor_chain 中每条边的 target 字段
```

例如：

```text
entertainment_annual
```


# 4. SFT 数据如何构造

## 4.1 SFT prompt

```text
You are a careful math reasoning assistant.

Solve the math word problem by selecting only the reasoning steps needed for the target question.

The problem may contain extra arithmetic chains that are valid but irrelevant.

Return valid JSON only. Do not include any extra text.

Required JSON schema:
{
  "target": "...",
  "selected_steps": [
    {
      "variable": "...",
      "description": "...",
      "expression": "...",
      "value": "..."
    }
  ],
  "final_expression": "...",
  "answer": "..."
}

Rules:
1. Identify the exact quantity asked by the problem.
2. Use only the computation steps needed for that target.
3. Do not include irrelevant distractor computations in selected_steps.
4. Each selected step must include variable, description, expression, and value.
5. The variable name should be a short meaningful name for the computed quantity.
6. The expression must be a valid arithmetic expression.
7. The final_expression must compute the final answer.
8. The answer must be a single number.
```

User：

```text
Problem:
{question_distracted}
```

---

## 4.2 SFT output

使用本地构造的 `gold_response`：

```json
{
  "target": "Jessica's annual debt payment",
  "selected_steps": [
    {
      "variable": "temp1",
      "description": "Compute an intermediate sum for the minimum monthly debt payment.",
      "expression": "300+200",
      "value": "500"
    },
    {
      "variable": "total_min",
      "description": "Compute the total minimum monthly debt payment.",
      "expression": "300+200+500",
      "value": "1000"
    },
    {
      "variable": "monthly_payment",
      "description": "Compute the monthly payment after paying 50% more than the minimum.",
      "expression": "(300+200+500)*1.5",
      "value": "1500"
    },
    {
      "variable": "annual_payment",
      "description": "Compute the annual debt payment.",
      "expression": "((300+200+500)*1.5)*12",
      "value": "18000"
    }
  ],
  "final_expression": "((300+200+500)*1.5)*12",
  "answer": "18000"
}
```

---

# 5. DPO 数据如何构造

## 5.1 DPO prompt

```text
Problem:
{question_distracted}

Return valid JSON only.
```

---

## 5.2 chosen response

直接使用 `gold_response`。

```json
{
  "target": "Jessica's annual debt payment",
  "selected_steps": [
    {
      "variable": "temp1",
      "description": "Compute an intermediate sum for the minimum monthly debt payment.",
      "expression": "300+200",
      "value": "500"
    },
    {
      "variable": "total_min",
      "description": "Compute the total minimum monthly debt payment.",
      "expression": "300+200+500",
      "value": "1000"
    },
    {
      "variable": "monthly_payment",
      "description": "Compute the monthly payment after paying 50% more than the minimum.",
      "expression": "(300+200+500)*1.5",
      "value": "1500"
    },
    {
      "variable": "annual_payment",
      "description": "Compute the annual debt payment.",
      "expression": "((300+200+500)*1.5)*12",
      "value": "18000"
    }
  ],
  "final_expression": "((300+200+500)*1.5)*12",
  "answer": "18000"
}
```

---

## 5.3 rejected response

使用 `distractor_trace` 构造：

```json
{
  "target": "Jessica's annual entertainment spending",
  "selected_steps": [
    {
      "variable": "entertainment_annual",
      "description": "Compute the annual entertainment spending.",
      "expression": "100*12",
      "value": "1200"
    }
  ],
  "final_expression": "100*12",
  "answer": "1200"
}
```

---

# 6. GRPO 奖励如何适配当前格式

使用：

[
R =
0.2r_{\text{fmt}}
+r_{\text{ans}}
+r_{\text{expr}}
+r_{\text{trace}}
-0.5r_{\text{dist}}
]

---

## 6.1 (r_{\text{fmt}})：格式奖励

变量名加入格式奖励，而不是核心语义奖励。

```text
valid JSON，包含 target / selected_steps / final_expression / answer，每个 step 包含 variable / description / expression / value
```

最终：

```text
r_fmt ∈ [0, 1]
```

注意：
`variable` 只要求存在、非空、结构合理，不要求和 gold variable 名称完全一致。

---

## 6.2 (r_{\text{ans}})：答案奖励

```text
r_ans = 1 if answer == gold answer else 0
```

样例：

```text
answer == 18000
```

则：

```text
r_ans = 1
```

---

## 6.3 (r_{\text{expr}})：表达式等价奖励

```text
r_expr = 1 if final_expression ≡ gold_expression else 0
```

样例中：

```text
gold_expression = ((300+200+500)*1.5)*12
```

下面都判为正确：

```text
((300+200+500)*1.5)*12
1500*12
1000*1.5*12
(300+200+500)*18
```

只要和 gold expression 符号等价即可。

---

## 6.4 (r_{\text{trace}})：核心计算轨迹奖励

不比较变量名 exact match。

比较模型输出的 `selected_steps[*].expression` 是否覆盖 `gold_trace[*].expression`。

定义：

```text
r_trace = matched_gold_trace_steps / total_gold_trace_steps
```

样例 gold_trace：

```text
300+200
300+200+500
(300+200+500)*1.5
((300+200+500)*1.5)*12
```

模型输出：

```json
[
  {
    "variable": "loan_credit_sum",
    "expression": "300+200",
    "value": "500"
  },
  {
    "variable": "minimum_total",
    "expression": "300+200+500",
    "value": "1000"
  },
  {
    "variable": "monthly_total",
    "expression": "1000*1.5",
    "value": "1500"
  },
  {
    "variable": "yearly_total",
    "expression": "1500*12",
    "value": "18000"
  }
]
```

虽然变量名不同，也应该接近满分。

推荐匹配规则：

```text
1. expression 与 gold step 符号等价：匹配；
2. expression 的 value 与 gold step value 相同，且顺序合理：弱匹配；
3. 否则不匹配。
```

可以设：

```text
符号等价匹配 = 1
value 相同且顺序合理 = 0.5
不匹配 = 0
```

这样允许小模型写出 `1000*1.5`，即使它没有展开成 `(300+200+500)*1.5`。

---

## 6.5 (r_{\text{dist}})：干扰链惩罚

基于 `distractor_expression` 和 `distractor_trace`，不基于关键词。

定义：

```text
r_dist = max(
  final_expression_equiv_distractor_expression,
  distractor_trace_coverage
)
```

样例中：

```text
distractor_expression = 100*12
```

如果模型输出：

```json
"final_expression": "100*12",
"answer": "1200"
```

则：

```text
r_dist = 1
```

如果模型 `selected_steps` 里包含：

```json
{
  "variable": "entertainment_annual",
  "expression": "100*12",
  "value": "1200"
}
```

即使最后答案写对，也说明干扰链进入 selected_steps，应给惩罚。

---

# 7. 四类训练方案

---

## 7.1 SFT

### 目标

让模型学会稳定输出：

```text
target
selected_steps
variable
description
expression
value
final_expression
answer
```

### 数据

```text
input = question_distracted
output = gold_response
```

### 训练目标

普通 next-token loss。

### 作用

SFT 负责：

```text
1. 学格式；
2. 学会保留中间变量；
3. 学会输出计算轨迹；
4. 初步学习忽略 distractor chain。
```

---

## 7.2 DPO

### 目标

让模型偏好 core trace，而不是 distractor trace。

### 数据

```text
prompt = question_distracted
chosen = gold_response
rejected = distractor_response
```

### chosen

来自 `gold_trace`。

### rejected

来自 `distractor_trace`。

### 作用

DPO 负责：

```text
1. 学习 core response > distractor response；
2. 降低模型输出 distractor answer 的概率；
3. 作为 GRPO 前的偏好 warm-up。
```

---

## 7.3 GRPO

### 目标

使用可验证奖励直接优化模型输出。

### Reward

```text
R = 0.2*r_fmt + r_ans + r_expr + r_trace - 0.5*r_dist
```

### 作用

GRPO 负责：

```text
1. 强化正确答案；
2. 强化 gold expression；
3. 强化 gold trace 覆盖；
4. 抑制 distractor expression / distractor trace。
```

---

## 7.4 SFT → GRPO

### 目标

这是主方法。

### Stage 1

SFT 学格式和基本 trace 输出。

### Stage 2

GRPO 用 reward 强化 chain-aware reasoning。

### 作用

```text
SFT 解决“怎么输出”；
GRPO 解决“输出什么更好”。
```

这是最推荐的最终训练方案。

---

# 8. 当前样例的最终构造结果

## 8.1 gold_trace

```json
[
  {
    "variable": "temp1",
    "description": "Compute an intermediate sum for the minimum monthly debt payment.",
    "expression": "300+200",
    "value": "500"
  },
  {
    "variable": "total_min",
    "description": "Compute the total minimum monthly debt payment.",
    "expression": "300+200+500",
    "value": "1000"
  },
  {
    "variable": "monthly_payment",
    "description": "Compute the monthly payment after paying 50% more than the minimum.",
    "expression": "(300+200+500)*1.5",
    "value": "1500"
  },
  {
    "variable": "annual_payment",
    "description": "Compute the annual debt payment.",
    "expression": "((300+200+500)*1.5)*12",
    "value": "18000"
  }
]
```

---

## 8.2 distractor_trace

```json
[
  {
    "variable": "entertainment_annual",
    "description": "Compute the annual entertainment spending.",
    "expression": "100*12",
    "value": "1200"
  }
]
```

---

## 8.3 gold_response

```json
{
  "target": "Jessica's annual debt payment",
  "selected_steps": [
    {
      "variable": "temp1",
      "description": "Compute an intermediate sum for the minimum monthly debt payment.",
      "expression": "300+200",
      "value": "500"
    },
    {
      "variable": "total_min",
      "description": "Compute the total minimum monthly debt payment.",
      "expression": "300+200+500",
      "value": "1000"
    },
    {
      "variable": "monthly_payment",
      "description": "Compute the monthly payment after paying 50% more than the minimum.",
      "expression": "(300+200+500)*1.5",
      "value": "1500"
    },
    {
      "variable": "annual_payment",
      "description": "Compute the annual debt payment.",
      "expression": "((300+200+500)*1.5)*12",
      "value": "18000"
    }
  ],
  "final_expression": "((300+200+500)*1.5)*12",
  "answer": "18000"
}
```

---

## 8.4 rejected_response

```json
{
  "target": "Jessica's annual entertainment spending",
  "selected_steps": [
    {
      "variable": "entertainment_annual",
      "description": "Compute the annual entertainment spending.",
      "expression": "100*12",
      "value": "1200"
    }
  ],
  "final_expression": "100*12",
  "answer": "1200"
}
```

---

# 9. 最终推荐的主 reward

最终就使用：

```text
R = 0.2*r_format + r_answer + r_expression + r_trace - 0.5*r_distractor
```

其中：

```text
r_format:
检查 JSON、字段、variable/description/expression/value 是否存在。

r_answer:
答案是否正确。

r_expression:
final_expression 是否等价于 gold_expression。

r_trace:
selected_steps 是否覆盖 gold_trace。
不要求变量名一致。

r_distractor:
final_expression 或 selected_steps 是否匹配 distractor_expression / distractor_trace。
```

这个版本满足你的要求：

```text
1. 保留 variable 输出；
2. 不让模型退化成纯数字；
3. 变量名只作为格式和语言逻辑约束；
4. reward 不依赖变量名 exact match；
5. gold_trace / distractor_trace 可以本地构造；
6. 不需要调用大模型；
7. 整体 reward 仍然简洁。 
```
