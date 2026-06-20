# v12 训练 Prompt：插入 1-2 个真实完整解题案例

> **作者**：训练组
> **日期**：2026-06-18
> **基础版本**：v11 cot_brackets（`train_pipeline/eval_vllm_chaingsm.py` 中的 `COT_BRACKETS_*`）
> **改动目标**：在 User Prompt 模板的 "Format" 和 "Rules" 之间插入 1-2 个真实 gold 案例，让 0.5B 模型从"看格式"升级到"看真实推理模式"。

---

## 一、设计动机

### 1.1 失败案例揭示的训练盲点

来自 v11 step_1000 评测的 4 类典型失败：

| 案例 | 错误 | 0-shot prompt 缺什么 |
|---|---|---|
| A path_competition | "除以 2 又乘 2" 自洽蒙混 | 缺「先算余蛋再乘单价」这种**两步推理**示范 |
| B target_scope | "150% 增长当 1.5×" 概念错 | 缺「百分数增长 = × (1+p)」这种**概念映射**示范 |
| C attribute_mismatch | "数量当重量" 单位混淆 | 缺「只问数量就别算重量」这种**单位区分**示范 |
| D independent_decoy | "+3-3 自抵消" 数字拼凑 | 缺「数字拼凑会引入错算」这种**负向示范** |

### 1.2 为什么 few-shot 能修

- **8-shot baseline 在 0.5B 上 original 43.29%**（v9 报告）——比 0-shot 的 v11 33.21% **高 10 pp**
- 8-shot 的核心红利是"看真实推理模式"而非"看格式"
- 但全 8-shot 会破坏 SFT/GRPO 已学到的 STEP 协议，引入 trace 风格不一致
- **折中方案**：在 0-shot 协议里加 **1-2 个** 完整示范（保留 STEP 协议 + 加少量示范）

### 1.3 与 SFT 数据的兼容性

- SFT 起点数据 `chaingsm_data/data/final/sft/all_sft_cot.jsonl` 的 assistant 段使用 `<<expr = val>>` + `<<FINAL: ...>>` + `ANSWER: N` 格式
- 下面设计的 2 个示范**完全沿用这个协议**，与 SFT 数据 100% 兼容
- 训练时如果用这个新 prompt，需要重新生成 SFT + GRPO 数据（**因为 prompt 是 user 模板的修改，不是 system 的修改**）

---

## 二、推荐版本：v12_prompt_fewshot_2（2 个示范）

在原 0-shot User 模板的"Format"之后、"Rules"之前插入 2 个真实 gold 案例。

### 2.1 System Prompt（与 v11 完全相同）

```
You are a careful grade-school math reasoning assistant. Solve the problem
using natural reasoning. Put every arithmetic derivation that is used for the
final answer inside double angle brackets in the exact form <<expression = value>>.
Put the final derivation inside <<FINAL: expression = answer>>. Do not put prose
inside the double angle brackets. Do not put ignored, hypothetical, optional,
separate-scope, or distractor calculations inside the double angle brackets;
mention them only in prose if needed.
```

### 2.2 User Prompt 模板（**修改版，标记 `[NEW]`**）

```text
Solve the following grade-school math problem.

Use natural language reasoning, but put each arithmetic derivation that
contributes to the final answer inside double angle brackets.

Format:
TARGET: ...

[Brief natural-language reasoning.]
<<expression = value>>

[Brief natural-language reasoning.]
<<expression = value>>

Add more derivations as needed.

<<FINAL: final_expression = answer>>
ANSWER: answer

[NEW] Examples:

Example 1 (standard problem):
Problem: Janet's ducks lay 16 eggs per day. She eats three for breakfast
every morning and bakes muffins for her friends every day with four. She
sells the remainder at the farmers' market daily for $2 per fresh duck
egg. How much in dollars does she make every day at the farmers' market?

Solution:
TARGET: dollars made at the farmers' market per day

First, subtract the breakfast and baking eggs from the daily total to find
how many eggs are left to sell.
<<16 - 3 - 4 = 9>>

Next, multiply the leftover eggs by the price per egg to get the daily
earnings.
<<9 * 2 = 18>>

<<FINAL: (16 - 3 - 4) * 2 = 18>>
ANSWER: 18

Example 2 (problem with distractor facts to ignore):
Problem: A robe takes 2 bolts of blue fiber and half that much white
fiber. Each bolt of blue fiber weighs 3 pounds and each bolt of white
fiber weighs 2 pounds. How many bolts in total does it take?

Solution:
TARGET: total number of bolts

The question asks for the number of bolts, so we use the quantities (2 blue
bolts, "half that much" = 1 white bolt), not the per-bolt weights.
<<2 / 2 = 1>>

Then add the two kinds of bolts.
<<2 + 1 = 3>>

Note: The per-bolt weights (3 lb blue, 2 lb white) are not needed because
the question asks for count, not weight.

<<FINAL: 2 + 2 / 2 = 3>>
ANSWER: 3

Rules:
- Only calculations used to answer the actual question should appear inside <<...>>.
- Do not put unused, hypothetical, optional, separate-scope, or distractor
  calculations inside <<...>>.
- If a fact is not used, you may explain in prose why it is excluded, but do not
  calculate it inside <<...>>.
- Keep all prose outside <<...>>.
- The final line must contain ANSWER: answer.

Problem:
{question}
```

---

## 三、轻量版本：v12_prompt_fewshot_1（1 个示范）

只插入 1 个 original 示范（**不覆盖 decoy**），适合作为**消融实验**的对照组。

### User Prompt 模板（只插入 Example 1）

把上一节 User 模板中的 `[NEW] Examples:` 段替换为：

```text
[NEW] Example:

Example 1 (standard problem):
Problem: Janet's ducks lay 16 eggs per day. She eats three for breakfast
every morning and bakes muffins for her friends every day with four. She
sells the remainder at the farmers' market daily for $2 per fresh duck
egg. How much in dollars does she make every day at the farmers' market?

Solution:
TARGET: dollars made at the farmers' market per day

First, subtract the breakfast and baking eggs from the daily total to find
how many eggs are left to sell.
<<16 - 3 - 4 = 9>>

Next, multiply the leftover eggs by the price per egg to get the daily
earnings.
<<9 * 2 = 18>>

<<FINAL: (16 - 3 - 4) * 2 = 18>>
ANSWER: 18
```

---

## 四、3 个示范的版本：v12_prompt_fewshot_3（推荐消融用）

如果你想加 1 个 decoy 示范但只针对一种类型（如 path_competition），可以加 1 个 path_competition 示范。在 v12_prompt_fewshot_2 的基础上追加 Example 3。

### Example 3（path_competition：单价/数量方向）

```text
Example 3 (problem with a "wrong path" distractor):
Problem: Janet's ducks lay 16 eggs per day. She eats three for breakfast
every morning and bakes muffins for her friends every day with four. She
could sell all 16 eggs for $32, but instead she sells the remainder at
the farmers' market daily for $2 per fresh duck egg. How much in dollars
does she make every day at the farmers' market?

Solution:
TARGET: dollars made at the farmers' market per day

First, find the leftover eggs after breakfast and baking.
<<16 - 3 - 4 = 9>>

Then multiply the leftover eggs by the per-egg price.
<<9 * 2 = 18>>

Note: The "$32 for all 16 eggs" path is a distractor — the question asks
for the actual per-day earnings at the market, not the alternative
all-eggs selling scenario.

<<FINAL: (16 - 3 - 4) * 2 = 18>>
ANSWER: 18
```

---

## 五、4 个示范的版本：v12_prompt_fewshot_4

如果你想覆盖**全部 4 类 decoy**，在 v12_prompt_fewshot_2 基础上追加 Example 3（path_competition）、Example 4（target_scope_misalignment）。

### Example 4（target_scope_misalignment：百分数概念）

```text
Example 4 (problem with a "percentage growth" concept):
Problem: Josh decides to try flipping a house. He buys a house for $80,000
and then puts in $50,000 in repairs. This increased the value of the house
by 150%. After selling the house, he donates 10% of his profit to charity.
How much profit did he make from the house flip?

Solution:
TARGET: profit

First, find the new value of the house after the 150% increase on the
purchase price. A 150% increase means the new value is the original plus
150% of it, i.e., 1 + 1.5 = 2.5 times the original.
<<80000 * 2.5 = 200000>>

Then compute the profit by subtracting total cost (purchase + repair) from
the sale value.
<<200000 - 80000 - 50000 = 70000>>

Note: The 10% charity donation happens after the sale and is NOT subtracted
from profit.

<<FINAL: 80000 * 2.5 - 80000 - 50000 = 70000>>
ANSWER: 70000
```

---

## 六、代码使用方式

### 6.1 直接修改 `eval_vllm_chaingsm.py`

在 `eval_vllm_chaingsm.py` 的 prompt 常量定义区（大约第 100-145 行）追加 few-shot 模板：

```python
# v12 few-shot 变体 (2026-06-18)
COT_BRACKETS_USER_TEMPLATE_FEWSHOT_2 = (
    "Solve the following grade-school math problem.\n\n"
    "Use natural language reasoning, but put each arithmetic derivation that "
    "contributes to the final answer inside double angle brackets.\n\n"
    "Format:\nTARGET: ...\n\n"
    "[Brief natural-language reasoning.]\n<<expression = value>>\n\n"
    "[Brief natural-language reasoning.]\n<<expression = value>>\n\n"
    "Add more derivations as needed.\n\n"
    "<<FINAL: final_expression = answer>>\nANSWER: answer\n\n"

    "Examples:\n\n"
    "Example 1 (standard problem):\n"
    "Problem: Janet's ducks lay 16 eggs per day. She eats three for breakfast "
    "every morning and bakes muffins for her friends every day with four. She "
    "sells the remainder at the farmers' market daily for $2 per fresh duck "
    "egg. How much in dollars does she make every day at the farmers' market?\n\n"
    "Solution:\n"
    "TARGET: dollars made at the farmers' market per day\n\n"
    "First, subtract the breakfast and baking eggs from the daily total to find "
    "how many eggs are left to sell.\n"
    "<<16 - 3 - 4 = 9>>\n\n"
    "Next, multiply the leftover eggs by the price per egg to get the daily "
    "earnings.\n"
    "<<9 * 2 = 18>>\n\n"
    "<<FINAL: (16 - 3 - 4) * 2 = 18>>\n"
    "ANSWER: 18\n\n"

    "Example 2 (problem with distractor facts to ignore):\n"
    "Problem: A robe takes 2 bolts of blue fiber and half that much white "
    "fiber. Each bolt of blue fiber weighs 3 pounds and each bolt of white "
    "fiber weighs 2 pounds. How many bolts in total does it take?\n\n"
    "Solution:\n"
    "TARGET: total number of bolts\n\n"
    "The question asks for the number of bolts, so we use the quantities (2 blue "
    "bolts, \"half that much\" = 1 white bolt), not the per-bolt weights.\n"
    "<<2 / 2 = 1>>\n\n"
    "Then add the two kinds of bolts.\n"
    "<<2 + 1 = 3>>\n\n"
    "Note: The per-bolt weights (3 lb blue, 2 lb white) are not needed because "
    "the question asks for count, not weight.\n\n"
    "<<FINAL: 2 + 2 / 2 = 3>>\n"
    "ANSWER: 3\n\n"

    "Rules:\n"
    "- Only calculations used to answer the actual question should appear inside <<...>>.\n"
    "- Do not put unused, hypothetical, optional, separate-scope, or distractor "
    "calculations inside <<...>>.\n"
    "- If a fact is not used, you may explain in prose why it is excluded, but do not "
    "calculate it inside <<...>>.\n"
    "- Keep all prose outside <<...>>.\n"
    "- The final line must contain ANSWER: answer.\n\n"
    "Problem:\n{question}\n"
)
```

然后在 `build_messages()` 末尾加方法选择：

```python
if method == "cot_brackets":
    return [
        {"role": "system", "content": COT_BRACKETS_SYSTEM_PROMPT},
        {"role": "user", "content": COT_BRACKETS_USER_TEMPLATE.format(question=question.strip())},
    ]
if method == "cot_brackets_fewshot_2":
    return [
        {"role": "system", "content": COT_BRACKETS_SYSTEM_PROMPT},
        {"role": "user", "content": COT_BRACKETS_USER_TEMPLATE_FEWSHOT_2.format(question=question.strip())},
    ]
```

并把 `--method` 的 choices 加上 `"cot_brackets_fewshot_2"`。

### 6.2 训练侧修改

**关键**：训练和评测 prompt 必须**完全一致**。如果用 few-shot 训练，需要：

1. **重新生成 SFT 数据**：
   ```bash
   # 用 few-shot prompt 重跑 SFT 数据准备
   python -m chaingsm_data.preprocess_sft_fewshot \
     --template cot_brackets_fewshot_2 \
     --output chaingsm_data/data/final/sft/all_sft_cot_fewshot2.jsonl
   ```

2. **重新生成 GRPO 数据**（同源 SFT）：
   ```bash
   python -m chaingsm_data.preprocess_grpo_fewshot \
     --sft-source all_sft_cot_fewshot2.jsonl \
     --output chaingsm_data/data/final/grpo/all_grpo_cot_fewshot2.parquet
   ```

3. **训练**：直接用新数据替换 v11 yaml 里的路径：
   ```yaml
   data:
     train_files: /home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/grpo/all_grpo_cot_fewshot2.parquet
   ```

### 6.3 评测侧（已有 ckpt 也可测）

**如果只想快速验证 few-shot 评测效果**（不需要重训），可以直接用 few-shot prompt 跑现有 v11 ckpt：

```bash
python -m train_pipeline.eval_vllm_chaingsm \
  --model-path outputs/.../global_step_1000/actor/huggingface \
  --data-path chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl \
  --output-dir outputs/v11_eval_fewshot2/step_1000 \
  --method cot_brackets_fewshot_2 \
  --batch-size 64 --gpu-memory-utilization 0.5 \
  --max-tokens 512 --max-model-len 2048 --seed 42
```

**注意**：用 few-shot prompt 测 0-shot 训练的 ckpt 可能**反而掉点**——因为模型在训练时没看过 few-shot 格式。**要稳的话，重训**。

---

## 七、预期影响 & 风险

### 7.1 预期提升

| 类别 | 0-shot 现状 | few-shot 预期 | 提升来源 |
|---|---|---|---|
| overall | 29.98% (v11 step_1000) | **32-35%** | 真实推理模式示范 |
| original | 33.21% | **35-37%** | 算式-语义对应示范 |
| path_competition | 28.03% | **30-33%** | Example 3/4 直击 |
| attribute_mismatch | 27.73% | **30-33%** | Example 2 直击 |
| target_scope | 28.83% | **31-34%** | Example 4 直击 |
| independent_decoy | 31.03% | **32-34%** | 间接提升 |

### 7.2 风险

| 风险 | 缓解 |
|---|---|
| **trace 风格不一致**（v11 训的 step 与 few-shot 示范的 step 风格不同） | 示范的 trace 严格沿用 v11 协议（`<<...>>` + `<<FINAL: ...>>` + `ANSWER: N`） |
| **位置偏差**（模型只学 Example 1 的范式） | 用 2-4 个不同 category 的示范分散 |
| **prompt 长度增加** | 4 个示范约 +800 tokens，max_prompt_length=768 需调到 1536 |
| **重训成本** | 1 epoch SFT (~2h) + 1000 步 GRPO (~3h) = **5h** |

### 7.3 建议路径

1. **第一步（0.5h）**：用 fewshot_2 prompt 跑 **现有 v11 step_1000 ckpt 评测**——验证 few-shot prompt 的边际效果，零训练成本
2. **第二步（5h）**：如果步骤 1 验证 few-shot 有效（预期 +1-3 pp），重训 SFT 1 epoch + GRPO 500 步
3. **第三步（消融）**：fewshot_1 / fewshot_2 / fewshot_3 / fewshot_4 各跑一次，定位最佳示范数

---

## 八、附录：完整 prompt 文本（v12_prompt_fewshot_2）

可以直接复制粘贴到 `eval_vllm_chaingsm.py` 使用。

### System Prompt

```
You are a careful grade-school math reasoning assistant. Solve the problem
using natural reasoning. Put every arithmetic derivation that is used for the
final answer inside double angle brackets in the exact form <<expression = value>>.
Put the final derivation inside <<FINAL: expression = answer>>. Do not put prose
inside the double angle brackets. Do not put ignored, hypothetical, optional,
separate-scope, or distractor calculations inside the double angle brackets;
mention them only in prose if needed.
```

### User Prompt Template

```text
Solve the following grade-school math problem.

Use natural language reasoning, but put each arithmetic derivation that
contributes to the final answer inside double angle brackets.

Format:
TARGET: ...

[Brief natural-language reasoning.]
<<expression = value>>

[Brief natural-language reasoning.]
<<expression = value>>

Add more derivations as needed.

<<FINAL: final_expression = answer>>
ANSWER: answer

Examples:

Example 1 (standard problem):
Problem: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

Solution:
TARGET: dollars made at the farmers' market per day

First, subtract the breakfast and baking eggs from the daily total to find how many eggs are left to sell.
<<16 - 3 - 4 = 9>>

Next, multiply the leftover eggs by the price per egg to get the daily earnings.
<<9 * 2 = 18>>

<<FINAL: (16 - 3 - 4) * 2 = 18>>
ANSWER: 18

Example 2 (problem with distractor facts to ignore):
Problem: A robe takes 2 bolts of blue fiber and half that much white fiber. Each bolt of blue fiber weighs 3 pounds and each bolt of white fiber weighs 2 pounds. How many bolts in total does it take?

Solution:
TARGET: total number of bolts

The question asks for the number of bolts, so we use the quantities (2 blue bolts, "half that much" = 1 white bolt), not the per-bolt weights.
<<2 / 2 = 1>>

Then add the two kinds of bolts.
<<2 + 1 = 3>>

Note: The per-bolt weights (3 lb blue, 2 lb white) are not needed because the question asks for count, not weight.

<<FINAL: 2 + 2 / 2 = 3>>
ANSWER: 3

Rules:
- Only calculations used to answer the actual question should appear inside <<...>>.
- Do not put unused, hypothetical, optional, separate-scope, or distractor calculations inside <<...>>.
- If a fact is not used, you may explain in prose why it is excluded, but do not calculate it inside <<...>>.
- Keep all prose outside <<...>>.
- The final line must contain ANSWER: answer.

Problem:
{question}
```
