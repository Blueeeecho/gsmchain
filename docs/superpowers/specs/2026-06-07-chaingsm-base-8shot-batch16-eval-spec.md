# ChainGSM 8-shot CoT 主线评测 spec(对齐 5,467 条干净集 + batch=16)

> **生成日期**：2026-06-07
> **作用**：定义"什么是 ChainGSM 主线评测"——不写历史、不写 batch=64 对照、只写当前规格。后续 SFT/DPO/GRPO checkpoint 全部沿用本 spec。
> **入口脚本**：`/home/wwq416/snap/wwq/math-chain/code/eval_chaingsm_base_8shot.py`

---

## 1. 目标

- 在 5,467 条干净测试集上,用 8-shot CoT 评估 6 个未微调 Instruct 模型。
- 同时给出 **overall** 与 **5 个 ChainGSM 类目**准确率。
- 数值答案提取与判等全仓库共用,口径与 AbstRaL 公开报告一致。
- 同脚本可被 SFT/DPO/GRPO checkpoint 沿用,产出"before/after"对比。

---

## 2. 测试集

### 2.1 文件

```text
/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl
```

### 2.2 规模

| 类别 | 数量 |
|---|---:|
| original | 1319 |
| independent_decoy | 1102 |
| attribute_mismatch | 1017 |
| path_competition | 999 |
| target_scope_misalignment | 1030 |
| **合计** | **5467** |

### 2.3 字段

每条记录包含:

```text
id, base_id, source_index, category,
question_original, question_distracted, answer,
solution_original, core_chain, distractor_chain,
gold_expression, distractor_expression,
difficulty_tags, metadata
```

### 2.4 题目选择规则

| category | 使用字段 |
|---|---|
| `original` | `question_original` |
| 其他 4 类 | `question_distracted` |

Gold 答案统一从 `answer` 字段取(整型字符串)。

---

## 3. 模型清单

| 模型 | 路径 | Profile |
|---|---|---|
| Qwen2.5-0.5B-Instruct | `/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct` | `qwen_multiturn_8shot_chat` |
| Qwen2.5-1.5B-Instruct | `/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-1.5B-Instruct` | `qwen_multiturn_8shot_chat` |
| Qwen2.5-Math-1.5B-Instruct | `/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-Math-1.5B-Instruct` | `qwen_math_completion_8shot` |
| Qwen2.5-3B-Instruct | `/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-3B-Instruct` | `qwen_multiturn_8shot_chat` |
| Llama-3.2-1B-Instruct | `/home/wwq416/snap/wwq/model/llama/Llama-3.2-1B-Instruct` | `llama_lm_eval_multiturn_8shot_single_bos` |
| Llama-3.2-3B-Instruct | `/home/wwq416/snap/wwq/model/llama/Llama-3.2-3B-Instruct` | `llama_lm_eval_multiturn_8shot_single_bos` |

> 后续 SFT/DPO/GRPO checkpoint 也按"模型名 = 目录名"路由,本表加一行即可。

---

## 4. 提示词规范

所有模型共享同一组 **8 条 CoT 示例**(同源于 AbstRaL 报告使用的 GSM8K 8-shot 集合,见 `EIGHT_SHOT_EXAMPLES` / `LM_EVAL_COT_EXAMPLES`),回答均以 `The final answer is N.` 收尾。

> 代码源:
> - [code/eval_chaingsm_base_8shot.py](file:///home/wwq416/snap/wwq/math-chain/code/eval_chaingsm_base_8shot.py) — Qwen Multiturn + Qwen-Math Completion + Llama 路由
> - [code/eval_official_gsm.py](file:///home/wwq416/snap/wwq/math-chain/code/eval_official_gsm.py) — Llama lm-eval 兼容 messages(`build_lm_eval_llama_messages`)

### 4.1 普通 Qwen 多轮 8-shot

**适用**:`qwen_multiturn_8shot_chat`,用于 Qwen2.5-0.5B / 1.5B / 3B-Instruct。

**构造步骤**(`build_qwen_messages`,再 `tokenizer.apply_chat_template(tokenize=False, add_generation_prompt=True)`):

```text
[1]  system:    "As an expert problem solver, solve step by step
                 the following mathematical questions."

[2]  user:      "Q: There are 15 trees in the grove. Grove workers will
                 plant trees in the grove today. After they are done,
                 there will be 21 trees. How many trees did the grove
                 workers plant today?\nA:"
[3]  assistant: "Let's think step by step. There are 15 trees originally.
                 Then there were 21 trees after some more were planted.
                 So there must have been 21 - 15 = 6.
                 The final answer is 6."

[4]  user:      "Q: If there are 3 cars in the parking lot and 2 more
                 cars arrive, how many cars are in the parking lot?\nA:"
[5]  assistant: "Let's think step by step. There are originally 3 cars.
                 2 more cars arrive. 3 + 2 = 5. The final answer is 5."

[6]  user:      "Q: Leah had 32 chocolates and her sister had 42. If
                 they ate 35, how many pieces do they have left in
                 total?\nA:"
[7]  assistant: "Let's think step by step. Originally, Leah had 32
                 chocolates. Her sister had 42. So in total they had
                 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.
                 The final answer is 39."

[8]  user:      "Q: Jason had 20 lollipops. He gave Denny some
                 lollipops. Now Jason has 12 lollipops. How many
                 lollipops did Jason give to Denny?\nA:"
[9]  assistant: "Let's think step by step. Jason started with 20
                 lollipops. Then he had 12 after giving some to Denny.
                 So he gave Denny 20 - 12 = 8. The final answer is 8."

[10] user:      "Q: Shawn has five toys. For Christmas, he got two
                 toys each from his mom and dad. How many toys does
                 he have now?\nA:"
[11] assistant: "Let's think step by step. Shawn started with 5 toys.
                 If he got 2 toys each from his mom and dad, then
                 that is 4 more toys. 5 + 4 = 9. The final answer is 9."

[12] user:      "Q: There were nine computers in the server room.
                 Five more computers were installed each day, from
                 monday to thursday. How many computers are now in
                 the server room?\nA:"
[13] assistant: "Let's think step by step. There were originally 9
                 computers. For each of 4 days, 5 more computers were
                 added. So 5 * 4 = 20 computers were added.
                 9 + 20 = 29. The final answer is 29."

[14] user:      "Q: Michael had 58 golf balls. On tuesday, he lost 23
                 golf balls. On wednesday, he lost 2 more. How many
                 golf balls did he have at the end of wednesday?\nA:"
[15] assistant: "Let's think step by step. Michael started with 58
                 golf balls. After losing 23 on tuesday, he had
                 58 - 23 = 35. After losing 2 more, he had
                 35 - 2 = 33 golf balls. The final answer is 33."

[16] user:      "Q: Olivia has $23. She bought five bagels for $3
                 each. How much money does she have left?\nA:"
[17] assistant: "Let's think step by step. Olivia had 23 dollars.
                 5 bagels for 3 dollars each will be 5 x 3 = 15
                 dollars. So she has 23 - 15 dollars left.
                 23 - 15 is 8. The final answer is 8."

[18] user:      "Q: {当前题}\nA: Let's think step by step."
```

vLLM 接渲染好的**字符串**(不是 token IDs)。Qwen2.5 的 chat template 会把每条消息包装成 `<|im_start|>{role}\n{content}<|im_end|>\n`,最终生成 prompt 形如:

```text
<|im_start|>system
As an expert problem solver, solve step by step the following mathematical questions.<|im_end|>
<|im_start|>user
Q: There are 15 trees in the grove. ... How many trees did the grove workers plant today?
A:<|im_end|>
<|im_start|>assistant
Let's think step by step. ... The final answer is 6.<|im_end|>
... (8 对 user/assistant) ...
<|im_start|>user
Q: {当前题}
A: Let's think step by step.<|im_end|>
<|im_start|>assistant
                                                                                 ← 模型从这里续
```

**Stop sequences**: `["<|im_end|>"]`(模型一产出 `<|im_end|>` 就停)。

> 字符级实测(用 Qwen2.5-0.5B-Instruct tokenizer 渲染 `SAMPLE_QUESTION_PLACEHOLDER`):每个 `Q: ...\nA:` 后接 `<|im_end|>`,assistant 答案以 `<|im_end|>` 收尾;最后一对 user 后,模型在 `<|im_start|>assistant\n` 之后续出。

---

### 4.2 Qwen-Math 纯 Completion 8-shot

**适用**:`qwen_math_completion_8shot`,**仅**用于 Qwen2.5-Math-1.5B-Instruct。

**构造步骤**(`build_qwen_math_completion_prompt`,**不调用** `apply_chat_template`):

```text
Q: There are 15 trees in the grove. Grove workers will plant trees in the
grove today. After they are done, there will be 21 trees. How many trees
did the grove workers plant today?
A: Let's think step by step. There are 15 trees originally. Then there
were 21 trees after some more were planted. So there must have been
21 - 15 = 6. The final answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how
many cars are in the parking lot?
A: Let's think step by step. There are originally 3 cars. 2 more cars
arrive. 3 + 2 = 5. The final answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how
many pieces do they have left in total?
A: Let's think step by step. Originally, Leah had 32 chocolates. Her
sister had 42. So in total they had 32 + 42 = 74. After eating 35, they
had 74 - 35 = 39. The final answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has
12 lollipops. How many lollipops did Jason give to Denny?
A: Let's think step by step. Jason started with 20 lollipops. Then he
had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.
The final answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his
mom and dad. How many toys does he have now?
A: Let's think step by step. Shawn started with 5 toys. If he got 2
toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.
The final answer is 9.

Q: There were nine computers in the server room. Five more computers
were installed each day, from monday to thursday. How many computers
are now in the server room?
A: Let's think step by step. There were originally 9 computers. For
each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers
were added. 9 + 20 = 29. The final answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On
wednesday, he lost 2 more. How many golf balls did he have at the end
of wednesday?
A: Let's think step by step. Michael started with 58 golf balls.
After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more,
he had 35 - 2 = 33 golf balls. The final answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money
does she have left?
A: Let's think step by step. Olivia had 23 dollars. 5 bagels for 3
dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars
left. 23 - 15 is 8. The final answer is 8.

Q: {当前题}
A: Let's think step by step.
```

每段之间用**两个换行**(`"\n\n"`)分隔。整段就是一个**纯字符串**,不注入 system 消息,不调用 Qwen chat template,也不附加 `<|im_end|>`。

**Stop sequences**: `["\nQ:", "\nQuestion:", "<|im_end|>"]`(模型若开始续下一题就停)。

---

### 4.3 Llama 单 BOS 多轮 8-shot

**适用**:`llama_lm_eval_multiturn_8shot_single_bos`,用于 Llama-3.2-1B-Instruct 与 Llama-3.2-3B-Instruct。

**构造步骤**(`build_lm_eval_llama_messages`,`eval_official_gsm.py`):

每对 user 的内容都套用 `LM_EVAL_LLAMA_DOC_TEMPLATE`:

```text
Given the following problem, reason and give a final answer to the problem.
Problem: {question}
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.
```

每条 assistant 的内容是 `LM_EVAL_COT_EXAMPLES` 中的样例,**但把文末的 `The answer is N.` 正则替换成 `The final answer is N.`**(`re.sub(r"\s*The answer is [-+]?\d+(?:\.\d+)?\.\s*$", f" The final answer is {final_answer}", shot_answer)`)。

完整 messages 结构(17 条):

```text
[1]  user:      "Given the following problem, reason and give a final answer to the problem.
                 Problem: There are 15 trees in the grove. ... How many trees did the grove
                 workers plant today?
                 Your response should end with \"The final answer is [answer]\" where
                 [answer] is the response to the problem."
[2]  assistant: "There are 15 trees originally. Then there were 21 trees after some
                 more were planted. So there must have been 21 - 15 = 6.
                 The final answer is 6."

[3]  user:      "Given the following problem, reason and give a final answer to the problem.
                 Problem: If there are 3 cars in the parking lot and 2 more cars arrive,
                 how many cars are in the parking lot?
                 Your response should end with \"The final answer is [answer]\" where
                 [answer] is the response to the problem."
[4]  assistant: "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.
                 The final answer is 5."

[5]  user:      "Given the following problem, reason and give a final answer to the problem.
                 Problem: Leah had 32 chocolates and her sister had 42. If they ate 35,
                 how many pieces do they have left in total?
                 Your response should end with \"The final answer is [answer]\" where
                 [answer] is the response to the problem."
[6]  assistant: "Originally, Leah had 32 chocolates. Her sister had 42. So in total
                 they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.
                 The final answer is 39."

[7]  user:      "Given the following problem, reason and give a final answer to the problem.
                 Problem: Jason had 20 lollipops. He gave Denny some lollipops. Now
                 Jason has 12 lollipops. How many lollipops did Jason give to Denny?
                 Your response should end with \"The final answer is [answer]\" where
                 [answer] is the response to the problem."
[8]  assistant: "Jason started with 20 lollipops. Then he had 12 after giving some
                 to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8."

[9]  user:      "Given the following problem, reason and give a final answer to the problem.
                 Problem: Shawn has five toys. For Christmas, he got two toys each
                 from his mom and dad. How many toys does he have now?
                 Your response should end with \"The final answer is [answer]\" where
                 [answer] is the response to the problem."
[10] assistant: "Shawn started with 5 toys. If he got 2 toys each from his mom and
                 dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9."

[11] user:      "Given the following problem, reason and give a final answer to the problem.
                 Problem: There were nine computers in the server room. Five more
                 computers were installed each day, from monday to thursday. How many
                 computers are now in the server room?
                 Your response should end with \"The final answer is [answer]\" where
                 [answer] is the response to the problem."
[12] assistant: "There were originally 9 computers. For each of 4 days, 5 more computers
                 were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.
                 The final answer is 29."

[13] user:      "Given the following problem, reason and give a final answer to the problem.
                 Problem: Michael had 58 golf balls. On tuesday, he lost 23 golf balls.
                 On wednesday, he lost 2 more. How many golf balls did he have at
                 the end of wednesday?
                 Your response should end with \"The final answer is [answer]\" where
                 [answer] is the response to the problem."
[14] assistant: "Michael started with 58 golf balls. After losing 23 on tuesday, he
                 had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.
                 The final answer is 33."

[15] user:      "Given the following problem, reason and give a final answer to the problem.
                 Problem: Olivia has $23. She bought five bagels for $3 each. How
                 much money does she have left?
                 Your response should end with \"The final answer is [answer]\" where
                 [answer] is the response to the problem."
[16] assistant: "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15
                 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.
                 The final answer is 8."

[17] user:      "Given the following problem, reason and give a final answer to the problem.
                 Problem: {当前题}
                 Your response should end with \"The final answer is [answer]\" where
                 [answer] is the response to the problem."
```

**关键:不传字符串,直接传 token IDs**(避免 vLLM 0.21.0 重复添加 BOS):

```python
tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
# → BatchEncoding(input_ids=[128000, 128006, 9125, 128007, ...], ...)
#   强校验:bos_token_id (128000) 必须出现恰好 1 次;否则 ValueError
```

Llama-3.2-Instruct 的 chat template 在每条消息前后自动注入特殊 token(实测前 12 个 token):

```text
[ 0] id=128000  text='<|begin_of_text|>'                       ← 单 BOS
[ 1] id=128006  text='<|start_header_id|>'
[ 2] id=9125    text='system'
[ 3] id=128007  text='<|end_header_id|>'
[ 4] id=271     text='\n\n'
[ 5] id=38766   text='Cut'  ← Llama chat template 默认注入的 system:
[ 6] id=1303    text='ting'     "Cutting Knowledge Date: December 2023\n
[ 7] id=33025   text=' Knowledge'    Today Date: 07 Jun 2026"
[ 8] id=2696    text=' Date'
[ 9] id=25      text=':'
[10] id=6790    text=' December'
[11] id=220     text=' '
```

每对 user/assistant 用 `<|eot_id|>`(id=128009)分隔,assistant 前是 `<|start_header_id|>assistant<|end_header_id|>\n\n`。最后一条 user 之后,token 化的最后 8 个 token 是:

```text
[1104] id=279     text=' the'
[1105] id=3575    text=' problem'
[1106] id=13      text='.'
[1107] id=128009  text='<|eot_id|>'
[1108] id=128006  text='<|start_header_id|>'
[1109] id=78191   text='assistant'
[1110] id=128007  text='<|end_header_id|>'
[1111] id=271     text='\n\n'                                   ← 模型从这里续
```

总长 **1,112 tokens**(含 1 个 BOS、18 个 EOT、17 个 user 头 + 8 个 assistant 头)。

**Stop sequences**: `["<|eot_id|>", "<|start_header_id|>user<|end_header_id|>", "Q:", "</s>", "<|im_end|>"]`(模型产出 EOT 或下一条 user 头就停)。

---

### 4.4 停止条件(总表)

| Profile | stop sequences |
|---|---|
| `qwen_multiturn_8shot_chat` | `["<\|im_end\|>"]` |
| `qwen_math_completion_8shot` | `["\nQ:", "\nQuestion:", "<\|im_end\|>"]` |
| `llama_lm_eval_multiturn_8shot_single_bos` | `["<\|eot_id\|>", "<\|start_header_id\|>user<\|end_header_id\|>", "Q:", "</s>", "<\|im_end\|>"]` |

---

### 4.5 三种 profile 的核心差异(对照表)

| 维度 | Qwen Multiturn | Qwen-Math Completion | Llama Multiturn |
|---|---|---|---|
| 适用模型 | 0.5B / 1.5B / 3B | Math-1.5B | 3.2-1B / 3.2-3B |
| 走 chat template? | ✅ | ❌ | ✅ |
| chat template 渲染 | `<\|im_start\|>{role}\n...<\|im_end\|>` | — | `<\|begin_of_text\|><\|start_header_id\|>{role}<\|end_header_id\|>\n\n...<\|eot_id\|>` |
| system 消息 | 注入数学专家提示 | 不注入 | 注入 chat template 自带 "Cutting Knowledge Date..." |
| user 格式 | `Q: ...\nA:` | `Q: ...\nA:` | `Given the following problem, reason...\nProblem: ...\nYour response should end with "The final answer is [answer]"...` |
| assistant 格式 | `Let's think step by step. ... The final answer is N.` | `Let's think step by step. ... The final answer is N.` | `... The final answer is N`(无 "Let's think step by step." 开头) |
| 当前 user 触发 | `Q: {q}\nA: Let's think step by step.` | `Q: {q}\nA: Let's think step by step.` | `Given the following problem, ...\nProblem: {q}\n...` |
| vLLM 接收 | 字符串 | 字符串 | **token IDs** |
| BOS 校验 | 无 | 无 | **必须恰好 1 个 (128000)**,否则 `ValueError` |
| stop | `<\|im_end\|>` | `\nQ:` / `\nQuestion:` / `<\|im_end\|>` | `<\|eot_id\|>` / `user` 头 / `Q:` / `</s>` / `<\|im_end\|>` |

 生成设置(全固定)

```text
temperature        = 0.0
top_p              = 1.0
max_tokens         = 512
seed               = 42
tensor_parallel_size = 1
gpu_memory_utilization = 0.4(候选 [0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25] 自动回退)
dtype              = auto
max_model_len      = 4096
enforce_eager      = False
batch_size         = 16
trust_remote_code  = True
```

---

## 5. 生成设置(全固定)

```text
temperature        = 0.0
top_p              = 1.0
max_tokens         = 512
seed               = 42
tensor_parallel_size = 1
gpu_memory_utilization = 0.4(候选 [0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25] 自动回退)
dtype              = auto
max_model_len      = 4096
enforce_eager      = False
batch_size         = 16
trust_remote_code  = True
```

---

## 6. 数值答案提取与判等

- 实现:`/home/wwq416/snap/wwq/math-chain/code/gsm_answer_extractor.py`
- `extract_answer(output)`:七级确定性回退(顺序固定,同输入同输出)。
- `is_correct(pred, gold)`:`fractions.Fraction` 严格化后比较,`tolerance=1e-6`。
- 所有评测入口必须 `from gsm_answer_extractor import extract_answer, is_correct`,不允许本地重复实现。

---

## 7. 评测流程

每个模型独立子进程(vLLM `LLM` 实例互不共享),由 `evaluate_model` 完成:

```text
1. 读 prompt_diagnostics.json(已有则复用)
2. 取 profile → 构造 SamplingParams + llm_kwargs
3. chunked(examples, batch_size=16) 分批
4. 每批 build_model_input × 16 → llm.generate → extract_answer → is_correct
5. 追加写 predictions.jsonl(可续跑)
6. 写 summary.json(overall + 5 类目)
```

主流程结束会合并所有模型到 `summary_overall.{json,csv}` 与 `summary_by_category.{json,csv}`。

---

## 8. 输出目录与文件

```text
code/results/chaingsm_base_8shot_batch16/<YYYYMMDD_HHMMSS>/
├── run_config.json
├── model_outputs/
│   ├── <safe_model_name>/
│   │   ├── predictions.jsonl
│   │   ├── prompt_diagnostics.json
│   │   └── summary.json
├── summary_overall.json / .csv
└── summary_by_category.json / .csv
```

`run_config.json` 必须包含:

```jsonc
{
  "data_path": ".../gsm8k_test_clean.jsonl",
  "example_count": 5467,
  "models": [
    {
      "model_name": "Llama-3.2-3B-Instruct",
      "model_path": "/home/wwq416/snap/wwq/model/llama/Llama-3.2-3B-Instruct",
      "prompt_profile": "llama_lm_eval_multiturn_8shot_single_bos"
    }
  ],
  "generation": { "temperature": 0.0, "top_p": 1.0, "max_tokens": 512, "seed": 42 },
  "runtime":   { "batch_size": 16, "tensor_parallel_size": 1, "gpu_memory_utilization": 0.4, "max_model_len": 4096, "dtype": "auto" },
  "resume_command": "..."
}
```

每条 prediction 记录:

```text
{id, base_id, category, question, gold_answer, model_name, model_path,
 prompt_profile, raw_output, pred_answer, correct, finish_reason, stop_reason}
```

`prompt_diagnostics.json` 必含:

```text
model_name, model_path, prompt_profile, input_type, first_prompt_tokens,
bos_token_id, bos_count, stop_sequences
```

`bos_count` 必为 1(对 Llama profile 强校验);`input_type` 必为 `token_ids`(Llama) 或 `text`(Qwen)。

---

## 9. 复现命令

```bash
conda activate math_chain_verl

# 全量(自动开新时间戳子目录,默认 4 个 Qwen 模型)
python code/eval_chaingsm_base_8shot.py \
  --data-path chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl \
  --output-root code/results/chaingsm_base_8shot_batch16 \
  --batch-size 16

# 续跑(自动跳过已完成模型)
python code/eval_chaingsm_base_8shot.py \
  --run-dir code/results/chaingsm_base_8shot_batch16/<timestamp>

# 选模型 + 限速烟测
python code/eval_chaingsm_base_8shot.py \
  --data-path chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl \
  --output-root code/results/chaingsm_base_8shot_batch16 \
  --batch-size 16 \
  --model Qwen2.5-0.5B-Instruct --model Llama-3.2-1B-Instruct \
  --limit 10
```

---

## 10. 成功标准

一个 run 算"完成"必须全部满足:

1. 每个模型 5,467 条 prediction 全部存在 `predictions.jsonl`,无 `errors.jsonl` 条目。
2. `summary_overall.json` 与 `summary_by_category.json` 都生成,且 5 类目齐全。
3. `run_config.json` 包含 `batch_size=16`、`example_count=5467`、每个模型的 `prompt_profile`。
4. Llama 模型的 `prompt_diagnostics.json` 中 `bos_count == 1` 且 `input_type == "token_ids"`。
5. 已通过对应 pytest:`pytest -q tests/test_chaingsm_base_8shot_eval.py`(6 个用例)。
6. 已通过全仓库 pytest:`pytest -q tests/`(共 45 个用例)。

---

## 11. 后续 checkpoint 沿用方式

SFT / DPO / GRPO checkpoint 跑同一脚本,只需:

- 把 `--model` 指向新 checkpoint 目录。
- 把 `--output-root` 切到 `code/results/chaingsm_base_8shot_sft/`、`..._dpo/`、`..._grpo/`。
- `model_name` 用 `safe_name(model_path.name)` 自动得到,不要硬改。

输出对比表由 `rescore_gsm_predictions.py` 直接吃 `predictions.jsonl` 生成,无需改主评测代码。
