# Llama 1-shot Prompt、vLLM 与 lm-eval 评测分析

生成日期：2026-06-06  
项目目录：`/home/wwq416/snap/wwq/math-chain`

## 1. 分析目标

本文分析下面这种手工构造的 Llama prompt：

```python
llama_prompt = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    f"{SOLUTION_PROMPT_1_SHOT_SYS.strip()}"
    "<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    f"{SOLUTION_PROMPT_1_SHOT_USER.format(
        puzzle=example['puzzle'],
        solution_header=final_grid['header'],
        attribute_values=attribute_values_from_solution(example['solution'])
    ).strip()}"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)
```

需要回答的问题包括：

1. 这是否属于标准 1-shot。
2. 当前结构为什么可能使 Llama 效果不及预期。
3. 使用 vLLM 时应该如何修改。
4. 不使用 vLLM、直接使用 Hugging Face 时应该如何修改。
5. `lm_eval` 是什么，以及它为什么经常得到更好的结果。

本文没有看到另一个任务中
`SOLUTION_PROMPT_1_SHOT_SYS`、`SOLUTION_PROMPT_1_SHOT_USER` 和完整推理调用代码的实际定义。
因此，关于模板内部字段边界的判断基于用户给出的代码。落地前应打印一条真实 prompt 和
token IDs，确认模板没有包含本文未覆盖的额外 target puzzle。

## 2. 当前 prompt 的真实消息结构

手工 token 展开后，当前 prompt 等价于：

```text
system:
  SOLUTION_PROMPT_1_SHOT_SYS

user:
  SOLUTION_PROMPT_1_SHOT_USER(
      puzzle=example puzzle,
      solution_header=example solution header,
      attribute_values=example solution values
  )

assistant:
  [模型从这里开始生成]
```

如果 `SOLUTION_PROMPT_1_SHOT_USER` 同时包含一个完整示例题目和它的完整答案，那么从
“一个上下文中出现过一组输入输出示范”的宽松定义看，它可以被称为 textual 1-shot。

但它不是 Llama Instruct chat 协议中推荐的标准 1-shot。标准 chat 1-shot 应该把示例输入
和示例输出分配给不同角色：

```text
system:    任务规则
user:      示例 puzzle
assistant: 示例答案
user:      当前 target puzzle
assistant: [模型生成当前答案]
```

这里的核心区别不是文本总量，而是 role boundary。Llama Instruct 在训练时学习的是
`user -> assistant` 的条件生成关系。把示例问题与示例答案全部放在同一个 `user` 消息中，
会削弱“这段答案是 assistant 应该模仿的行为”这一信号。

## 3. 当前设置的主要问题

### 3.1 示例答案放在 user role

当前的 `SOLUTION_PROMPT_1_SHOT_USER` 看起来同时接收：

```text
example["puzzle"]
final_grid["header"]
attribute_values_from_solution(example["solution"])
```

这意味着示例题和答案很可能都位于同一个 user turn。模型看到的是：

```text
user: 题目 + 答案
assistant: 请继续生成
```

而不是：

```text
user: 题目
assistant: 答案
```

后果可能包括：

- 模型不明确应该复现答案格式还是继续讨论 user 提供的答案。
- 模型可能把答案部分理解为当前用户的附加条件。
- 示例中的格式约束没有通过 assistant role 被强化。
- 复杂 puzzle 上更容易出现结构不完整、字段重复或直接复述示例。

### 3.2 可能缺少独立的 target puzzle

给出的代码只出现一次 `puzzle=example["puzzle"]`。如果这里的 `example` 是 few-shot 示例，
那么 prompt 中没有看到需要模型实际求解的 target puzzle。

如果 `example` 实际上是当前测试样本，那么代码又把：

```python
example["solution"]
```

放进了 prompt，这会造成 gold solution 泄漏。此时模型不是在求解未知 puzzle，而是在读取
或改写已提供答案，评测结果将失去意义。

必须在数据结构上明确区分：

```python
fewshot_example  # 有 puzzle 和 solution，用于示范
target_example   # 只向模型提供 puzzle，solution 仅用于评分
```

建议增加防御性检查：

```python
assert fewshot_example["id"] != target_example["id"]
assert target_example["solution"] not in rendered_target_prompt
```

### 3.3 手工拼接特殊 token

当前代码手工写入：

```text
<|begin_of_text|>
<|start_header_id|>
<|end_header_id|>
<|eot_id|>
```

这会绕开 tokenizer 自带的 chat template。风险包括：

- 模型版本变化后模板不一致。
- system message 中模型卡要求的日期或工具字段缺失。
- 换用其他 Llama tokenizer 时 token 边界不一致。
- prompt 已含 BOS，但后端再次自动添加 BOS。
- 缺少 generation prompt 或错误添加 `<|eot_id|>`。

除非是在做严格的 token-level 复现实验，否则应使用：

```python
tokenizer.apply_chat_template(...)
```

### 3.4 vLLM 字符串输入可能产生重复 BOS

本项目已在 Llama-3.2-1B-Instruct、vLLM 0.21.0 上确认：

```text
chat-template 字符串已经包含 BOS
+ vLLM 字符串分词再次添加 BOS
= 双 BOS
```

GSM8K 实验中的直接证据：

```text
错误字符串路径：
  1171 tokens
  BOS count = 2
  首题预测错误

正确 token-ID 路径：
  1170 tokens
  BOS count = 1
  首题预测正确
```

全量 GSM8K Original 结果：

```text
旧手写 vLLM 路径：约 38%
单 BOS token-ID vLLM 路径：603 / 1319 = 45.72%
AbstRaL 参考：45.19%
lm_eval HF strict-match：46.32%
```

这些数据证明“双 BOS 会伤害当前 Llama GSM8K 评测”。对于本文的 puzzle 任务，双 BOS
仍然是明确的协议错误，但其具体精度影响应在该任务上单独做 A/B 测试，不能直接照搬
GSM8K 的百分点差值。

### 3.5 示例格式与目标输出格式可能不一致

如果示例答案中包含：

```text
solution_header
attribute_values
完整推理过程
最终 grid
```

那么 target user message也必须明确要求同样的输出协议。否则模型可能只生成推理、不生成
grid，或者生成 grid 但缺少可解析的 header。

建议把输出契约写在 system message 中，并让示例 assistant answer 完整遵守该契约：

```text
1. Reasoning
2. Final grid header
3. Final attribute rows
4. 不得继续生成下一道题
```

评分器应分别记录：

- 格式是否可解析。
- grid 是否完整。
- 约束是否满足。
- 最终答案是否正确。

不要只用一个 overall accuracy 掩盖 prompt 格式错误。

### 3.6 缺少 stop sequence

Llama 可能在答案结束后继续生成：

- 下一轮 user header。
- 新的 puzzle。
- 对答案的二次修改。
- 重复 grid。

建议至少停止于：

```text
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
</s>
```

如果任务有稳定的最终 XML/JSON/grid 结束标记，也应将其作为业务层截断边界，或在解析器中
只提取第一个完整答案块。

## 4. 标准 1-shot 消息设计

推荐将模板拆成四部分：

```python
messages = [
    {
        "role": "system",
        "content": SOLUTION_SYSTEM_PROMPT.strip(),
    },
    {
        "role": "user",
        "content": PUZZLE_USER_PROMPT.format(
            puzzle=fewshot_example["puzzle"],
        ).strip(),
    },
    {
        "role": "assistant",
        "content": SOLUTION_ASSISTANT_PROMPT.format(
            solution_header=fewshot_grid["header"],
            attribute_values=attribute_values_from_solution(
                fewshot_example["solution"]
            ),
        ).strip(),
    },
    {
        "role": "user",
        "content": PUZZLE_USER_PROMPT.format(
            puzzle=target_example["puzzle"],
        ).strip(),
    },
]
```

推荐的模板职责：

```python
SOLUTION_SYSTEM_PROMPT
# 只定义任务、约束、推理要求和输出格式。

PUZZLE_USER_PROMPT
# 只呈现一个 puzzle，不包含答案。

SOLUTION_ASSISTANT_PROMPT
# 呈现示例答案，格式与模型对 target 应生成的格式完全一致。
```

不要继续用名称为 `SOLUTION_PROMPT_1_SHOT_USER` 的单一模板同时承担示例 question、示例
answer 和 target question 三种职责。职责混合会使 prompt 难以审计，也容易引入答案泄漏。

## 5. 使用 vLLM 的推荐版本

### 5.1 构造单 BOS token IDs

```python
from collections.abc import Mapping


def build_llama_vllm_token_ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    # transformers 5.x 可能返回 BatchEncoding，而不是裸 list。
    if isinstance(encoded, Mapping):
        token_ids = list(encoded["input_ids"])
    else:
        token_ids = list(encoded)

    bos_count = token_ids.count(tokenizer.bos_token_id)
    if bos_count != 1:
        raise ValueError(f"Expected exactly one BOS token, got {bos_count}")

    return token_ids
```

### 5.2 调用 vLLM

```python
from vllm import LLM, SamplingParams


prompt_token_ids = build_llama_vllm_token_ids(tokenizer, messages)

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=1024,
    stop=[
        "<|eot_id|>",
        "<|start_header_id|>user<|end_header_id|>",
        "</s>",
    ],
)

outputs = llm.generate(
    [prompt_token_ids],
    sampling_params,
    use_tqdm=False,
)

raw_output = outputs[0].outputs[0].text
```

当前 vLLM 版本允许直接传入 `list[int]`。这样可以跳过 vLLM 的字符串重新分词，确保送入
模型的 token IDs 与 Hugging Face 参考路径一致。

### 5.3 不推荐的 vLLM 调用

```python
rendered_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

outputs = llm.generate([rendered_prompt], sampling_params)
```

这条路径的风险是 `rendered_prompt` 已包含 BOS，而 vLLM tokenizer 又自动添加 BOS。即使
未来 vLLM 版本修复或改变该行为，token-ID 输入仍然更适合做严格的 HF/vLLM 对齐评测。

### 5.4 vLLM 路径必须记录的诊断信息

每次实验至少保存：

```python
diagnostics = {
    "prompt_token_count": len(prompt_token_ids),
    "bos_count": prompt_token_ids.count(tokenizer.bos_token_id),
    "prompt_prefix_ids": prompt_token_ids[:16],
    "prompt_suffix_ids": prompt_token_ids[-16:],
    "decoded_prompt": tokenizer.decode(prompt_token_ids),
}
```

这可以避免只保存“prompt_style=one_shot”，却无法知道模型实际收到了什么。

## 6. 不使用 vLLM 的 Hugging Face 版本

Hugging Face 路径适合作为正确性参考实现：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
).eval()

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

input_ids = inputs["input_ids"]
bos_count = input_ids[0].tolist().count(tokenizer.bos_token_id)
if bos_count != 1:
    raise ValueError(f"Expected exactly one BOS token, got {bos_count}")

eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
eos_ids = [tokenizer.eos_token_id]
if eot_id is not None and eot_id >= 0:
    eos_ids.append(eot_id)

with torch.inference_mode():
    generated = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=1024,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.eos_token_id,
    )

new_tokens = generated[0, input_ids.shape[1]:]
raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
```

这条路径比 vLLM 慢，但优点是：

- 与 tokenizer/chat template 的集成直接。
- 便于检查实际 `input_ids`。
- 适合建立小样本 golden baseline。
- 可以逐 token 比较 HF 与 vLLM 的输入和输出。

建议先用 HF 跑固定的 20 至 100 条诊断集，再使用相同 token IDs 跑 vLLM。两边应固定：

```text
模型权重
dtype
prompt token IDs
do_sample=False
最大生成长度
停止条件
答案解析器
样本顺序
```

即使输入完全一致，HF 与 vLLM 仍可能因 kernel、数值精度或调度实现产生少量 greedy
分叉，因此评估目标不应是逐字输出 100% 相同，而应关注：

- 输入 token 是否完全一致。
- 最终答案准确率是否接近。
- 格式成功率是否接近。
- 分叉是否集中在低置信度样本。

## 7. `lm_eval` 是什么

`lm_eval` 通常指 EleutherAI 的
`lm-evaluation-harness`，即 Language Model Evaluation Harness。

它不是模型，也不是新的推理算法，而是标准化评测框架。它主要负责：

- 加载数据集和指定 split。
- 定义 `doc_to_text` 和 `doc_to_target`。
- 选择 few-shot 示例。
- 把 few-shot 组织成文本或多轮 conversation。
- 应用 tokenizer chat template。
- 调用 HF、vLLM 等模型后端。
- 设置 generation kwargs 和 stop sequences。
- 执行答案 filter/extractor。
- 计算 exact match、multiple choice accuracy 等指标。
- 保存配置、样本输出和复现元数据。

一个 task 通常会明确描述：

```text
dataset
prompt template
few-shot samples
num_fewshot
generation kwargs
stop sequences
answer filters
metrics
```

## 8. 为什么 `lm_eval` 经常看起来更好

严格来说，`lm_eval` 不会让模型“更聪明”。它经常得到更好结果，是因为它减少了评测协议
偏差。

### 8.1 正确的 chat template

Instruct 模型不是只依赖自然语言内容，它还依赖：

```text
BOS
system/user/assistant role
turn boundary
generation header
EOS/EOT
```

`lm_eval --apply_chat_template` 会调用模型 tokenizer 的模板，而不是依赖手工拼接。

### 8.2 正确的 few-shot 角色组织

`fewshot_as_multiturn=true` 会把示例组织为：

```text
user: example question
assistant: example answer
user: target question
assistant: generate
```

这正是当前 puzzle prompt 缺少的关键结构。

### 8.3 明确的 BOS 和 stop 策略

`lm_eval` 的 Hugging Face wrapper 会检查 prompt 是否已经以 BOS 开头，避免再次添加 BOS。
task 还会定义 stop sequences，防止模型继续生成下一道题。

### 8.4 task 专属答案提取

一个模型可能给出正确答案，但通用“取最后一个数字”解析器会因后续续写而判错。
`lm_eval` task 通常定义与答案格式匹配的 regex/filter，从而减少评测脚本自身造成的误差。

### 8.5 可复现性更好

`lm_eval` 会记录 task、模型参数、few-shot 数量、随机种子和样本输出。手写脚本如果只保存
最终 accuracy，而不保存最终 prompt/token IDs，就很难解释结果变化。

## 9. `lm_eval`、HF 和 vLLM 的关系

三者不是互斥概念：

```text
lm_eval
  ├── HF backend
  ├── vLLM backend
  └── 其他模型 API/backend
```

`lm_eval` 是评测编排层；HF 和 vLLM 是模型执行后端。

本项目 GSM8K 的历史差异并不是：

```text
lm_eval 聪明
vLLM 不聪明
```

而是：

```text
lm_eval + HF：
  正确的多轮 prompt
  单 BOS
  完整 stop
  task filter

旧手写 vLLM：
  prompt 结构不完全一致
  字符串路径产生双 BOS
  stop/filter 不完全一致
```

当 vLLM 改为接收与 `lm_eval` 对齐的单 BOS token IDs 后，GSM8K 全量达到 45.72%，已经
超过 45.19% 参考值，并接近 HF strict-match 46.32%。因此应把 `lm_eval` 当作评测协议
参考，而不是把 HF backend 当成唯一正确后端。

## 10. 推荐的重新评价实验

不要一次修改多个变量。建议按下面的顺序做消融：

### 实验 A：复现当前手写 prompt

保存：

```text
完整 rendered prompt
token IDs
BOS count
输入 token 数
输出
格式解析结果
最终准确率
```

这建立当前结果基线。

### 实验 B：只修正 role boundary

保持示例内容、模型、生成参数不变，仅改为：

```text
user example
assistant example answer
user target
```

这验证“标准 chat 1-shot”本身的收益。

### 实验 C：只改为 tokenizer chat template

不再手写特殊 token，检查：

```text
BOS count == 1
最后一个 token/角色是 assistant generation prompt
```

### 实验 D：HF 与 vLLM 使用同一 token IDs

选择固定诊断集，例如 50 条：

```text
HF greedy
vLLM greedy
```

比较：

```text
答案准确率
格式成功率
约束满足率
逐样本分叉
```

### 实验 E：0-shot 与 1-shot

1-shot 不一定总是优于 0-shot。如果示例：

- 与 target 分布不一致。
- 解法有错误或冗余。
- 输出格式不稳定。
- 占用过多上下文。

它可能反而降低结果。因此应同时报告：

```text
0-shot standard chat
1-shot standard chat
```

### 实验 F：示例选择

固定 1-shot 后，至少比较：

```text
固定代表性示例
随机示例，多个 seed
按 puzzle 规模匹配的示例
```

单一示例可能带来明显的 selection bias。

## 11. 推荐验收标准

在宣称 Llama 适配完成前，建议满足：

```text
[ ] few-shot example 与 target 样本明确分离
[ ] target solution 未进入 prompt
[ ] 示例答案位于 assistant role
[ ] target puzzle 位于最后一个 user role
[ ] 使用 tokenizer.apply_chat_template
[ ] 最终输入恰好一个 BOS
[ ] vLLM 接收 token IDs，而不是已含 BOS 的字符串
[ ] HF 与 vLLM 的 prompt token IDs 完全一致
[ ] do_sample=False
[ ] stop sequences 明确
[ ] 最大生成长度一致
[ ] 输出解析器一致
[ ] 保存逐样本 prompt、输出和判分结果
[ ] 报告 0-shot 与标准 1-shot 对照
[ ] 报告 HF 与 vLLM 小样本对照
```

## 12. 最终结论

1. 当前 prompt 在宽松的文本定义下可以称为 1-shot，但不是 Llama chat 协议中推荐的标准
   1-shot。
2. 最大的结构问题是示例 question 和 answer 很可能都放在 user role，同时没有明确看到
   独立 target puzzle。
3. 如果 `example` 是当前测试样本，使用 `example["solution"]` 会造成答案泄漏。
4. 不应手工拼接 Llama 特殊 token；应使用 tokenizer chat template。
5. 使用 vLLM 时应传入单 BOS token IDs，避免字符串路径重复添加 BOS。
6. 不使用 vLLM 时，可用 Hugging Face greedy generation 建立正确性参考。
7. `lm_eval` 的优势来自标准化 prompt、few-shot、BOS、stop 和答案 filter，而不是它改变了
   模型能力。
8. 下一轮评价应先做 role boundary、单 BOS 和同 token IDs 的单变量消融，再扩大到全量。
