# GSM8K / GSM-Plus 官方基线复现报告

生成日期：2026-06-05  
项目目录：`/home/wwq416/snap/wwq/math-chain`  
本地环境：`/home/wwq416/miniconda3/envs/math_chain_verl/bin/python`

> 2026-06-06 更新：已定位此前 Llama vLLM 结果偏低的直接原因。vLLM 0.21.0
> 会对已经包含 `<|begin_of_text|>` 的字符串 prompt 再添加一次 BOS，形成双 BOS。
> 将 `lm_eval` 多轮 chat template 直接 tokenize，并把单 BOS token IDs 交给 vLLM 后，
> 前 200 条达到 46.0%，全量 1319 条达到 45.72%。因此 Llama 并非必须使用
> HF backend；关键是 prompt token IDs、BOS、few-shot 多轮结构和 stop sequences 对齐。

## 1. 背景与目标

本轮工作的目标是复现 AbstRaL Table 5 中的 GSM8K / GSM-Plus CoT-8S 基线，先确认 official GSM8K Original 的基础评测链路，再用同一类设置扩展到 GSM-Plus Rephrase 和 Distract。

参考指标如下：

| Model | CoT-8S Original | CoT-8S Rephrase | CoT-8S Distract |
|---|---:|---:|---:|
| Llama-3.2-1B-Instruct | 45.19% | 50.42% | 27.90% |
| Qwen2.5-0.5B-Instruct | 42.38% | 43.59% | 22.67% |

本轮固定的基础生成设置：

```text
do_sample = false
temperature = 0
top_p = 1.0
max_new_tokens / max_gen_toks = 512
8-shot CoT
```

最终结论是：Qwen 和 Llama 不能使用完全相同的 prompt 包装。Qwen 可以用当前仓库的
vLLM chat 风格脚本达到参考值；Llama 需要对齐 `lm_eval` 的 Llama 专用配置，尤其是
单 BOS、`apply_chat_template`、`fewshot_as_multiturn` 和 task stop sequences。
Llama 可以继续使用 vLLM，但必须传入已经对齐的 token IDs，不能把已含 BOS 的
chat-template 字符串直接交给 vLLM 再次分词。

## 2. 模型与路径

本轮使用的本地模型均已提前下载，不需要重新下载。

| Model | Local Path |
|---|---|
| Qwen2.5-0.5B-Instruct | `/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct` |
| Llama-3.2-1B-Instruct | `/home/wwq416/snap/wwq/model/llama/Llama-3.2-1B-Instruct` |

官方 GSM8K Original 本地数据：

```text
/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/raw/test-00000-of-00001.jsonl
```

该文件包含 1319 条 GSM8K test 样本。

## 3. Qwen 与 Llama 的核心区别

### 3.1 Qwen：当前 vLLM chat 评测链路足够接近参考配置

Qwen2.5-0.5B-Instruct 在仓库脚本 `code/eval_official_gsm.py` 的 `prompt_style=chat` 下表现稳定。该路径使用：

- vLLM 后端
- tokenizer 的 chat template
- 8-shot CoT prompt
- `temperature=0.0`
- `top_p=1.0`
- `max_tokens=512`
- 仓库统一 numeric answer extractor

该配置下 Qwen 在 GSM8K Original 全量结果为 43.06%，超过 AbstRaL 参考值 42.38%。

### 3.2 Llama：不能套用 Qwen 的 chat prompt

Llama-3.2-1B-Instruct 对 prompt 包装非常敏感。将 Qwen 可用的 `prompt_style=chat` 直接套到 Llama 上，历史全量只有 25.63%。这说明问题不是单纯的模型能力，而是评测格式与模型指令调优方式不匹配。

Llama 更接近官方的配置是 EleutherAI `lm_eval` 中的 `gsm8k_cot_llama` 任务，并且需要：

- `add_bos_token=True`
- `--apply_chat_template`
- `--fewshot_as_multiturn true`
- task 自带 Llama 专用 doc_to_text
- task 自带 strict / flexible extractor
- 完整 stop sequences
- 若使用 vLLM，传入单 BOS token IDs，避免字符串 prompt 被重复添加 BOS

使用真正 `lm_eval` 后，Llama GSM8K Original 达到：

```text
strict-match:     46.32%
flexible-extract: 46.47%
```

该结果超过 AbstRaL 参考值 45.19%，也高于 Meta 模型卡中 GSM8K CoT 8-shot 的 44.4%。

## 4. Prompt 设计差异

### 4.1 Qwen 使用的 prompt 设计

Qwen 的有效路径是当前脚本中的 `prompt_style=chat`。其构造逻辑位于：

```text
/home/wwq416/snap/wwq/math-chain/code/eval_official_gsm.py
```

核心构造函数包括：

```text
build_fewshot_prompt(...)
build_qwen_chat_prompt(...)
build_model_input(..., prompt_style="chat")
```

Qwen 的 prompt 形态可以概括为：把 8 个 CoT 示例拼成一个 user prompt，再交给 tokenizer 的 chat template。

示例结构如下：

```text
System:
You are a helpful assistant.

User:
Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today.
After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: Let's think step by step. There are 21 trees now and there were 15 trees before.
So the workers planted 21 - 15 = 6 trees.
#### 6

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: Let's think step by step. There are 3 cars at first. 2 more cars arrive.
Now there are 3 + 2 = 5 cars.
#### 5

...

Question: {test_question}
Answer: Let's think step by step.
```

随后通过：

```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

变成模型真实输入。

Qwen 适配这种 prompt 的原因：

- Qwen Instruct 模型对单轮 chat 中包含长 few-shot 文本的格式容忍度较高。
- `#### answer` 这种 GSM8K 原始答案标记对 Qwen 输出有较强诱导。
- 仓库统一 extractor 优先识别 `The final answer is`、`####`、`\boxed{}`，最后 fallback 到数值抽取；对 Qwen 输出足够稳。

### 4.2 已删除的 Llama 手写路径为何失败

早期诊断使用过手工字符串 prompt。这些实现和中间结果已从仓库删除，只保留以下根因结论：

- 误读题目中的数量语义，例如把 “with four” 错当成 “4 days”。
- 生成过程中发生重复推理或自我修正，最后 extractor 抓到错误数字。
- 纯文本 prompt 没有稳定 stop 时，模型可能继续生成下一道虚构题，导致 fallback extractor 抓错最后数字。
- 字符串 prompt 未完整对齐 chat-template、few-shot 多轮、BOS token 和任务 filter。

### 4.3 Llama 最终有效 prompt：lm_eval 的 gsm8k_cot_llama

Llama 最终采用 EleutherAI `lm_eval` 的 `gsm8k_cot_llama` 任务。其官方配置的题面模板是：

```text
Given the following problem, reason and give a final answer to the problem.
Problem: {question}
Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.
```

8-shot 示例使用 `first_n`，每个示例 target 以 `The final answer is ...` 结束。例如：

```text
Question:
There are 15 trees in the grove. Grove workers will plant trees in the grove today.
After they are done, there will be 21 trees. How many trees did the grove workers plant today?

Target:
There are 15 trees originally. Then there were 21 trees after some more were planted.
So there must have been 21 - 15 = 6. The final answer is 6
```

在 `lm_eval` 中配合：

```text
--apply_chat_template
--fewshot_as_multiturn true
```

之后，8 个 few-shot 示例不会简单拼成一个大字符串，而是转成多轮 conversation：

```text
user:    Given the following problem...
assistant: There are 15 trees originally... The final answer is 6

user:    Given the following problem...
assistant: There are originally 3 cars... The final answer is 5

...

user:    Given the following problem...
assistant: [model generates here]
```

这点是 Llama 复现的关键。Llama Instruct 训练时高度依赖 chat-template 边界和 role 结构；如果只是把示例塞进一个 user 文本里，模型行为会明显偏离官方评测。

## 5. Answer Extractor 差异

### 5.1 仓库统一 extractor

仓库脚本 `eval_official_gsm.py` 中的统一 extractor 大致优先级为：

```text
1. The final answer is ...
2. #### ...
3. \boxed{...}
4. 结论句中的答案
5. The answer is ...
6. fallback 到最后一个数字
```

优点是兼容 Qwen、Llama、GSM8K 原始答案、LaTeX boxed 等多种输出格式。

缺点是对于 Llama 这种容易继续生成或重复推理的模型，fallback 到最后一个数字可能受到续写污染。

### 5.2 lm_eval 的 Llama extractor

`gsm8k_cot_llama` 自带两个 filter：

```text
strict-match:
The final answer is ((-?[$0-9.,]{2,})|(-?[0-9]+))

flexible-extract:
(-?[$0-9.,]{2,})|(-?[0-9]+)
```

并使用 exact_match 做指标，忽略逗号、美元符号、末尾句点等。

本轮 Llama 全量结果：

```text
strict-match:     46.32%
flexible-extract: 46.47%
```

strict 与 flexible 只差 0.15 个百分点，说明在真正 `lm_eval` 链路中，大多数输出已经能稳定产生 `The final answer is ...`。

## 6. 实验结果汇总

### 6.1 Qwen2.5-0.5B-Instruct

| Split | Result | Reference | Notes |
|---|---:|---:|---|
| Original | 43.06% | 42.38% | 本轮复跑，达标 |
| Rephrase | 44.28% | 43.59% | 本轮复跑，达标 |
| Distract | 21.46% | 22.67% | 保留的完整对照，接近但略低 |

关键结果文件：

```text
code/results/official_gsm/qwen_original_8shot_vllm/summary.json
code/results/official_gsm/qwen_rephrase_8shot_vllm/summary.json
code/results/official_gsm/qwen_distract_8shot_vllm/summary.json
```

Qwen 推荐命令：

```bash
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python \
  /home/wwq416/snap/wwq/math-chain/code/eval_official_gsm.py \
  --split original \
  --model Qwen2.5-0.5B-Instruct=/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct \
  --prompt-style chat \
  --batch-size 64 \
  --gpu-memory-utilization 0.8
```

Rephrase / Distract 可改 split：

```bash
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python \
  /home/wwq416/snap/wwq/math-chain/code/eval_official_gsm.py \
  --split rephrase \
  --split distract \
  --model Qwen2.5-0.5B-Instruct=/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct \
  --prompt-style chat \
  --batch-size 64 \
  --gpu-memory-utilization 0.8
```

注意：建议单独运行 `--split distract`，避免同进程连续加载多个 split 带来的
vLLM/NCCL 资源释放问题。

### 6.2 Llama-3.2-1B-Instruct

| Route | Result | Notes |
|---|---:|---|
| `lm_eval` HF + Llama task | strict 46.32% / flexible 46.47% | 官方 task 对照 |
| token 对齐后的 vLLM，全量 1319 条 | 45.72% | 超过 AbstRaL 参考 45.19% |

关键结果文件：

```text
code/results/official_gsm/llama_original_8shot_vllm/summary.json
code/results/official_gsm/llama_original_8shot_hf_lm_eval/__home__wwq416__snap__wwq__model__llama__Llama-3.2-1B-Instruct/results_2026-06-05T17-21-05.051435.json
code/results/official_gsm/llama_original_8shot_hf_lm_eval/__home__wwq416__snap__wwq__model__llama__Llama-3.2-1B-Instruct/samples_gsm8k_cot_llama_2026-06-05T17-21-05.051435.jsonl
```

Llama 推荐命令：

```bash
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python -m lm_eval run \
  --model hf \
  --model_args pretrained=/home/wwq416/snap/wwq/model/llama/Llama-3.2-1B-Instruct,dtype=bfloat16,add_bos_token=True \
  --tasks gsm8k_cot_llama \
  --num_fewshot 8 \
  --apply_chat_template \
  --fewshot_as_multiturn true \
  --batch_size 8 \
  --device cuda:0 \
  --gen_kwargs do_sample=False temperature=0.0 top_p=1.0 max_gen_toks=512 \
  --output_path code/results/official_gsm/llama_original_8shot_hf_lm_eval \
  --log_samples \
  --seed 42
```

本轮安装了：

```text
lm-eval==0.4.12
```

安装位置为 `math_chain_verl` 环境。

## 7. 为什么 Llama 的 lm_eval 结果高于手写 vLLM

### 7.1 BOS token 差异

Llama 推荐配置显式使用：

```text
add_bos_token=True
```

Llama tokenizer 的 BOS token 是：

```text
<|begin_of_text|>
```

Instruct 模型对开头特殊 token 和 chat template 边界很敏感。缺少 BOS 或重复/错误插入 BOS 都可能改变生成行为。

### 7.2 few-shot 组织方式不同

错误的字符串路径主要有两类：

- 把 8 个样例拼进一个 user prompt
- 手写 `<|start_header_id|>user...assistant...` 多轮片段

真正 `lm_eval` 的路径则由 tokenizer 的 `apply_chat_template` 统一生成最终输入，并由 `fewshot_as_multiturn` 把每个示例作为独立 user/assistant turn。这个差异看似只是格式，但对 Llama Instruct 属于核心输入协议。

### 7.3 task filter 和 stop sequence 完整对齐

`gsm8k_cot_llama` 的 stop sequences 是：

```text
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Q:
</s>
<|im_end|>
```

这可以防止模型在回答完后继续生成下一轮 user 或下一道题。手写 prompt 如果 stop 不完整，输出续写会污染 fallback 数字抽取。

### 7.4 HF backend 与 vLLM 行为差异

2026-06-06 的 token 对齐实验确认，主要差距不是 vLLM 模型计算本身，而是字符串 prompt
在 vLLM 中被再次添加 BOS。诊断结果如下：

```text
错误字符串路径：1171 tokens，BOS count = 2，首题预测 0
正确 token 路径：1170 tokens，BOS count = 1，首题预测 18
```

正确路径需要对齐完整评测栈：

```text
tokenizer chat template
+ exactly one BOS token
+ fewshot_as_multiturn
+ task filter
+ task stop sequence
+ vLLM token-ID input
```

前 200 条对照：

```text
HF strict-match: 94 / 200 = 47.0%
vLLM shared extractor: 92 / 200 = 46.0%
逐题正确性一致: 184 / 200
HF only correct: 9
vLLM only correct: 7
```

## 8. 后续复现建议

### 8.1 保留模型专属入口

建议不要强行维护一个“所有模型完全共用”的官方复现入口。更合理的是：

```text
Qwen official baseline:
  code/eval_official_gsm.py
  vLLM
  prompt_style=chat
  shared numeric extractor

Llama official baseline:
  code/eval_official_gsm.py
  vLLM token-ID input
  prompt_style=lm_eval_llama_chat_multiturn
  exactly one BOS token
  apply_chat_template
  fewshot_as_multiturn
  complete task stop sequences
```

Llama 的 HF `lm_eval` 结果仍保留为官方 task filter 的权威全量对照。vLLM 全量结果
为 45.72%，使用仓库 shared numeric extractor；最终论文表格应明确标注两条路径的
extractor 差异。

Llama vLLM 诊断命令：

```bash
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python \
  /home/wwq416/snap/wwq/math-chain/code/eval_official_gsm.py \
  --split original \
  --model Llama-3.2-1B-Instruct=/home/wwq416/snap/wwq/model/llama/Llama-3.2-1B-Instruct \
  --prompt-style lm_eval_llama_chat_multiturn \
  --batch-size 64 \
  --gpu-memory-utilization 0.8 \
  --dtype bfloat16 \
  --max-model-len 4096
```

### 8.2 GSM-Plus Rephrase / Distract 的处理

Qwen 的 GSM-Plus Rephrase 已达到 44.28%，Distract 完整对照为 21.46%，接近参考
22.67%。建议后续单独复跑 Qwen Distract，避免连续加载 split 导致 vLLM 显存启动检查失败。

Llama 的 GSM-Plus Rephrase / Distract 不能直接使用 `gsm8k_cot_llama`，因为该 task 绑定的是 `openai/gsm8k` test split。要复现 GSM-Plus，需要：

1. 新建或临时生成一个 `lm_eval` 自定义 task，dataset 指向 GSM-Plus。
2. 复用 `gsm8k_cot_llama` 的 `doc_to_text`、few-shot samples、filter_list、generation_kwargs。
3. 对 Rephrase 只选择 `perturbation_type == "problem understanding"`。
4. 对 Distract 只选择 `perturbation_type == "distraction insertion"`。
5. 使用相同命令参数：

```text
--model hf
--model_args pretrained=...,dtype=bfloat16,add_bos_token=True
--apply_chat_template
--fewshot_as_multiturn true
--gen_kwargs do_sample=False temperature=0.0 top_p=1.0 max_gen_toks=512
```

### 8.3 报告指标时建议同时给 strict 与 flexible

对于 Llama，建议报告：

```text
strict-match
flexible-extract
```

其中 strict 更接近 `The final answer is ...` 规范，flexible 更接近宽松数字抽取。两者本轮只差 0.15%，说明评测稳定。

对于 Qwen，当前脚本使用统一 numeric extractor，建议在最终论文表格中注明 extractor 与 Llama lm_eval filter 的差别，避免审稿或内部复核时误以为二者是完全相同代码路径。

## 9. 最终结论

本轮复现说明：

1. Qwen2.5-0.5B-Instruct 的基础评测链路没有明显问题，GSM8K Original 和 GSM-Plus Rephrase 均达到或超过 AbstRaL 参考。
2. Llama-3.2-1B-Instruct 之前偏低的主要原因是评测包装和 token 输入不匹配，而不是模型能力不足。
3. Llama 的官方复现需要单 BOS、`apply_chat_template`、`fewshot_as_multiturn` 和完整 stop sequences；HF 与 vLLM 都可以满足。
4. vLLM 字符串 prompt 路径会重复添加 BOS；使用 chat template 生成的 token IDs 后，全量结果从约 38% 恢复到 45.72%。
5. Llama vLLM 的 1319 条 GSM8K Original 已复跑完成；下一阶段将同一 token-ID 路径扩展到 GSM-Plus Rephrase / Distract。

## 10. 参考来源

- EleutherAI `gsm8k_cot_llama` 配置：`https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/gsm8k/gsm8k-cot-llama.yaml`
- Meta / Hugging Face `Llama-3.2-1B-Instruct` 模型卡：`https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct`
- 本地 Llama lm_eval 结果：`code/results/official_gsm/llama_original_8shot_hf_lm_eval/__home__wwq416__snap__wwq__model__llama__Llama-3.2-1B-Instruct/results_2026-06-05T17-21-05.051435.json`
- 本地 Qwen official GSM 结果：`code/results/official_gsm/`
