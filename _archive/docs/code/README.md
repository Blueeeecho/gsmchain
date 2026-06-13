# ChainGSM 本地模型评测说明

本文档说明如何使用 `eval_chaingsm.py` 对 ChainGSM 数据进行本地模型评测。脚本使用 `vLLM` 加速推理，默认在 `math_chain_verl` 虚拟环境中运行。

## 1. 快速开始

默认命令只评测小规模测试文件：

```bash
conda run --no-capture-output -n math_chain_verl python /home/wwq416/snap/wwq/math-chain/code/eval_chaingsm.py
```

默认数据路径是：

```text
/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/test.jsonl
```

如果希望进行全量测试，请使用下面的命令：

```bash
conda run --no-capture-output -n math_chain_verl python /home/wwq416/snap/wwq/math-chain/code/eval_chaingsm.py \
  --data-path /home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/chaingsm_train_1000.jsonl
```

如果只想先用某个模型做快速检查，可以加 `--model-filter` 和 `--limit`：

```bash
conda run --no-capture-output -n math_chain_verl python /home/wwq416/snap/wwq/math-chain/code/eval_chaingsm.py \
  --data-path /home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/chaingsm_train_1000.jsonl \
  --model-filter Qwen2.5-0.5B-Instruct \
  --limit 5
```

## 2. 评测对象

脚本会从下面的模型根目录中自动发现模型：

```text
/home/wwq416/snap/wwq/model
```

模型发现规则如下：

- 递归查找所有包含 `config.json` 的模型目录。
- 只保留目录名包含 `Instruct` 的模型。
- 默认排除路径中包含 `ministral` 的模型，因此不会测试 `Ministral-3-3B-Instruct-2512`。
- 非 instruct 模型、encoder 模型、base 模型不会参与评测。

当前默认会覆盖的主要模型包括：

- `Qwen2.5-0.5B-Instruct`
- `Qwen2.5-1.5B-Instruct`
- `Qwen2.5-3B-Instruct`
- `Qwen2.5-Math-1.5B-Instruct`
- `Llama-3.2-1B-Instruct`

可以通过 `--model-filter` 只运行路径中包含指定字符串的模型。该参数可以重复传入：

```bash
conda run --no-capture-output -n math_chain_verl python /home/wwq416/snap/wwq/math-chain/code/eval_chaingsm.py \
  --model-filter Qwen2.5 \
  --model-filter Math
```

## 3. 数据格式与使用方式

脚本读取 JSONL 格式数据。每行是一道题或一道变体题，关键字段包括：

- `id`：样本 ID。
- `base_id`：原始问题 ID。
- `category`：题目类别。
- `question_original`：原始问题。
- `question_distracted`：加入干扰信息后的变体问题。
- `answer`：标准答案。原始问题和变体问题使用同一个标准答案。

脚本中的问题选择规则如下：

- 当 `category == "original"` 时，使用 `question_original`。
- 当 `category != "original"` 时，使用 `question_distracted`。
- 所有类别都使用 `answer` 作为 gold answer。

目前数据中包含原题和 4 类变体：

- `original`
- `independent_decoy`
- `attribute_mismatch`
- `target_scope_misalignment`
- `path_competition`

## 4. 评测方法

脚本统一评测 3 种 prompt 方法：

### Direct

System prompt：

```text
You are a helpful assistant.
```

User prompt：

```text
Q: {question}
A:
```

### Zero-shot CoT

System prompt：

```text
You are a helpful assistant.
```

User prompt：

```text
Q: {question}
A: Let's think step by step.
```

### 8-shot CoT

System prompt：

```text
As an expert problem solver, solve step by step the following mathematical questions.
```

User prompt 中包含 8 个 GSM8K 风格的 step-by-step 示例，最后追加：

```text
Q: {question}
A: Let's think step by step.
```

如果只想运行某一种或某几种方法，可以使用 `--method`。该参数可以重复传入：

```bash
conda run --no-capture-output -n math_chain_verl python /home/wwq416/snap/wwq/math-chain/code/eval_chaingsm.py \
  --method direct \
  --method zero_shot_cot
```

## 5. 推理配置

所有方法使用同一套生成配置：

```text
temperature = 0.0
top_p = 1.0
max_tokens = 512
do_sample = false
```

在 vLLM 中，`do_sample = false` 对应确定性采样设置，也就是 `temperature=0.0`。

其他常用参数：

- `--batch-size`：每批 prompt 数量，默认 `64`。
- `--tensor-parallel-size`：张量并行大小，默认 `1`。
- `--gpu-memory-utilization`：vLLM 显存利用率，默认 `0.9`。
- `--dtype`：模型 dtype，默认 `auto`。
- `--max-model-len`：可选，限制最大上下文长度。

示例：

```bash
conda run --no-capture-output -n math_chain_verl python /home/wwq416/snap/wwq/math-chain/code/eval_chaingsm.py \
  --data-path /home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/chaingsm_train_1000.jsonl \
  --batch-size 32 \
  --gpu-memory-utilization 0.85
```

## 6. 输出目录

每次运行都会创建一个带时间戳的结果目录：

```text
/home/wwq416/snap/wwq/math-chain/code/results/chaingsm_test/<timestamp>/
```

例如：

```text
/home/wwq416/snap/wwq/math-chain/code/results/chaingsm_test/20260525_075842/
```

结果目录包含：

```text
run_config.json
summary_by_category.csv
summary_by_category.json
summary_overall.csv
summary_overall.json
model_outputs/
```

### 过程文件

过程文件按模型和方法两级目录分别存储：

```text
model_outputs/<model_name>/<method>/predictions.jsonl
```

例如：

```text
model_outputs/Qwen2.5-0.5B-Instruct/direct/predictions.jsonl
model_outputs/Qwen2.5-0.5B-Instruct/zero_shot_cot/predictions.jsonl
model_outputs/Qwen2.5-0.5B-Instruct/eight_shot_cot/predictions.jsonl
model_outputs/Llama-3.2-1B-Instruct/direct/predictions.jsonl
```

每个 `predictions.jsonl` 只包含该模型在该方法下的预测结果。`predictions.jsonl` 每完成一条预测就立即写入并 flush，字段包括：

- `model_name`：模型名称。
- `model_path`：模型路径。
- `method`：评测方法。
- `id`：样本 ID。
- `base_id`：原始问题 ID。
- `category`：类别。
- `question`：实际输入的问题。
- `gold_answer`：标准答案。
- `raw_output`：模型原始输出。
- `pred_answer`：抽取出的答案。
- `correct`：是否答对。

如果某个模型加载或生成失败，会在该模型目录下写入：

```text
model_outputs/<model_name>/errors.jsonl
model_outputs/<model_name>/<method>/errors.jsonl
```

注意：预测记录数按“样本数 × 方法数 × 模型数”计算。对单个模型来说，如果数据文件有 879 道题，并且同时运行 3 种方法，那么该模型总体预测记录数会是 `879 × 3 = 2637`，并会拆成每个方法各 879 条。实际样本数以每次结果目录中的 `run_config.json` 里的 `example_count` 为准；例如 `gsm8k_test_all.jsonl` 当前记录为 6575 条。

### 汇总文件

`summary_by_category.csv/json` 按下面的维度统计准确率：

```text
model_name + method + category
```

字段包括：

- `model_name`
- `method`
- `category`
- `correct`
- `total`
- `accuracy`

`summary_overall.csv/json` 按下面的维度统计总体准确率：

```text
model_name + method
```

字段包括：

- `model_name`
- `method`
- `correct`
- `total`
- `accuracy`

`run_config.json` 保存本次运行配置，包括数据路径、模型列表、prompt 方法、采样参数、输出目录、排除模型关键词等。

## 7. 答案抽取与判分

模型输出会按下面的优先级抽取答案：

1. 匹配 `The final answer is ...`
2. 匹配 `#### ...`
3. 匹配 `\boxed{...}`
4. 使用输出中的最后一个数字、小数或分数

抽取后会进行归一化：

- 去掉逗号、美元符号、百分号和尾部标点。
- 支持整数、小数和简单分数。
- 使用数值比较。
- 默认容差为 `1e-6`。

例如，下面这些形式都可以被归一化为数值答案：

```text
14,000
$14000
14000.
1/2
0.5
```

## 8. 推荐运行流程

第一次运行建议先做小样本检查：

```bash
conda run --no-capture-output -n math_chain_verl python /home/wwq416/snap/wwq/math-chain/code/eval_chaingsm.py \
  --limit 1 \
  --model-filter Qwen2.5-0.5B-Instruct
```

确认输出目录和汇总文件正常后，再运行全量测试：

```bash
conda run --no-capture-output -n math_chain_verl python /home/wwq416/snap/wwq/math-chain/code/eval_chaingsm.py \
  --data-path /home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/chaingsm_train_1000.jsonl
```

如果全量运行时显存压力较大，可以降低 batch size：

```bash
conda run --no-capture-output -n math_chain_verl python /home/wwq416/snap/wwq/math-chain/code/eval_chaingsm.py \
  --data-path /home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/chaingsm_train_1000.jsonl \
  --batch-size 16
```

## 9. 注意事项

- 当前脚本默认使用 `test.jsonl`，全量测试必须显式传入 `--data-path`。
- 当前脚本默认排除 `Ministral`，因为此前本地 tokenizer 配置无法被当前环境正常加载。
- 每个模型评测完成后，脚本会释放 vLLM 对象并清理 CUDA cache。
- vLLM 首次加载某个模型时可能会进行编译和 CUDA graph 初始化，启动阶段耗时较长是正常现象。
- 如果使用普通 `conda run -n math_chain_verl ...`，conda 可能会捕获输出，导致 tqdm 进度条不实时显示。建议使用 `conda run --no-capture-output -n math_chain_verl ...`。
- 过程文件是逐条写入的，即使中途某个模型失败，已经完成的预测也会保留。
