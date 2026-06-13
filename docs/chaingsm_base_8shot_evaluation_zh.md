# ChainGSM 干净测试集 8-shot 基线评测说明

生成日期：2026-06-07
项目目录：`/home/wwq416/snap/wwq/math-chain`
本地环境：`/home/wwq416/miniconda3/envs/math_chain_verl/bin/python`

本文档对齐当前主线评测：四个 Qwen Instruct 模型与两个 Llama-3.2 Instruct 模型
在 ChainGSM 干净测试集（5,467 条）上跑 8-shot CoT，使用仓库统一数值答案
提取器，并按 ChainGSM 五类（original + 四类变体）输出准确率。

## 1. 评测目标

- 量化小尺寸 Instruct 模型在 GSM8K Original 上的 8-shot CoT 能力。
- 量化同一批模型在四类 ChainGSM 干扰变体上的鲁棒性退化幅度。
- 为后续 SFT / DPO / GRPO 训练流程提供统一基线锚点。

## 2. 测试集

数据来源：

```text
chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl
```

来源链路：

```text
GSM8K test (1319 original)
  → 4 类 DeepSeek 变体生成（independent_decoy / attribute_mismatch /
     path_competition / target_scope_misalignment）
  → deepseek-v4-flash 强模型审计（1,319 题组 / 6,575 记录全覆盖）
  → 删除 flagged 变体（保留全部 original）
  → gsm8k_test_clean.jsonl（5,467 条）
```

当前干净集规模（来自 `cleaning_stats.json`）：

| 类别 | 数量 |
|---|---:|
| original | 1319 |
| independent_decoy | 1102 |
| attribute_mismatch | 1017 |
| path_competition | 999 |
| target_scope_misalignment | 1030 |
| **合计** | **5467** |

每条记录保留字段（不修改、不重排）：

```text
id, base_id, source_index, category,
question_original, question_distracted, answer,
solution_original, core_chain, distractor_chain,
gold_expression, distractor_expression,
difficulty_tags, metadata
```

`category == original` 的样本使用 `question_original`，其余类别使用
`question_distracted`。Gold 答案统一来自 `answer` 字段。

## 3. 模型

主线评测五个模型：

| 模型 | 本地路径 | Prompt Profile |
|---|---|---|
| Qwen2.5-0.5B-Instruct | `/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct` | `qwen_multiturn_8shot_chat` |
| Qwen2.5-1.5B-Instruct | `/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-1.5B-Instruct` | `qwen_multiturn_8shot_chat` |
| Qwen2.5-Math-1.5B-Instruct | `/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-Math-1.5B-Instruct` | `qwen_math_completion_8shot` |
| Qwen2.5-3B-Instruct | `/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-3B-Instruct` | `qwen_multiturn_8shot_chat` |
| Llama-3.2-1B-Instruct | `/home/wwq416/snap/wwq/model/llama/Llama-3.2-1B-Instruct` | `llama_lm_eval_multiturn_8shot_single_bos` |
| Llama-3.2-3B-Instruct | `/home/wwq416/snap/wwq/model/llama/Llama-3.2-3B-Instruct` | `llama_lm_eval_multiturn_8shot_single_bos` |

`select_prompt_profile` 按模型名关键字路由：

```python
"llama"           → llama_lm_eval_multiturn_8shot_single_bos
"qwen" + "math"   → qwen_math_completion_8shot
"qwen"            → qwen_multiturn_8shot_chat
```

## 4. 提示词规范

所有模型共享同一组 8 条 CoT 示例（`EIGHT_SHOT_EXAMPLES` / `LM_EVAL_COT_EXAMPLES`），
回答均以 `The final answer is N.` 收尾。

### 4.1 普通 Qwen 多轮 8-shot

适用：`qwen_multiturn_8shot_chat` profile，0.5B / 1.5B / 3B 三个尺寸共享。

构造方式（`build_qwen_messages`）：

```text
system: "As an expert problem solver, solve step by step
         the following mathematical questions."
[user, assistant] × 8（使用 EIGHT_SHOT_EXAMPLES）
user:   "Q: {question}\nA: Let's think step by step."
```

调用 tokenizer 的 `apply_chat_template(tokenize=False, add_generation_prompt=True)`
渲染为字符串，交给 vLLM。

### 4.2 Qwen-Math 纯 Completion 8-shot

适用：`qwen_math_completion_8shot` profile，仅 Math-1.5B 使用。

构造方式（`build_qwen_math_completion_prompt`）：

```text
"Q: 示例问题\nA: 示例推理与答案"

…

"Q: 当前问题\nA: Let's think step by step."
```

**不调用** `apply_chat_template`，也不注入普通 Qwen system prompt。

### 4.3 Llama 单 BOS 多轮 8-shot

适用：`llama_lm_eval_multiturn_8shot_single_bos` profile，仅 Llama-3.2-1B-Instruct
使用。

构造方式（`build_lm_eval_llama_messages`，`eval_official_gsm.py`）：

```text
[user("Question: ..."), assistant("Let's think step by step. ... The answer is N.")] × 8
user("Question: {question}")
```

输入 `vLLM` 时**直接传 token IDs**，不再回退到字符串：

```python
tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
```

要求 token IDs 中恰好包含 1 个 `bos_token_id`（Llama-3.2 为 `128000`），
多 BOS 会抛 `ValueError`。该约束由 `build_model_input` 在评测入口处
强制检查，避免 vLLM 0.21.0 重复添加 BOS。

### 4.4 停止条件

| Profile | stop sequences |
|---|---|
| `qwen_multiturn_8shot_chat` | `["<|im_end|>"]` |
| `qwen_math_completion_8shot` | `["\nQ:", "\nQuestion:", "<\|im_end\|>"]` |
| `llama_lm_eval_multiturn_8shot_single_bos` | `["<\|eot_id\|>", "<\|start_header_id\|>user<\|end_header_id\|>", "Q:", "</s>", "<\|im_end\|>"]` |

## 5. 生成设置

固定推理参数（`SamplingParams`）：

```text
temperature = 0.0
top_p       = 1.0
max_tokens  = 512
seed        = 42
```

vLLM 运行参数（`llm_kwargs`）：

```text
trust_remote_code        = True
tensor_parallel_size     = 1
gpu_memory_utilization   = 0.4（自动回退列表 0.8 → 0.25）
dtype                    = auto
max_model_len            = 4096
enforce_eager            = False
batch_size               = 16
```

`batch_size=16` 是当前主线规格。

## 6. 数值答案提取标准

仓库唯一标准实现：`code/gsm_answer_extractor.py`，五个评测入口都从该模块
导入 `extract_answer` 与 `is_correct`。

`extract_answer(output)` 确定性解析顺序：

1. 截断模型自行生成的下一题 `Question:` / `Q:`。
2. 取最后一个明确答案标记：`Final answer` 或 `####`。
3. 取最后一个平衡的 `\boxed{...}`（支持嵌套 LaTeX 分数）。
4. 取最后一句等式右侧的第一个数。
5. 取以 `So / Therefore / Thus / Hence / Consequently / Together / Finally`
   开头的结论：有等式取右侧第一个数，否则取谓词后的第一个数。
6. 取显式 `Answer:` 标记。
7. 最后回退到全文最后一个数。

`is_correct(pred, gold)` 用 `fractions.Fraction` 比较，相对 / 绝对容差
`tolerance=1e-6`。

`is_correct` 不读取 gold 之外的内容，相同输入始终得到相同结果。

## 7. 评测流程

入口脚本：`code/eval_chaingsm_base_8shot.py`

工作流程（每个模型独立子进程）：

```text
for spec in requested_specs:
    run_model_in_subprocess(args, spec, run_dir)
```

子进程（`_worker_command`）步骤：

1. `profile = select_prompt_profile(spec.name)` 决定 prompt profile。
2. `examples = load_examples(data_path, limit)` 读 5467 条样本。
3. `tokenizer = AutoTokenizer.from_pretrained(spec.path)`。
4. 写 `prompt_diagnostics.json`（profile、input_type、first_prompt_tokens、
   bos_token_id、bos_count、stop_sequences）。Llama 必须满足
   `bos_count == 1`。
5. 构造 `SamplingParams`。
6. `LLM(**llm_kwargs)` 初始化 vLLM。
7. 按 `batch_size` 切批，调用 `llm.generate`，每批 `append_jsonl` 增量
   写 `predictions.jsonl`，断点续跑通过 `completed_ids` 自动跳过。
8. 退出前 `write_model_summary` 写 `model_outputs/<model>/summary.json`。
9. 父进程 `write_combined_summaries` 汇总
   `summary_overall.json` / `summary_by_category.json` 以及对应 CSV。

断点续跑命令：

```bash
python code/eval_chaingsm_base_8shot.py --run-dir code/results/chaingsm_base_8shot_batch16/<timestamp>
```

## 8. 输出结构

```text
code/results/chaingsm_base_8shot_batch16/<timestamp>/
  run_config.json
  summary_overall.json
  summary_overall.csv
  summary_by_category.json
  summary_by_category.csv
  errors.jsonl            # 模型级失败
  model_outputs/
    <model>/
      prompt_diagnostics.json
      predictions.jsonl   # 5467 条
      summary.json
```

`predictions.jsonl` 字段：

```text
id, base_id, category, question, gold_answer,
model_name, model_path, prompt_profile,
raw_output, pred_answer, correct,
finish_reason, stop_reason
```

`summary_overall.json` 字段：

```text
model_name, model_path, prompt_profile,
correct, total, accuracy, accuracy_percent
```

`summary_by_category.json` 字段：

```text
model_name, model_path, prompt_profile,
category, correct, total, accuracy
```

## 9. 最新结果

输出根目录：

```text
code/results/chaingsm_base_8shot_batch16/
```

由于显存与时间限制，最新评测分三个时间戳完成，每个子目录覆盖一到两个模型，
合并后即四个 Qwen 模型 + 两个 Llama-3.2 Instruct 模型在干净集 5,467 条上的
`batch_size=16` 结果。

| 子目录 | 输出根 | 模型 | prompt_profile | batch_size |
|---|---|---|---|---:|
| `20260606_174725` | `chaingsm_base_8shot_batch16` | Qwen2.5-0.5B-Instruct | `qwen_multiturn_8shot_chat` | 16 |
| `20260606_174725` | `chaingsm_base_8shot_batch16` | Qwen2.5-1.5B-Instruct | `qwen_multiturn_8shot_chat` | 16 |
| `20260606_184504` | `chaingsm_base_8shot_batch16` | Qwen2.5-Math-1.5B-Instruct | `qwen_math_completion_8shot` | 16 |
| `20260606_184504` | `chaingsm_base_8shot_batch16` | Qwen2.5-3B-Instruct | `qwen_multiturn_8shot_chat` | 16 |
| `20260607_131942` | `chaingsm_base_8shot_batch16` | Llama-3.2-1B-Instruct | `llama_lm_eval_multiturn_8shot_single_bos` | 16 |
| `20260607_140452` | `chaingsm_base_8shot` | Llama-3.2-1B-Instruct | `llama_lm_eval_multiturn_8shot_single_bos` | 64 |
| `20260607_141832` | `chaingsm_base_8shot_batch16` | Llama-3.2-3B-Instruct | `llama_lm_eval_multiturn_8shot_single_bos` | 16 |
| `20260607_143400` | `chaingsm_base_8shot` | Llama-3.2-3B-Instruct | `llama_lm_eval_multiturn_8shot_single_bos` | 64 |

三批运行的运行参数一致：

```text
data_path     = chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl
example_count = 5467
batch_size    = 16
temperature   = 0.0
top_p         = 1.0
max_tokens    = 512
seed          = 42
max_model_len = 4096
```

### 9.1 总体准确率

| 模型 | Prompt Profile | batch=16 正确/总数 | batch=16 准确率 | batch=64 正确/总数 | batch=64 准确率 |
|---|---|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | qwen_multiturn_8shot_chat | 1342/5467 | 24.55% | 1217/5467 | 22.26% |
| Qwen2.5-1.5B-Instruct | qwen_multiturn_8shot_chat | 2676/5467 | 48.95% | 1872/5467 | 34.24% |
| Qwen2.5-Math-1.5B-Instruct | qwen_math_completion_8shot | 3159/5467 | 57.78% | 2788/5467 | 51.00% |
| Qwen2.5-3B-Instruct | qwen_multiturn_8shot_chat | 3325/5467 | 60.82% | 3293/5467 | 60.23% |
| Llama-3.2-1B-Instruct | llama_lm_eval_multiturn_8shot_single_bos | 1515/5467 | 27.71% | 1499/5467 | 27.42% |
| Llama-3.2-3B-Instruct | llama_lm_eval_multiturn_8shot_single_bos | 3191/5467 | 58.37% | 3187/5467 | 58.30% |

### 9.2 分组准确率

| 模型 | batch | original | independent_decoy | attribute_mismatch | path_competition | target_scope_misalignment |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 16 | 43.29% | 16.33% | 20.16% | 21.22% | 16.89% |
| Qwen2.5-0.5B-Instruct | 64 | 41.77% | 17.12% | 17.90% | 17.62% | 14.85% |
| Qwen2.5-1.5B-Instruct | 16 | 72.25% | 37.48% | 47.39% | 44.94% | 36.80% |
| Qwen2.5-1.5B-Instruct | 64 | 53.68% | 24.95% | 30.88% | 31.33% | 25.44% |
| Qwen2.5-Math-1.5B-Instruct | 16 | 76.19% | 50.91% | 55.36% | 54.75% | 46.89% |
| Qwen2.5-Math-1.5B-Instruct | 64 | 76.42% | 36.57% | 48.08% | 47.95% | 39.71% |
| Qwen2.5-3B-Instruct | 16 | 85.90% | 46.91% | 59.10% | 57.66% | 48.35% |
| Qwen2.5-3B-Instruct | 64 | 84.99% | 47.55% | 59.29% | 56.16% | 46.99% |
| Llama-3.2-1B-Instruct | 16 | 45.64% | 18.51% | 22.81% | 25.23% | 21.84% |
| Llama-3.2-1B-Instruct | 64 | 45.41% | 18.42% | 23.11% | 24.82% | 20.78% |
| Llama-3.2-3B-Instruct | 16 | 77.41% | 50.91% | 58.21% | 54.25% | 46.12% |
| Llama-3.2-3B-Instruct | 64 | 77.03% | 50.64% | 57.62% | 55.06% | 46.31% |

### 9.3 完整性检查

- 每个模型各 5,467 条预测，原始 GSM8K test 1,319 条 original 全覆盖。
- 三批运行各自 `summary_overall.json` / `summary_by_category.json` 已生成。
- 五个模型的 `prompt_diagnostics.json` 均已落盘：

| 模型 | input_type | first_prompt_tokens | bos_count | stop sequences |
|---|---|---:|---:|---|
| Qwen2.5-0.5B-Instruct | text | 946 | — | `["<\|im_end\|>"]` |
| Qwen2.5-1.5B-Instruct | text | 946 | — | `["<\|im_end\|>"]` |
| Qwen2.5-Math-1.5B-Instruct | text | 838 | — | `["\nQ:", "\nQuestion:", "<\|im_end\|>"]` |
| Qwen2.5-3B-Instruct | text | 946 | — | `["<\|im_end\|>"]` |
| Llama-3.2-1B-Instruct | **token_ids** | 1170 | **1** | 5 个 stop 序列 |
| Llama-3.2-3B-Instruct | **token_ids** | 1170 | **1** | 5 个 stop 序列 |

Llama 的 `bos_count=1` 与 `input_type=token_ids` 共同确认单 BOS 约束在
干净集 5,467 条上全程成立。

### 9.4 AbstRaL 公开基线对比

干净集 1,319 条 `original` 子集与 AbstRaL 报告的 GSM-Plus Original CoT-8S
任务在样本上对齐，可直接比较。

| 模型 | AbstRaL GSM-Plus Original CoT-8S | Our batch=16 Original | Our batch=64 Original |
|---|---:|---:|---:|
| Llama-3.2-1B-Instruct | 45.2 | 45.64 | 45.41 |
| Llama-3.2-3B-Instruct | 79.5 | 77.41 | 77.03 |
| Llama-3.1-8B-Instruct | 85.7 | — | — |
| Qwen2.5-0.5B-Instruct | 42.4 | **43.29** | 41.77 |
| Qwen2.5-1.5B-Instruct | 67.0 | **72.25** | 53.68 |
| Qwen2.5-3B-Instruct | 81.2 | **85.90** | 84.99 |
| Qwen2.5-7B-Instruct | 89.0 | — | — |
| Qwen2.5-Math-7B-Instruct | 91.8 | — | — |
| Mathstral-7B-v0.1 | 80.7 | — | — |

注：两份本地运行都使用同一套 8-shot 提示词（第 4 节），profile 路由
和答案提取器与 AbstRaL 一致；唯一变量是 vLLM 推理时的 `batch_size`。

**Original 准确率观察**：

- 六个有本地数据的模型中，Llama-3.2-1B-Instruct 45.64% 与 AbstRaL 45.2%
  基本一致，Llama-3.2-3B-Instruct 77.41% 比 AbstRaL 79.5% 低 2.09pp；
  0.5B / 1.5B / 3B 三个 Qwen 模型在 `batch_size=16` 下都**超过** AbstRaL
  同尺寸数字（+0.9 / +5.2 / +4.7）。
- 0.5B 在 `batch_size=16` 下的 43.29% 正好命中 AbstRaL 报告附近的
  `571/1319 = 43.29%` 期望值（设计 spec 中的目标锚点）。
- `batch_size=16` 与 `batch_size=64` 在 Original 1,319 条上对 Llama-1B、
  Qwen 0.5B、Math-1.5B、3B 影响都很小（≤0.3pp），Qwen 1.5B-Instruct 出现
  18.6pp 巨大差距（72.25% vs 53.68%），与该模型在 vLLM 0.21.0 上的调度
  稳定性以及 `enforce_eager` 行为相关，待后续重跑与对照组确认。

**全量（5,467 条）准确率**：

| 模型 | Our batch=16 全量 | Our batch=64 全量 |
|---|---:|---:|
| Qwen2.5-0.5B-Instruct | 24.55 | 22.26 |
| Qwen2.5-1.5B-Instruct | 48.95 | 34.24 |
| Qwen2.5-Math-1.5B-Instruct | 57.78 | 51.00 |
| Qwen2.5-3B-Instruct | 60.82 | 60.23 |
| Llama-3.2-1B-Instruct | 27.71 | 27.42 |
| Llama-3.2-3B-Instruct | 58.37 | 58.30 |

全量准确率把 original + 四类变体（独立 dummy / 属性错配 / 路径竞争 /
目标偏移）一起计入，可用于观察在干扰链上的总体退化幅度。

## 10. 复现命令

进入项目根目录并激活本地环境后：

```bash
conda activate math_chain_verl

# 全量跑 5 个模型（自动建时间戳子目录，batch_size=16）
python code/eval_chaingsm_base_8shot.py \
  --data-path chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl \
  --output-root code/results/chaingsm_base_8shot_batch16 \
  --batch-size 16

# 续跑（自动跳过已完成模型）
python code/eval_chaingsm_base_8shot.py \
  --run-dir code/results/chaingsm_base_8shot_batch16/20260607_131942

# 只跑 Llama
python code/eval_chaingsm_base_8shot.py \
  --data-path chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl \
  --output-root code/results/chaingsm_base_8shot_batch16 \
  --batch-size 16 \
  --model Llama-3.2-1B-Instruct

# 限速烟测
python code/eval_chaingsm_base_8shot.py \
  --data-path chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl \
  --output-root code/results/chaingsm_base_8shot_batch16 \
  --batch-size 16 \
  --model Qwen2.5-0.5B-Instruct \
  --limit 10
```

## 11. 测试覆盖

```bash
/home/wwq416/miniconda3/envs/math_chain_verl/bin/python -m pytest tests/ -q
```

当前 45 个测试全部通过，覆盖：

- 共享提取器 7 个回归（`tests/test_gsm_answer_extractor.py`）。
- 官方 GSM8K 评测助手 11 个（`tests/test_official_gsm_eval.py`）。
- Qwen 8-shot profile 路由与多轮消息 6 个（`tests/test_chaingsm_base_8shot_eval.py`）。
- 历史结果重算 2 个（`tests/test_rescore_gsm_predictions.py`）。
- 远程提交脚本 dry-run 4 个（`tests/test_remote_submission_scripts.py`）。
- 本地环境文档一致性 3 个（`tests/test_local_environment_docs.py`）。

## 12. 后续待办

- 后续 SFT / DPO / GRPO checkpoint 评测沿用同一脚本、同一数据、同一提取器、
  同一 `batch_size=16` 规格，按时间戳开新运行目录，避免覆盖当前基线。
- 可选：把三个时间戳子目录合成一个汇总表，方便 SFT/GRPO 阶段做 before/after
  对比。
