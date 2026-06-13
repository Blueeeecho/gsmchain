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

所有模型共享同一组 8 条 CoT 示例 (`EIGHT_SHOT_EXAMPLES` / `LM_EVAL_COT_EXAMPLES`),回答均以 `The final answer is N.` 收尾。

### 4.1 普通 Qwen 多轮 8-shot

适用:`qwen_multiturn_8shot_chat`,0.5B / 1.5B / 3B 共享。

构造 (`build_qwen_messages`):

```text
system: "As an expert problem solver, solve step by step
         the following mathematical questions."
[user("Q: 示例题\nA:"), assistant("示例推理 ... The final answer is N.")] × 8
user: "Q: {question}\nA: Let's think step by step."
```

调用 `tokenizer.apply_chat_template(tokenize=False, add_generation_prompt=True)`,vLLM 接字符串。

### 4.2 Qwen-Math 纯 Completion 8-shot

适用:`qwen_math_completion_8shot`,仅 Math-1.5B。

构造 (`build_qwen_math_completion_prompt`):

```text
"Q: 示例题\nA: 示例推理 ... The final answer is N."

"Q: {question}\nA: Let's think step by step."
```

不调用 `apply_chat_template`,不注入普通 Qwen system prompt。

### 4.3 Llama 单 BOS 多轮 8-shot

适用:`llama_lm_eval_multiturn_8shot_single_bos`,仅 Llama-3.2-1B/3B。

构造 (`build_lm_eval_llama_messages`,`eval_official_gsm.py`):

```text
[user("Given the following problem, reason and give a final answer to the problem.\nProblem: {example_question}\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."),
 assistant("Let's think step by step. ... The final answer is N.")] × 8
user("Given the following problem, reason and give a final answer to the problem.\nProblem: {question}\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.")
```

vLLM 接 **token IDs**(`tokenize=True`),要求其中恰好 1 个 `bos_token_id`(Llama-3.2 = `128000`)。多 BOS 抛 `ValueError`。

### 4.4 停止条件

| Profile | stop sequences |
|---|---|
| `qwen_multiturn_8shot_chat` | `["<\|im_end\|>"]` |
| `qwen_math_completion_8shot` | `["\nQ:", "\nQuestion:", "<\|im_end\|>"]` |
| `llama_lm_eval_multiturn_8shot_single_bos` | `["<\|eot_id\|>", "<\|start_header_id\|>user<\|end_header_id\|>", "Q:", "</s>", "<\|im_end\|>"]` |

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
