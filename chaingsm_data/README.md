# ChainGSM Data 数据生成项目

`chaingsm_data` 用于基于 GSM8K 构造 chain-level distractor benchmark。项目会为每条基础样本保留原题，并生成四类链级干扰变体，便于后续做 original-vs-distracted 对比。

当前主线方案固定为：

```text
原版 verbose JSON prompt + OpenAI-compatible JSON mode + DeepSeek thinking disabled
```

不再保留 text generator 或 compact prompt 分支，避免后续误用。

## 当前数据源

默认输入文件：

```text
data/raw/test-00000-of-00001.jsonl
```

该文件包含 1319 条 GSM8K test 样本，字段为：

```json
{
  "base_id": "gsm8k_test_000001",
  "source_index": 1,
  "question": "...",
  "answer": "... #### 18",
  "final_answer": "18"
}
```

脚本也支持通过 `--input-path` 指定其他 JSONL 或 parquet 文件。

## 数据规模

对 `N` 条基础样本，完整生成规模为：

```text
N original + N × 4 generated variants = 5N records
```

四类 generated variants：

- `independent_decoy`
- `attribute_mismatch`
- `path_competition`
- `target_scope_misalignment`

test 集理论完整规模为：

```text
1319 original + 1319 × 4 variants = 6595 records
```

当前合并后的 test 数据文件为：

```text
data/final/gsm8k_test_full/gsm8k_test_all.jsonl
```

该文件的真实统计数量为 6575 条：

```text
original: 1319
independent_decoy: 1314
attribute_mismatch: 1316
path_competition: 1312
target_scope_misalignment: 1314
```

相对理论完整规模，当前 test 合并文件少 20 个 generated variants：

```text
independent_decoy: missing 5
attribute_mismatch: missing 3
path_competition: missing 7
target_scope_misalignment: missing 5
```

## 训练集单变体均衡生成

训练集输入文件为：

```text
data/raw/train-00000-of-00001.jsonl
```

该文件包含 7473 条 GSM8K train 基础样本，字段格式与 test 输入一致。训练集不按 test 全量设置为每个 base problem 生成 4 类变体，而是每个 base problem 只采样 1 个 distracted variant，同时在全局保持四类变体尽量均衡：

```text
7473 original + 7473 × 1 generated variant = 14946 records
```

四类变体的类别分配由 `--balanced-one-variant` 和 `--seed 42` 固定，完整训练集的目标类别数量为：

```text
original: 7473
independent_decoy: 1869
attribute_mismatch: 1868
path_competition: 1868
target_scope_misalignment: 1868
```

类别分配会写入 selected cache，便于断点续跑和复现：

```text
data/raw/selected_gsm8k_train_balanced_one_variant_all.jsonl
```

训练集生成沿用 test 数据生成的同一套 verbose JSON generator prompt、JSON mode、本地结构校验和最终 record schema；prompt 内容不在此重复。生成时同样关闭 DeepSeek thinking，避免 reasoning token 开销和空 `content` 问题。

推荐命令：

```bash
python generate_dataset.py \
  --mode full \
  --confirm-pilot-ok \
  --input-path data/raw/train-00000-of-00001.jsonl \
  --input-format jsonl \
  --run-name gsm8k_train_balanced_one_variant \
  --balanced-one-variant \
  --thinking disabled \
  --seed 42 \
  --max-workers 8 \
  --api-retries 2 \
  --max-retries 2 \
  --output-dir data/final/train_balanced_one_variant \
  --reports-dir reports/train_balanced_one_variant \
  --force
```

主要输出路径：

```text
data/final/train_balanced_one_variant/gsm8k_train_balanced_one_variant/gsm8k_train_balanced_one_variant_14946.jsonl
data/final/train_balanced_one_variant/gsm8k_train_balanced_one_variant/gsm8k_train_balanced_one_variant_14946.sorted.jsonl
reports/train_balanced_one_variant/gsm8k_train_balanced_one_variant/final_stats.json
reports/train_balanced_one_variant/gsm8k_train_balanced_one_variant/final_summary.md
reports/train_balanced_one_variant/gsm8k_train_balanced_one_variant/failed_generation.jsonl
```

旧 full run 中失败的 3090 个 sample+category 请求已提取到：

```text
data/raw/failed_requests_gsm8k_test_full_all.jsonl
```

这个文件必须保留，用于补齐旧实验失败项。

## 项目结构

```text
chaingsm_data/
  README.md
  requirements.txt
  generate_dataset.py
  src/
    llm_client.py
    prompts.py
    utils.py
    validators.py
    schemas.py
  data/
    raw/
    pilot/
    final/
  reports/
```

核心文件：

- `generate_dataset.py`：CLI、并发生成、实时写入、断点续跑、失败提取、summary。
- `src/llm_client.py`：OpenAI-compatible DeepSeek client，支持 JSON mode、thinking 开关、fatal error 早停。
- `src/prompts.py`：原版 verbose JSON generator prompt 和 validator prompt。
- `src/utils.py`：JSONL 读写、答案解析、样本选择、排序。
- `src/validators.py`：本地结构校验。
- `src/schemas.py`：类别、字段、默认 difficulty tags。

## 输出 JSONL 字段

每条最终 record 都包含：

```json
{
  "id": "gsm8k_test_000001_path_competition",
  "base_id": "gsm8k_test_000001",
  "source_index": 1,
  "category": "path_competition",
  "question_original": "...",
  "question_distracted": "...",
  "answer": "18",
  "solution_original": "...",
  "core_chain": [
    ["source_quantity", "target_quantity", "operation"]
  ],
  "distractor_chain": [
    ["source_quantity", "target_quantity", "operation"]
  ],
  "gold_expression": "string",
  "distractor_expression": "string",
  "difficulty_tags": {
    "entity_overlap": "low | medium | high | unknown",
    "operation_similarity": "low | medium | high | unknown",
    "answer_proximity": "far | near | same | unknown",
    "computational_complexity": "simple | multi_step | aggregate | unknown"
  },
  "metadata": {
    "generator_model": "deepseek-v4-flash",
    "seed": 42,
    "variant_type": "generated"
  }
}
```

最终 JSONL 不包含：

- `why_answer_unchanged`
- `distractor_answer`
- LLM validator 详情
- text-mode 中间格式

## 四类干扰定义

### independent_decoy

添加一条与原始核心解题链无关、但完整可计算的额外算术链。通常使用新人物、新物品或新场景。新增链不能改变原题答案。

### attribute_mismatch

添加一条与原题共享部分实体、但讨论不同属性或单位的干扰链。例如原题问钱，干扰链讨论书、票、积分；原题问年龄，干扰链讨论徽章、苹果等。

### path_competition

添加一条从原题起点、中间量、人物或物体分叉出去的竞争路径。它看起来像合理的继续计算路径，但不通向原题真正询问的目标。

### target_scope_misalignment

添加一条围绕目标实体之后、过去/未来时间、故事/假设语境展开的干扰链。真实当前目标和答案不变。

## Prompt

Prompt 实现在：

```text
src/prompts.py
```

当前只保留 JSON generator prompt。

Generator system message：

```text
You are a careful dataset generator for grade-school math word problems. Your job is to add one coherent arithmetic distractor chain to a given GSM8K problem without changing the original correct answer. The distractor must be natural, internally computable, and misleading, but it must not contradict the original problem or change the target quantity. You must also provide structured chain annotations.
```

Generator user message 模板：

```text
Given the original GSM8K problem, create one distracted version according to the specified category.

Original question:
{question}

Original solution:
{solution_original}

Original final answer:
{final_answer}

Category:
{category}

Category definitions:
- independent_decoy: add a separate arithmetic chain using mostly unrelated entities/items.
- attribute_mismatch: add a chain that may share entities but uses a different attribute or unit.
- path_competition: add a chain that branches from an original entity/quantity but leads to a non-target quantity.
- target_scope_misalignment: add a chain after the target, or in a different temporal/hypothetical/story scope, while the original target remains unchanged.

Rules:
1. Preserve the original problem and its correct answer.
2. The final answer must remain exactly: {final_answer}.
3. Add 1 to 4 natural sentences as the distractor chain.
4. The added distractor chain must be arithmetically coherent and computable.
5. Do not introduce contradictions.
6. Do not make the problem ambiguous.
7. Do not directly tell the solver that the added chain is irrelevant.
8. The generated question should still be a natural grade-school math word problem.
9. Keep the original question's target quantity unchanged.
10. The generated question should include the original problem content plus the added distractor chain.
11. Return JSON only.
12. All required fields must be present.
13. core_chain and distractor_chain must be lists of triples: [source_quantity, target_quantity, operation].
14. Use readable symbolic names such as "Li_age", "Zhang_age", "Jung_age", "Ming_badges".
15. operation should be a compact string such as "*2", "+3", "-5", "/2", "+out0", "aggregate_sum", or a short natural operation label if needed.
16. gold_expression should be a compact expression that computes the original final answer.
17. distractor_expression should be a compact expression that computes the plausible distractor answer.
18. Return one complete JSON object only.
19. Keep string values concise.
20. If the original problem has few named entities, branch from a known quantity such as an item count, rate, subtotal, time amount, or intermediate computed value.
21. Do not add facts that change the original computation conditions, such as changed rates, extra required materials, discounts, boosts, or new required quantities.
22. The generated question's final sentence should ask the original target quantity.

Return exactly this JSON schema:

{
  "question_distracted": "string",
  "answer": "{final_answer}",
  "core_chain": [
    ["source_quantity", "target_quantity", "operation"]
  ],
  "distractor_chain": [
    ["source_quantity", "target_quantity", "operation"]
  ],
  "gold_expression": "string",
  "distractor_expression": "string",
  "difficulty_tags": {
    "entity_overlap": "low | medium | high | unknown",
    "operation_similarity": "low | medium | high | unknown",
    "answer_proximity": "far | near | same | unknown",
    "computational_complexity": "simple | multi_step | aggregate | unknown"
  }
}

Important:
- The JSON must not contain markdown.
- The JSON must not contain comments.
- The JSON must be valid and parseable.
- Do not omit any field.
```

## 为什么禁用 thinking

实测 `deepseek-v4-flash` 在 thinking 开启或默认开启时，可能把 output token 消耗在 `reasoning_content`，导致 `content` 为空。禁用 thinking 后：

- 减少 completion token 消耗；
- 避免 `content=""`；
- 保持 JSON mode 输出更稳定；
- 速度更快。

因此所有推荐命令都使用：

```bash
--thinking disabled
```

## 安装依赖

```bash
cd /home/wwq416/snap/wwq/math-chain/chaingsm_data
pip install -r requirements.txt
```

## API 配置

默认：

```text
OPENAI_BASE_URL=https://api.deepseek.com
GENERATOR_MODEL=deepseek-v4-flash
VALIDATOR_MODEL=deepseek-v4-flash
```

环境变量会覆盖脚本默认值：

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.deepseek.com
export GENERATOR_MODEL=deepseek-v4-flash
export VALIDATOR_MODEL=deepseek-v4-flash
```

## Pilot

```bash
python generate_dataset.py \
  --mode pilot \
  --input-path data/raw/test-00000-of-00001.jsonl \
  --run-name gsm8k_test_pilot_json_no_think \
  --seed 42 \
  --with-validator \
  --thinking disabled
```

输出：

```text
data/pilot/gsm8k_test_pilot_json_no_think/pilot_samples.jsonl
data/pilot/gsm8k_test_pilot_json_no_think/pilot_samples.sorted.jsonl
reports/gsm8k_test_pilot_json_no_think/pilot_report.md
reports/gsm8k_test_pilot_json_no_think/pilot_failed_generation.jsonl
```

## Test 全量生成

```bash
python generate_dataset.py \
  --mode full \
  --input-path data/raw/test-00000-of-00001.jsonl \
  --run-name gsm8k_test_json_no_think_full \
  --seed 42 \
  --confirm-pilot-ok \
  --thinking disabled \
  --max-workers 4 \
  --max-retries 2 \
  --api-retries 1 \
  --generation-max-tokens 3000
```

输出：

```text
data/final/gsm8k_test_json_no_think_full/
reports/gsm8k_test_json_no_think_full/
```

## 提取旧失败请求

提取旧 full run 中仍未成功的失败项：

```bash
python generate_dataset.py \
  --mode extract-failures \
  --input-path data/raw/test-00000-of-00001.jsonl \
  --failed-log-path reports/gsm8k_test_full/failed_generation.jsonl \
  --success-path data/final/gsm8k_test_full/gsm8k_test_full_6595.jsonl \
  --failed-requests-path data/raw/failed_requests_gsm8k_test_full_all.jsonl
```

该命令已生成：

```text
data/raw/failed_requests_gsm8k_test_full_all.jsonl
```

文件包含 3090 条旧失败 sample+category 请求。不要删除。

## 3090 条旧失败案例补全

当前正在使用/推荐使用的补全命令：

```bash
python generate_dataset.py \
  --mode full \
  --input-path data/raw/failed_requests_gsm8k_test_full_all.jsonl \
  --run-name failed_retry_json_full_no_think_all3090 \
  --use-input-categories \
  --skip-originals \
  --thinking disabled \
  --seed 42 \
  --confirm-pilot-ok \
  --max-workers 4 \
  --max-retries 2 \
  --api-retries 1 \
  --generation-max-tokens 3000
```

预期输出为 3090 条 generated variants，不写 original：

```text
data/final/failed_retry_json_full_no_think_all3090/failed_retry_json_full_no_think_all3090_3090.jsonl
data/final/failed_retry_json_full_no_think_all3090/failed_retry_json_full_no_think_all3090_3090.sorted.jsonl
reports/failed_retry_json_full_no_think_all3090/final_summary.md
reports/failed_retry_json_full_no_think_all3090/final_stats.json
reports/failed_retry_json_full_no_think_all3090/failed_generation.jsonl
```

运行时间预估：

- `--max-workers 4` 实测约 50-55 条 / 分钟；
- 3090 条预计约 55-65 分钟；
- 中途断掉后，重跑同一命令即可断点续跑。

完成后检查：

```bash
wc -l data/final/failed_retry_json_full_no_think_all3090/failed_retry_json_full_no_think_all3090_3090.jsonl
cat reports/failed_retry_json_full_no_think_all3090/final_stats.json
```

理想情况：

```text
total_records = 3090
actual_variant_count = 3090
failed_count = 0
```

## 断点续跑

默认不加 `--force` 时，脚本会：

- 读取已有 output JSONL；
- 跳过已成功写入的 `id`；
- 继续生成未成功的 sample+category；
- 根据最终 output JSONL 重新生成 sorted 文件和 summary。

如果要从头重来，加：

```bash
--force
```

## 排序文件

实时写入顺序取决于并发完成顺序。脚本结束后会生成 `.sorted.jsonl`。

排序规则：

```text
base_id ascending
category order:
  original
  independent_decoy
  attribute_mismatch
  path_competition
  target_scope_misalignment
```

## 常用检查

```bash
wc -l data/final/failed_retry_json_full_no_think_all3090/failed_retry_json_full_no_think_all3090_3090.jsonl
cat reports/failed_retry_json_full_no_think_all3090/final_summary.md
cat reports/failed_retry_json_full_no_think_all3090/final_stats.json
cat reports/failed_retry_json_full_no_think_all3090/failed_generation.jsonl
```
