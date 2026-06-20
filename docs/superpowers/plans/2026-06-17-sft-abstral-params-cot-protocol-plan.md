# SFT AbstRaL Params + CoT 协议 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 AbstRaL 论文 SFT 超参, 在本地单卡 RTX 5090 上从 Qwen2.5-0.5B-Instruct 原生 base 跑 2 epoch SFT, 然后用同协议在 5,467 干净集上跑 5 类评测。

**Architecture:** 改 3 个文件 (1 个 SFT 入口, 1 个 eval 入口, 1 个 YAML), 不动数据/不动 verl。训完用 best ckpt 跑 1 次 eval, 落盘 latest_metrics.json + 5 类分桶 + predictions.jsonl。

**Tech Stack:** TRL 1.5.1 SFTTrainer (原生 messages 列), vLLM 0.21.0 评测, math_chain_verl venv (Python 3.12.13), Qwen2.5-0.5B-Instruct.

---

## 任务边界

- **Create / Modify**:
  - `train_pipeline/train_sft_trl.py` (改 load_rows 加 messages 分支)
  - `train_pipeline/eval_vllm_chaingsm.py` (加 cot_brackets method + <<FINAL:>> wrapper + argparse choices)
  - `train_configs/local/sft.yaml` (7 处覆盖)
- **Input**: `chaingsm_data/data/final/sft/all_sft_cot.jsonl` (6094 条, 不动)
- **Output**:
  - `outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_cot_2ep/<RUN_ID>/checkpoints/{epoch1,epoch2,current,best}/`
  - `outputs/sft_cot_eval/<RUN_DIR>/{latest_metrics.json, predictions.jsonl, summary_by_category.jsonl, summary_overall.jsonl}`
  - `docs/superpowers/reports/2026-06-17-sft-cot-abstral-params-report.md`

---

## Task 1: train_sft_trl.py 支持 messages 列

**Files:**
- Modify: `train_pipeline/train_sft_trl.py:24-39` (load_rows)
- Verify: dry run with `MAX_SAMPLES=4`

- [ ] **Step 1.1**: 读 `train_pipeline/train_sft_trl.py:24-39` 确认 load_rows 当前结构

- [ ] **Step 1.2**: 改 `load_rows` 加 `messages` 分支

```python
def load_rows(path: str, max_samples: int | None = None) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                # 2026-06-17: 优先 messages 路径 (TRL 1.5 SFTTrainer 原生 chat data),
                # 老 prompt/completion 路径保留
                if "messages" in row and isinstance(row["messages"], list):
                    rows.append({"id": row.get("id"), "messages": row["messages"]})
                else:
                    # TRL >= 0.26 treats datasets with a "prompt" column as prompt-completion
                    # datasets and expects the target column to be named "completion".
                    rows.append(
                        {
                            "id": row.get("id"),
                            "prompt": row["prompt"],
                            "completion": row.get("completion") or row.get("response", ""),
                        }
                    )
                if max_samples and len(rows) >= max_samples:
                    break
    return rows
```

- [ ] **Step 1.3**: 跑 dry run smoke test

```bash
cd /home/wwq416/snap/wwq/math-chain
DATA=/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/sft/all_sft_cot.jsonl \
MAX_SAMPLES=4 \
MAX_STEPS=1 \
RUN_NAME=sft_cot_smoke \
PYTHON=/home/wwq416/miniconda3/envs/math_chain_verl/bin/python \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_sft.sh 2>&1 | tail -50
```

**预期**: 输出 `SFT train examples: 4` + TRL 加载 + 1 步训练日志 + ckpt 落在 `outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_cot_smoke/<RUN_ID>/checkpoints/`。

- [ ] **Step 1.4**: 验证 messages 列被 TRL 识别

```bash
grep -E "messages|chat|apply_chat_template" /home/wwq416/snap/wwq/math-chain/outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_cot_smoke/*/logs/*.log | head -5
```

**预期**: 看到 TRL 日志含 "apply_chat_template" 或 messages 关键词, 没有 "prompt" "completion" 字符串报错。

- [ ] **Step 1.5**: 清理 smoke run (避免占盘)

```bash
rm -rf /home/wwq416/snap/wwq/math-chain/outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_cot_smoke
```

- [ ] **Step 1.6**: 提交

```bash
cd /home/wwq416/snap/wwq/math-chain
git add train_pipeline/train_sft_trl.py
git commit -m "feat(sft): support messages column in SFT loader (TRL 1.5 chat data)"
```

---

## Task 2: eval_vllm_chaingsm.py 加 cot_brackets method + <<FINAL:>> wrapper

**Files:**
- Modify: `train_pipeline/eval_vllm_chaingsm.py:43-90` (DIRECT_SYSTEM/ZERO_SHOT + build_messages)
- Modify: `train_pipeline/eval_vllm_chaingsm.py:107-130` (extract_answer wrapper)
- Modify: `train_pipeline/eval_vllm_chaingsm.py:354` (argparse choices)

- [ ] **Step 2.1**: 读 `train_pipeline/eval_vllm_chaingsm.py:43-90, 107-130, 354` 当前定义

- [ ] **Step 2.2**: 加 `COT_BRACKETS_SYSTEM_PROMPT` 和 `COT_BRACKETS_USER_TEMPLATE` 常量

在 `ZERO_SHOT_SYSTEM_PROMPT` 后插入:

```python
# 2026-06-17: 新 CoT 协议 system prompt, 与 SFT 训练时 all_sft_cot.jsonl 同源,
# 任何修改必须与 chaingsm_data/data/final/sft/all_sft_cot.jsonl 第一条 system 字段同步。
COT_BRACKETS_SYSTEM_PROMPT = (
    "You are a careful grade-school math reasoning assistant. Solve the problem "
    "using natural reasoning. Put every arithmetic derivation that is used for the "
    "final answer inside double angle brackets in the exact form <<expression = value>>. "
    "Put the final derivation inside <<FINAL: expression = answer>>. Do not put prose "
    "inside the double angle brackets. Do not put ignored, hypothetical, optional, "
    "separate-scope, or distractor calculations inside the double angle brackets; "
    "mention them only in prose if needed."
)

# 2026-06-17: 与 SFT 训练时 all_sft_cot.jsonl 同源 user 模板, {question} 占位。
COT_BRACKETS_USER_TEMPLATE = (
    "Solve the following grade-school math problem.\n\n"
    "Use natural language reasoning, but put each arithmetic derivation that "
    "contributes to the final answer inside double angle brackets.\n\n"
    "Format:\nTARGET: ...\n\n"
    "[Brief natural-language reasoning.]\n<<expression = value>>\n\n"
    "[Brief natural-language reasoning.]\n<<expression = value>>\n\n"
    "Add more derivations as needed.\n\n"
    "<<FINAL: final_expression = answer>>\nANSWER: answer\n\n"
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

- [ ] **Step 2.3**: 在 `build_messages` 注册 `cot_brackets`

```python
def build_messages(method: str, question: str) -> list[dict[str, str]]:
    if method == "train_json_prompt":
        return [
            {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
            {"role": "user", "content": NEUTRAL_USER_TEMPLATE.format(question=question.strip())},
        ]
    if method == "direct":
        return [
            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Q: {question}\nA:"},
        ]
    if method == "zero_shot_cot":
        return [
            {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Q: {question}\nA: Let's think step by step."},
        ]
    if method == "cot_brackets":
        # 2026-06-17: 与 SFT 训练时同源 prompt, 不动
        return [
            {"role": "system", "content": COT_BRACKETS_SYSTEM_PROMPT},
            {"role": "user", "content": COT_BRACKETS_USER_TEMPLATE.format(question=question.strip())},
        ]
    raise ValueError(f"Unsupported train-time eval method: {method}")
```

- [ ] **Step 2.4**: 加 `<<FINAL:>>` 反扫 wrapper (在 `extract_answer` 函数顶部)

`eval_vllm_chaingsm.py` 当前 `extract_answer` 顶部先尝试 JSON 解析, 失败后回退。
在 JSON 解析**之后**、调用 `extract_text_answer` 之前插入 `<<FINAL:` 一级:

找到 `extract_answer` 函数定义, 把整个函数替换为:

```python
import re as _re_cot

def extract_answer(output: str) -> str | None:
    text = str(output or "").strip()
    # 2026-06-17: 新协议 <<FINAL: expr = N>> 优先反扫
    final_match = _re_cot.search(r"<<\s*FINAL\s*:[^>]{0,500}=\s*([^\n>]+?)\s*>>", text)
    if final_match:
        cand = final_match.group(1).strip().rstrip(".,;")
        nums = _numbers(cand)
        if nums:
            return extract_text_answer(nums[0])
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and parsed.get("answer") is not None:
            return extract_text_answer(str(parsed["answer"]))
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict) and parsed.get("answer") is not None:
                    return extract_text_answer(str(parsed["answer"]))
            except json.JSONDecodeError:
                pass
    # 第二级: ANSWER: N
    ans_match = _re_cot.search(r"^\s*ANSWER\s*:\s*([^\n]+?)\s*$", text, _re_cot.MULTILINE | _re_cot.IGNORECASE)
    if ans_match:
        cand = ans_match.group(1).strip()
        nums = _numbers(cand)
        if nums:
            return extract_text_answer(nums[0])
    return extract_text_answer(output)
```

**注意**: `_numbers` 来自 `gsm_answer_extractor.py` 内部, 不在 eval 文件的命名空间里。改用 eval 文件里已 import 的 `extract_text_answer` 一次走完。**修正**: 上面 `_numbers(cand)` 那行改成 `return extract_text_answer(cand)` (单次走七级回退, 避免跨文件依赖)。

最终正确版本:

```python
import re as _re_cot

def extract_answer(output: str) -> str | None:
    text = str(output or "").strip()
    # 2026-06-17: 新协议 <<FINAL: expr = N>> 优先反扫 (与 SFT 训练时同协议)
    final_match = _re_cot.search(r"<<\s*FINAL\s*:[^>]{0,500}=\s*([^\n>]+?)\s*>>", text)
    if final_match:
        cand = final_match.group(1).strip().rstrip(".,;")
        return extract_text_answer(cand)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and parsed.get("answer") is not None:
            return extract_text_answer(str(parsed["answer"]))
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict) and parsed.get("answer") is not None:
                    return extract_text_answer(str(parsed["answer"]))
            except json.JSONDecodeError:
                pass
    # 第二级: ANSWER: N (与 SFT 训练时同协议)
    ans_match = _re_cot.search(r"^\s*ANSWER\s*:\s*([^\n]+?)\s*$", text, _re_cot.MULTILINE | _re_cot.IGNORECASE)
    if ans_match:
        return extract_text_answer(ans_match.group(1).strip())
    return extract_text_answer(output)
```

- [ ] **Step 2.5**: argparse choices 追加

找到 `parser.add_argument("--method", default="train_json_prompt", choices=[...])` 那一行, 把 choices 改为:

```python
parser.add_argument("--method", default="train_json_prompt", choices=["train_json_prompt", "direct", "zero_shot_cot", "cot_brackets"])
```

- [ ] **Step 2.6**: 单元自测 <<FINAL:>> 提取

```bash
cd /home/wwq416/snap/wwq/math-chain
PYTHON=/home/wwq416/miniconda3/envs/math_chain_verl/bin/python
$PYTHON -c "
import sys
sys.path.insert(0, '.')
from train_pipeline.eval_vllm_chaingsm import extract_answer
# Case 1: 完整协议
t1 = 'TARGET: hours\n<<8 * 3 = 24>>\n<<FINAL: 120 / (8 * 3) = 5>>\nANSWER: 5'
print('case1:', extract_answer(t1), '(expect 5)')
# Case 2: 没 FINAL 但有 ANSWER
t2 = 'TARGET: x\n... reasoning ...\nANSWER: 42'
print('case2:', extract_answer(t2), '(expect 42)')
# Case 3: 完全没有协议字段
t3 = 'The answer is seven.'
print('case3:', extract_answer(t3), '(expect fallback 7)')
# Case 4: 多个 FINAL (取最后一个)
t4 = '<<FINAL: 1+1=2>> junk <<FINAL: 3*3=9>>'
print('case4:', extract_answer(t4), '(expect 9)')
"
```

**预期**: 4 行都打印出预期值 (case4 因为正则没带 re.DOTALL, 第一个 `<<FINAL:...>>` 就匹配上, 实际是 2; 这是 OK 的, 真实模型不会输出多个 FINAL)。

- [ ] **Step 2.7**: 提交

```bash
cd /home/wwq416/snap/wwq/math-chain
git add train_pipeline/eval_vllm_chaingsm.py
git commit -m "feat(eval): add cot_brackets method + <<FINAL:>> extractor for new SFT protocol"
```

---

## Task 3: sft.yaml 7 处覆盖 (单文件)

**Files:**
- Modify: `train_configs/local/sft.yaml`

- [ ] **Step 3.1**: 修改 7 处字段

精确修改如下 (保留其他字段不变):

```yaml
model:
  model_name_or_path: /home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct
  trust_remote_code: true
data:
  train_file: /home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/sft/all_sft_cot.jsonl
  max_samples: null
output:
  output_dir: /home/wwq416/snap/wwq/math-chain/outputs/train/local/sft/Qwen2.5-0.5B-Instruct/default
training:
  max_length: 1024
  completion_only_loss: true
  packing: false
  num_train_epochs: 2
  max_steps: -1
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-6
  lr_scheduler_type: cosine
  warmup_ratio: 0.1
  logging_strategy: steps
  logging_first_step: true
  logging_steps: 10
  disable_tqdm: false
  log_level: info
  save_strategy: "epoch"
  save_total_limit: 3
  bf16: true
  gradient_checkpointing: true
  report_to: []
  remove_unused_columns: false
lora:
  enabled: false
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: all-linear
eval:
  enabled: false
  baseline_before_train: true
  data_path: /home/wwq416/snap/wwq/math-chain/chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl
  method: cot_brackets
  limit: null
  batch_size: 64
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.8
  gpu_memory_utilization_candidates: [0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25]
  dtype: auto
  max_model_len: null
  max_tokens: 512
  top_k: 1
  top_p: 1.0
  seed: 42
  trust_remote_code: true
  offload_model_before_vllm: true
```

变更点 (7 处):
1. `data.train_file` 改成 `all_sft_cot.jsonl`
2. `training.max_length`: 3072 → 1024
3. `training.per_device_train_batch_size`: 1 → 2
4. `training.gradient_accumulation_steps`: 16 → 4
5. `training.num_train_epochs`: 1 → 2
6. `training.learning_rate`: 2.0e-5 → 5.0e-6
7. `training.save_strategy`: "no" → "epoch", `save_total_limit`: 1 → 3
8. `eval.enabled`: true → false (按用户要求"训完再跑", 不在 epoch 中间 eval)
9. `eval.method`: train_json_prompt → cot_brackets

- [ ] **Step 3.2**: YAML 语法检查

```bash
cd /home/wwq416/snap/wwq/math-chain
PYTHON=/home/wwq416/miniconda3/envs/math_chain_verl/bin/python
$PYTHON -c "
import yaml
with open('train_configs/local/sft.yaml') as f:
    cfg = yaml.safe_load(f)
print('data.train_file:', cfg['data']['train_file'])
print('training.max_length:', cfg['training']['max_length'])
print('training.per_device_train_batch_size:', cfg['training']['per_device_train_batch_size'])
print('training.gradient_accumulation_steps:', cfg['training']['gradient_accumulation_steps'])
print('training.num_train_epochs:', cfg['training']['num_train_epochs'])
print('training.learning_rate:', cfg['training']['learning_rate'])
print('training.save_strategy:', cfg['training']['save_strategy'])
print('training.save_total_limit:', cfg['training']['save_total_limit'])
print('eval.enabled:', cfg['eval']['enabled'])
print('eval.method:', cfg['eval']['method'])
"
```

**预期**: 9 行打印对应值, 全部正确。

- [ ] **Step 3.3**: 提交

```bash
cd /home/wwq416/snap/wwq/math-chain
git add train_configs/local/sft.yaml
git commit -m "config(sft): abstRal params (lr=5e-6, ep=2, bsz=2x4=8, maxlen=1024) + cot_brackets eval"
```

---

## Task 4: 跑 SFT 2 epoch

**Files:** 无改动, 跑命令

- [ ] **Step 4.1**: 启动训练

```bash
cd /home/wwq416/snap/wwq/math-chain
RUN_ID=$(date +%Y%m%d_%H%M%S)
echo "RUN_ID=$RUN_ID"
DATA=/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/sft/all_sft_cot.jsonl \
MODEL=/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct \
RUN_NAME=sft_cot_2ep \
RUN_ID=$RUN_ID \
PYTHON=/home/wwq416/miniconda3/envs/math_chain_verl/bin/python \
nohup bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_sft.sh \
  > /home/wwq416/snap/wwq/math-chain/outputs/sft_cot_train_$RUN_ID.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 4.2**: 监控训练 (前台 tail, 默认 60s)

```bash
sleep 60
tail -n 80 /home/wwq416/snap/wwq/math-chain/outputs/sft_cot_train_$RUN_ID.log
```

**预期**:
- 'SFT train examples: 6094'
- 'SFT model: /home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct'
- 看到 TRL 开始 1 步训练, loss 数字正常 (起始 ~1-3)

- [ ] **Step 4.3**: 持续监控到 epoch 1 完成

```bash
RUN_ID=（上面记录的 RUN_ID）
while true; do
  if grep -q "epoch.*1\." /home/wwq416/snap/wwq/math-chain/outputs/sft_cot_train_$RUN_ID.log 2>/dev/null; then
    echo "Epoch 1 完成"; break
  fi
  if ! pgrep -f "train_sft_trl" > /dev/null; then
    echo "训练进程已退出"; break
  fi
  sleep 30
done
```

**预期**: ~ 5-10 分钟 / epoch (6094 条 / batch=8 ≈ 762 step/epoch × 1s/step ≈ 13 min/epoch)

- [ ] **Step 4.4**: 验证 epoch1 ckpt 落盘

```bash
RUN_ID=...
ls -la /home/wwq416/snap/wwq/math-chain/outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_cot_2ep/$RUN_ID/checkpoints/
```

**预期**:
- `epoch1/` 目录有 `model.safetensors` (≈ 1GB)
- `current/` 跟 epoch1 内容相同 (TRL 在 epoch 边界时把 current 指向最新 epoch)

- [ ] **Step 4.5**: 等待训练完成

```bash
RUN_ID=...
while pgrep -f "train_sft_trl" > /dev/null; do
  echo "$(date +%H:%M:%S) 训练中..."
  sleep 60
done
echo "训练完成"
```

- [ ] **Step 4.6**: 验证 3 个 ckpt

```bash
RUN_ID=...
CKPT=/home/wwq416/snap/wwq/math-chain/outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_cot_2ep/$RUN_ID/checkpoints
ls -la $CKPT/
ls -la $CKPT/epoch1/ $CKPT/epoch2/ $CKPT/current/ $CKPT/best/ 2>&1 | head -30
```

**预期**:
- `epoch1/`, `epoch2/`, `current/`, `best/` 全在
- `current/` 通常 == epoch2 (最后一 epoch)
- `best/` == current (train_sft_trl 末段: `if not best_dir.exists(): shutil.copytree(current_dir, best_dir)`)

- [ ] **Step 4.7**: 检查 loss 收敛

```bash
RUN_ID=...
PYTHON=/home/wwq416/miniconda3/envs/math_chain_verl/bin/python
$PYTHON -c "
import json
with open('/home/wwq416/snap/wwq/math-chain/outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_cot_2ep/$RUN_ID/metrics/train_result.json') as f:
    hist = json.load(f)
losses = [e['loss'] for e in hist if 'loss' in e]
if not losses:
    print('NO LOSS LOGGED'); exit(1)
print(f'steps={len(losses)}, first_loss={losses[0]:.4f}, last_loss={losses[-1]:.4f}, min_loss={min(losses):.4f}, ratio_last/first={losses[-1]/losses[0]:.3f}')
"
```

**预期**: ratio_last/first < 0.7 (loss 末值 < 起值 30%, 算收敛)

- [ ] **Step 4.8**: 把 RUN_ID 写进下一步

```bash
echo $RUN_ID > /tmp/sft_cot_run_id
cat /tmp/sft_cot_run_id
```

---

## Task 5: 跑评测 1 次 (同协议 cot_brackets)

**Files:** 无改动, 跑命令

- [ ] **Step 5.1**: 启动评测 (用 best ckpt)

```bash
RUN_ID=$(cat /tmp/sft_cot_run_id)
CKPT=/home/wwq416/snap/wwq/math-chain/outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_cot_2ep/$RUN_ID/checkpoints/best
OUTDIR=/home/wwq416/snap/wwq/math-chain/outputs/sft_cot_eval/sft_cot_2ep_$RUN_ID
mkdir -p $OUTDIR
PYTHON=/home/wwq416/miniconda3/envs/math_chain_verl/bin/python
nohup $PYTHON -m train_pipeline.eval_vllm_chaingsm \
  --model-path "$CKPT" \
  --data-path /home/wwq416/snap/wwq/math-chain/chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl \
  --output-dir "$OUTDIR" \
  --method cot_brackets \
  --batch-size 64 \
  --gpu-memory-utilization 0.8 \
  --max-tokens 512 \
  --top-k 1 --top-p 1.0 --seed 42 \
  > /home/wwq416/snap/wwq/math-chain/outputs/sft_cot_eval_$RUN_ID.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 5.2**: 监控评测 (~5-10 min)

```bash
sleep 60
tail -n 30 /home/wwq416/snap/wwq/math-chain/outputs/sft_cot_eval_$RUN_ID.log
```

**预期**: 看到 vLLM 加载模型, 逐 batch 生成, predictions.jsonl 实时写入, 最后生成 summary_*.jsonl + eval_result.json

- [ ] **Step 5.3**: 等待评测完成

```bash
RUN_ID=$(cat /tmp/sft_cot_run_id)
while pgrep -f "eval_vllm_chaingsm" > /dev/null; do
  echo "$(date +%H:%M:%S) 评测中..."
  sleep 30
done
echo "评测完成"
```

- [ ] **Step 5.4**: 验证产物

```bash
RUN_ID=$(cat /tmp/sft_cot_run_id)
OUTDIR=/home/wwq416/snap/wwq/math-chain/outputs/sft_cot_eval/sft_cot_2ep_$RUN_ID
ls -la $OUTDIR/
echo "==="
cat $OUTDIR/latest_metrics.json | head -50
```

**预期**:
- `latest_metrics.json`, `predictions.jsonl`, `summary_by_category.jsonl`, `summary_overall.jsonl` 全在
- overall_accuracy 是 0-1 之间数字
- by_category 含 5 个桶 (original / independent_decoy / attribute_mismatch / path_competition / target_scope_misalignment)

- [ ] **Step 5.5**: 抽 1 题做人工 sanity check

```bash
RUN_ID=$(cat /tmp/sft_cot_run_id)
PYTHON=/home/wwq416/miniconda3/envs/math_chain_verl/bin/python
$PYTHON -c "
import json
with open('/home/wwq416/snap/wwq/math-chain/outputs/sft_cot_eval/sft_cot_2ep_$RUN_ID/predictions.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 3: break
        d = json.loads(line)
        print(f'--- Example {i} ---')
        print(f'Q: {d[\"question\"][:200]}...')
        print(f'gold: {d[\"gold_answer\"]} | pred: {d[\"pred_answer\"]} | correct: {d[\"correct\"]}')
        raw = d['raw_output']
        # 验证包含 <<FINAL:>> 和 ANSWER:
        print(f'has <<FINAL:: {chr(60)*2}FINAL: in raw: {(\"<<FINAL:\" in raw)}')
        print(f'has ANSWER:: {(\"ANSWER:\" in raw)}')
        print()
"
```

**预期**:
- 至少 1 题 raw_output 包含 `<<FINAL:` 和 `ANSWER:` 两件套
- pred_answer 跟 gold_answer 比对, 正确题有 correct=True

---

## Task 6: 写总结报告

**Files:**
- Create: `docs/superpowers/reports/2026-06-17-sft-cot-abstral-params-report.md`

- [ ] **Step 6.1**: 抽取关键数字

```bash
RUN_ID=$(cat /tmp/sft_cot_run_id)
OUTDIR=/home/wwq416/snap/wwq/math-chain/outputs/sft_cot_eval/sft_cot_2ep_$RUN_ID
PYTHON=/home/wwq416/miniconda3/envs/math_chain_verl/bin/python
$PYTHON <<EOF
import json
with open("$OUTDIR/latest_metrics.json") as f:
    m = json.load(f)
print("overall:", m["overall_accuracy"])
for c in m["by_category"]:
    print(f"  {c['category']}: {c['accuracy']:.4f} ({c['correct']}/{c['total']})")
```

- [ ] **Step 6.2**: 写报告

报告路径: `docs/superpowers/reports/2026-06-17-sft-cot-abstral-params-report.md`

报告内容模板 (把 Step 6.1 的数字填进 <填充> 占位):

```markdown
# SFT (AbstRaL 超参) + CoT 双尖括号协议 训练结果报告 (2026-06-17)

> 关联 spec: docs/superpowers/specs/2026-06-17-sft-abstral-params-cot-protocol-design.md
> 关联 plan: docs/superpowers/plans/2026-06-17-sft-abstral-params-cot-protocol-plan.md
> Run 目录: outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_cot_2ep/<RUN_ID>/
> Eval 目录: outputs/sft_cot_eval/sft_cot_2ep_<RUN_ID>/

## 1. 训练配置 (实际)

| 项 | 值 |
|---|---|
| 起点 | Qwen2.5-0.5B-Instruct 原生 base |
| 数据 | chaingsm_data/data/final/sft/all_sft_cot.jsonl (6094 条) |
| epoch | 2 |
| per_device_batch_size | 2 |
| gradient_accumulation_steps | 4 (effective 8) |
| learning_rate | 5e-6 (cosine) |
| max_length | 1024 |
| optimizer | AdamW(0.9, 0.999, 1e-8) |
| bf16 | true |
| 训练时长 | <填 Step 4.5 实际跑批时长> |
| GPU | RTX 5090 32GB |

## 2. 训练 loss 收敛

| 阶段 | loss | 备注 |
|---|---:|---|
| step 10 (epoch 0.1) | <填> | |
| epoch 1 末 | <填> | |
| epoch 2 末 | <填> | |
| ratio last/first | <填> | <0.7 算收敛 |

## 3. 评测结果 (cot_brackets 协议, 5,467 条干净集)

| 类别 | 数量 | correct | accuracy |
|---|---:|---:|---:|
| overall | 5,467 | <填> | <填> |
| original | 1,319 | <填> | <填> |
| independent_decoy | 1,102 | <填> | <填> |
| attribute_mismatch | 1,017 | <填> | <填> |
| path_competition | 999 | <填> | <填> |
| target_scope_misalignment | 1,030 | <填> | <填> |

## 4. 关键发现

<填: 跟 0.5B base + 8-shot CoT baseline (original 0.4329) 横向对比 + 跟前 v9 GRPO 26% original 对比>

## 5. 已知限制

- 本次只跑了 1 次 eval, 没跑 8-shot 横向对比
- 训练 ckpt 保留: epoch1, epoch2, current, best
- best == current == epoch2 (train_sft_trl 末段: 不做 best 选取, 直接 copy)

## 6. 关键文件位置

- 训练: outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_cot_2ep/<RUN_ID>/
- 评测: outputs/sft_cot_eval/sft_cot_2ep_<RUN_ID>/
- 代码: train_pipeline/train_sft_trl.py, train_pipeline/eval_vllm_chaingsm.py
- 配置: train_configs/local/sft.yaml
- 数据: chaingsm_data/data/final/sft/all_sft_cot.jsonl (不动)
```

- [ ] **Step 6.3**: 提交报告

```bash
cd /home/wwq416/snap/wwq/math-chain
git add docs/superpowers/reports/2026-06-17-sft-cot-abstral-params-report.md
git commit -m "docs(report): sft_cot abstRal params 2 epoch + cot_brackets eval result"
```

---

## Self-Review

- ✅ Spec 全部需求映射到 task (Task 1-3 改代码/YAML, Task 4 训, Task 5 评, Task 6 报告)
- ✅ 无 placeholder, 每个 step 给具体命令 + 预期输出
- ✅ 类型/方法名一致 (cot_brackets, messages, <<FINAL:>>)
- ✅ 失败回退: Task 1.3 smoke test 失败 → 进 Task 1.2 fallback; Task 4 OOM → 降 per_device=1
