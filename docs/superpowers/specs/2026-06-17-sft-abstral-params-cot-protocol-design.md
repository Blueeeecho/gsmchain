# SFT (AbstRaL 超参) + COT 双尖括号协议 设计

> **日期**: 2026-06-17
> **作者**: 主线接力 (Brainstorming with user)
> **目的**: 抛弃 v9/v8.2 那套 STEP 模板/8-shot CoT 路线, 改用新协议
> (`<<expression = value>>` + `<<FINAL: expression = answer>>` + `ANSWER:`),
> 完全照搬 AbstRaL 论文 SFT 超参, 在本地单卡 RTX 5090 32GB 跑 2 epoch SFT
> (起点 = Qwen2.5-0.5B-Instruct 原生 base), 然后用同协议评测
> (不用 8-shot, 不用 train_json_prompt / NEUTRAL prompt)。

---

## 1. 目标

| 维度 | 数值 / 行为 |
|---|---|
| 主线目标 | 把 0.5B base + 新 CoT 协议训到 SFT ckpt, 在 5,467 条干净集上 **5 类** + **overall** 数字全部跑出来 |
| 起点 | `Qwen2.5-0.5B-Instruct` 原生 (无任何 LoRA / SFT 前置) |
| 数据 | `chaingsm_data/data/final/sft/all_sft_cot.jsonl` (6094 条) |
| 协议 | system + user 模板 (与训练时同源), 自然语言 + `<<expr=val>>` + `<<FINAL:>>` + `ANSWER:` |
| 训练 epoch | 2 (AbstRaL 论文) |
| 评测 | 2 epoch 训完后跑 1 次同协议 5 类评测; 每 epoch 末存 checkpoint 但不跑 eval (按用户要求"训完再跑") |

**显式不做**:
- 不接 GRPO (本期不训 v9 reward)
- 不接 8-shot CoT 评测
- 不接 NEUTRAL / train_json_prompt 评测
- 不动 verl, 不动 v9 reward

## 2. 训练超参 (AbstRaL 论文 → 本地单卡适配)

### 2.1 论文原值 (4 × A100 80GB)

| 项 | 值 |
|---|---|
| 训练数据 | GranulAR data, 6386 条 |
| 训练目标 | causal LM loss |
| 输入 | 抽象题目 X_A |
| 输出 | GranulAR 抽象答案 Y_A |
| 全局 batch size | 8 |
| GPU | 4 × A100 80GB |
| 每卡 batch size | 2 |
| learning rate | 5e-6 |
| optimizer | AdamW |
| β1, β2, ε | 0.9, 0.999, 1e-8 |
| epoch | 2 |

### 2.2 本地单卡 RTX 5090 32GB 适配 (1 卡)

| 项 | 论文 → 本地 |
|---|---|
| `per_device_train_batch_size` | 2 → **2** (RTX 5090 32GB 装得下, 不降) |
| `gradient_accumulation_steps` | 4 (论文) → **4** (4 × 2 = 8, 等价 effective batch=8) |
| 有效 batch size | 8 (= 2 × 4) |
| learning rate | 5e-6 (沿用, 不缩放) |
| optimizer | AdamW(0.9, 0.999, 1e-8) |
| epoch | 2 |
| `max_length` | 1024 (新数据 max 2498 字符 ≈ ≤ 800 tokens, 1024 留余量) |
| `packing` | **false** (chat 数据, completion_only_loss 才有意义) |
| `completion_only_loss` | **true** (只对 assistant 段计算 loss) |
| `bf16` | true |
| `gradient_checkpointing` | true |
| `warmup_ratio` | 0.1 (沿用仓库 sft.yaml 默认) |
| `lr_scheduler_type` | cosine (沿用) |
| `logging_steps` | 10 |
| `save_strategy` | "epoch" (每 epoch 存 1 个) |
| `save_total_limit` | 3 (epoch1 + epoch2 + best, 留余量) |
| 报告 | 仓库 sft.yaml 默认 `report_to: []` (不开 wandb) |

**显存估算**:
- 模型 0.5B ≈ 1GB bf16
- AdamW state 2 × = 2GB
- grad ckpt 激活 ≈ 0.5GB
- seq=1024 × batch=2 × chat 数据 ≈ 2GB
- 总计 ≈ 5.5GB, 远低于 32GB 限额
- **保守降级策略**: 若 OOM, 退回 `per_device_train_batch_size=1, grad_accum=8`, 显存预算减半

## 3. 数据契约

### 3.1 输入文件

- 路径: `chaingsm_data/data/final/sft/all_sft_cot.jsonl`
- 大小: 6,094 条
- 类目分布 (已验证):

  | 类目 | 数量 |
  |---|---:|
  | original | 3,051 |
  | attribute_mismatch | 905 |
  | independent_decoy | 837 |
  | path_competition | 811 |
  | target_scope_misalignment | 490 |

- schema (已验证):
  ```json
  {
    "id": "gsm8k_train_000015_original_sft",
    "source_id": "gsm8k_train_000015_original",
    "category": "original",
    "messages": [
      {"role": "system", "content": "You are a careful ..."},
      {"role": "user", "content": "Solve the following ..."},
      {"role": "assistant", "content": "TARGET: hours\n\nFirst, ...\n<<8 * 3 = 24>>\n...\n<<FINAL: 120 / (8 * 3) = 5>>\nANSWER: 5"}
    ],
    "reference": {...}
  }
  ```
- 6,093/6,094 包含 `TARGET:` / `<<FINAL:` / `ANSWER:` 三件套; 1 条缺, 已统计可见但不影响主流程 (容差)。

### 3.2 加载路径 (要点)

`train_pipeline/train_sft_trl.py` 现有 `load_rows` **只支持 `prompt`/`completion` 两列**。
要按用户选择 A: **在 `load_rows` 中加 `messages` 分支**, 当行内有 `messages` 字段时直接保留 `messages`, TRL `SFTTrainer` 原生识别为 chat data。

```python
def load_rows(path, max_samples=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            if "messages" in row:
                # TRL 1.5.x SFTTrainer 原生支持 messages 列
                rows.append({"id": row.get("id"), "messages": row["messages"]})
            else:
                # 老 prompt / completion 路径保留
                rows.append({
                    "id": row.get("id"),
                    "prompt": row["prompt"],
                    "completion": row.get("completion") or row.get("response", ""),
                })
            if max_samples and len(rows) >= max_samples: break
    return rows
```

**关键**: 不要 `apply_chat_template` 在外侧拼字符串, TRL 内部会自己处理 chat template + completion_only_loss。

## 4. 评测契约

### 4.1 测试集

- 路径: `chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl` (5,467 条, 5 类)
- 字段: `id / base_id / category / question_original / question_distracted / answer / ...`
- 加载逻辑: 已存在 `eval_vllm_chaingsm.py:load_examples` (按 category 选 `question_original` 或 `question_distracted`)。

### 4.2 新 method: `cot_brackets`

- system prompt: 与训练同源 (从 `all_sft_cot.jsonl` 首条 system 字段读出, 落到 `train_pipeline/eval_vllm_chaingsm.py` 的常量, 注释里标"与 SFT 训练时同源, 不要改")
- user prompt: 与训练同源 (从首条 user 字段读出, 模板化 `{question}` 占位)
- 评测解码参数: `temperature=0.0, top_k=1, top_p=1.0, max_tokens=512, batch_size=64` (与 sft.yaml 一致)
- argparse choices 追加 `cot_brackets`

### 4.3 答案提取

`code/gsm_answer_extractor.py:extract_answer` 现有七级回退**不包含 `<<FINAL:` 协议**。
在 `eval_vllm_chaingsm.py:extract_answer` (wrapper) 顶部新增**第一级**:

```python
# 1. 优先反扫 <<FINAL: expr = N>>
m = re.search(r"<<\s*FINAL\s*:[^>]*=\s*([-+]?\d[\d,\./\s]*)", text)
if m:
    candidate = m.group(1).strip().rstrip(".,;")
    # 用 NUMBER_PATTERN 再清洗一次, 防 "1,200" 这类
    nums = _numbers(candidate)
    if nums:
        return extract_text_answer(nums[0])
```

注意: 这个 wrapper 跟 `code/gsm_answer_extractor.py:extract_answer` **同名** (eval 内部已经 import 成 `extract_text_answer` 避免冲突), 改 wrapper 即可, 不动核心 extractor。

### 4.4 评测产物

- 路径: `outputs/sft_cot_eval/<model_dir>/latest_metrics.json` + `predictions.jsonl` + `summary_by_category.jsonl` + `summary_overall.jsonl`
- 字段格式: 与 `outputs/sft_v4/eval/3epoch/latest_metrics.json` 一致 (overall accuracy + 5 类桶)

## 5. 配置改动 (单文件)

`train_configs/local/sft.yaml`:
- `data.train_file` → `chaingsm_data/data/final/sft/all_sft_cot.jsonl`
- `training.max_length`: 3072 → **1024**
- `training.per_device_train_batch_size`: 1 → **2**
- `training.gradient_accumulation_steps`: 16 → **4**
- `training.num_train_epochs`: 1 → **2** (AbstRaL 论文)
- `training.learning_rate`: 2e-5 → **5e-6**
- `training.save_strategy`: "no" → **"epoch"**
- `training.save_total_limit`: 1 → **3**
- `eval.enabled`: true → **false** (按用户要求"训完再跑", 不在 epoch 中间 eval)
- `eval.data_path` / `method` 保留 (用于后续手动评测时套用同 YAML)

**YAML 不引入新 LoRA 字段**, `lora.enabled` 保持 false (AbstRaL 论文全参数微调)。

## 6. 入口脚本

复用 `train_scripts/local/run_sft.sh`, 透传环境变量:

```bash
DATA=/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/final/sft/all_sft_cot.jsonl \
CONFIG=/home/wwq416/snap/wwq/math-chain/train_configs/local/sft.yaml \
MODEL=/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct \
RUN_NAME=sft_cot_2ep \
RUN_ID=$(date +%Y%m%d_%H%M%S) \
PYTHON=/home/wwq416/miniconda3/envs/math_chain_verl/bin/python \
bash /home/wwq416/snap/wwq/math-chain/train_scripts/local/run_sft.sh
```

(用环境变量覆盖 `CONFIG` 让 YAML 改后跑的是新文件, 不污染旧的 rl_preprocessed 路径。)

## 7. 验证 (verification-before-completion)

训练完做以下检查 (失败必须修, 不允许"差不多就行"):

1. **ckpt 落盘**: `outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_cot_2ep/<RUN_ID>/checkpoints/`
   - 必有 `epoch1/`, `epoch2/`, `current/` (current = final), `best/` (按 eval overall 选)
2. **train_result.json 存在** 且 log_history 至少有 2 个 epoch 边界 (loss 收敛, train_loss 末值 < 起值 30%)
3. **评测产物齐全**: `outputs/sft_cot_eval/<RUN_DIR>/` 有
   - `latest_metrics.json` overall accuracy > 0
   - 5 类桶 `summary_by_category.jsonl` 全 5 行
   - 3 个 **GT 抽 1 题**做人工 sanity check (assistant 输出含 `<<FINAL:` 与 `ANSWER:` 两件套, 数值等于 gold)
4. **环境证据**: 训练 + 评测日志中, `python` path 与 `math_chain_verl` venv 一致
5. **不留 stale**: `__pycache__/` 不在仓库内 (`.gitignore` 已配, 但跑前 `find . -name __pycache__ -prune -exec rm -rf {} +`)

## 8. 风险与回退

| 风险 | 应对 |
|---|---|
| OOM at per_device=2 | 降到 per_device=1, grad_accum=8, 重新跑 |
| `messages` 列 TRL 不识别 | fallback: 在 SFTConfig 加 `dataset_kwargs={"skip_prepare_dataset": False}` + `max_seq_length=1024`; 还不行就退回方案 B (拼 prompt/completion) |
| loss 不收敛 (末值 > 起值) | 把 LR 调到 2e-6 重跑 (AbstRaL 论文的 2/5) |
| ckpt 太大磁盘撑爆 | 0.5B fp16 ≈ 1GB/epoch × 3 keep = 3GB, 远低于 `/home` 配额, 风险低 |
| `<<FINAL:>>` 提取器把 1.5e3 之类误抽 | 加正则"必须是 `=` 后纯数字 (无字母 e)" 防误伤; 若真实数据有 e 表示, 改 `_numbers` 容差 |

## 9. 不在本期

- GRPO 训练 (v9 reward / verl)
- 8-shot CoT 评测 / NEUTRAL 评测
- 数据再生成 (用现有 `all_sft_cot.jsonl` 不动)
- v8.1/v8.2 的 trace_overlap / irrelevant_eq 机制
- 远程服务器路径 (登录仍不可用)

## 10. 任务清单 (落到 writing-plans)

1. 改 `train_pipeline/train_sft_trl.py:load_rows` 支持 `messages` 分支
2. 改 `train_pipeline/eval_vllm_chaingsm.py`:
   - 加 `cot_brackets` method + system/user 常量
   - 加 `<<FINAL:>>` 反扫 wrapper
   - argparse choices 追加
3. 改 `train_configs/local/sft.yaml` (单文件 7 处)
4. 跑 SFT 2 epoch (单脚本)
5. 跑评测 1 次 (单脚本, 同协议)
6. 写总结报告 `docs/superpowers/reports/2026-06-17-sft-cot-abstral-params-report.md`
