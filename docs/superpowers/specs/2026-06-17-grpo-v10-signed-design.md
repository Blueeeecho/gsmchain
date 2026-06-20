# GRPO v10-signed Reward 设计

> **日期**: 2026-06-17
> **作者**: 主线接力 (user 提需求)
> **目的**: 实现允许负奖励的 GRPO reward 函数 v10-signed, 适配新 CoT 协议
> (`<<expr=val>>` + `<<FINAL:>>` + `ANSWER:`), 在 SFT epoch3 末 ckpt 上跑 GRPO 训练.

---

## 1. 目标与基线

| 指标 | 值 |
|---|---:|
| 起点 ckpt | SFT epoch3 末 (overall 26.87% / original 31.16%) |
| 训练数据 | `chaingsm_data/data/final/grpo/all_grpo_cot.jsonl` (6094 条) |
| 测试集 | `chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl` (5467 条) |
| 框架 | verl + vLLM rollout |
| Reward | **v10-signed** (本 spec 主体) |

## 2. v10-signed 公式

```
R = 0.2·r_format + 2.5·r_answer + 1.2·r_core + 0.3·r_calc − 0.5·r_distractor
```

**理论范围**: `R ∈ [-0.5, +4.2]`

| 项 | 权重 | 范围 | 含义 |
|---|---:|---|---|
| r_format | +0.2 | [0,1] | 4 个 marker 齐备 (TARGET/STEP/FINAL/ANSWER) |
| r_answer | +2.5 | {0,1} | ANSWER == gold (Fraction 比较) |
| r_core | +1.2 | [0,1] | 0.7·sim_trace + 0.3·sim_final |
| r_calc | +0.3 | [0,1] | 0.8·r_step_calc + 0.2·r_final_calc (Fraction 等式自洽) |
| r_distractor | -0.5 | [0,1] | max(0, sim_distractor − r_core), original 类别 = 0 |

**关键差异 (vs v9)**:
1. **允许负奖励** (去掉 v9 的 `max(0.0, total)`)
2. **加 r_calc** (v9 没有)
3. **r_core 权重 1.5 → 1.2**, **sim_final 权重 0.2 → 0.3** (更稳)
4. **r_distractor 范围不变** (max(0, sim_dist - r_core) 避免误伤)
5. **新增 r_calc 0.3** (辅助项, 权重低于 answer + core)

## 3. 数据契约

### 3.1 训练数据 (parquet)
输入 JSONL: `chaingsm_data/data/final/grpo/all_grpo_cot.jsonl` (6094 条)

每条 schema:
```json
{
  "id": "gsm8k_train_000015_original_grpo",
  "source_id": "...",
  "category": "original" | "...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "reward_meta": {
    "format_version": "nattrace_v1",
    "answer": "5",
    "gold_expression": "120 / (8 * 3)",
    "gold_trace_tokens": ["8","*","3","=","24","<step>","120","/","(","8","*","3",")","=","5"],
    "distractor_expression": null | "...",
    "distractor_trace_tokens": [],
    "distractor_enabled": true | false,
    "category": "...",
    "source_id": "..."
  }
}
```

### 3.2 parquet 转换
verl 需要 parquet 格式. 转换脚本: `chaingsm_data/data/final/sft/all_grpo_to_verl_parquet.py`
(仓库已有模板, 改 `input_jsonl` 路径 + 字段映射即可)

**verl parquet 每行字段**:
- `prompt`: 拼接 system+user 的 chat 字符串 (tokenizer.apply_chat_template)
- `reward_model.ground_truth`: 把 reward_meta 整个塞进去 (v10 reward 从这里读)

### 3.3 参考读取路径
v9 reward 读 `reference["gold_trace_tokens"]` 等. v10 沿用同样路径.
**关键**: parquet 写入时 `ground_truth` 字段必须是 dict-like, 包含:
- `answer` / `gold_answer`
- `gold_expression`
- `gold_trace_tokens` (list[str])
- `distractor_expression`
- `distractor_trace_tokens` (list[str])
- `distractor_enabled`
- `category`

## 4. v10 reward 实现

### 4.1 文件位置
新文件: `train_pipeline/reward_chaingsm_v10_verl.py` (仿 v9, 但适配新协议)

### 4.2 复用与改造

| 组件 | 来源 | v10 改造 |
|---|---|---|
| `extract_answer` (ANSWER 反扫) | `code/gsm_answer_extractor` | 复用 |
| `is_correct` (Fraction 比较) | `code/gsm_answer_extractor` | 复用 |
| `Levenshtein` (edit distance) | python-Levenshtein | 复用 |
| `_normalized_edit_similarity` | v9 复制 | 复用 (加括号版本) |
| `_tokenize_expr` | v9 复制 | **改**: 不删括号 (跟 GRPO 数据带括号一致) |
| `_extract_pred_target` | v9 复制 | **改**: 协议从 `TARGET:` → v10 用 `TARGET:` (兼容) |
| `_extract_pred_final_expr` | v9 复制 | **改**: `<<FINAL:>>` 协议 |
| `_extract_pred_trace_tokens` | v9 复制 | **改**: `<<expr=val>>` 协议 |
| `_has_step_with_expr_value` | v9 复制 | **改**: 走新协议 |
| `r_distractor` 计算 | v9 复制 | 复用 (max(0, sim - r_core) 逻辑) |
| `r_calc` | **新** | 新加: 等式自洽性, Fraction 求值 |

### 4.3 关键正则 (新协议)

```python
# 协议 1: 普通 step 块 <<expr = value>> (单步, 不含 FINAL)
STEP_PATTERN = re.compile(r"<<\s*([^>]+?)\s*>>")
# 协议 2: FINAL 块 <<FINAL: expr = answer>>
FINAL_PATTERN = re.compile(r"<<\s*FINAL\s*:\s*([^>]+?)\s*>>")
# 协议 3: ANSWER: N
ANSWER_PATTERN = re.compile(r"^\s*ANSWER\s*:\s*([^\n]+?)\s*$", re.MULTILINE | re.IGNORECASE)
# 协议 4: TARGET: ...
TARGET_PATTERN = re.compile(r"^\s*TARGET\s*:\s*([^\n]+?)\s*$", re.MULTILINE | re.IGNORECASE)
```

### 4.4 tokenize (带括号)

```python
TOKEN_PATTERN = re.compile(r"\*\*|[()]|(?<![\d.])-?\d+(?:\.\d+)?|[+\-*/]")

def _tokenize_expr(expr: str) -> list[str]:
    if not expr: return []
    e = expr.replace(" ", "")
    return TOKEN_PATTERN.findall(e)
```

### 4.5 r_calc (新机制)

```python
def _is_math_correct(expr: str, value: str) -> bool:
    """eval(expr) == eval(value) 用 Fraction."""
    fe, fv = _safe_frac(expr), _safe_frac(value)
    if fe is not None and fv is not None:
        return fe == fv
    return False
```

对每个 step `<<expr = value>>`:
- r_step_calc = 1 if Fraction(eval(expr)) == Fraction(value) else 0
- 取所有 step 的平均值

对 FINAL `<<FINAL: expr = answer>>`:
- r_final_calc = 1 if Fraction(eval(expr)) == Fraction(answer) else 0

公式: `r_calc = 0.8 * r_step_calc + 0.2 * r_final_calc`

### 4.6 完整 score_response 骨架

```python
def score_response(completion, reference, ...):
    # 1. 解析 (4 个 marker + step trace + final expr + answer)
    # 2. r_format (4 个 marker / 4)
    # 3. r_answer (ANSWER == gold)
    # 4. r_core (0.7 * sim_trace + 0.3 * sim_final)
    # 5. r_calc (Fraction 自洽)
    # 6. r_distractor (max(0, sim_dist - r_core), original=0)
    # 7. R = 0.2*format + 2.5*answer + 1.2*core + 0.3*calc - 0.5*distractor
    #    **不再 max(0, R)** ← v10 signed 关键
    # 8. 返回 (R, metrics_dict)
```

## 5. 训练计划

### 5.1 起点 + 步数
- 起点: `outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_cot_2ep_resume/20260617_180315/checkpoints/checkpoint-762`
- MAX_STEPS: 500 (经验值, v9 500 步是 0.5B + SFT 真实最佳点)
- SAVE_FREQ: 100 (5 个 eval 节点)
- 训练集: all_grpo_cot.parquet (verl 格式)

### 5.2 verl 配置
- ROLLOUT_N: 4 (v9 用过, OK)
- KL_COEF: 0.02
- ACTOR_LR: 5e-7
- TEMPERATURE: 0.9
- 入口: `train_scripts/local/run_grpo_verl.sh` + `--reward-path train_pipeline/reward_chaingsm_v10_verl.py`

### 5.3 评测
- 5 个 eval 节点 (step 100/200/300/400/500)
- 测试集: gsm8k_test_clean.jsonl
- method: cot_brackets (同 SFT 评测协议)
- 重点: 跟 SFT epoch3 (overall 26.87 / original 31.16) 对比, 看 GRPO 涨不涨

## 6. 风险与回退

| 风险 | 应对 |
|---|---|
| r_calc 误判 (Fraction 解析失败) | _safe_frac 容错, 失败返 0 (不加分) |
| 协议解析不到 (模型输出奇怪格式) | r_format=0, r_calc=0, r_core 低, r_distractor 可能高 → R 可能负 |
| GRPO 500 步内撞 26% 天花板 (v9 历史) | 看 step 100/200 是否突破, 没有就停 |
| verl 1525 步死锁 (v9 2000 步报告) | 限制 500 步, 不会触发 |
| OOM at per_device=2 | 降到 1 |

## 7. 不在本期

- 不改 v9 reward (历史可比性)
- 不动 SFT 数据 / 训练 (epoch3 ckpt 已经是最佳点)
- 不写 8-shot / NEUTRAL 评测
- 不接远程服务器

## 8. 任务清单 (落到 plan)

1. 改 `chaingsm_data/data/final/sft/all_grpo_to_verl_parquet.py` (改 input + 字段映射)
2. 新建 `train_pipeline/reward_chaingsm_v10_verl.py` (实现 v10 reward)
3. 写单元测试 `train_pipeline/test_reward_v10.py` (5+ case)
4. 改 `train_configs/local/grpo_verl.yaml` (用 v10 reward, 步数 500)
5. 跑 GRPO 500 步 + 5 节点 eval
6. 写报告 `docs/superpowers/reports/2026-06-17-grpo-v10-signed-report.md`
