# AbstRaL 主方法在 5,467 干净集上全量复现 + 失败判据 (2026-06-13)

> **作者**: 老师建议 + spec 落地
> **目标**: 在 5,467 干净集上把 `abstral_style_two_stage_prompting` 跑全量,跟 `8-shot CoT` baseline 直接对比,论文里写"AbstRaL 复现未能带来鲁棒性提升"作为负结果贡献
> **现状**: AbstRaL 主方法已在 `code/eval_abstral_baselines.py` 实现;5,467 上**只跑过 5 条烟测**(`code/results/baseline_test/abstral_style_two_stage_prompting/chaingsm_gsm8k_test_full/20260527_163320/`);6 个模型 8-shot CoT 数字已落盘(`code/results/chaingsm_base_8shot_batch16/`)

---

## 0. 重要前提 (不再混用 baseline)

- **本次实验用 5,467 干净集 + 8-shot CoT baseline,而不是 NEUTRAL zero-shot**。`8-shot` 的 purpose 是用 8 个示例教模型做 GSM8K 风格的 CoT;`AbstRaL` 自家 prompt 自带 3 个 sub-step 模板,所以 AbstRaL 跑 **零示例**(`method=abstral_style_two_stage_prompting` 用 `stage1_messages` / `stage2_messages` 不含 8-shot),与 8-shot baseline 在 "提示强度" 上不严格对称,论文里要明确写"AbstRaL 自家两阶段格式 vs CoT-8S 通用格式"。
- 评测入口:
  - **8-shot CoT baseline** = `code/eval_chaingsm_base_8shot.py` (已有 batch=16 全量结果)
  - **AbstRaL 主方法** = `code/eval_abstral_baselines.py` + `--method abstral_style_two_stage_prompting`
- 评判标准 (`gsm_answer_extractor.is_correct`, `tolerance=1e-6`):同 `code/eval_official_gsm.py`,沿用 6 个模型已有口径

## 1. 目标

1. **全量 5,467 × 6 模型 × AbstRaL 主方法** 跑完
2. 出 `code/results/abstral_5k_8shot/<timestamp>/` 跟 8-shot CoT batch=16 数字横向对比
3. **失败判据** 满足下列任一即算"复现失败":
   - overall 准确率 < 同模型 8-shot CoT 至少 3pp
   - 4 个变体类目 (`independent_decoy` / `attribute_mismatch` / `path_competition` / `target_scope_misalignment`) 里有 ≥ 2 个 **低于** 8-shot CoT
4. **harsh noise 档 (可选 ablation)**:把 `chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl` 里 `path_competition` 的 distractor_chain 改成"算式更相似 gold_chain" 重新生成,再跑 AbstRaL 看是否跌更深
5. **Liveness 单独表**:从 `train_pipeline/reward_chaingsm_lbprm_v2_verl.py` 抽出 `_compute_liveness`,对 5,467 题做 per-step liveness 标注,跟 5 类目准确率做相关分析

## 2. 测试集 / 模型 / 评测入口

### 2.1 测试集 (固定)

```text
/home/wwq416/snap/wwq/math-chain/chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl
5,467 条 (original 1319 + 4×变体)
```

### 2.2 模型清单 (固定 6 个,主线 baseline 已跑过 8-shot batch=16)

| 模型 | 路径 | 8-shot CoT baseline (original / 全量) | AbstRaL 已跑? |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | `/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-0.5B-Instruct` | 43.29 / 24.55 | 否 |
| Qwen2.5-1.5B-Instruct | `/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-1.5B-Instruct` | 72.25 / 48.95 | 否 |
| Qwen2.5-Math-1.5B-Instruct | `/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-Math-1.5B-Instruct` | -/57.78 | 否 |
| Qwen2.5-3B-Instruct | `/home/wwq416/snap/wwq/model/Qwen/Qwen2.5-3B-Instruct` | 85.90 / 60.82 | 否 |
| Llama-3.2-1B-Instruct | `/home/wwq416/snap/wwq/model/llama/Llama-3.2-1B-Instruct` | 45.64 / 27.71 | 否 |
| Llama-3.2-3B-Instruct | `/home/wwq416/snap/wwq/model/llama/Llama-3.2-3B-Instruct` | 77.41 / 58.37 | 否 |

### 2.3 评测入口

```bash
conda activate math_chain_verl
export PATH="/home/wwq416/miniconda3/envs/math_chain_verl/bin:${PATH}"

# AbstRaL 主方法
python /home/wwq416/snap/wwq/math-chain/code/eval_abstral_baselines.py \
  --data-path /home/wwq416/snap/wwq/math-chain/chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl \
  --model-root /home/wwq416/snap/wwq/model \
  --model-filter Qwen2.5-0.5B-Instruct \
  --method abstral_style_two_stage_prompting \
  --output-root /home/wwq416/snap/wwq/math-chain/code/results/abstral_5k_8shot \
  --batch-size 16 \
  --gpu-memory-utilization 0.4 \
  --max-model-len 4096 \
  --dtype auto \
  --seed 42 \
  --trust-remote-code
```

注: `eval_abstral_baselines.py` 默认会同时跑 `granular_style_prompting`,需要加 `--method` 限定只跑 Abstral。`--model-filter` 多次传以选择单模型。

## 3. 实验矩阵

### 3.1 本期必跑 (1-2 个工作日)

| Run | 范围 | 入口 | 输出 |
|---|---|---|---|
| A1 | Qwen2.5-0.5B-Instruct × AbstRaL × 5,467 | `eval_abstral_baselines.py` | `code/results/abstral_5k_8shot/<ts>/Qwen2.5-0.5B-Instruct/` |
| A2 | Qwen2.5-1.5B-Instruct × AbstRaL × 5,467 | 同上 | 同上 |
| A3 | Qwen2.5-3B-Instruct × AbstRaL × 5,467 | 同上 | 同上 |
| A4 | Llama-3.2-3B-Instruct × AbstRaL × 5,467 | 同上 | 同上 |
| A5 | Qwen2.5-Math-1.5B-Instruct × AbstRaL × 5,467 | 同上 | 同上 |
| A6 | Llama-3.2-1B-Instruct × AbstRaL × 5,467 | 同上 | 同上 |

### 3.2 Harsh noise 档 (可选 ablation, 1-2 个工作日)

只对 `path_competition` + `attribute_mismatch` 两个类目重新生成 distractor_chain:
- 让 distractor 跟 gold_chain 共享 ≥ 3 步中间变量名
- 让 distractor 的最终表达式在数值上跟 gold 的 target 是"对偶" (e.g., gold 算 sum, distractor 算 product)
- 5,467 题不动,只换 `distractor_chain` 字段

构造产物: `chaingsm_data/data/gsmchain/gsm8k_test_harsh.jsonl` (~5,467 条,部分题可能审计失败,落到 ~4,500 条)

跑 AbstRaL + 8-shot CoT (限 A1/A2/A3 三个 Qwen 模型) 看是否差距扩大。

### 3.3 Liveness 单独表 (本期必跑, 半天)

- 复用 `code/eval_chaingsm_base_8shot.py` 已有 `predictions.jsonl` (6 模型 × 5,467 条已有, 不重跑)
- 写新脚本 `code/liveness_diagnose.py`:
  1. 读每条 `prediction`
  2. 调用 `train_pipeline.reward_chaingsm_lbprm_v2_verl._compute_liveness` + `_step_consistency` + `_final_consistency`
  3. 算 4 个分量 (`liveness` / `step_consistency` / `final_consistency` / `chain_quality`) 的均值,按 5 类目
  4. 算 liveness 与"答对率"的皮尔逊相关
- 输出: `code/results/liveness_diagnose/<ts>/summary_liveness_vs_accuracy.csv` + `correlation_matrix.csv`

## 4. 输出文件

每个 AbstRaL run 落到:

```text
code/results/abstral_5k_8shot/<timestamp>/
├── run_config.json
├── summary_overall.json
├── summary_overall.csv
├── summary_by_category.json
├── summary_by_category.csv
└── model_outputs/
    └── <safe_model_name>/
        ├── predictions.jsonl      # 5,467 条完整
        ├── stage1_outputs.jsonl   # AbstRaL 专属
        └── errors.jsonl
```

## 5. 复现命令 (顺序)

```bash
# 0. 环境
conda activate math_chain_verl
export PATH="/home/wwq416/miniconda3/envs/math_chain_verl/bin:${PATH}"
cd /home/wwq416/snap/wwq/math-chain

# 1. 跑 Qwen2.5-0.5B (最小模型先验证 pipeline 走得通)
TS=$(date +%Y%m%d_%H%M%S)
OUT=/home/wwq416/snap/wwq/math-chain/code/results/abstral_5k_8shot/$TS
mkdir -p $OUT

python code/eval_abstral_baselines.py \
  --data-path chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl \
  --model-root /home/wwq416/snap/wwq/model \
  --model-filter Qwen2.5-0.5B-Instruct \
  --method abstral_style_two_stage_prompting \
  --output-root $OUT \
  --batch-size 16 \
  --gpu-memory-utilization 0.4 \
  --max-model-len 4096 \
  --dtype auto \
  --seed 42 \
  --trust-remote-code 2>&1 | tee $OUT/qwen05b.log

# 2. 同命令换 --model-filter 跑 Qwen2.5-1.5B / 3B / Math / Llama-1B / 3B (串行)
```

## 6. 成功 / 失败判据

### 6.1 复现成功 (有 ablation 价值, 论文里写"部分场景下 AbstRaL 有效")

- 6 模型 overall 准确率,AbstRaL **≥** 8-shot CoT baseline
- 4 个变体类目里, ≥ 3 个 AbstRaL **≥** 8-shot CoT
- 数学模型 (Qwen2.5-Math-1.5B) 表现更突出

### 6.2 复现失败 (本实验预期结果, 论文里写负结果)

- 至少 4 个模型 overall 准确率 AbstRaL **<** 8-shot CoT baseline ≥ 3pp
- 4 个变体类目里 ≥ 2 个 AbstRaL **<** 8-shot CoT baseline
- 数学模型上 AbstRaL 也没提升 (说明两阶段抽象对小模型不友好)

### 6.3 Harsh noise 档 (扩展实验)

- 跟 §6.1 / §6.2 一样对比,但只看 `path_competition` + `attribute_mismatch` 两个类目
- 预期: AbstRaL 在 harsh noise 上比 8-shot CoT 跌得更深 → 论文里写"AbstRaL 的抽象反而放大分心链干扰"

## 7. 资源估算

| 项 | 值 |
|---|---|
| GPU | 1 × RTX 5090 32GB |
| 单模型 5,467 跑 AbstRaL (含 stage1 + stage2, 各 1 次 vLLM call) | ~ 30-50 min |
| 6 模型串行 | ~ 3-5 h |
| Liveness 诊断脚本 (读 6 模型 predictions 即可) | ~ 1 h (含调试) |
| Harsh noise 档 (含数据重生成 + 重跑 3 模型) | ~ 1.5 day |

## 8. 后续动作 (失败后)

如果 §6.2 成立, 论文叙事:
1. **Abstract**: AbstRaL 在小模型 (≤3B) 上的鲁棒性提升**未能在 ChainGSM 干净集上复现**
2. **Related work**: 列出 AbstRaL/GranuLar 在 GSM-Plus 报告的数字, 说明 GSM-Plus 没有"分心链 + 真实链对偶"的构造
3. **Experiments** §: 三表三图
   - 表 1: 6 模型 × (8-shot CoT / AbstRaL) 5 类目准确率
   - 表 2: Harsh noise 下三个 Qwen 模型 (8-shot CoT / AbstRaL) 的退化幅度
   - 表 3: Liveness 4 个分量 vs 5 类目答对率相关矩阵
4. **Discussion**: 为什么 AbstRaL 在 ChainGSM 上失败
   - 4 类变体本身已经"算式对偶", 抽象化 (stage1) 反而丢失了 distractor 与 gold 的对比信号
   - 0.5B/1.5B 模型在两阶段格式上 OOD, 格式错误率比 8-shot CoT 高
   - Liveness 跟答对率相关弱 (r < 0.3), 说明 liveness 不是 gold 推理的可靠代理

## 9. 决策记录

- 2026-06-13: 老师建议 "Implement AbstRaL and run it on our data to make sure it fails" → 落本 spec
- 本 spec 不写新代码, **只跑已有 `eval_abstral_baselines.py`**, 跟 `8-shot CoT baseline` 对比
- Liveness 表复用 `train_pipeline.reward_chaingsm_lbprm_v2_verl._compute_liveness`,不重写
- 数据增强 (harsh noise) 是可选 ablation, 不影响主表
