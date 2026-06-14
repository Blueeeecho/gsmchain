# v5 step 100 评测启动 + v6 准备就绪 (2026-06-14 05:55)

> 关键决策: v5 训练在 step 103 终止 (撞天花板信号明显), v6 全部就绪待启动

---

## 1. v5 训练终止决策

### 1.1 终止原因

| 现象 | 数据 | 解读 |
|---|---|---|
| 训练内 liveness 1.0 持续 | step 101/103 都是 1.0 | **松绑过头**, 跟 v2 失败模式 (0.89-0.97) 信号一致 |
| 训练内 accuracy 震荡 | step 101=0.75 → step 103=0.31 | 50% 区间 noise, 单步 8 sample 数据不显著 |
| entropy 收窄 | v3 step 100=0.056 → v5 step 101=0.026 | policy 收窄, 训练不健康 |
| 0.5B + JSON schema 协议天花板 | v3 best 30.47% / 29.95% Original (300 步撞平台) | v5 在同协议下继续微调, 撞天花板是必然 |
| v6 是更优路径 | 0.5B + 8-shot CoT 原生 43.29% (无 RL) | **v6 协议换完, 起点水位 +13pp** |

### 1.2 终止动作

- **kill v5 训练 (PID 3323532)**: SIGTERM 发送, ray cleanup 自动跑, GPU 完全释放 (19GB → 0.5GB)
- **保留 v5 step 100 ckpt**: `outputs/train/local/grpo_verl_lbprm_v5/Qwen2.5-0.5B-Instruct/grpo_verl_v5/20260614_052636/checkpoints/global_step_100/actor/huggingface` (actor/huggingface 完整, 可独立 eval)
- **手动启动 v5 step 100 eval**: 独立 vLLM 进程, 跑 5467 条 5 类目评测, 跟 v3 best 直接对比

---

## 2. v5 step 100 评测启动 (2026-06-14 05:49)

### 2.1 评测命令

```bash
V5_CKPT=/home/wwq416/snap/wwq/math-chain/outputs/train/local/grpo_verl_lbprm_v5/Qwen2.5-0.5B-Instruct/grpo_verl_v5/20260614_052636/checkpoints/global_step_100/actor/huggingface
python -m train_pipeline.eval_vllm_chaingsm \
  --model-path "$V5_CKPT" \
  --data-path /home/wwq416/snap/wwq/math-chain/chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl \
  --output-dir /home/wwq416/snap/wwq/math-chain/outputs/train/local/grpo_verl_lbprm_v5/Qwen2.5-0.5B-Instruct/grpo_verl_v5/20260614_052636/eval/step_100 \
  --method train_json_prompt \
  --batch-size 16 \
  --gpu-memory-utilization 0.3 \
  --max-tokens 2048 \
  --top-k 1
```

### 2.2 评测进程

- PID: 3447100
- 启动时间: 2026-06-14 05:49
- 预计完成: 2026-06-14 06:00 (实际 10-11 min)
- 日志: `/tmp/v5_step100_eval.log`
- 输出: `eval/step_100/eval_result.json` + `predictions.jsonl`

### 2.3 评测进度 (05:52)

- predictions.jsonl: 3.7MB (约 1000/5467 条, 18%)
- 预估剩余时间: 8-10 min
- GPU: 10.2GB / 32GB, util 99%

---

## 3. v5 训练内 step 100 详细 metrics (前 100 步)

### 3.1 step 1 vs step 100 对比

| 指标 | step 1 | step 100 (last 10 步) | Δ |
|---|---:|---:|---:|
| accuracy | 0.500 | 0.475 (均值) | -0.025 (几乎没动) |
| reward | 0.625 | 0.567 (均值) | -0.058 |
| answer | 0.500 | 0.475 | -0.025 |
| c2a | 0.4375 | 0.475 | +0.038 |
| liveness | 1.0 (松绑后) | 0.99 | 几乎满 |
| step_calc | 1.0 | 0.97 | -0.03 |
| no_degenerate | 1.0 | 0.92 | -0.08 |
| chain_quality | 1.0 | 0.97 | -0.03 |
| gated_chain_quality | 0.4375 | 0.475 | +0.038 |
| response_length | 161 | 195 | +34 |
| entropy | 0.351 | 0.026 | **-0.325 (收窄严重)** |
| kl_loss | 0.0 | 0.0007 | - |
| length_bonus_flag | 0.5 | 0.5 | 0 |

### 3.2 v5 vs v3 step 1-100 mean 对比

| 指标 | v3 step 1-50 | v3 step 51-100 | v4 step 1-50 | v4 step 51-100 | v5 step 1-50 | v5 step 51-100 |
|---|---:|---:|---:|---:|---:|---:|
| accuracy | 0.422 | 0.426 | 0.393 | 0.407 | 0.490 | 0.475 |
| format | 0.981 | 0.99 | 0.981 | 0.99 | 1.0 | 1.0 |
| answer | 0.422 | 0.426 | 0.393 | 0.407 | 0.490 | 0.475 |
| c2a | 0.458 | 0.456 | 0.431 | 0.435 | 0.555 | 0.475 |
| liveness | 0.346 | 0.346 | 0.346 | 0.346 | **0.975** | **0.99** |
| step_calc | 0.890 | 0.890 | 0.890 | 0.900 | 0.928 | 0.97 |
| no_degenerate | 0.932 | 0.932 | 0.932 | 0.948 | 0.981 | 0.92 |
| response_length | 214 | 192 | 210 | 197 | 192 | 195 |

**v5 step 1-50 看起来比 v3 同期强** (accuracy 0.490 vs 0.422), 但 **step 51-100 掉回 0.475 (跟 v3 同期持平)**.

**这印证了 v3 报告 §4 的判断**: v3 在 300 步撞平台, v5 在 100 步内已经接近天花板.

---

## 4. v6 启动计划 (v5 eval 完后立即)

### 4.1 v6 准备状态

| 文件 | 状态 |
|---|---|
| `docs/superpowers/specs/2026-06-14-lbprm-v6-design.md` | ✅ 200 行 |
| `docs/superpowers/plans/2026-06-14-lbprm-v6-plan.md` | ✅ 157 行 |
| `train_pipeline/reward_chaingsm_lbprm_v6_verl.py` | ✅ 292 行 |
| `train_pipeline/test_lbprm_v6.py` | ✅ **26/26 全过** |
| `train_pipeline/preprocess_chaingsm_8shot_cot.py` | ✅ 164 行 |
| `train_scripts/local/run_grpo_verl_lbprm_v6.sh` | ✅ |
| `verl_grpo_train_8shot_cot.parquet` | ✅ 7419 条, prompt 长度 921-1029 |

### 4.2 v6 启动动作

```bash
# v5 step 100 eval 完后, 立即启动 v6
RUN_NAME=grpo_verl_v6 \
  bash train_scripts/local/run_grpo_verl_lbprm_v6.sh 2>&1 | tee /tmp/v6_run.log
```

### 4.3 预期 v6 时间表

- **smoke test** (5 步 + 1 次 eval): ~10 min (验证 v6 reward + 8-shot CoT 拼装正确)
- **完整 400 步训练**: 400 * 10s = ~67 min
- **5 次 eval** (step 80/160/240/320/400): 5 * 5 min = ~25 min
- **总计**: ~1h45min

### 4.4 v6 期望结果

- 0.5B + 8-shot CoT 原生 Original 43.29% (无 RL)
- v6 400 步后 Original 预期 **45-55%** (RL 涨 2-12pp)
- 0.46 目标 = **0.46 overall / 0.46 Original**
- v6 整体预期 **35-45%** (4 类变体涨 10-20pp, Original 涨 2-12pp)

---

## 5. 决策记录 (2026-06-14 05:55)

- ✅ 2026-06-14 05:26: v5 完整 500 步训练启动
- ✅ 2026-06-14 05:48: v5 step 100 ckpt 已保存
- ✅ 2026-06-14 05:48: v5 step 101 训练内 accuracy 0.75 (单步 noise 大)
- ✅ 2026-06-14 05:49: v5 step 100 eval 独立启动 (PID 3447100)
- ✅ 2026-06-14 05:53: **v5 训练 SIGTERM 终止 (PID 3323532)**
- ✅ 2026-06-14 05:53: GPU 完全释放 (19GB → 0.5GB)
- ✅ 2026-06-14 05:50: v6 spec/plan/reward/test/preprocess/entry 全部就绪 (26/26 行为测试通过)
- ⏳ 2026-06-14 06:00: v5 step 100 eval 预计完成
- ⏳ 2026-06-14 06:00: v5 step 100 eval 结果分析 + v5 报告落盘
- ⏳ 2026-06-14 06:05: v6 smoke test 启动 (5 步 + 1 eval)
- ⏳ 2026-06-14 06:15: v6 smoke test 完成
- ⏳ 2026-06-14 06:15: v6 完整 400 步训练启动
- ⏳ 2026-06-14 07:25: v6 训练完成
- ⏳ 2026-06-14 07:50: v6 5 次 eval 全部完成
- ⏳ 2026-06-14 08:00: v6 报告 + 0.46 目标达成判断

---

**结论**:
- **v5 训练终止** (撞天花板信号明显, 保留 step 100 ckpt 可独立 eval)
- **v6 全部就绪** (26/26 行为测试通过, 7419 条训练数据已生成)
- **v5 step 100 eval 即将完成**, **v6 smoke test 紧随其后**
- **v6 是用户约束 (0.5B) 唯一可达 0.46 的路径**
