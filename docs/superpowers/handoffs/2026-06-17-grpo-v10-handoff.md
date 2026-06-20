# GRPO v10-signed 训练 Handoff (中断于 2026-06-17 21:06)

## 当前状态

- **训练任务**: v10-signed GRPO 训练 500 步
- **运行 ID**: `20260617_205539`
- **进度**: **47 / 500 步完成** (~9.4%)
- **状态**: 中断（不是代码问题，是被同机器其他用户插入 GPU 占用导致 OOM）
- **最后一次成功状态**: step 47 metrics 写入完成，step 48 update_weights 时 vLLM `resume(wake_up)` 撞 CUDA OOM

## 中断原因（不是 bug）

- 训练开始时 GPU 32GB 全可用
- 训练进行到 5-10 分钟时，用户 `cw` 的 `agrifood/ECAN/run.py` 进程插入，**额外占用 18.6GB**
- 我们当时的总占用是 actor 2GB + vLLM 9.8GB (gpu_memory_utilization=0.3) = 11.8GB
- update_weights 期间 vLLM 试图重新激活 KV cache，**撞 32GB 上限 → OOM → 训练崩溃**

## 已保存的 47 步 metrics

- 路径: `outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/grpo_v10_signed/20260617_205539/metrics/train_metrics.jsonl`
- 格式: 47 行 JSONL，每行一个 step
- v10 奖励组件**全部正常工作**:
  - `critic/score/mean`: 1.99 (max 4.2, min 1.16, 区间内)
  - `reward/accuracy`: 0.35 (mean), 0.81 (max)
  - `reward_components/core/mean`: 0.59 (上升趋势)
  - `reward_components/distractor/mean`: 0.03 (有触发)

## 任务清单

- [x] v10 奖励实现 (`train_pipeline/reward_chaingsm_v10_verl.py`)
- [x] v10 奖励 6/6 单元测试通过 (`train_pipeline/test_reward_v10.py`)
- [x] GRPO 数据生成 (6094 行, `chaingsm_data/data/final/grpo/all_grpo_cot.jsonl` + `.parquet`)
- [x] v10 配置 (`train_configs/local/grpo_verl_v10.yaml`)
- [x] v10 启动脚本 (`train_scripts/local/run_grpo_verl_v10.sh`, 默认 0.5 mem util 抗插入)
- [ ] **500 步训练 (47/500 完成)**
- [ ] **eval at step 500 (中断后未跑)**
- [ ] **v10 报告 (`docs/superpowers/reports/2026-06-17-grpo-v10-signed-report.md`)**

## 配置文件

- `train_configs/local/grpo_verl_v10.yaml`: v10 配置 (model=SFT epoch3, data=GRPO parquet, reward=v10)
- `train_scripts/local/run_grpo_verl_v10.sh`: v10 启动脚本 (已改为 ROLLOUT_GPU_MEM_UTIL=0.5 防插入)
- `train_pipeline/reward_chaingsm_v10_verl.py`: v10 奖励函数 (signed, [-0.5, 4.2] 区间)
- `train_pipeline/test_reward_v10.py`: 6/6 PASS

## 评测方法

- 测试集: `chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl` (5467 行)
- 方法: `cot_brackets` (在 `train_pipeline/eval_vllm_chaingsm.py` 中已注册)

## SFT 起点 (GRPO 训练 baseline)

- SFT epoch3 ckpt: `outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_cot_2ep_resume/20260617_180315/checkpoints/checkpoint-762/`
- SFT epoch3 eval (baseline 已知):
  - overall: 26.87%
  - original: 31.16%

## 下次继续训练

1. **确认 GPU 空闲** (32GB 全可用):
   ```bash
   nvidia-smi --query-gpu=memory.used,memory.free --format=csv
   # 应该显示 ~0 MiB used, ~32GB free
   ```

2. **清理残留进程** (若 GPU 仍有 20GB+ 占用):
   ```bash
   pkill -9 -f "wwq416.*(verl|ray|run_grpo|vllm)"
   sleep 5
   nvidia-smi --query-gpu=memory.used --format=csv
   ```

3. **启动 v10 训练** (用 0.5 mem util 防插入):
   ```bash
   setsid bash -c '
     cd /home/wwq416/snap/wwq/math-chain
     export PATH="/home/wwq416/miniconda3/envs/math_chain_verl/bin:${PATH}"
     export CUDA_HOME="/home/wwq416/miniconda3/envs/math_chain_verl"
     export FLASHINFER_CUDA_ARCH_LIST="12.0f"
     export LD_LIBRARY_PATH="/home/wwq416/miniconda3/envs/math_chain_verl/lib:${LD_LIBRARY_PATH:-}"
     export CUDA_MODULE_LOADING=LAZY
     export TOTAL_GRPO_STEPS=500
     export SAVE_FREQ=100
     export RUN_NAME=grpo_v10_signed
     export TOTAL_EPOCHS=1
     export ROLLOUT_GPU_MEM_UTIL=0.5
     export EVAL_BASELINE=0
     export RUN_ID=$(date +%Y%m%d_%H%M%S)
     bash train_scripts/local/run_grpo_verl_v10.sh
   ' < /dev/null > /tmp/grpo_v10_launch4.log 2>&1 &
   disown
   ```

4. **监控训练**:
   ```bash
   # 实时跟踪
   tail -f /tmp/grpo_v10_launch4.log
   # 或
   tail -f /home/wwq416/snap/wwq/math-chain/outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/grpo_v10_signed/{最新RUN_ID}/metrics/train_metrics.jsonl
   ```

5. **训练完成后评测** (auto, 脚本会跑一次 cot_brackets eval 在 gsm8k_test_clean.jsonl):
   ```bash
   # 脚本会跑一次 eval on global_step_500
   # 结果在: outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/grpo_v10_signed/{RUN_ID}/eval/epoch_0001/eval_result.json
   ```

## 期望完成时间

- 500 步 × 11s/step (0.5 mem util 比 0.3 略慢) ≈ **92 分钟**
- + 1 次 eval ≈ 5 分钟
- 总计 ≈ **97 分钟**

## 风险

- 仍可能被他人的 GPU 任务插入并 OOM
- 若需要更激进防御，可加 CUDA 预占 tensor 主动 reserve 30GB（但需要改 vLLM 启动顺序，复杂）
