#!/usr/bin/env bash
# 一键查看 v9 训练实时状态
RUN_DIR=$(ls -td /home/wwq416/snap/wwq/math-chain/outputs/train/local/grpo_v9/Qwen2.5-0.5B-Instruct/grpo_v9/*/ 2>/dev/null | head -1)
LOG=$RUN_DIR/logs/run.log
METRICS=$RUN_DIR/metrics/train_metrics.jsonl

echo "=== run_id ==="; basename "$RUN_DIR"
echo "=== process ==="
ps -p $(pgrep -f verl.trainer.main_ppo | head -1) -o pid,etime,stat 2>/dev/null
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== progress ==="
grep "Training Progress" "$LOG" 2>/dev/null | tail -1
echo "=== step 1-最新 timing ==="
tail -50 "$METRICS" 2>/dev/null | python3 -c "
import json, sys
steps = []
for line in sys.stdin:
    d = json.loads(line)
    if d.get('step', 0) > 0 and 'timing_s/step' in d.get('data', {}):
        steps.append(d)
if not steps: print('  no steps yet'); exit()
last3 = steps[-3:]
for d in last3:
    data = d['data']
    print(f'  step {d[\"step\"]}: {data[\"timing_s/step\"]:.1f}s  reward={data.get(\"critic/score/mean\", 0):.3f}  acc={data.get(\"reward/accuracy\", 0):.3f}  core={data.get(\"reward_components/core/mean\", 0):.3f}')
" 2>&1
echo "=== ckpt 列表 ==="
ls -d "$RUN_DIR"/checkpoints/global_step_* 2>/dev/null | xargs -n1 basename
