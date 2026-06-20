#!/usr/bin/env bash
# Watchdog: 等 GPU 空闲到 <8GB 使用时, 自动启动 v11 续训
# 检查频率 60s, 启动后 30 分钟无 OOM 视为稳
#
# 退出条件:
#   - 成功启动训练 (sleep 30min 持续存活)
#   - 启动后 30 分钟内崩, 回到 watch 循环 (最多 5 次, 避免无限 loop)
#
# 启动:
#   nohup bash train_scripts/local/watch_v11_resume.sh > /tmp/v11_watchdog.log 2>&1 &

set -u

ROOT="/home/wwq416/snap/wwq/math-chain"
LOG="/tmp/v11_watchdog.log"
STATE="/tmp/v11_resume_state"
TRAIN_SCRIPT="$ROOT/train_scripts/local/run_grpo_verl_v11_resume.sh"

GPU_FREE_THRESHOLD_MB=8000
CHECK_INTERVAL=60
STARTUP_GRACE_SEC=1800   # 启动后 30 分钟
MAX_RESTART=5

mkdir -p "$(dirname "$LOG")"
: > "$LOG"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

restart_count=0
if [[ -f "$STATE" ]]; then
  restart_count=$(cat "$STATE" 2>/dev/null || echo 0)
fi

log "=== v11 resume watchdog started ==="
log "ROOT=$ROOT"
log "GPU_FREE_THRESHOLD_MB=$GPU_FREE_THRESHOLD_MB"
log "CHECK_INTERVAL=${CHECK_INTERVAL}s"
log "TRAIN_SCRIPT=$TRAIN_SCRIPT"
log "previous restart_count=$restart_count"

while [[ $restart_count -lt $MAX_RESTART ]]; do
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  log "GPU free: ${free_mb} MiB (threshold ${GPU_FREE_THRESHOLD_MB})"

  if [[ $free_mb -ge $GPU_FREE_THRESHOLD_MB ]]; then
    log "GPU free enough, launching v11 resume"
    # 启动前再校验 ckpt
    RESUME_CKPT="$ROOT/outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/grpo_v11_stepvalue/20260618_134820/checkpoints/global_step_200"
    if [[ ! -d "$RESUME_CKPT/actor" ]]; then
      log "ERROR: resume ckpt missing at $RESUME_CKPT/actor, exiting"
      exit 1
    fi

    cd "$ROOT"
    nohup bash "$TRAIN_SCRIPT" > /tmp/v11_resume_attempt_$((restart_count+1)).log 2>&1 &
    TRAIN_PID=$!
    log "Launched train pid=$TRAIN_PID, grace ${STARTUP_GRACE_SEC}s"

    # 监控 grace 期: 每 60s 看进程 + GPU 是否有 v11 训练占用
    elapsed=0
    alive=true
    while [[ $elapsed -lt $STARTUP_GRACE_SEC ]]; do
      sleep "$CHECK_INTERVAL"
      elapsed=$((elapsed + CHECK_INTERVAL))
      if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        log "ERROR: train pid=$TRAIN_PID died at +${elapsed}s"
        alive=false
        break
      fi
      # 检查 metrics 是否在更新 (说明真正在训练)
      METRICS="$ROOT/outputs/train/local/grpo_verl/Qwen2.5-0.5B-Instruct/grpo_v11_stepvalue/20260618_134820/metrics/train_metrics.jsonl"
      if [[ -f "$METRICS" ]]; then
        last_step=$(tail -1 "$METRICS" 2>/dev/null | python3 -c "import json,sys
try: print(json.loads(sys.stdin.read())['step'])
except: pass" 2>/dev/null)
        log "+${elapsed}s pid=$TRAIN_PID alive, last_step=$last_step"
      fi
    done

    if [[ "$alive" == "true" ]]; then
      log "SUCCESS: training survived grace period, detaching"
      echo "$TRAIN_PID" > /tmp/v11_train_pid
      log "training pid written to /tmp/v11_train_pid"
      exit 0
    else
      restart_count=$((restart_count+1))
      echo "$restart_count" > "$STATE"
      log "training failed, restart_count=$restart_count, waiting 120s before retry"
      sleep 120
    fi
  else
    log "GPU not free, sleeping ${CHECK_INTERVAL}s"
    sleep "$CHECK_INTERVAL"
  fi
done

log "Reached MAX_RESTART=$MAX_RESTART, exiting"
