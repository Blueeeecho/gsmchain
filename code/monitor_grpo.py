"""监控 GRPO 训练内 reward 分布(实时)。

Usage:
  python code/monitor_grpo.py --run-dir outputs/train/local/grpo_verl_lbprm_v3/Qwen2.5-0.5B-Instruct/<run>/<id>
  python code/monitor_grpo.py --jsonl <path>/train_metrics.jsonl
"""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", help="run root (auto-find train_metrics.jsonl)")
    ap.add_argument("--jsonl", help="explicit path to train_metrics.jsonl")
    ap.add_argument("--last", type=int, default=50, help="show last N step aggregates")
    args = ap.parse_args()

    if args.jsonl:
        p = Path(args.jsonl)
    else:
        p = Path(args.run_dir) / "metrics" / "train_metrics.jsonl"

    if not p.exists():
        print(f"not found: {p}", file=sys.stderr); sys.exit(1)

    KEYS = [
        "reward/accuracy",
        "reward_components/format/mean",
        "reward_components/answer/mean",
        "reward_components/chain_to_answer_ok/mean",  # 可能不存在(verl 不一定直传所有 metric)
        "reward_components/causal_liveness_score/mean",
        "reward_components/step_calc_score/mean",
        "reward_components/no_degenerate_score/mean",
        "reward_components/chain_quality_score/mean",
        "reward_components/gated_chain_quality/mean",
        "critic/rewards/mean",
        "critic/rewards/min",
        "critic/rewards/max",
        "response_length/mean",
        "response_length/min",
        "response_length/max",
        "response/aborted_ratio",
        "actor/entropy",
        "actor/pg_loss",
    ]
    rows = []
    with p.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            data = d.get("data", d)
            step = data.get("training/global_step") or d.get("step") or 0
            row = {"step": int(step) if step else 0}
            for k in KEYS:
                row[k] = data.get(k)
            rows.append(row)
    rows.sort(key=lambda r: r["step"])
    if not rows:
        print("no rows"); sys.exit(0)

    print(f"file: {p}")
    print(f"total step entries: {len(rows)} (step {rows[0]['step']}..{rows[-1]['step']})")
    print()

    # 50-step windows
    win = args.last
    print(f"=== last {win} steps (or all if < {win}) ===")
    window = rows[-win:]
    for k in KEYS:
        vals = [r[k] for r in window if isinstance(r[k], (int, float))]
        if not vals:
            continue
        print(f"  {k:>50}  mean={mean(vals):.3f}  median={median(vals):.3f}  min={min(vals):.3f}  max={max(vals):.3f}")
    print()
    # 100-step windows
    n = len(rows)
    print(f"=== {min(5, n//100 + 1)} windows of 100 steps each ===")
    for i in range(0, n, 100):
        chunk = rows[i:i+100]
        if len(chunk) < 10: continue
        acc = mean([r["reward/accuracy"] or 0 for r in chunk if isinstance(r.get("reward/accuracy"), (int, float))])
        rwd = mean([r["critic/rewards/mean"] or 0 for r in chunk if isinstance(r.get("critic/rewards/mean"), (int, float))])
        fmt = mean([r["reward_components/format/mean"] or 0 for r in chunk if isinstance(r.get("reward_components/format/mean"), (int, float))])
        ans = mean([r["reward_components/answer/mean"] or 0 for r in chunk if isinstance(r.get("reward_components/answer/mean"), (int, float))])
        c2a = mean([r.get("reward_components/chain_to_answer_ok/mean") or 0 for r in chunk if isinstance(r.get("reward_components/chain_to_answer_ok/mean"), (int, float))])
        cq = mean([r["reward_components/chain_quality_score/mean"] or 0 for r in chunk if isinstance(r.get("reward_components/chain_quality_score/mean"), (int, float))])
        rl = mean([r["response_length/mean"] or 0 for r in chunk if isinstance(r.get("response_length/mean"), (int, float))])
        print(f"  step {chunk[0]['step']:>4}-{chunk[-1]['step']:>4}:  acc={acc:.3f}  rwd={rwd:.3f}  fmt={fmt:.3f}  ans={ans:.3f}  c2a={c2a:.3f}  chain={cq:.3f}  len={rl:.1f}")

if __name__ == "__main__":
    main()
