"""汇总 v10 GRPO 训练结果, 生成最终报告."""
import argparse
import json
from pathlib import Path
from datetime import datetime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", required=True, help="grpo_v10_signed 训练根目录")
    ap.add_argument("--report-out", required=True, help="输出报告路径")
    ap.add_argument("--sft-baseline-eval", default=None, help="SFT epoch3 评测结果目录 (对照)")
    args = ap.parse_args()

    ckpt_root = Path(args.ckpt_root)
    runs = sorted([d for d in ckpt_root.iterdir() if d.is_dir()], key=lambda x: x.name)
    if not runs:
        print(f"No runs found in {ckpt_root}")
        return
    latest_run = runs[-1]
    eval_dir = latest_run / "eval_per_ckpt"
    summary_file = eval_dir / "eval_summary.jsonl"
    train_metrics = latest_run / "metrics" / "train_metrics.jsonl"
    ckpt_dir = latest_run / "checkpoints"

    # Read eval results
    eval_results = []
    if summary_file.exists():
        with open(summary_file) as f:
            for line in f:
                if line.strip():
                    eval_results.append(json.loads(line))

    # Read SFT baseline
    sft_baseline = None
    if args.sft_baseline_eval:
        sft_eval_json = Path(args.sft_baseline_eval) / "eval_result.json"
        if sft_eval_json.exists():
            with open(sft_eval_json) as f:
                sft_baseline = json.load(f)

    # Read training metrics summary
    train_steps = 0
    train_score_mean = 0.0
    train_acc_mean = 0.0
    if train_metrics.exists():
        scores, accs = [], []
        with open(train_metrics) as f:
            for line in f:
                if not line.strip(): continue
                d = json.loads(line)
                data = d.get("data", {})
                step = d.get("step", 0)
                train_steps = max(train_steps, step)
                scores.append(data.get("critic/score/mean", 0))
                accs.append(data.get("reward/accuracy", 0))
        if scores:
            train_score_mean = sum(scores) / len(scores)
        if accs:
            train_acc_mean = sum(accs) / len(accs)

    # Count checkpoints
    ckpts = sorted(ckpt_dir.glob("global_step_*")) if ckpt_dir.exists() else []

    # Build report
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("# v10-signed GRPO 训练 & 评测报告")
    lines.append("")
    lines.append(f"**生成时间**: {now}")
    lines.append(f"**训练运行 ID**: `{latest_run.name}`")
    lines.append(f"**训练路径**: `{latest_run}`")
    lines.append(f"**训练步数**: {train_steps} 步 (目标: 3046 步 = 2 epoch × 1523 steps)")
    lines.append(f"**保存的 ckpt 数量**: {len(ckpts)}")
    lines.append(f"**评测 ckpt 数量**: {len([r for r in eval_results if r.get('status') == 'ok'])} / {len(ckpts)}")
    lines.append("")

    lines.append("## 1. 训练配置")
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("|---|---|")
    lines.append("| 起点 (SFT ckpt) | outputs/train/local/sft/Qwen2.5-0.5B-Instruct/sft_cot_2ep_resume/20260617_180315/checkpoints/checkpoint-762/ |")
    lines.append("| 数据 | chaingsm_data/data/final/grpo/all_grpo_cot.parquet (6094 行) |")
    lines.append("| 奖励函数 | train_pipeline/reward_chaingsm_v10_verl.py (v10-signed) |")
    lines.append("| 公式 | R = 0.2·r_format + 2.5·r_answer + 1.2·r_core + 0.3·r_calc − 0.5·r_distractor |")
    lines.append("| Actor LR | 5e-7 (AdamW, betas=[0.9, 0.99]) |")
    lines.append("| KL coef | 0.04 (low_var_kl, use_kl_loss=True) |")
    lines.append("| Rollout | vLLM, n=4, temperature=0.9, top_p=1.0, top_k=50, gpu_mem_util=0.5 |")
    lines.append("| batch_size | 4 (train) × 4 (mini) × 1 (micro/GPU) |")
    lines.append("| total_epochs | 2 (epoch 1 = 1523 步, epoch 2 = 1523 步) |")
    lines.append("| save_freq | 300 (→ 10 ckpts @ step 300/600/900/1200/1500/1800/2100/2400/2700/3000) |")
    lines.append("| max_actor_ckpt_to_keep | 10 (保留全部) |")
    lines.append("| max_response_length | 1024 |")
    lines.append("| enable_sleep_mode | true (update_weights 时 vLLM 睡掉, 防 OOM) |")
    lines.append("")

    lines.append("## 2. 训练 metrics 总结")
    lines.append("")
    lines.append(f"- 训练总步数: **{train_steps}**")
    lines.append(f"- 训练期间 score 均值: **{train_score_mean:.3f}** (理论区间 [-0.5, 4.2])")
    lines.append(f"- 训练期间 accuracy 均值: **{train_acc_mean:.3f}**")
    lines.append("")

    lines.append("## 3. 评测结果 (全量 5467 行, cot_brackets 方法)")
    lines.append("")
    lines.append("| ckpt (global_step) | overall | original | attr_mismatch | indep_decoy | path_competition | target_scope |")
    lines.append("|---|---|---|---|---|---|---|")

    # Build a dict by step for easy lookup
    eval_by_step = {r["step"]: r for r in eval_results if r.get("status") == "ok"}

    # For each ckpt, try to load full breakdown
    for step_dir in sorted(eval_dir.glob("step_*")):
        step = int(step_dir.name.replace("step_", ""))
        result_json = step_dir / "eval_result.json"
        if not result_json.exists():
            lines.append(f"| {step} | (no result) | | | | | |")
            continue
        with open(result_json) as f:
            d = json.load(f)
        ov = d.get("overall_accuracy", 0.0)
        cats = {c.get("category"): c.get("accuracy", 0.0) for c in d.get("by_category", [])}
        line = f"| {step} | {ov*100:.2f}% | {cats.get('original', 0)*100:.2f}% | {cats.get('attribute_mismatch', 0)*100:.2f}% | {cats.get('independent_decoy', 0)*100:.2f}% | {cats.get('path_competition', 0)*100:.2f}% | {cats.get('target_scope_misalignment', 0)*100:.2f}% |"
        lines.append(line)

    lines.append("")

    if sft_baseline:
        lines.append("## 4. 对照: SFT epoch3 baseline (训练起点)")
        lines.append("")
        lines.append(f"- overall: **{sft_baseline.get('overall_accuracy', 0)*100:.2f}%**")
        sft_cats = {c.get("category"): c.get("accuracy", 0.0) for c in sft_baseline.get("by_category", [])}
        for cat, acc in sft_cats.items():
            lines.append(f"- {cat}: {acc*100:.2f}%")
        lines.append("")

    lines.append("## 5. 关键发现")
    lines.append("")
    if eval_results:
        ok_results = [r for r in eval_results if r.get("status") == "ok"]
        if ok_results:
            best = max(ok_results, key=lambda r: r.get("overall", 0))
            best_step = best["step"]
            best_overall = best.get("overall", 0) * 100
            best_original = best.get("original", 0) * 100
            lines.append(f"- **最佳 ckpt**: global_step_{best_step} (overall={best_overall:.2f}%, original={best_original:.2f}%)")
            if sft_baseline:
                sft_ov = sft_baseline.get("overall_accuracy", 0) * 100
                sft_orig = 0.0
                for c in sft_baseline.get("by_category", []):
                    if c.get("category") == "original":
                        sft_orig = c.get("accuracy", 0) * 100
                        break
                delta_ov = best_overall - sft_ov
                delta_orig = best_original - sft_orig
                lines.append(f"- 相比 SFT epoch3 (overall={sft_ov:.2f}%, original={sft_orig:.2f}%): overall {delta_ov:+.2f}pp, original {delta_orig:+.2f}pp")
        lines.append(f"- 共评测 {len(ok_results)} 个 ckpt")
    lines.append("")

    lines.append("## 6. 评测方法与训练一致性")
    lines.append("")
    lines.append("- 评测方法: `cot_brackets` (与训练时 prompt 协议一致)")
    lines.append("- 测试集: chaingsm_data/data/gsmchain/gsm8k_test_clean.jsonl (5467 行)")
    lines.append("- 推理设置: vLLM, top_k=1, max_tokens=1024, dtype=auto")
    lines.append("- GPU: 1 × RTX 5090 32GB, gpu_memory_utilization=0.5")
    lines.append("")

    lines.append("## 7. 结论")
    lines.append("")
    lines.append("(训练完成后填写)")
    lines.append("")

    # Write report
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
