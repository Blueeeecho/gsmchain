from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from train_pipeline.config_utils import resolve_path
from train_pipeline.eval_constants import DEFAULT_TEST_DATA


def _copy_or_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _save_model_dir(model: Any, tokenizer: Any, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model = getattr(model, "module", model)
    save_model.save_pretrained(str(output_dir))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(output_dir))


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


class EpochVLLMEvalCallback(TrainerCallback):
    def __init__(self, run_dir: Path, eval_config: dict[str, Any], tokenizer: Any):
        self.run_dir = Path(run_dir)
        self.eval_config = eval_config or {}
        self.tokenizer = tokenizer
        self.enabled = bool(self.eval_config.get("enabled", True))
        self.best_accuracy = -1.0
        self.evaluated_epochs: set[int] = set()
        self.baseline_done = False

    def _gpu_memory_candidates(self) -> list[float]:
        candidates = self.eval_config.get("gpu_memory_utilization_candidates")
        if candidates:
            return [float(x) for x in candidates]
        start = float(self.eval_config.get("gpu_memory_utilization", 0.8))
        defaults = [start, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25]
        seen = set()
        return [x for x in defaults if not (x in seen or seen.add(x))]

    def _print_accuracy(self, label: str, metrics: dict[str, Any]) -> None:
        print(
            f"[eval] {label} overall_accuracy={metrics['overall_accuracy']:.6f} "
            f"predictions={metrics['prediction_count']} "
            f"gpu_memory_utilization={metrics.get('gpu_memory_utilization')}",
            flush=True,
        )
        for row in metrics.get("by_category", []):
            print(
                f"[eval] {label} category={row['category']} "
                f"accuracy={row['accuracy']:.6f} correct={row['correct']}/{row['total']}",
                flush=True,
            )

    def _run_eval(self, model_dir: Path, eval_dir: Path) -> dict[str, Any]:
        """Run vLLM evaluation in a subprocess to ensure GPU memory is fully released on exit."""
        cmd = [
            sys.executable, "-m", "train_pipeline.eval_vllm_chaingsm",
            "--model-path", str(model_dir),
            "--output-dir", str(eval_dir),
            "--method", self.eval_config.get("method", "train_json_prompt"),
            "--batch-size", str(int(self.eval_config.get("batch_size", 64))),
            "--tensor-parallel-size", str(int(self.eval_config.get("tensor_parallel_size", 1))),
            "--gpu-memory-utilization", str(float(self.eval_config.get("gpu_memory_utilization", 0.8))),
            "--dtype", self.eval_config.get("dtype", "auto"),
            "--seed", str(int(self.eval_config.get("seed", 42))),
            "--max-tokens", str(int(self.eval_config.get("max_tokens", 2048))),
            "--top-k", str(int(self.eval_config.get("top_k", 1))),
            "--top-p", str(float(self.eval_config.get("top_p", 1.0))),
        ]

        data_path = resolve_path(self.eval_config.get("data_path", DEFAULT_TEST_DATA))
        cmd.extend(["--data-path", str(data_path)])

        if self.eval_config.get("limit") is not None:
            cmd.extend(["--limit", str(int(self.eval_config["limit"]))])

        if self.eval_config.get("max_model_len") is not None:
            cmd.extend(["--max-model-len", str(int(self.eval_config["max_model_len"]))])

        if not self.eval_config.get("trust_remote_code", True):
            cmd.append("--no-trust-remote-code")

        for candidate in self._gpu_memory_candidates():
            cmd.extend(["--gpu-memory-utilization-candidate", str(candidate)])

        print(f"[eval] Launching eval subprocess: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"Eval subprocess failed with return code {proc.returncode}")

        result_path = eval_dir / "eval_result.json"
        if not result_path.exists():
            raise RuntimeError(f"Eval result file not found: {result_path}")

        with result_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _append_summary(self, row: dict[str, Any]) -> None:
        _append_jsonl(
            self.run_dir / "eval" / "epoch_summary.jsonl",
            row,
        )
        with (self.run_dir / "eval" / "latest_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, ensure_ascii=False)

    def _offload_model(self, model: Any) -> Any:
        if not self.eval_config.get("offload_model_before_vllm", True):
            return None
        try:
            import torch

            original_device = next(model.parameters()).device
            model.to("cpu")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            return original_device
        except Exception:
            return None

    def _restore_model(self, model: Any, original_device: Any) -> None:
        if original_device is not None:
            try:
                model.to(original_device)
            except Exception as exc:
                print(f"[eval] WARNING: failed to restore model to {original_device}: {exc!r}", flush=True)

    def _evaluate_saved_model(
        self,
        stage: str,
        epoch: int,
        global_step: int,
        model: Any,
        latest_dir: Path,
        best_dir: Path,
        eval_dir: Path,
    ) -> None:
        original_device = self._offload_model(model)
        try:
            try:
                metrics = self._run_eval(latest_dir, eval_dir)
                overall_accuracy = float(metrics["overall_accuracy"])
                if overall_accuracy > self.best_accuracy:
                    self.best_accuracy = overall_accuracy
                    _copy_or_replace(latest_dir, best_dir)
                self._print_accuracy(stage, metrics)
                self._append_summary(
                    {
                        "stage": stage,
                        "epoch": epoch,
                        "global_step": global_step,
                        "overall_accuracy": overall_accuracy,
                        "best_accuracy": self.best_accuracy,
                        "eval_dir": str(eval_dir),
                        "current_model_dir": str(latest_dir),
                        "best_model_dir": str(best_dir),
                        "eval_status": "ok",
                        "eval_error": "",
                        "gpu_memory_utilization": metrics.get("gpu_memory_utilization"),
                    }
                )
            except Exception as exc:
                print(f"[eval] {stage} failed after all retries: {exc!r}", flush=True)
                eval_dir.mkdir(parents=True, exist_ok=True)
                with (eval_dir / "eval_failed.json").open("w", encoding="utf-8") as f:
                    json.dump({"stage": stage, "error": repr(exc)}, f, indent=2, ensure_ascii=False)
                self._append_summary(
                    {
                        "stage": stage,
                        "epoch": epoch,
                        "global_step": global_step,
                        "overall_accuracy": "",
                        "best_accuracy": self.best_accuracy,
                        "eval_dir": str(eval_dir),
                        "current_model_dir": str(latest_dir),
                        "best_model_dir": str(best_dir),
                        "eval_status": "failed",
                        "eval_error": repr(exc),
                        "gpu_memory_utilization": "",
                    }
                )
        finally:
            self._restore_model(model, original_device)

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.enabled or self.baseline_done or not self.eval_config.get("baseline_before_train", True):
            return control
        model = kwargs.get("model")
        if model is None:
            return control
        self.baseline_done = True
        latest_dir = self.run_dir / "checkpoints" / "current"
        best_dir = self.run_dir / "checkpoints" / "best"
        print("[eval] Running baseline evaluation before training.", flush=True)
        _save_model_dir(model, self.tokenizer, latest_dir)
        self._evaluate_saved_model(
            stage="baseline",
            epoch=0,
            global_step=state.global_step,
            model=model,
            latest_dir=latest_dir,
            best_dir=best_dir,
            eval_dir=self.run_dir / "eval" / "baseline",
        )
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if not self.enabled:
            return control
        epoch = int(state.epoch or 0)
        if epoch <= 0:
            epoch = len(self.evaluated_epochs) + 1
        if epoch in self.evaluated_epochs:
            return control
        self.evaluated_epochs.add(epoch)

        model = kwargs.get("model")
        if model is None:
            return control

        latest_dir = self.run_dir / "checkpoints" / "current"
        best_dir = self.run_dir / "checkpoints" / "best"
        print(f"[eval] Running epoch {epoch} evaluation.", flush=True)
        _save_model_dir(model, self.tokenizer, latest_dir)
        self._evaluate_saved_model(
            stage=f"epoch_{epoch:04d}",
            epoch=epoch,
            global_step=state.global_step,
            model=model,
            latest_dir=latest_dir,
            best_dir=best_dir,
            eval_dir=self.run_dir / "eval" / f"epoch_{epoch:04d}",
        )
        return control
