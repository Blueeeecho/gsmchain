#!/usr/bin/env python
"""Generate ChainGSM distractor benchmark data."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from src.llm_client import FatalLLMError, LLMClient, LLMJSONParseError
from src.prompts import build_generation_prompt, build_validation_prompt
from src.schemas import CATEGORIES, CATEGORY_ORDER, DIFFICULTY_KEYS, make_original_record
from src.utils import (
    append_jsonl,
    existing_ids,
    load_or_select_samples,
    read_jsonl,
    sanitize_name,
    sort_records,
    write_jsonl,
)
from src.validators import validate_final_record, validate_generated_payload

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments.
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "test-00000-of-00001.jsonl"

DEFAULT_DEEPSEEK_API_KEY = "sk-310998e7790f4b388415106f9f01e2f3"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


def project_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def sorted_path_for(output_path):
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}.sorted{output_path.suffix}")


def default_run_name(args):
    source_name = sanitize_name(project_path(args.input_path))
    sample_part = f"n{args.num_samples}" if args.num_samples is not None else "all"
    if args.mode == "pilot":
        sample_part = f"pilot{args.pilot_samples}"
    return f"{source_name}_{args.mode}_{sample_part}_json_{args.thinking}"


def remove_if_exists(paths):
    for path in paths:
        path = Path(path)
        if path.exists():
            path.unlink()


def parse_categories(value):
    if value == "all":
        return list(CATEGORIES)
    categories = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [category for category in categories if category not in CATEGORIES]
    if invalid:
        raise ValueError(f"Invalid categories: {', '.join(invalid)}")
    return categories


def run_variant_tasks(tasks, args, config, paths, write_lock, generator_client, validator_client):
    stop_event = threading.Event()
    success_count = 0
    failed_count = 0
    skipped_count = 0
    fatal_error = None
    task_iter = iter(tasks)
    in_flight = set()

    def submit_next(executor):
        try:
            sample, category = next(task_iter)
        except StopIteration:
            return False
        future = executor.submit(
            generate_variant_task,
            sample,
            category,
            args,
            config,
            paths["output_path"],
            paths["failed_path"],
            write_lock,
            generator_client,
            validator_client,
            stop_event,
        )
        in_flight.add(future)
        return True

    progress = None
    if tqdm is not None:
        progress = tqdm(total=len(tasks), desc="Generating variants", unit="variant")

    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for _ in range(min(args.max_workers, len(tasks))):
                submit_next(executor)

            while in_flight:
                done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    if result.get("ok"):
                        success_count += 1
                    elif result.get("skipped"):
                        skipped_count += 1
                    else:
                        failed_count += 1
                    if result.get("fatal"):
                        fatal_error = result.get("error")
                        stop_event.set()
                    if progress is not None:
                        progress.update(1)
                    else:
                        completed = success_count + failed_count + skipped_count
                        print(f"Generating variants: {completed}/{len(tasks)}", flush=True)

                if stop_event.is_set():
                    for future in in_flight:
                        future.cancel()
                    break
                while len(in_flight) < args.max_workers and submit_next(executor):
                    pass
    finally:
        if progress is not None:
            progress.close()

    print(
        f"Finished variant tasks: {success_count} ok, "
        f"{failed_count} failed, {skipped_count} skipped."
    )
    if fatal_error:
        print(f"Stopped early because of fatal API error: {fatal_error}")


def load_config():
    generator_model = os.getenv("GENERATOR_MODEL", DEFAULT_MODEL)
    return {
        "api_key": os.getenv("OPENAI_API_KEY", DEFAULT_DEEPSEEK_API_KEY),
        "base_url": os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        "generator_model": generator_model,
        "validator_model": os.getenv("VALIDATOR_MODEL", generator_model),
    }


def make_generated_record(sample, category, payload, seed, generator_model):
    difficulty_tags = payload.get("difficulty_tags") or {}
    difficulty_tags = {key: difficulty_tags.get(key, "unknown") for key in DIFFICULTY_KEYS}
    return {
        "id": f"{sample['base_id']}_{category}",
        "base_id": sample["base_id"],
        "source_index": sample["source_index"],
        "category": category,
        "question_original": sample["question"],
        "question_distracted": payload["question_distracted"],
        "answer": sample["final_answer"],
        "solution_original": sample["answer"],
        "core_chain": payload["core_chain"],
        "distractor_chain": payload["distractor_chain"],
        "gold_expression": payload["gold_expression"],
        "distractor_expression": payload["distractor_expression"],
        "difficulty_tags": difficulty_tags,
        "metadata": {
            "generator_model": generator_model,
            "seed": seed,
            "variant_type": "generated",
        },
    }


def validate_with_llm(validator_client, original_question, record):
    payload = validator_client.generate_json(
        build_validation_prompt(original_question, record),
        temperature=0.0,
        max_tokens=700,
    )
    if not isinstance(payload, dict):
        raise ValueError("validator returned non-object JSON")
    if payload.get("pass") is not True:
        reason = payload.get("reason", "validator rejected generated record")
        raise ValueError(f"validator rejected generated record: {reason}")
    return payload


def generate_variant_task(
    sample,
    category,
    args,
    config,
    output_path,
    failed_path,
    write_lock,
    generator_client,
    validator_client=None,
    stop_event=None,
):
    record_id = f"{sample['base_id']}_{category}"
    last_error = None
    raw_response_snippet = None

    if stop_event is not None and stop_event.is_set():
        return {"id": record_id, "ok": False, "skipped": True}

    for attempt in range(1, args.max_retries + 1):
        try:
            if stop_event is not None and stop_event.is_set():
                return {"id": record_id, "ok": False, "skipped": True}
            payload = generator_client.generate_json(
                build_generation_prompt(
                    category,
                    sample["question"],
                    sample["answer"],
                    sample["final_answer"],
                ),
                temperature=args.temperature,
                max_tokens=args.generation_max_tokens,
            )
            errors = validate_generated_payload(
                payload,
                category,
                sample["final_answer"],
                original_question=sample["question"],
            )
            if errors:
                raise ValueError("; ".join(errors))

            record = make_generated_record(
                sample,
                category,
                payload,
                args.seed,
                config["generator_model"],
            )
            final_errors = validate_final_record(record)
            if final_errors:
                raise ValueError("; ".join(final_errors))

            if validator_client is not None:
                validate_with_llm(validator_client, sample["question"], record)

            append_jsonl(output_path, record, write_lock)
            return {"id": record_id, "ok": True}
        except FatalLLMError as exc:
            last_error = str(exc)
            if stop_event is not None:
                stop_event.set()
            failure = {
                "id": record_id,
                "base_id": sample["base_id"],
                "source_index": sample["source_index"],
                "category": category,
                "error": last_error,
                "attempts": attempt,
                "fatal": True,
                "question_original": sample["question"],
                "answer": sample["final_answer"],
            }
            append_jsonl(failed_path, failure, write_lock)
            return {"id": record_id, "ok": False, "fatal": True, "error": last_error}
        except LLMJSONParseError as exc:
            last_error = str(exc)
            raw_response_snippet = exc.raw_content[:2000]
        except Exception as exc:  # noqa: BLE001 - generation must continue globally.
            last_error = str(exc)

    failure = {
        "id": record_id,
        "base_id": sample["base_id"],
        "source_index": sample["source_index"],
        "category": category,
        "error": last_error,
        "attempts": args.max_retries,
        "question_original": sample["question"],
        "answer": sample["final_answer"],
    }
    if raw_response_snippet:
        failure["raw_response_snippet"] = raw_response_snippet
    append_jsonl(failed_path, failure, write_lock)
    return {"id": record_id, "ok": False, "error": last_error}


def write_original_records(samples, seed, output_path, seen_ids, write_lock):
    written = 0
    for sample in samples:
        record = make_original_record(sample, seed)
        if record["id"] in seen_ids:
            continue
        errors = validate_final_record(record)
        if errors:
            raise ValueError(f"Invalid original record {record['id']}: {'; '.join(errors)}")
        append_jsonl(output_path, record, write_lock)
        seen_ids.add(record["id"])
        written += 1
    return written


def summarize_records(output_path, failed_path, expected_original_count, expected_variant_count):
    records = read_jsonl(output_path)
    successful_ids = {record.get("id") for record in records}
    unresolved_failures = {}
    for failure in read_jsonl(failed_path):
        failure_id = failure.get("id")
        if failure_id not in successful_ids:
            unresolved_failures[failure_id] = failure
    failures = list(unresolved_failures.values())
    category_counts = Counter(record.get("category") for record in records)
    return {
        "expected_original_count": expected_original_count,
        "actual_original_count": category_counts.get("original", 0),
        "expected_variant_count": expected_variant_count,
        "actual_variant_count": sum(category_counts.get(category, 0) for category in CATEGORIES),
        "total_records": len(records),
        "failed_count": len(failures),
        "category_counts": dict(category_counts),
    }


def write_sorted_output(output_path, sorted_path):
    records = sort_records(read_jsonl(output_path), CATEGORY_ORDER)
    write_jsonl(sorted_path, records)


def write_final_summary(
    summary,
    stats_path,
    summary_path,
    output_path,
    sorted_path,
    selected_path,
    input_path,
    args,
    config,
    validator_used,
):
    stats = {
        **summary,
        "output_path": str(output_path),
        "sorted_output_path": str(sorted_path),
        "selected_path": str(selected_path),
        "input_path": str(input_path),
        "run_name": args.run_name,
        "thinking": args.thinking,
        "seed": args.seed,
        "max_workers": args.max_workers,
        "generator_model": config["generator_model"],
    }
    if validator_used:
        stats["validator_model"] = config["validator_model"]

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Final Summary",
        "",
        f"- total original samples: {summary['actual_original_count']}",
        f"- total generated variants: {summary['actual_variant_count']}",
        f"- total final records: {summary['total_records']}",
        f"- category counts: {json.dumps(summary['category_counts'], ensure_ascii=False)}",
        f"- failed generation count: {summary['failed_count']}",
        f"- output path: {output_path}",
        f"- sorted output path: {sorted_path}",
        f"- selected path: {selected_path}",
        f"- input path: {input_path}",
        f"- run_name: {args.run_name}",
        f"- thinking: {args.thinking}",
        f"- seed: {args.seed}",
        f"- max_workers: {args.max_workers}",
        f"- generator_model: {config['generator_model']}",
    ]
    if validator_used:
        lines.append(f"- validator_model: {config['validator_model']}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_failed_requests(args):
    failed_path = project_path(args.failed_log_path)
    success_path = project_path(args.success_path)
    input_path = project_path(args.input_path)
    if not failed_path.exists():
        raise FileNotFoundError(f"Failed log not found: {failed_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input/source file not found: {input_path}")

    source_samples = load_or_select_samples(
        input_path,
        PROJECT_ROOT / "data" / "raw" / "_tmp_extract_source_cache.jsonl",
        count=None,
        force=True,
        input_format=args.input_format,
        base_id_prefix=args.base_id_prefix,
    )
    tmp_cache = PROJECT_ROOT / "data" / "raw" / "_tmp_extract_source_cache.jsonl"
    if tmp_cache.exists():
        tmp_cache.unlink()

    by_base_id = {sample["base_id"]: sample for sample in source_samples}
    success_ids = existing_ids(success_path) if success_path.exists() else set()
    failures_by_id = {}
    failure_counts = Counter()
    for failure in read_jsonl(failed_path):
        failure_id = failure.get("id")
        if not failure_id or failure_id in success_ids:
            continue
        error = failure.get("error", "")
        if args.exclude_balance_errors and "Insufficient Balance" in error:
            continue
        failures_by_id[failure_id] = failure
        failure_counts[failure_id] += 1

    extracted = []
    for failure_id, failure in sorted(failures_by_id.items()):
        sample = by_base_id.get(failure.get("base_id"))
        if sample is None:
            continue
        extracted.append(
            {
                "failed_id": failure_id,
                "base_id": sample["base_id"],
                "source_index": sample["source_index"],
                "category": failure["category"],
                "question": sample["question"],
                "answer": sample["answer"],
                "final_answer": sample["final_answer"],
                "previous_error_count": failure_counts[failure_id],
                "latest_error": failure.get("error", ""),
            }
        )

    output_path = project_path(args.failed_requests_path)
    write_jsonl(output_path, extracted)
    print(
        json.dumps(
            {
                "failed_log_path": str(failed_path),
                "success_path": str(success_path),
                "output_path": str(output_path),
                "records": len(extracted),
                "category_counts": dict(Counter(row["category"] for row in extracted)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def write_pilot_report(samples, output_path, failed_path, report_path, validator_used):
    records = read_jsonl(output_path)
    successful_ids = {record.get("id") for record in records}
    unresolved_failures = {}
    for failure in read_jsonl(failed_path):
        failure_id = failure.get("id")
        if failure_id not in successful_ids:
            unresolved_failures[failure_id] = failure
    failures = list(unresolved_failures.values())
    by_base = defaultdict(dict)
    for record in records:
        by_base[record["base_id"]][record["category"]] = record

    generated_count = sum(1 for record in records if record.get("category") != "original")
    validator_rate = "N/A"
    if validator_used:
        total_attempted = generated_count + len(failures)
        validator_rate = f"{generated_count}/{total_attempted}" if total_attempted else "0/0"

    lines = [
        "# Pilot Report",
        "",
        "## Summary",
        "",
        f"- selected base samples: {', '.join(sample['base_id'] for sample in samples)}",
        f"- total records: {len(records)}",
        f"- generated variants: {generated_count}",
        f"- failed variants: {len(failures)}",
        f"- validator pass rate: {validator_rate}",
    ]

    for sample in samples:
        lines.extend(
            [
                "",
                f"## Base Sample {sample['base_id']}",
                "",
                "### Original Question",
                "",
                sample["question"],
                "",
                "### Original Answer",
                "",
                sample["answer"],
                "",
                "### Original Final Answer",
                "",
                sample["final_answer"],
            ]
        )
        for category in CATEGORIES:
            record = by_base.get(sample["base_id"], {}).get(category)
            lines.extend(["", f"### Variant: {category}", ""])
            if not record:
                lines.append("Not generated.")
                continue
            lines.extend(
                [
                    "Generated question:",
                    "",
                    record["question_distracted"],
                    "",
                    "Core chain:",
                    "",
                    "```json",
                    json.dumps(record["core_chain"], ensure_ascii=False, indent=2),
                    "```",
                    "",
                    "Distractor chain:",
                    "",
                    "```json",
                    json.dumps(record["distractor_chain"], ensure_ascii=False, indent=2),
                    "```",
                    "",
                    f"Gold expression: {record['gold_expression']}",
                    "",
                    f"Distractor expression: {record['distractor_expression']}",
                    "",
                    "Difficulty tags:",
                    "",
                    "```json",
                    json.dumps(record["difficulty_tags"], ensure_ascii=False, indent=2),
                    "```",
                ]
            )

    if failures:
        lines.extend(["", "## Failed Generations", ""])
        for failure in failures:
            lines.append(
                f"- {failure.get('id')}: {failure.get('error')} "
                f"(attempts: {failure.get('attempts')})"
            )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_paths(args, sample_count=None):
    run_dir_name = sanitize_name(args.run_name)
    report_dir = project_path(args.reports_dir) / run_dir_name

    if args.selected_path:
        selected_path = project_path(args.selected_path)
    else:
        count_label = args.pilot_samples if args.mode == "pilot" else args.num_samples
        if count_label is None:
            count_label = "all"
        selected_path = PROJECT_ROOT / "data" / "raw" / f"selected_{run_dir_name}_{count_label}.jsonl"

    if args.output_path:
        output_path = project_path(args.output_path)
    elif args.mode == "pilot":
        output_path = project_path(args.output_dir) / run_dir_name / "pilot_samples.jsonl"
    else:
        if sample_count is None:
            total_label = "unknown"
        else:
            category_count = 1 if args.use_input_categories else len(parse_categories(args.categories))
            original_count = 0 if args.skip_originals else sample_count
            total_label = original_count + (sample_count * category_count)
        output_path = project_path(args.output_dir) / run_dir_name / f"{run_dir_name}_{total_label}.jsonl"

    if args.mode == "pilot":
        failed_path = report_dir / "pilot_failed_generation.jsonl"
        report_path = report_dir / "pilot_report.md"
        stats_path = None
    else:
        failed_path = report_dir / "failed_generation.jsonl"
        report_path = report_dir / "final_summary.md"
        stats_path = report_dir / "final_stats.json"

    return {
        "output_path": output_path,
        "sorted_path": sorted_path_for(output_path),
        "selected_path": selected_path,
        "failed_path": failed_path,
        "report_path": report_path,
        "stats_path": stats_path,
        "report_dir": report_dir,
    }


def run_generation(args):
    if args.mode == "extract-failures":
        extract_failed_requests(args)
        return

    if args.mode == "full" and not args.confirm_pilot_ok:
        raise SystemExit("full mode requires --confirm-pilot-ok")

    input_path = project_path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.run_name is None:
        args.run_name = default_run_name(args)

    sample_count = args.pilot_samples if args.mode == "pilot" else args.num_samples
    paths = build_paths(args, sample_count)
    validator_used = args.with_validator or args.mode == "pilot"

    if args.force:
        removable = [
            paths["output_path"],
            paths["sorted_path"],
            paths["failed_path"],
            paths["report_path"],
        ]
        if paths["selected_path"].resolve() != input_path.resolve():
            removable.append(paths["selected_path"])
        if paths["stats_path"]:
            removable.append(paths["stats_path"])
        remove_if_exists(removable)

    samples = load_or_select_samples(
        input_path,
        paths["selected_path"],
        count=sample_count,
        force=args.force,
        input_format=args.input_format,
        base_id_prefix=args.base_id_prefix,
    )

    if args.output_path is None and args.mode == "full" and sample_count is None:
        paths = build_paths(args, len(samples))
        if args.force:
            removable = [
                paths["output_path"],
                paths["sorted_path"],
                paths["failed_path"],
                paths["report_path"],
            ]
            if paths["stats_path"]:
                removable.append(paths["stats_path"])
            remove_if_exists(removable)

    config = load_config()
    generator_client = LLMClient(
        model=config["generator_model"],
        api_key=config["api_key"],
        base_url=config["base_url"],
        api_retries=args.api_retries,
        thinking=None if args.thinking == "default" else args.thinking,
    )
    validator_client = None
    if validator_used:
        validator_client = LLMClient(
            model=config["validator_model"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            api_retries=args.api_retries,
            thinking=None if args.thinking == "default" else args.thinking,
        )

    write_lock = threading.Lock()
    seen = existing_ids(paths["output_path"])
    if not args.skip_originals:
        originals_written = write_original_records(samples, args.seed, paths["output_path"], seen, write_lock)
        if originals_written:
            print(f"Wrote {originals_written} original records.")

    tasks = []
    requested_categories = parse_categories(args.categories)
    expected_variant_count = 0
    for sample in samples:
        if args.use_input_categories and sample.get("category"):
            sample_categories = [sample["category"]]
        else:
            sample_categories = requested_categories
        expected_variant_count += len(sample_categories)
        for category in sample_categories:
            record_id = f"{sample['base_id']}_{category}"
            if record_id in seen:
                continue
            tasks.append((sample, category))

    print(f"Variant tasks to run: {len(tasks)}")
    if tasks:
        run_variant_tasks(
            tasks,
            args,
            config,
            paths,
            write_lock,
            generator_client,
            validator_client,
        )

    write_sorted_output(paths["output_path"], paths["sorted_path"])
    expected_original_count = 0 if args.skip_originals else len(samples)
    summary = summarize_records(
        paths["output_path"],
        paths["failed_path"],
        expected_original_count,
        expected_variant_count,
    )

    if args.mode == "pilot":
        write_pilot_report(samples, paths["output_path"], paths["failed_path"], paths["report_path"], validator_used)
    else:
        write_final_summary(
            summary,
            paths["stats_path"],
            paths["report_path"],
            paths["output_path"],
            paths["sorted_path"],
            paths["selected_path"],
            input_path,
            args,
            config,
            validator_used,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate ChainGSM distractor data.")
    parser.add_argument("--mode", choices=["pilot", "full", "extract-failures"], required=True)
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--input-format", choices=["auto", "jsonl", "parquet"], default="auto")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--base-id-prefix", default="chaingsm")
    parser.add_argument("--categories", default="all")
    parser.add_argument("--use-input-categories", action="store_true")
    parser.add_argument("--skip-originals", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--generation-max-tokens", type=int, default=1600)
    parser.add_argument("--thinking", choices=["disabled", "enabled", "default"], default="disabled")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--pilot-samples", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--api-retries", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--with-validator", action="store_true")
    parser.add_argument("--confirm-pilot-ok", action="store_true")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--selected-path", default=None)
    parser.add_argument("--failed-log-path", default="reports/gsm8k_test_full/failed_generation.jsonl")
    parser.add_argument("--success-path", default="data/final/gsm8k_test_full/gsm8k_test_full_6595.jsonl")
    parser.add_argument("--failed-requests-path", default="data/raw/failed_requests_gsm8k_test_full.jsonl")
    parser.add_argument("--exclude-balance-errors", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = "data/pilot" if args.mode == "pilot" else "data/final"
    return args


def main(argv=None):
    args = parse_args(argv)
    run_generation(args)


if __name__ == "__main__":
    main(sys.argv[1:])
