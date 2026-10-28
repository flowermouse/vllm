#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reproduce AdaFlash datasets on vanilla, fixed and adaptive vLLM."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.benchmark.run import (  # noqa: E402
    GSM8K_REASONING_INSTRUCTION,
    _answers_match,
    _extract_boxed_answer,
    _extract_gsm8k_prediction,
    _load_jsonl_entries,
)

DATASETS = {
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "math500": ("HuggingFaceH4/MATH-500", None, "test"),
    "aime25": ("yentinglin/aime_2025", None, "train"),
    "humaneval": ("openai/openai_humaneval", None, "test"),
    "mbpp": ("google-research-datasets/mbpp", "full", "test"),
    "mt-bench": ("HuggingFaceH4/mt_bench_prompts", None, "train"),
}
DEFAULT_DATASETS = [name for name in DATASETS if name != "aime25"]
TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DRAFT_REVISION = "6d9b5c2fd6a0c7bab8436bbca67c380dcdfc547f"
DATASET_REVISIONS = {
    "gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "math500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    "aime25": "6f71d77b0b89b9dabe07ab466c51df33f514df7f",
    "humaneval": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
    "mbpp": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    "mt-bench": "e3a795c5e9a82ee40611c416b8a7786c73198991",
}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def comparison_configuration(configuration: dict) -> dict:
    # Suite membership does not change an individual dataset's sampling inputs.
    return {key: value for key, value in configuration.items() if key != "datasets"}


def prepare(args) -> None:
    from datasets import load_dataset

    manifest = {}
    for name in args.datasets:
        output = args.data_dir / f"{name}_test.jsonl"
        metadata = output.with_suffix(".source.json")
        if output.exists() and metadata.exists():
            entry = json.loads(metadata.read_text())
            if (
                entry["sha256"] != digest(output.read_bytes())
                or entry["revision"] != DATASET_REVISIONS[name]
            ):
                raise ValueError(f"Changed dataset: {output}")
            manifest[name] = entry
            continue
        repo, config, split = DATASETS[name]
        revision = DATASET_REVISIONS[name]
        dataset = load_dataset(repo, config, split=split, revision=revision)
        rows = []
        for index, row in enumerate(dataset):
            answer = None
            if name == "gsm8k":
                turns = [row["question"]]
                answer = row["answer"].split("####")[-1].strip().replace(",", "")
            elif name == "math500":
                turns, answer = [row["problem"]], row["answer"]
            elif name == "aime25":
                turns = [row.get("problem", row.get("question"))]
                answer = str(row["answer"])
            elif name == "humaneval":
                turns = [
                    "Write a solution to the following problem and make sure"
                    " that it passes the tests:\n```python\n" + row["prompt"] + "\n```"
                ]
            elif name == "mbpp":
                tests = "\n".join(row["test_list"])
                turns = [f"{row['text']}\nYour code should pass these tests:\n{tests}"]
            else:
                turns = row["prompt"]
            if not turns or not isinstance(turns[0], str):
                raise ValueError(f"Missing prompt in {name} row {index}")
            entry = {"id": str(index), "turns": turns}
            if answer is not None:
                entry["answer"] = answer
            rows.append(entry)
        # Preserve this repository's held-out MATH500 split (rows 451–500).
        if name == "math500":
            rows = rows[450:500]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
        entry = dict(
            repository=repo,
            revision=revision,
            config=config,
            split=split,
            rows=len(rows),
            sha256=digest(output.read_bytes()),
            selection="indices 450:500" if name == "math500" else "full split",
        )
        save(metadata, entry)
        manifest[name] = entry
        print(f"Prepared {name}: {len(rows)} rows at {revision}", flush=True)
    save(args.data_dir / "manifest.json", manifest)


def metric_counts(llm) -> dict[str, float]:
    from vllm.v1.metrics.reader import Counter

    return {
        metric.name: float(metric.value)
        for metric in llm.get_metrics()
        if isinstance(metric, Counter) and "spec_decode" in metric.name
    }


def run(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"{args.mode}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run_mode(args)


def run_mode(args) -> None:
    import torch

    import vllm
    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig

    missing = [
        name
        for name in args.datasets
        if not (args.data_dir / f"{name}_test.jsonl").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Run prepare first; missing {missing}")
    commit = subprocess.check_output(
        ["git", "-C", str(Path(vllm.__file__).parents[1]), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    configuration = dict(
        vllm_commit=commit,
        target=args.target_model,
        target_revision=args.target_revision,
        draft=args.draft_model,
        draft_revision=args.draft_revision,
        concurrency=args.concurrency,
        num_prompts=args.num_prompts,
        max_samples=args.max_samples,
        repeats=args.repeats,
        max_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_batched_tokens,
        kv_cache_gib=args.kv_cache_gib,
        batch_invariant=os.environ.get("VLLM_BATCH_INVARIANT", "0"),
        seed=args.seed,
        enable_thinking=False,
        prefix_caching=False,
        backend="TRITON_ATTN",
        datasets=args.datasets,
    )
    if args.resume and all(
        (args.output_dir / f"{name}-{args.mode}.json").is_file()
        for name in args.datasets
    ):
        for name in args.datasets:
            old = json.loads((args.output_dir / f"{name}-{args.mode}.json").read_text())
            data_sha = digest((args.data_dir / f"{name}_test.jsonl").read_bytes())
            if (
                comparison_configuration(old["configuration"])
                != comparison_configuration(configuration)
                or old["data_sha256"] != data_sha
            ):
                raise ValueError(
                    "Resume configuration differs; use a new output directory"
                )
        print(f"Already complete: {args.mode}", flush=True)
        return
    spec = (
        None
        if args.mode == "vanilla"
        else dict(
            method="dflash",
            model=args.draft_model,
            revision=args.draft_revision,
            num_speculative_tokens=15,
            enable_adaptive_verification=args.mode == "adaptive",
        )
    )
    captures = [2**i for i in range((16 * args.concurrency).bit_length())]
    llm = LLM(
        model=args.target_model,
        revision=args.target_revision,
        speculative_config=spec,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=args.concurrency,
        max_num_batched_tokens=args.max_batched_tokens,
        kv_cache_memory_bytes=args.kv_cache_gib * 1024**3,
        attention_config={"backend": "TRITON_ATTN"},
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        disable_log_stats=False,
        seed=args.seed,
        compilation_config=CompilationConfig(
            cudagraph_mode="FULL_DECODE_ONLY",
            cudagraph_capture_sizes=captures,
        ),
    )
    tokenizer = llm.get_tokenizer()
    for name in args.datasets:
        output_file = args.output_dir / f"{name}-{args.mode}.json"
        data_file = args.data_dir / f"{name}_test.jsonl"
        data_sha = digest(data_file.read_bytes())
        if args.resume and output_file.exists():
            old = json.loads(output_file.read_text())
            if (
                comparison_configuration(old["configuration"])
                != comparison_configuration(configuration)
                or old["data_sha256"] != data_sha
            ):
                raise ValueError(f"Resume inputs differ for {name}")
            continue
        entries = _load_jsonl_entries(data_file, dataset=name, need_answers=True)
        if len(entries) > args.max_samples:
            random.Random(args.seed).shuffle(entries)
            entries = entries[: args.max_samples]
        count = args.concurrency + args.num_prompts
        selected = [entries[i % len(entries)] for i in range(count)]
        prompts = []
        for entry in selected:
            content = entry["turns"][0]
            if name == "gsm8k":
                content += GSM8K_REASONING_INSTRUCTION
            prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        if (
            max(len(tokenizer.encode(p)) for p in prompts) + args.max_new_tokens
            > args.max_model_len
        ):
            raise ValueError(
                f"{name}: increase --max-model-len; inputs will not be truncated"
            )
        warmup = SamplingParams(temperature=0, max_tokens=64, seed=args.seed)
        llm.generate(prompts[: args.concurrency], warmup, use_tqdm=False)
        prompts, selected = prompts[args.concurrency :], selected[args.concurrency :]
        params = SamplingParams(
            temperature=0, max_tokens=args.max_new_tokens, seed=args.seed
        )
        samples, rows = [], []
        for repetition in range(args.repeats):
            before = metric_counts(llm)
            torch.accelerator.synchronize()
            start = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            torch.accelerator.synchronize()
            elapsed = time.perf_counter() - start
            after = metric_counts(llm)
            current = [
                dict(
                    id=entry["id"],
                    reference=entry.get("answer"),
                    text=output.outputs[0].text,
                    token_ids=list(output.outputs[0].token_ids),
                    finish_reason=output.outputs[0].finish_reason,
                )
                for entry, output in zip(selected, outputs, strict=True)
            ]
            if configuration["batch_invariant"] == "1" and rows:
                assert [r["token_ids"] for r in rows] == [
                    r["token_ids"] for r in current
                ]
            rows = current
            tokens = sum(len(row["token_ids"]) for row in rows)
            if not tokens or any(not row["token_ids"] for row in rows):
                raise ValueError(f"Empty completion: {name}/{args.mode}")
            counters = {key: value - before.get(key, 0) for key, value in after.items()}
            samples.append(
                dict(
                    seconds=elapsed,
                    output_tokens=tokens,
                    output_tokens_per_second=tokens / elapsed,
                    truncated=sum(row["finish_reason"] == "length" for row in rows),
                    metrics=counters,
                )
            )
            print(
                f"{name}/{args.mode} run {repetition + 1}: "
                f"{tokens / elapsed:.1f} tokens/s",
                flush=True,
            )
        throughput = [sample["output_tokens_per_second"] for sample in samples]
        accuracy = None
        if name in {"gsm8k", "math500", "aime25"}:
            correct = 0
            for row in rows:
                prediction = _extract_boxed_answer(
                    row["text"]
                ) or _extract_gsm8k_prediction(row["text"])
                correct += _answers_match(prediction, row["reference"] or "", name)
            accuracy = dict(
                correct=correct,
                total=len(rows),
                evaluator="repository normalized extracted-answer exact match",
            )
        save(
            output_file,
            dict(
                dataset=name,
                mode=args.mode,
                configuration=configuration,
                vllm_commit=commit,
                gpu=torch.cuda.get_device_name(0),
                data_sha256=data_sha,
                prompt_sha256=digest(json.dumps(prompts).encode()),
                unique_source_rows=len(entries),
                samples=samples,
                mean_tps=statistics.mean(throughput),
                stdev_tps=statistics.stdev(throughput) if len(throughput) > 1 else 0,
                accuracy=accuracy,
                outputs=rows,
            ),
        )


def summarize(args) -> None:
    rows, details = [], []
    for name in args.datasets:
        results = [
            json.loads((args.output_dir / f"{name}-{mode}.json").read_text())
            for mode in ("vanilla", "fixed", "adaptive")
        ]
        base, fixed, adaptive = results
        for result in results:
            if (
                comparison_configuration(result["configuration"])
                != comparison_configuration(base["configuration"])
                or result["prompt_sha256"] != base["prompt_sha256"]
            ):
                raise ValueError(f"Mismatched benchmark inputs: {name}")
            matches = sum(
                a["token_ids"] == b["token_ids"]
                for a, b in zip(base["outputs"], result["outputs"], strict=True)
            )
            if base["configuration"]["batch_invariant"] == "1" and matches != len(
                base["outputs"]
            ):
                raise ValueError(f"Token mismatch: {name}/{result['mode']}: {matches}")

            def total(key, result=result):
                return sum(
                    sample["metrics"].get(key, 0) for sample in result["samples"]
                )

            drafts = total("vllm:spec_decode_num_drafts")
            proposed = total("vllm:spec_decode_num_draft_tokens")
            accepted = total("vllm:spec_decode_num_accepted_tokens")
            if result["mode"] != "vanilla" and (drafts <= 0 or accepted <= 0):
                raise ValueError(
                    f"Speculation was not exercised: {name}/{result['mode']}"
                )
            details.append(
                dict(
                    dataset=name,
                    mode=result["mode"],
                    mean_tps=result["mean_tps"],
                    stdev_tps=result["stdev_tps"],
                    exact_matches=matches,
                    requests=len(result["outputs"]),
                    mean_scheduled_drafts=proposed / drafts if drafts else None,
                    mean_acceptance_length=1 + accepted / drafts if drafts else None,
                    truncated=result["samples"][-1]["truncated"],
                    accuracy=result["accuracy"],
                )
            )
        cell = lambda result: f"{result['mean_tps']:.1f} ± {result['stdev_tps']:.1f}"
        rows.append(
            f"| {name} | {len(base['outputs'])} | {cell(base)} | {cell(fixed)} "
            f"| {cell(adaptive)} | {adaptive['mean_tps'] / base['mean_tps']:.2f}× "
            f"| {adaptive['mean_tps'] / fixed['mean_tps']:.2f}× |"
        )
    table = "\n".join(
        [
            (
                "| Dataset | Requests | AR tok/s | Fixed SD tok/s | AdaFlash tok/s "
                "| Ada / AR | Ada / fixed |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
        ]
    )
    (args.output_dir / "table.md").write_text(table)
    save(
        args.output_dir / "summary.json",
        dict(
            configuration={
                **comparison_configuration(base["configuration"]),
                "datasets": args.datasets,
            },
            results=details,
        ),
    )
    print(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "summarize"])
    parser.add_argument(
        "--datasets", nargs="+", choices=list(DATASETS), default=DEFAULT_DATASETS
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "test_data/vllm-six")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "benchmark_results/vllm-c64"
    )
    parser.add_argument(
        "--mode", choices=["vanilla", "fixed", "adaptive"], default="vanilla"
    )
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--target-revision", default=TARGET_REVISION)
    parser.add_argument("--draft-model", default="AdaFlash/Qwen3-8B-AdaFlash")
    parser.add_argument("--draft-revision", default=DRAFT_REVISION)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-batched-tokens", type=int, default=2048)
    parser.add_argument("--kv-cache-gib", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize}[args.action](args)


if __name__ == "__main__":
    main()
