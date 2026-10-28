# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inspect AdaFlash's real verification layouts and host waits in a separate run."""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
os.environ.setdefault("VLLM_GPU_SYNC_CHECK", "error")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig


def start_audit(worker):
    import time

    manager = worker.model_runner.adaptive_verification
    original_wait = manager._wait_for_draft_lengths
    original_reallocate = manager.reallocate_drafts
    manager.adaflash_audit = {"wait_ms": [], "offsets": []}

    def wait():
        pending = manager._lengths_pending
        start = time.perf_counter()
        original_wait()
        if pending:
            manager.adaflash_audit["wait_ms"].append(
                1000 * (time.perf_counter() - start)
            )

    def reallocate(req_ids, idx_mapping):
        result = original_reallocate(req_ids, idx_mapping)
        # Snapshot the offsets actually consumed by the target. Read only at end.
        manager.adaflash_audit["offsets"].append(result[0].clone())
        return result

    manager._wait_for_draft_lengths = wait
    manager.reallocate_drafts = reallocate


def finish_audit(worker):
    import torch

    stats = worker.model_runner.adaptive_verification.adaflash_audit
    lengths = torch.cat([torch.diff(x) - 1 for x in stats["offsets"]]).cpu().tolist()
    waits = stats["wait_ms"]
    return {
        "verification_steps": len(stats["offsets"]),
        "request_steps": len(lengths),
        "actual_mean_verified_drafts": sum(lengths) / len(lengths),
        "actual_min_verified_drafts": min(lengths),
        "actual_max_verified_drafts": max(lengths),
        "shortened_request_steps": sum(x < 15 for x in lengths),
        "wait_calls": len(waits),
        "wait_ms_total": sum(waits),
        "wait_ms_mean": sum(waits) / len(waits),
        "wait_ms_max": max(waits),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    captures = [2**i for i in range((16 * args.concurrency).bit_length())]
    llm = LLM(
        model="Qwen/Qwen3-8B",
        revision="b968826d9c46dd6066d109eabc6255188de91218",
        speculative_config={
            "method": "dflash",
            "model": "AdaFlash/Qwen3-8B-AdaFlash",
            "revision": "6d9b5c2fd6a0c7bab8436bbca67c380dcdfc547f",
            "num_speculative_tokens": 15,
            "enable_adaptive_verification": True,
        },
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=2048,
        max_num_seqs=args.concurrency,
        max_num_batched_tokens=2048,
        kv_cache_memory_bytes=32 * 1024**3,
        attention_config={"backend": "TRITON_ATTN"},
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        disable_log_stats=False,
        seed=42,
        compilation_config=CompilationConfig(
            cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=captures
        ),
    )
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"Calculate {i + 13} * 7 + 4. Explain briefly.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for i in range(max(args.concurrency * 2, 8))
    ]
    llm.generate(
        prompts[: args.concurrency],
        SamplingParams(temperature=0, max_tokens=32),
        use_tqdm=False,
    )
    llm.collective_rpc(start_audit)
    start = time.perf_counter()
    results = llm.generate(
        prompts, SamplingParams(temperature=0, max_tokens=128), use_tqdm=False
    )
    elapsed = time.perf_counter() - start
    stats = llm.collective_rpc(finish_audit)[0]
    assert stats["shortened_request_steps"] > 0
    stats.update(
        concurrency=args.concurrency,
        seconds=elapsed,
        output_tokens=sum(len(r.outputs[0].token_ids) for r in results),
        instrumented=True,
        note="Instrumentation run; not used for uninstrumented throughput claims.",
    )
    args.output.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
