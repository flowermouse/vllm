# H800 validation and benchmark

Single NVIDIA H800 80GB. The three modes run sequentially on the same GPU.
The measured implementation is `0499c8077307edb4c40a32d3f432cbb54c794e2e`.
The original A800 records remain unchanged at the snapshot root.

## Results

See [table.md](table.md), [summary.json](summary.json), [raw timings](timings.json), and compressed per-dataset
JSON files. Each JSON includes all three raw timing samples, token counts,
configuration, input hashes, finish reasons, generated text and token IDs.
The summary checks identical outputs across all modes; repeated outputs are also
checked during each run. These are throughput and token-equivalence measurements,
not full math-equivalence or executable code pass@1 evaluations.

## Protocol

Target: `Qwen/Qwen3-8B`, revision
`b968826d9c46dd6066d109eabc6255188de91218`. Drafter:
`AdaFlash/Qwen3-8B-AdaFlash`, revision
`6d9b5c2fd6a0c7bab8436bbca67c380dcdfc547f`.
Vanilla disables speculation; fixed and adaptive use the same drafter with
15 speculative tokens, differing only in `enable_adaptive_verification`.

TP1, BF16, MRV2, TRITON_ATTN, full decode CUDA Graphs; 64 concurrent requests,
256 requests per dataset, 3 timed repetitions. Maximum output 2048, context 8192,
scheduled tokens 2048, fixed 32 GiB KV cache. Greedy, no thinking or prefix cache,
batch invariance enabled. Each dataset has a 64-request/64-output-token warmup.
Timing covers offline `LLM.generate()` including prefill and queue tail, excluding
startup, compilation and warmup. Performance runs have no audit callbacks or GPU
sync-check instrumentation. Small datasets are cycled to 256 requests; MATH500
uses the original AdaFlash protocol's final 50 held-out rows. MT-Bench uses first
turns. See [pinned data manifest](data-manifest.json); all 15 result files match
its prepared-data hashes.

[environment.json](environment.json) records the installed packages and driver.
Editable vLLM uses precompiled artifacts from
`68eca9af01b8b323667cb4b87148547e8b98eef6`; compiled sources match the merged base.
The host driver requires CUDA compatibility libraries for this CUDA 13 runtime.
Set `LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat:$LD_LIBRARY_PATH` when applicable.

From an installed checkout of the measured implementation:

```bash
VLLM_DIR=/path/to/vllm CUDA_VISIBLE_DEVICES=0 \
  bash adaflash_reproduction/scripts/tools/reproduce_vllm.sh
```

## Validation

- Manager suite: 33 passed; [JUnit](manager.xml).
- Fixed/adaptive real-checkpoint generation: 2 passed; [JUnit](e2e-final.xml).
  The adaptive case checks actual target query offsets and leaves a complete
  draft budget available, so scheduler output-limit clipping cannot fake success.
- All seven changed files pass pre-commit, including mypy and CI tethering.
- All six released BF16 head tensors produce bit-identical training/vLLM results
  at batches 1, 3 and 16; [head-equivalence.json](head-equivalence.json).
  The reference class is `ThreshHeadTwoModel` from
  [AdaFlash's training implementation](https://github.com/ZinYY/AdaFlash/blob/507b7aba5fbccaa99a400f161e4f2794ba6a388c/specforge/modeling/draft/dflash.py).
- The checkpoint's full SHA256 is recorded in [checkpoint-sha256.txt](checkpoint-sha256.txt).

```bash
.venv/bin/python -m pytest tests/v1/spec_decode/test_adaptive_verification.py -v
.venv/bin/python -m pytest tests/v1/e2e/spec_decode/acceptance_rates/dflash/test_dflash.py -k adaflash_checkpoint_generation -v -s
git diff --name-only -z upstream/main...HEAD | xargs -0 .venv/bin/python -m pre_commit run --files
```

The existing DFlash nightly selection includes the new real-checkpoint tests.
Upstream GPU CI has not run for this draft. Contributor gating blocked the
previous pre-commit and documentation jobs before actual checks/builds; those
failures were not lint or documentation errors.

## Separate synchronization audit

[audit.py](audit.py) records actual target offsets and CPU event waits in separate
instrumented runs; its throughput is not used in the benchmark table.
At concurrency 64, 1502/1516 request-steps shortened verification, averaging
6.0 draft positions excluding the anchor. At concurrency 1, 75/75 shortened,
averaging 5.64. Raw records: [c64](audit-c64.json), [c1](audit-c1.json).

The current implementation synchronizes on predicted lengths before CPU shape
selection. The c64 event wait averages 21.51 ms across 28 calls; c1 averages
6.33 ms across 83 calls. These durations include GPU work being awaited and do
not measure incremental synchronization overhead. The explicit wait is allowed
by the sync checker; passing that checker does not establish an asynchronous
hot path. Net performance comes from the separate uninstrumented comparison.
Upstream scheduled-draft counters are not actual verified-length counters.
