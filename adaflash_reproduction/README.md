# AdaFlash benchmark reproduction snapshot

These files snapshot the local AdaFlash benchmark implementation; original helper sources are from ZinYY/AdaFlash at 507b7aba5fbccaa99a400f161e4f2794ba6a388c. They live on a separate fork branch, outside the core vLLM PR diff.

The recorded table uses implementation c01935becfdc3d9fa7d0f3f3d26ce636271f1e87. Use an installed checkout of that version for an exact match. Default datasets: GSM8K, MATH500 (50 held-out rows), HumanEval, MBPP and MT-Bench. AIME25 is excluded. See [protocol](docs/vllm_benchmarks.md) and [results](RESULTS.md).

From this directory, activate your vLLM Python environment and run:

```bash
VLLM_DIR=/path/to/vllm CUDA_VISIBLE_DEVICES=0 \
  bash scripts/tools/reproduce_vllm.sh
```

Runtime used: Python 3.12.15, Torch 2.14.0+cu130, Triton 3.8.0, Transformers 5.20.2, datasets 3.6.0, huggingface_hub 2.3.0, pydantic 2.12.5. Models and datasets are pinned in the script. Default concurrency is 64; 256 requests per dataset; 3 measured repetitions; max output 2048; max context 8192; fixed 32 GiB KV; TP1/BF16/TRITON_ATTN; full decode CUDA Graph; batch invariance on; no thinking or prefix caching.
