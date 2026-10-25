#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VLLM_DIR=${VLLM_DIR:-${TASK_ROOT}/external/vllm}
VLLM_PYTHON=${VLLM_PYTHON:-${VLLM_DIR}/.venv/bin/python}
OUTPUT_DIR=${OUTPUT_DIR:-${TASK_ROOT}/benchmark_results/vllm-c64}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-1}
export PYTHONPATH="${VLLM_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUTPUT_DIR}/logs"
"${VLLM_PYTHON}" "${TASK_ROOT}/bin/reproduce_vllm.py" prepare "$@"
for mode in vanilla fixed adaptive; do
    "${VLLM_PYTHON}" "${TASK_ROOT}/bin/reproduce_vllm.py" run \
        --mode "${mode}" --resume --output-dir "${OUTPUT_DIR}" "$@" \
        >> "${OUTPUT_DIR}/logs/${mode}.log" 2>&1
done
"${VLLM_PYTHON}" "${TASK_ROOT}/bin/reproduce_vllm.py" summarize \
    --output-dir "${OUTPUT_DIR}" "$@"
