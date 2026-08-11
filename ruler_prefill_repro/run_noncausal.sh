#!/bin/sh
# Run from anywhere; the aiter checkout root is assumed to be the parent
# directory of this script (blasst-block-skip branch).
# Usage: ./run_noncausal.sh [/path/to/ruler_prefill_repro] [/path/to/aiter]
set -eu

REPRO_DIR="${1:-$(cd "$(dirname "$0")" && pwd)}"
AITER_DIR="${2:-$(cd "${REPRO_DIR}/.." && pwd)}"

mkdir -p "${REPRO_DIR}/hf_cache"

VIDEO_GID="$(getent group video | cut -d: -f3)"
RENDER_GID="$(getent group render | cut -d: -f3)"

docker run --rm \
  --name aiter_blasst_bench_noncausal \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add "${VIDEO_GID}" \
  --group-add "${RENDER_GID}" \
  --security-opt seccomp=unconfined \
  -e ROCR_VISIBLE_DEVICES \
  -e AITER_TRITON_ONLY=1 \
  -e BLASST_CAUSAL=0 \
  -e BLASST_INPUT_FILE=/data/ruler_qa1_ctx32768.jsonl \
  --ipc=host \
  --shm-size=16g \
  -v "${AITER_DIR}:/workspace/aiter-blasst" \
  -v "${REPRO_DIR}/hf_cache:/root/.cache/huggingface" \
  -v "${REPRO_DIR}/data/ruler_qa1_ctx32768.jsonl:/data/ruler_qa1_ctx32768.jsonl:ro" \
  --entrypoint bash \
  rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0 \
  -c "cd /workspace/aiter-blasst && pip install -q transformers accelerate && pip install -e . -q --no-build-isolation && python3 op_tests/op_benchmarks/triton/bench_mha_blasst.py"
