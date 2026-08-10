#!/bin/sh
# Run op_tests/op_benchmarks/triton/bench_mha_blasst.py (RULER qa_1/ctx=32768
# real-prompt input, from ../ruler_prefill_repro/) against the
# chain-dot-force-patch image (Triton main HEAD + patch), toggling
# TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF.
#
# Usage: ./run_ruler.sh <causal|noncausal> <0|1> [aiter_blasst_dir] [ruler_repro_dir]
set -eu

MODE="${1:?Usage: run_ruler.sh <causal|noncausal> <0|1> [aiter_blasst_dir] [ruler_repro_dir]}"
FORCE="${2:?Usage: run_ruler.sh <causal|noncausal> <0|1> [aiter_blasst_dir] [ruler_repro_dir]}"
AITER_DIR="${3:-$(cd "$(dirname "$0")/.." && pwd)}"
REPRO_DIR="${4:-${AITER_DIR}/ruler_prefill_repro}"
CACHE_ROOT="${CACHE_ROOT:-$(cd "$(dirname "$0")" && pwd)/triton_cache}"
IMAGE_TAG="${IMAGE_TAG:-aiter-blasst/triton-chaindot-patch-main:rocm7.2.4-py3.12-torch2.10.0}"

case "${MODE}" in
  causal) BLASST_CAUSAL=1 ;;
  noncausal) BLASST_CAUSAL=0 ;;
  *) echo "Unknown mode: ${MODE} (expected causal or noncausal)" >&2; exit 1 ;;
esac

case "${FORCE}" in
  0|1) ;;
  *) echo "Unknown force value: ${FORCE} (expected 0 or 1)" >&2; exit 1 ;;
esac

CACHE_DIR="${CACHE_ROOT}/force${FORCE}"
mkdir -p "${CACHE_DIR}" "${REPRO_DIR}/hf_cache"

VIDEO_GID="$(getent group video | cut -d: -f3)"
RENDER_GID="$(getent group render | cut -d: -f3)"

docker run --rm \
  --name "aiter_blasst_chaindot_patch_ruler_${MODE}_force${FORCE}" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add "${VIDEO_GID}" \
  --group-add "${RENDER_GID}" \
  --security-opt seccomp=unconfined \
  -e ROCR_VISIBLE_DEVICES \
  -e AITER_TRITON_ONLY=1 \
  -e AITER_USE_SYSTEM_TRITON=1 \
  -e BLASST_CAUSAL="${BLASST_CAUSAL}" \
  -e BLASST_INPUT_FILE=/data/ruler_qa1_ctx32768.jsonl \
  -e TRITON_CACHE_DIR=/triton_cache \
  -e TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF="${FORCE}" \
  --ipc=host \
  --shm-size=16g \
  -v "${AITER_DIR}:/workspace/aiter-blasst" \
  -v "${CACHE_DIR}:/triton_cache" \
  -v "${REPRO_DIR}/hf_cache:/root/.cache/huggingface" \
  -v "${REPRO_DIR}/data/ruler_qa1_ctx32768.jsonl:/data/ruler_qa1_ctx32768.jsonl:ro" \
  --entrypoint bash \
  "${IMAGE_TAG}" \
  -c "set -eu
      cd /workspace/aiter-blasst
      pip install -e . -q --no-build-isolation
      python3 -c 'import torch, triton; assert torch.cuda.is_available(), \"ROCm support lost\"; print(\"Triton in use:\", triton.__version__); print(\"Torch in use:\", torch.__version__)'
      cat /opt/triton-src-commit.txt
      echo \"TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF=\${TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF:-}\"
      python3 op_tests/op_benchmarks/triton/bench_mha_blasst.py
     "
