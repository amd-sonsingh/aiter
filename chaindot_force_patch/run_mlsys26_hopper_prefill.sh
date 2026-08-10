#!/bin/sh
# Run op_tests/op_benchmarks/triton/bench_mlsys26_hopper_prefill.py against
# the chain-dot-force-patch image (Triton main HEAD + patch), toggling
# TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF. No RULER data or HF model needed
# (pure random-Q/K/V kernel microbenchmark) -- unlike run_ruler.sh.
#
# Usage: ./run_mlsys26_hopper_prefill.sh <0|1> [aiter_blasst_dir]
set -eu

FORCE="${1:?Usage: run_mlsys26_hopper_prefill.sh <0|1> [aiter_blasst_dir]}"
AITER_DIR="${2:-$(cd "$(dirname "$0")/.." && pwd)}"
CACHE_ROOT="${CACHE_ROOT:-$(cd "$(dirname "$0")" && pwd)/triton_cache}"
IMAGE_TAG="${IMAGE_TAG:-aiter-blasst/triton-chaindot-patch-main:rocm7.2.4-py3.12-torch2.10.0}"

case "${FORCE}" in
  0|1) ;;
  *) echo "Unknown force value: ${FORCE} (expected 0 or 1)" >&2; exit 1 ;;
esac

CACHE_DIR="${CACHE_ROOT}/force${FORCE}"
mkdir -p "${CACHE_DIR}"

VIDEO_GID="$(getent group video | cut -d: -f3)"
RENDER_GID="$(getent group render | cut -d: -f3)"

docker run --rm \
  --name "aiter_blasst_chaindot_patch_mlsys26_force${FORCE}" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add "${VIDEO_GID}" \
  --group-add "${RENDER_GID}" \
  --security-opt seccomp=unconfined \
  -e ROCR_VISIBLE_DEVICES \
  -e AITER_TRITON_ONLY=1 \
  -e AITER_USE_SYSTEM_TRITON=1 \
  -e TRITON_CACHE_DIR=/triton_cache \
  -e TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF="${FORCE}" \
  --ipc=host \
  --shm-size=16g \
  -v "${AITER_DIR}:/workspace/aiter-blasst" \
  -v "${CACHE_DIR}:/triton_cache" \
  --entrypoint bash \
  "${IMAGE_TAG}" \
  -c "set -eu
      cd /workspace/aiter-blasst
      pip install -e . -q --no-build-isolation
      python3 -c 'import torch, triton; assert torch.cuda.is_available(), \"ROCm support lost\"; print(\"Triton in use:\", triton.__version__); print(\"Torch in use:\", torch.__version__)'
      cat /opt/triton-src-commit.txt
      echo \"TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF=\${TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF:-}\"
      python3 op_tests/op_benchmarks/triton/bench_mlsys26_hopper_prefill.py
     "
