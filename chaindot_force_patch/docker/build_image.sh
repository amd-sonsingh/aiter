#!/bin/sh
# Build the chain-dot-force-patch experiment image (Triton main HEAD +
# 0001-force-chain-dot-across-scf-if.patch on top of the ROCm PyTorch base).
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_TAG="${IMAGE_TAG:-aiter-blasst/triton-chaindot-patch-main:rocm7.2.4-py3.12-torch2.10.0}"

docker build \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE_TAG}" \
  "${SCRIPT_DIR}/../patch"

echo "Built image: ${IMAGE_TAG}"
