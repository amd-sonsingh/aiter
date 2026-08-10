# Builds a Triton compiler from triton-lang/triton's current `main` HEAD
# with 0001-force-chain-dot-across-scf-if.patch applied on top, then installs
# it over the ROCm PyTorch image's stock Triton.
#
# Why: BLASST's P@V dot sits inside an `scf.if all_skip: ...` region, which
# breaks the AMD backend's isChainDotHead/isChainDotTail detection (they only
# look within a single MLIR region), so planWarps() falls back to a worse
# warpsPerCTA=[2,2] layout instead of the FA-optimized [4,1] layout dense
# gets. This patch adds an opt-in env var, TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF,
# that lets the chain-dot detectors see across exactly one level of scf.if
# nesting, so [4,1] can be recovered without restructuring the kernel source
# and without giving up the V-load/P@V-matmul skip. See ../../CHAINDOT_FORCE_PATCH.md
# for the full write-up.
#
# Unlike the original scratch experiment (which pinned a specific
# triton-lang/triton commit before applying the patch), this Dockerfile
# builds against whatever `main` HEAD is at image-build time, matching the
# same-repo convention used for unrelated Triton-regression investigations
# (see blasst_vllm's docker/Dockerfile.triton_master).
FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0

ENV MAX_JOBS=16

# transformers/accelerate must be installed BEFORE the Triton swap below:
# torch's install metadata pins an exact Triton version, and installing
# transformers/accelerate after breaking that pin makes pip's resolver
# silently re-resolve torch+triton to non-ROCm wheels.
RUN pip install --no-cache-dir transformers accelerate \
    && python3 -c "import torch; assert torch.version.hip is not None, 'lost ROCm torch build'; print('torch (ROCm):', torch.__version__)"

COPY 0001-force-chain-dot-across-scf-if.patch /opt/0001-force-chain-dot-across-scf-if.patch

# --no-build-isolation skips pyproject.toml's [build-system].requires, so
# ninja/cmake/nanobind must be installed explicitly first (nanobind version
# pinned to match triton's own pyproject.toml requirement).
RUN pip uninstall -y triton pytorch-triton pytorch-triton-rocm triton-rocm amd-triton || true \
    && pip install --no-cache-dir ninja cmake 'nanobind==2.10.2' \
    && git clone --depth 1 https://github.com/triton-lang/triton.git /opt/triton-src \
    && cd /opt/triton-src \
    && git apply /opt/0001-force-chain-dot-across-scf-if.patch \
    && git rev-parse HEAD > /opt/triton-src-commit.txt \
    && echo "patched: 0001-force-chain-dot-across-scf-if.patch" >> /opt/triton-src-commit.txt \
    && pip install --no-cache-dir -e . --no-build-isolation \
    && python3 -c "import triton; print('Triton built from source:', triton.__version__)" \
    && python3 -c "import torch; assert torch.version.hip is not None, 'lost ROCm torch build after triton swap'; print('torch (ROCm, post-triton-swap):', torch.__version__)"

WORKDIR /workspace
