# Recovering the FA warp layout for BLASST via a Triton compiler patch

## What this is

BLASST is a block-skip attention kernel (see `BLASST_BLOCK_SKIP.md`): it
wraps the P@V matmul in `if all_skip: ...` so that fully-masked-out blocks
skip both the V-load and the matmul entirely. On AMD GPUs (ROCm/Triton),
this kernel ends up with a worse register layout than a plain dense
attention kernel, even before any blocks are actually skipped.
`chaindot_force_patch/` contains a small, opt-in Triton compiler patch that
fixes the register layout without touching the kernel's own source code or
giving up the skip logic, plus the Docker image and run scripts used to
build and benchmark it against this repo's fixed BLASST kernel.

## Why the layout regresses

Triton's AMD backend has a fast path for flash-attention-style kernels: when
it recognizes that a QK matmul and a P@V matmul form a "chain" (the QK
result feeds into softmax, which feeds into the P@V matmul, and all of this
happens in one accumulation loop), it assigns a `warpsPerCTA=[num_warps, 1]`
layout that is tuned for exactly this pattern. This detection
(`isChainDotHead` / `isChainDotTail` in
`third_party/amd/lib/TritonAMDGPUToLLVM/Utility.cpp`) works by walking the
def-use chain between the two matmuls and checking that every operation in
between lives in the same MLIR region as the matmuls themselves.

The block-skip kernel breaks this assumption: because the P@V matmul is
wrapped in `if all_skip: ...`, it sits in a nested `scf.if` region, one
level below the QK matmul's region. The chain-dot walk stops the instant it
would have to cross that region boundary, so detection fails and the
compiler falls back to a generic `warpsPerCTA=[2, 2]` layout instead of the
tuned `[4, 1]` layout a dense kernel gets. This is a fixed cost paid on
every launch of the block-skip kernel, regardless of how many blocks
actually get skipped at runtime, on top of whatever speedup the skip itself
provides.

## The patch

`chaindot_force_patch/patch/0001-force-chain-dot-across-scf-if.patch` adds a
new, default-off environment variable, `TRITON_HIP_FORCE_CHAIN_DOT_ACROSS_IF`.
When set, the chain-dot detection's region-equality check is relaxed to also
accept two regions related by exactly one level of `scf.if` nesting, in
either direction. This is enough for `isChainDotHead`/`isChainDotTail` to
see through the `if all_skip: ...` wrapper and report the same chain-dot
match a dense kernel would get, so the compiler's existing tuned-layout
logic fires unmodified once detection succeeds. No change is needed to the
layout-selection logic itself.

Two design choices worth noting:

- The patch is scoped to exactly one level of `scf.if` nesting (matching
  this kernel's structure: one loop body containing one conditional around
  the second matmul). It is not a general multi-level chain-dot
  generalization, and would need re-verification against a kernel with
  deeper nesting.
- The flag is opt-in and off by default, so it cannot change behavior for
  any other kernel or build unless explicitly set. This also means both
  "forced" and "unforced" behavior can be compared from the same built
  image, just by toggling the environment variable at container run time.

`chaindot_force_patch/docker/Dockerfile` builds Triton from
triton-lang/triton's current `main` HEAD (cloned fresh at image-build time
via `git clone --depth 1`) with this patch applied on top, rather than a
pinned commit -- so the image always reflects a recent upstream Triton.

## Verified results (reference run, pinned commit `71d121b0690ad2615f19c85a1c29b08b55a9f80e`)

The numbers below are from an earlier exploratory run of this same patch
against a pinned triton-lang/triton commit (self-reporting as `3.8.0`), not
against the `main`-HEAD image this directory now builds. They are kept here
as a reference for the kind of effect to expect; re-running against current
`main` may shift the exact numbers, though the underlying mechanism (and
thus the qualitative pattern of low-threshold overhead vs. clear wins as
threshold increases) is not expected to change.

### Register pressure (compiled kernel, layer 0, causal, ctx=32768)

| | Block-skip kernel, unpatched | **Block-skip kernel, patched (forced)** | Dense kernel (reference) |
|---|---|---|---|
| `warpsPerCTA` | [2, 2] | **[4, 1]** | [4, 1] |
| `ScratchSize` | 260 B | **60 B** | 156 B |
| `vgpr_spill_count` | 92 | **15** | 38 |
| `sgpr_spill_count` | 53 | **44** | 49 |
| `scratch_load`/`scratch_store` count | 169 | **16** | 69 |

Direct inspection of the compiled kernel (TTGIR) confirms the P@V matmul is
still located inside `if all_skip: ...` after patching, meaning the actual
compute-skip is preserved, not given up to get the better layout.

### Correctness

Ran the kernel's existing correctness test suite (bf16, causal and
non-causal, several thresholds) with the patch active. Result: same pass
rate and same numerical tolerances as the unpatched kernel.

### Isolated branch overhead

Even with a threshold so low the skip almost never triggers, the patched
kernel still pays a small fixed cost relative to dense, roughly 8 to 12
percent, on most sampled points -- the cost of the hardware branch
instruction (`if all_skip: ...` compiles to an actual conditional branch)
being present at all, independent of whether it is ever taken. This shows
up most clearly at low thresholds and explains why the patched kernel can
be a few percent under dense there even though it always beats the
unpatched kernel at the same point.

Full task/context grid-sweep numbers from that exploratory run are not
reproduced here since they predate this repo's `m_ij` -> `m_i` NaN fix (see
`BLASST_BLOCK_SKIP_NAN_FIX.md`); re-run the sweep against the fixed kernel
before relying on absolute numbers.

## Contents

- `chaindot_force_patch/patch/0001-force-chain-dot-across-scf-if.patch` --
  the Triton patch.
- `chaindot_force_patch/docker/Dockerfile` +
  `chaindot_force_patch/docker/build_image.sh` -- builds a Triton compiler
  from triton-lang/triton `main` HEAD with the patch applied, on top of the
  same ROCm PyTorch base image used elsewhere in this repo.
- `chaindot_force_patch/run_mlsys26_hopper_prefill.sh <0|1>` -- runs
  `op_tests/op_benchmarks/triton/bench_mlsys26_hopper_prefill.py` (random
  Q/K/V, no external data needed) against this checkout inside the patched
  image, toggling the patch on (`1`) or off (`0`).
- `chaindot_force_patch/run_ruler.sh <causal|noncausal> <0|1>` -- runs
  `op_tests/op_benchmarks/triton/bench_mha_blasst.py` against the RULER
  `qa_1`/ctx=32768 prompt bundled in `ruler_prefill_repro/`, inside the
  patched image, toggling the patch on or off.

## How to run it

1. Build the image:

   ```
   ./chaindot_force_patch/docker/build_image.sh
   ```

2. MLSys 2026 Hopper-prefill-shape benchmark, patch on vs. off:

   ```
   ./chaindot_force_patch/run_mlsys26_hopper_prefill.sh 0    # unpatched
   ./chaindot_force_patch/run_mlsys26_hopper_prefill.sh 1    # patched
   ```

3. RULER-based benchmark, patch on vs. off:

   ```
   ./chaindot_force_patch/run_ruler.sh causal 0      # unpatched
   ./chaindot_force_patch/run_ruler.sh causal 1      # patched
   ./chaindot_force_patch/run_ruler.sh noncausal 1
   ```

Both run scripts assume this repo is checked out with `chaindot_force_patch/`
directly under its root (matching where these files live); pass the aiter
checkout path explicitly as an argument if your layout differs. Each script
writes its Triton kernel cache to `chaindot_force_patch/triton_cache/`,
created automatically on first run and reused afterward (separate `force0`/
`force1` subdirectories, since the two configurations compile differently).
`run_ruler.sh` reuses `ruler_prefill_repro/hf_cache/` and
`ruler_prefill_repro/data/ruler_qa1_ctx32768.jsonl` rather than maintaining
its own copies.
