# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MLSys 2026 Hopper-prefill-shape speedup benchmark for BLASST block-skipping.

Reproduces the fixed benchmark shape used by the MLSys 2026 BLASST paper's
artifact evaluation, Hopper prefill benchmark (`blasst-ae-mlsys26/hopper_prefill`):
BS=1, head_dim=128, 64 query heads / 4 KV heads (GQA), non-causal, random
Q/K/V, seqlen in {16384, 65536} -- for direct comparison against the paper's
own reported speedup numbers at matched sparsity.

Unlike op_tests/op_benchmarks/triton/bench_mha_blasst.py (which captures
REAL Qwen3-8B activations, since block skipping depends on the attention
score distribution and random tensors rarely produce whole skippable
blocks), this benchmark intentionally uses random Q/K/V to match the
paper artifact's own methodology. Random-tensor sparsity curves are
sharper/steeper than real-model ones (see BLASST_BLOCK_SKIP.md), so absolute
sparsity% here is not directly comparable in kind to bench_mha_blasst.py's
per-layer numbers -- both are reported as measured, not adjusted.

See BLASST_BLOCK_SKIP.md, BLASST_BLOCK_SKIP_NAN_FIX.md, and
MLSYS26_HOPPER_PREFILL_REPRO.md for the full write-up.

Env:
  AITER_TRITON_ONLY=1    recommended so `import aiter` skips the C++ ops build.

Run:
  AITER_TRITON_ONLY=1 python op_tests/op_benchmarks/triton/bench_mlsys26_hopper_prefill.py
"""

import time

import torch

from aiter.ops.triton.attention.mha import flash_attn_func

NUM_Q_HEADS = 64
NUM_KV_HEADS = 4
HEAD_DIM = 128
SEQLENS = [16384, 65536]

# Shared 15-point threshold list, applied identically across both seqlens
# (matching the Hopper artifact's own convention of one fixed list reused
# across seqlens, rather than per-seqlen calibration), plus a tiny point
# that isolates the fixed skip-check overhead at ~0% real sparsity.
TINY_THRESHOLD = 1e-9
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.3, 1.7, 2, 4, 6, 8, 10, 12]

WARMUP = 5
REPEAT = 20


def benchmark_fn(fn, warmup=WARMUP, repeat=REPEAT):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / repeat * 1000.0


def make_qkv(seqlen, seed):
    torch.manual_seed(seed)
    g = lambda h: torch.randn(1, seqlen, h, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    return g(NUM_Q_HEADS), g(NUM_KV_HEADS), g(NUM_KV_HEADS)


def main():
    if not torch.cuda.is_available():
        print("ERROR: GPU required.")
        return

    print("#" * 100)
    print("# MLSys 2026 BLASST Hopper-prefill-shape reproduction")
    print(f"# BS=1  dH={HEAD_DIM}  num_q_heads={NUM_Q_HEADS}  num_kv_heads={NUM_KV_HEADS} (GQA)  causal=False")
    print(f"# Device: {torch.cuda.get_device_name(0)}")
    print("#" * 100)

    for seqlen in SEQLENS:
        q, k, v = make_qkv(seqlen, seed=seqlen)
        dense_ms = benchmark_fn(lambda: flash_attn_func(q, k, v, causal=False, block_skip_threshold=0.0))

        print(f"\n=== seqlen = {seqlen // 1024}k ===")
        print(f"{'threshold':>12} {'time/ms':>10} {'Speedup':>8}")
        print("-" * 34)
        print(f"{'0(dense)':>12} {dense_ms:>10.3f} {'1.000x':>8}")
        for t in [TINY_THRESHOLD] + THRESHOLDS:
            out = flash_attn_func(q, k, v, causal=False, block_skip_threshold=t)
            assert torch.isfinite(out).all(), f"seqlen={seqlen} threshold={t}: non-finite output"
            ms = benchmark_fn(lambda t=t: flash_attn_func(q, k, v, causal=False, block_skip_threshold=t))
            speedup = dense_ms / ms
            label = f"{t:.3g}" if t != TINY_THRESHOLD else f"{t:g}(tiny)"
            print(f"{label:>12} {ms:>10.3f} {speedup:>7.3f}x")


if __name__ == "__main__":
    main()
