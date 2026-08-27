# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Random-input speedup benchmark for BLASST block-skipping in the Triton MHA kernel.

Times the SAME kernel at `block_skip_threshold=0` (dense) against threshold > 0,
on synthetic Q/K/V.

Shape, causality and thresholds are all flags, so this can be matched against
bench_unified_attention_blasst.py for a like-for-like comparison.
Each row also carries `row_skip` and `tile_elide` from an offline replay of the
kernel's own skip decision (see `measure_sparsity`), so a speedup can be read
against how much the kernel actually skipped rather than assumed. Timing alone
cannot distinguish "skipped little" from "skipped a lot but could not cash it in".

Env:
  AITER_TRITON_ONLY=1    recommended so `import aiter` skips the C++ ops build.

Run:
  python op_tests/op_benchmarks/triton/bench_mha_blasst.py \
      --nq 64 --nkv 4 --seqlens 65536 --causal
"""

import math
import time

import torch

from aiter.ops.triton.attention.mha import flash_attn_func

DEFAULT_NUM_Q_HEADS = 64
DEFAULT_NUM_KV_HEADS = 4
DEFAULT_HEAD_DIM = 128
DEFAULT_SEQLENS = [16384, 65536]

# Shared 15-point threshold list, applied identically across both seqlens
# (one fixed threshold list reused
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


LOG2E = 1.4426950408889634


def kernel_blocks(head_dim):
    """BLOCK_M / BLOCK_N the kernel will actually use, from its own config loader.

    BLOCK_N is the K/V tile width and therefore the granularity of every skip
    decision; BLOCK_M is how many query rows share a program and so must all
    agree before a tile is actually elided. Hardcoding either would silently
    mis-measure sparsity on any arch whose config differs.
    """
    from aiter.ops.triton._triton_kernels.attention.mha import _get_config

    cfg = _get_config(False, torch.bfloat16, has_pe=False, head_dim_v=head_dim)
    return int(cfg["BLOCK_M"]), int(cfg["BLOCK_N"])


def measure_sparsity(q, k, threshold, block_m, block_n, causal, max_blocks=8):
    """Replay the kernel's skip decision in PyTorch and report how much it skips.

    Mirrors the kernel: scores in log2 space, causal mask, and a running max `M`
    carried across tiles that only non-skipped tiles update. The comparison is
    `tile_max - M` with M the max BEFORE this tile -- using the post-update max is
    the form that yields NaN for thresholds above 1.0.

    Returns (row_skip, tile_elide):
      row_skip    fraction of (row, tile) pairs whose row contributes nothing
      tile_elide  fraction of tiles where ALL block_m rows skip -- only this one
                  saves work, since the V load and P@V are elided per tile.
    Query blocks are sampled; a full replay at 64K costs more than the benchmark.
    """
    if threshold <= 0:
        return 0.0, 0.0
    log2_thr = math.log(threshold) * LOG2E
    # q,k are (B, S, H, D) here; GQA means several q heads share one kv head.
    _, seqlen, nq, head_dim = q.shape
    nkv = k.shape[2]
    qpkv = nq // nkv
    scale = head_dim**-0.5 * LOG2E

    n_blocks = seqlen // block_m
    if n_blocks == 0:
        return 0.0, 0.0
    stride = max(1, n_blocks // max_blocks)
    sampled = list(range(0, n_blocks, stride))[:max_blocks]

    rows_skipped = rows_total = tiles_elided = tiles_total = 0
    for h in range(nq):
        kv_h = h // qpkv
        k_h = k[0, :, kv_h, :].float()
        for qb in sampled:
            q0 = qb * block_m
            qblk = q[0, q0:q0 + block_m, h, :].float()
            qpos = torch.arange(q0, q0 + block_m, device=q.device).unsqueeze(1)
            M = torch.full((block_m,), float("-inf"), device=q.device)
            n_tiles = ((q0 + block_m) if causal else seqlen)
            n_tiles = (n_tiles + block_n - 1) // block_n
            for ti in range(n_tiles):
                k0, k1 = ti * block_n, min((ti + 1) * block_n, seqlen)
                s = (qblk @ k_h[k0:k1].T) * scale
                if causal:
                    kpos = torch.arange(k0, k1, device=q.device).unsqueeze(0)
                    s = s.masked_fill(kpos > qpos, float("-inf"))
                tile_max = s.max(dim=1).values
                if not torch.isfinite(tile_max).any():
                    continue
                skip = (tile_max - M) < log2_thr
                M = torch.where(skip, M, torch.maximum(M, tile_max))
                rows_skipped += int(skip.sum()); rows_total += skip.numel()
                tiles_elided += int(bool(skip.all())); tiles_total += 1
    return (rows_skipped / max(rows_total, 1), tiles_elided / max(tiles_total, 1))


def make_qkv(seqlen, seed, nq, nkv, head_dim):
    torch.manual_seed(seed)

    def g(h):
        return torch.randn(1, seqlen, h, head_dim, dtype=torch.bfloat16, device="cuda")

    return g(nq), g(nkv), g(nkv)


def parse_args():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nq", type=int, default=DEFAULT_NUM_Q_HEADS)
    ap.add_argument("--nkv", type=int, default=DEFAULT_NUM_KV_HEADS)
    ap.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    ap.add_argument("--seqlens", type=str,
                    default=",".join(str(s) for s in DEFAULT_SEQLENS))
    # unified_attention is causal-only, so --causal is what makes the two
    # benchmarks comparable. Off by default to preserve prior behaviour.
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--thresholds", type=str, default="")
    ap.add_argument("--skip-sparsity", action="store_true",
                    help="skip the offline skip-rate replay (timings only)")
    ap.add_argument("--sparsity-blocks", type=int, default=8,
                    help="query blocks sampled per head by the replay. The replay "
                         "is O(heads x blocks x tiles) and at 64K costs more than "
                         "the benchmark itself, so it samples. Whatever value is "
                         "used is printed in the header -- never assume 8.")
    ap.add_argument("--digest", action="store_true",
                    help="also print a content hash of each BLASST output. Inputs "
                         "are seeded, so a kernel change that is meant to be a "
                         "pure no-op elision must leave every digest identical; "
                         "rel_diff at 4 decimals can hide a small real change.")
    return ap.parse_args()


def rel_diff(a, b):
    """Mean relative difference between two outputs.

    `b` is the SAME kernel at threshold=0, so this is the size of the BLASST
    approximation, not an error against ground truth (op_tests/triton_tests/
    attention/test_mha_blasst.py answers that). It doubles as a regression
    tripwire: an optimisation that only removes no-ops must not move it at all.
    """
    return (
        (a.float() - b.float()).abs().mean() / b.float().abs().mean().clamp_min(1e-6)
    ).item()


def digest(t):
    """Short content hash of a tensor's raw bytes (bit-exactness check)."""
    import hashlib

    return hashlib.md5(t.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()[:10]


def main():
    args = parse_args()
    seqlens = [int(s) for s in args.seqlens.split(",")]
    thresholds = ([float(x) for x in args.thresholds.split(",")]
                  if args.thresholds else THRESHOLDS)
    # TINY_THRESHOLD is prepended so the fixed skip-check overhead is always
    # measured -- but an explicit --thresholds list may already contain it, and
    # running the same point twice makes the grid silently differ from the one
    # that was asked for. Dedupe, keeping the requested order.
    grid, seen = [], set()
    for t in [TINY_THRESHOLD] + thresholds:
        if t not in seen:
            seen.add(t)
            grid.append(t)
    if not torch.cuda.is_available():
        print("ERROR: GPU required.")
        return

    block_m, block_n = kernel_blocks(args.head_dim)

    print("#" * 100)
    print("# BLASST block-skipping, MHA kernel, random inputs")
    print(f"# BS=1  dH={args.head_dim}  num_q_heads={args.nq}  "
          f"num_kv_heads={args.nkv} (GQA)  causal={args.causal}")
    print(f"# Device: {torch.cuda.get_device_name(0)}")
    print(f"# Kernel tiling (from _get_config): BLOCK_M={block_m} BLOCK_N={block_n}")
    if args.skip_sparsity:
        print("# Sparsity replay: OFF (--skip-sparsity)")
    else:
        print(f"# Sparsity replay: ON, sampling {args.sparsity_blocks} query "
              f"block(s) of {block_m} rows per head, all {args.nq} heads")
    print(f"# Thresholds: {', '.join(f'{t:g}' for t in grid)}")
    print("#" * 100)

    for seqlen in seqlens:
        q, k, v = make_qkv(seqlen, seqlen, args.nq, args.nkv, args.head_dim)
        dense_ms = benchmark_fn(lambda: flash_attn_func(q, k, v, causal=args.causal, block_skip_threshold=0.0))
        dense_out = flash_attn_func(q, k, v, causal=args.causal, block_skip_threshold=0.0)

        print(f"\n=== seqlen = {seqlen // 1024}k ===")
        dig_hdr = f" {'digest':>11}" if args.digest else ""
        print(f"{'threshold':>12} {'time/ms':>10} {'Speedup':>8} {'rel_diff':>9} "
              f"{'row_skip':>9} {'tile_elide':>11}{dig_hdr}")
        print("-" * (66 + (12 if args.digest else 0)))
        print(f"{'0(dense)':>12} {dense_ms:>10.3f} {'1.000x':>8} {0.0:>9.4f} "
              f"{'0.0%':>9} {'0.0%':>11}"
              f"{(' ' + format(digest(dense_out), '>11')) if args.digest else ''}")
        replay_s = 0.0
        for t in grid:
            out = flash_attn_func(q, k, v, causal=args.causal, block_skip_threshold=t)
            assert torch.isfinite(out).all(), f"seqlen={seqlen} threshold={t}: non-finite output"
            err = rel_diff(out, dense_out)
            dig = f" {digest(out):>11}" if args.digest else ""
            ms = benchmark_fn(lambda t=t: flash_attn_func(q, k, v, causal=args.causal, block_skip_threshold=t))
            speedup = dense_ms / ms
            label = f"{t:.3g}" if t != TINY_THRESHOLD else f"{t:g}(tiny)"
            if args.skip_sparsity:
                # Not measured, not zero -- printing 0.0% here would read as "no
                # skipping happened", which is a different claim entirely.
                row_str, tile_str = f"{'-':>9}", f"{'-':>11}"
            else:
                t0 = time.time()
                row_s, tile_s = measure_sparsity(
                    q, k, t, block_m, block_n, args.causal,
                    max_blocks=args.sparsity_blocks)
                row_str, tile_str = f"{row_s:>9.1%}", f"{tile_s:>11.1%}"
                replay_s += time.time() - t0
            print(f"{label:>12} {ms:>10.3f} {speedup:>7.3f}x {err:>9.4f} "
                  f"{row_str} {tile_str}{dig}")
        del dense_out, out, q, k, v
        torch.cuda.empty_cache()

        if replay_s:
            print(f"\n  (sparsity replay cost {replay_s:.1f}s for this seqlen, "
                  f"{args.sparsity_blocks} sampled query block(s)/head; "
                  f"--skip-sparsity omits it)")


if __name__ == "__main__":
    main()
