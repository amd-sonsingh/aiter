# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Speedup benchmark for BLASST block skipping in Triton `unified_attention`.

Times the SAME kernel at `block_skip_threshold=0` (dense) against threshold > 0
(BLASST) and reports speedup, how far the answer moved, and how much skipping
actually occurred.

Random inputs only: no dependency on `transformers` or a model on disk.

Read the sparsity columns rather than the threshold. The threshold is calibrated
against a given activation distribution, so on random inputs it produces a
different amount of skipping than it would on a model's own activations. The
kernel responds to the achieved sparsity, so compare results at matched
`tile_elide`.

Sparsity is replayed in PyTorch (`simulate_sparsity`) for both kernels the
wrapper can dispatch, rather than counted in-kernel: a counter would add an
atomic per tile and perturb the timings.

Run:
    python op_tests/op_benchmarks/triton/bench_unified_attention_blasst.py
    python op_tests/op_benchmarks/triton/bench_unified_attention_blasst.py \\
        --shapes 64x4 --seqlens 16384,65536 --shapes-3d 128x16384
"""

import argparse
import csv
import logging
import math
import os
import sys
import time

import torch

logger = logging.getLogger("aiter")

LOG2E = 1.4426950408889634

# 1e-9 is not a useful sparsity setting; it isolates the fixed cost of the skip
# check itself (a max, a compare, a where per tile) from any benefit.
DEFAULT_THRESHOLDS = [1e-9, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]

WARMUP = 5
REPEAT = 20


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------
def benchmark_fn(fn, warmup=WARMUP, repeat=REPEAT):
    """Median-of-repeats wall time in ms. Median, not mean: an occasional
    scheduler hiccup should not decide a speedup claim."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def make_paged_inputs(seqlen, num_q_heads, num_kv_heads, head_dim, block_size, seed=0):
    """Random full-prefill inputs in the paged layout (query_len == kv_len)."""
    torch.manual_seed(seed)
    num_blocks = (seqlen + block_size - 1) // block_size

    query = torch.randn(
        seqlen, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda"
    )
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_dim,
        dtype=torch.bfloat16, device="cuda",
    )
    value_cache = torch.randn_like(key_cache)
    # Identity block table: block i holds positions [i*block_size, ...). Keeps
    # the sparsity replay and the kernel reading the same K/V order.
    block_tables = torch.arange(
        num_blocks, dtype=torch.int32, device="cuda"
    ).unsqueeze(0)

    return _pack(query, key_cache, value_cache, block_tables, seqlen, head_dim)


def make_paged_inputs_qkv(q_len, kv_len, num_q_heads, num_kv_heads, head_dim,
                          block_size, seed=0):
    """Random inputs where query length and KV length differ.

    make_paged_inputs() ties them together (full prefill), which always routes to
    the 2D kernel. Few queries over long KV is what sends the wrapper down the 3D
    path, so measuring 3D needs this.
    """
    torch.manual_seed(seed)
    num_blocks = (kv_len + block_size - 1) // block_size
    query = torch.randn(q_len, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_dim,
        dtype=torch.bfloat16, device="cuda",
    )
    value_cache = torch.randn_like(key_cache)
    block_tables = torch.arange(
        num_blocks, dtype=torch.int32, device="cuda"
    ).unsqueeze(0)
    d = _pack(query, key_cache, value_cache, block_tables, q_len, head_dim)
    d["seqused_k"] = torch.tensor([kv_len], dtype=torch.int32, device="cuda")
    d["max_seqlen_k"] = kv_len
    d["seqlen"] = kv_len
    return d


def _pack(query, key_cache, value_cache, block_tables, seqlen, head_dim):
    return dict(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        output=torch.empty_like(query),
        cu_seqlens_q=torch.tensor([0, seqlen], dtype=torch.int32, device="cuda"),
        seqused_k=torch.tensor([seqlen], dtype=torch.int32, device="cuda"),
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        scale=head_dim**-0.5,
        seqlen=seqlen,
    )


def run_attention(inp, threshold):
    from aiter.ops.triton.attention.unified_attention import unified_attention

    unified_attention(
        q=inp["query"],
        k=inp["key_cache"],
        v=inp["value_cache"],
        out=inp["output"],
        cu_seqlens_q=inp["cu_seqlens_q"],
        seqused_k=inp["seqused_k"],
        max_seqlen_q=inp["max_seqlen_q"],
        max_seqlen_k=inp["max_seqlen_k"],
        softmax_scale=inp["scale"],
        causal=True,
        window_size=(-1, -1),
        block_table=inp["block_tables"],
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        block_skip_threshold=float(threshold),
    )
    return inp["output"]


# --------------------------------------------------------------------------
# sparsity
# --------------------------------------------------------------------------
class _DispatchCapture:
    """Record which kernel the wrapper launches, and its tiling.

    `unified_attention` dispatches to either a 2D kernel or a 3D kernel plus
    `reduce_segments`. The two skip differently, so results are only meaningful
    when labelled with the kernel that produced them. Wraps the wrapper's own
    selectors and reads back what it chose.
    """

    def __init__(self):
        self.kernel = None
        self.cfg = None
        self.num_segments = None

    def __enter__(self):
        import aiter.ops.triton.attention.unified_attention as ua

        self._ua = ua
        self._o2, self._o3 = ua.select_2d_config, ua.select_3d_config

        def w2(*a, **k):
            cfg = self._o2(*a, **k)
            self.kernel, self.cfg = "2D", cfg
            return cfg

        def w3(*a, **k):
            # returns (attn_config, reduce_config) -- a tuple, not a dict
            attn_cfg, reduce_cfg = self._o3(*a, **k)
            self.kernel, self.cfg = "3D", attn_cfg
            self.num_segments = attn_cfg.get("NUM_SEGMENTS_PER_SEQ")
            return attn_cfg, reduce_cfg

        ua.select_2d_config, ua.select_3d_config = w2, w3
        return self

    def __exit__(self, *exc):
        self._ua.select_2d_config, self._ua.select_3d_config = self._o2, self._o3
        return False

    def describe(self, num_q_heads=None, num_kv_heads=None):
        """One line naming the kernel and its tiling.

        Pass the head counts to fill in BLOCK_M/BLOCK_Q on the 3D path, where the
        selector does not return them (the wrapper derives them). Without them
        those two read `None`, which looks like a bug rather than a division of
        labour between the selector and the wrapper.
        """
        if self.kernel is None:
            return "kernel: <not captured>"
        c = self.cfg or {}
        derived = ""
        if c.get("BLOCK_M") is None and num_q_heads and num_kv_heads:
            c = self.effective_cfg(num_q_heads, num_kv_heads)
            derived = " (BLOCK_M/BLOCK_Q derived: not returned by the 3D selector)"
        seg = f" NUM_SEGMENTS={self.num_segments}" if self.kernel == "3D" else ""
        return (
            f"kernel: {self.kernel}{seg}  BLOCK_M={c.get('BLOCK_M')} "
            f"BLOCK_Q={c.get('BLOCK_Q')} TILE_SIZE={c.get('TILE_SIZE')} "
            f"num_warps={c.get('num_warps')} num_stages={c.get('num_stages')}"
            f"{derived}"
        )

    def tiles_per_segment(self, kv_len):
        """Tiles each 3D segment covers, mirroring the kernel's own arithmetic
        (`tiles_per_segment = cdiv(seq_len, num_segments * TILE_SIZE)`).

        Returns None for the 2D kernel, which makes one continuous pass.
        """
        if self.kernel != "3D" or not self.num_segments:
            return None
        tile = (self.cfg or {}).get("TILE_SIZE")
        if not tile:
            return None
        denom = self.num_segments * tile
        return (kv_len + denom - 1) // denom

    def effective_cfg(self, num_q_heads, num_kv_heads):
        """Config with BLOCK_M/BLOCK_Q guaranteed present.

        `select_2d_config` returns them, but `select_3d_config` does NOT -- on the
        3D path the wrapper computes them itself and passes them straight to the
        launch (aiter/ops/triton/attention/unified_attention.py, the
        `BLOCK_M = 16 if num_queries_per_kv <= 16 else next_power_of_2(...)`
        block). Reading them from the captured config therefore yields None for
        3D. Mirror the
        wrapper's arithmetic instead.

        The values differ sharply between paths and that is the point: at 64q/4kv
        the 2D kernel uses BLOCK_M=128, the 3D kernel BLOCK_M=16. A 3D tile needs
        only 16 rows to agree before it can be elided, not 128 -- so 3D is
        *easier* to elide per tile, and its weakness is the per-segment
        running-max restart rather than the block width.
        """
        cfg = dict(self.cfg or {})
        if cfg.get("BLOCK_Q") and cfg.get("BLOCK_M"):
            return cfg
        qpkv = max(1, num_q_heads // num_kv_heads)
        block_m = 16 if qpkv <= 16 else 1 << (qpkv - 1).bit_length()
        cfg["BLOCK_M"] = block_m
        cfg["BLOCK_Q"] = max(1, block_m // qpkv)
        return cfg


def simulate_sparsity(inp, cfg, num_q_heads, num_kv_heads, threshold,
                      tiles_per_segment=None, max_blocks=8):
    """Replay the kernel's skip decision in PyTorch.

    Mirrors the kernel exactly: scores in log2 space, causal mask, and a running
    max `M` carried across tiles that is updated only by non-skipped tiles. The
    comparison is `tile_max - M` where M is the max BEFORE this tile -- using the
    post-update max instead yields NaN for thresholds above 1.0.

    `tiles_per_segment` models the 3D kernel: it resets `M` to -inf at every
    segment boundary, exactly as the kernel does, so the first tile of each
    segment can never skip. Pass None for the 2D kernel. Without this the 3D
    numbers would be the 2D numbers wearing a 3D label.

    Returns two very different numbers, or None if the shape is too small to
    sample honestly:
      row_skip   fraction of (row, tile) pairs whose row contributes nothing.
      tile_elide fraction of (query_block, tile) pairs where ALL BLOCK_M rows
                 skip. **Only this one saves work**: the V load and P@V are
                 elided per tile, not per row.

    Query blocks are sampled (`max_blocks`) because a full replay at 64K is
    slower than the benchmark itself.
    """
    if threshold <= 0:
        return 0.0, 0.0

    log2_thr = math.log(threshold) * LOG2E
    q = inp["query"]
    kv_len = inp["seqlen"]
    head_dim = q.shape[-1]
    kc = inp["key_cache"]
    keys = kc.view(-1, kc.shape[2], head_dim)[:kv_len]  # identity block table

    BLOCK_Q, TILE = cfg["BLOCK_Q"], cfg["TILE_SIZE"]
    qpkv = num_q_heads // num_kv_heads
    scale = inp["scale"] * LOG2E

    # Queries are the LAST q_len positions of the sequence, which is how the
    # kernel indexes them (context_len = seq_len - cur_batch_query_len). For full
    # prefill context_len is 0; for the few-query 3D shapes it is not, and
    # ignoring it would mask away nearly everything and report fake sparsity.
    q_len = q.shape[0]
    context_len = kv_len - q_len

    n_qblocks = q_len // BLOCK_Q  # whole blocks only, so every sample is BLOCK_M rows
    if n_qblocks == 0:
        # Fewer query rows than one block. We could pad, but sparsity would
        # then need fewer rows to agree than the kernel does, overstating it.
        return None
    # Spread samples across the sequence: early blocks see few tiles and skip
    # almost nothing, late blocks see many. Sampling only the start would flatter.
    stride = max(1, n_qblocks // max_blocks)
    sampled = list(range(0, n_qblocks, stride))[:max_blocks]

    # Batch every (query block, kv head) pair into one axis. The running max
    # forces a sequential loop over tiles, but nothing forces one over pairs --
    # looping over both is what made this slower than the benchmark it annotates.
    qblks, qposs, khs = [], [], []
    for qb in sampled:
        q0 = qb * BLOCK_Q
        for kvh in range(num_kv_heads):
            heads = slice(kvh * qpkv, (kvh + 1) * qpkv)
            qblks.append(q[q0 : q0 + BLOCK_Q, heads, :].reshape(-1, head_dim))
            qposs.append(
                torch.arange(
                    context_len + q0, context_len + q0 + BLOCK_Q, device=q.device
                ).repeat_interleave(qpkv)
            )
            khs.append(kvh)
    qb_t = torch.stack(qblks).float()  # (P, BLOCK_M, D)
    qpos = torch.stack(qposs).unsqueeze(2)  # (P, BLOCK_M, 1)
    k_sel = keys[:, khs, :].permute(1, 0, 2).float()  # (P, S, D)

    P, block_m, _ = qb_t.shape
    M = torch.full((P, block_m), float("-inf"), device=q.device)
    rows_skipped = rows_total = 0
    tiles_elided = tiles_total = 0

    # Causality: a tile beyond the last query position of every sampled block
    # contributes nothing, but blocks differ, so mask per pair instead.
    n_tiles = (kv_len + TILE - 1) // TILE
    for t in range(n_tiles):
        if tiles_per_segment and t % tiles_per_segment == 0:
            # 3D kernel: every segment starts with M = -inf, so tile_max - M is
            # +inf and the first tile of the segment is never skipped.
            M = torch.full((P, block_m), float("-inf"), device=q.device)

        k0, k1 = t * TILE, min((t + 1) * TILE, kv_len)
        s = torch.bmm(qb_t, k_sel[:, k0:k1, :].transpose(1, 2)) * scale
        kpos = torch.arange(k0, k1, device=q.device).view(1, 1, -1)
        s = s.masked_fill(kpos > qpos, float("-inf"))
        tile_max = s.max(dim=2).values  # (P, BLOCK_M)

        # A fully-masked tile (entirely in the future for this block) is not a
        # BLASST skip; exclude it so sparsity is not inflated by causality.
        live = torch.isfinite(tile_max).any(dim=1)  # (P,)
        if not bool(live.any()):
            break

        skip = (tile_max - M) < log2_thr
        M = torch.where(skip, M, torch.maximum(M, tile_max))

        rows_skipped += int((skip & live.unsqueeze(1)).sum())
        rows_total += int(live.sum()) * block_m
        tiles_elided += int((skip.all(dim=1) & live).sum())
        tiles_total += int(live.sum())

    return (
        rows_skipped / max(rows_total, 1),
        tiles_elided / max(tiles_total, 1),
    )


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def rel_diff(a, b):
    """Mean relative difference between two outputs.

    Here `b` is always the SAME kernel at threshold=0, so this reports how far
    BLASST moved the answer away from dense -- the size of the approximation,
    NOT an error against ground truth. Correctness is a separate question,
    answered by op_tests/triton_tests/attention/test_unified_attention_blasst.py.
    """
    return (
        (a.float() - b.float()).abs().mean() / b.float().abs().mean().clamp_min(1e-6)
    ).item()


def sweep(label, inp, nq, nkv, head_dim, block_size, thresholds, skip_sparsity, rows):
    print(f"\n=== {label} ===")
    # Observe the dispatch on a real call rather than assuming 2D.
    with _DispatchCapture() as cap:
        run_attention(inp, 0.0)
    # Not `cap.cfg`: the 3D selector omits BLOCK_M/BLOCK_Q, and the replay needs them.
    cfg = cap.effective_cfg(nq, nkv)
    tps = cap.tiles_per_segment(inp["seqlen"])
    print(f"    {cap.describe(nq, nkv)}")
    if tps:
        print(
            f"    sparsity replay is segment-aware: "
            f"tiles_per_segment={tps}"
        )

    dense_ms = benchmark_fn(lambda: run_attention(inp, 0.0))
    dense_out = run_attention(inp, 0.0).clone()
    print(f"    dense: {dense_ms:.3f} ms   <- speedup and rel_diff are both against this")
    print(
        f"    {'threshold':>11} {'ms':>9} {'speedup':>8} {'rel_diff':>9} "
        f"{'row_skip':>9} {'tile_elide':>11}"
    )

    for thr in thresholds:
        try:
            ms = benchmark_fn(lambda t=thr: run_attention(inp, t))
            out = run_attention(inp, thr).clone()
        except Exception as exc:  # noqa: BLE001
            print(f"    {thr:>11g} FAILED {type(exc).__name__}: {exc}")
            continue
        err = rel_diff(out, dense_out)
        finite = bool(torch.isfinite(out).all())
        if skip_sparsity:
            # Not measured, not zero. Printing nan% here reads like a numerical
            # failure rather than "the replay was switched off".
            row_s = tile_s = float("nan")
            row_str, tile_str = f"{'-':>8}", f"{'-':>10}"
        else:
            got = simulate_sparsity(inp, cfg, nq, nkv, thr, tiles_per_segment=tps)
            if got is None:
                row_s = tile_s = float("nan")
                row_str, tile_str = f"{'<blk':>8}", f"{'<blk':>10}"
            else:
                row_s, tile_s = got
                row_str, tile_str = f"{row_s:>8.1%}", f"{tile_s:>10.1%}"
        note = "" if finite else "  <-- NON-FINITE"
        print(
            f"    {thr:>11g} {ms:>9.3f} {dense_ms / ms:>7.3f}x {err:>9.4f} "
            f"{row_str} {tile_str}{note}"
        )
        rows.append(
            dict(
                case=label, threshold=thr, dense_ms=round(dense_ms, 4),
                ms=round(ms, 4), speedup=round(dense_ms / ms, 4),
                rel_diff=round(err, 6), row_skip=round(row_s, 4),
                tile_elide=round(tile_s, 4), finite=finite,
                kernel=cap.kernel, block_m=cfg.get("BLOCK_M"),
                tile_size=cfg.get("TILE_SIZE"),
                tiles_per_segment=tps,
            )
        )


def banner(args):
    import triton

    # Non-default Triton codegen knobs change this kernel's speedup, so echo any
    # that are set. Printed only when present: a knob a build does not have is
    # noise, not information.
    codegen = {
        k: v for k, v in os.environ.items()
        if k.startswith("TRITON_") and k.endswith(("_ACROSS_IF", "_SWIZZLE"))
    }
    # The chain-dot toggle changes codegen (warpsPerCTA [2,2] vs [4,1]), so a run
    # is not interpretable without it.
    print("=" * 78)
    print("BLASST block skipping in Triton unified_attention  (random inputs)")
    print("=" * 78)
    print(f"  device            {torch.cuda.get_device_name(0)}")
    print(f"  torch / triton    {torch.__version__} / {triton.__version__}")
    print(f"  baseline          same kernel at threshold=0 (Triton dense)")
    print(f"  causal            True (unified_attention supports causal only)")
    print(f"  dtype             bf16      block_size {args.block_size}")
    for k, v in sorted(codegen.items()):
        print(f"  codegen           {k}={v}")
    print(f"  sparsity replay   {'off (--skip-sparsity)' if args.skip_sparsity else 'on (2D and 3D)'}")
    print("=" * 78)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="BLASST unified_attention benchmark (random inputs)"
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seqlens", type=str, default="16384,32768,65536",
                        help="comma-separated")
    parser.add_argument("--shapes", type=str, default="64x4",
                        help="comma-separated NQxNKV head configs")
    parser.add_argument("--thresholds", type=str, default="")
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument(
        "--shapes-3d", type=str, default="",
        help="comma-separated QxKV token counts that route to the 3D kernel, "
             "e.g. 128x16384. Few queries over long KV; the wrapper picks 3D when "
             "there are too few query blocks to fill the GPU. BLASST is expected to "
             "do WORSE here: each segment restarts the running max, so the first "
             "tile of every segment can never skip.")
    parser.add_argument("--skip-sparsity", action="store_true",
                        help="skip the sparsity replay (much faster)")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("ERROR: GPU required.")
        return 1

    thresholds = (
        [float(x) for x in args.thresholds.split(",")]
        if args.thresholds
        else list(DEFAULT_THRESHOLDS)
    )
    banner(args)
    rows = []

    shapes = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",")]
    for nq, nkv in shapes:
        for seqlen in [int(s) for s in args.seqlens.split(",")]:
            inp = make_paged_inputs(seqlen, nq, nkv, args.head_dim, args.block_size)
            sweep(
                f"random  {nq}q/{nkv}kv  seqlen={seqlen}",
                inp, nq, nkv, args.head_dim, args.block_size,
                thresholds, args.skip_sparsity, rows,
            )
            del inp
            torch.cuda.empty_cache()

    if args.shapes_3d:
        print("\n" + "=" * 78)
        print("3D kernel shapes (few queries, long KV)")
        print("=" * 78)
        # Use the SAME head config as the 2D sweep. A different GQA ratio changes
        # BLOCK_M/BLOCK_Q and therefore how much sparsity is cashable, so hard-
        # coding one here would confound every 2D-vs-3D comparison.
        nq, nkv = (int(v) for v in args.shapes.split(",")[0].split("x"))
        for spec in args.shapes_3d.split(","):
            q_len, kv_len = (int(v) for v in spec.split("x"))
            inp = make_paged_inputs_qkv(
                q_len, kv_len, nq, nkv, args.head_dim, args.block_size
            )
            sweep(f"3d-shape  {nq}q/{nkv}kv  q={q_len} kv={kv_len}",
                  inp, nq, nkv, args.head_dim, args.block_size,
                  thresholds, args.skip_sparsity, rows)
            del inp
            torch.cuda.empty_cache()

    # Split by kernel: 2D and 3D are different regimes and averaging them
    # together would hide exactly the gap that matters.
    if rows:
        print("\n" + "=" * 78)
        print("OVERALL  (sum dense / sum BLASST, across the cases measured)")
        print("=" * 78)
        by = {}
        for r in rows:
            by.setdefault((r.get("kernel") or "?", r["threshold"]), [0.0, 0.0, 0])
            e = by[(r.get("kernel") or "?", r["threshold"])]
            e[0] += r["dense_ms"]; e[1] += r["ms"]; e[2] += 1
        for kern in sorted({k for k, _ in by}):
            n = max(v[2] for (k, _), v in by.items() if k == kern)
            print(f"  kernel {kern}  ({n} cases)")
            print(f"    {'threshold':>11} {'dense ms':>10} {'blasst ms':>10} {'speedup':>8}")
            for (k, thr), (d, b, _) in sorted(by.items()):
                if k != kern:
                    continue
                print(f"    {thr:>11g} {d:>10.1f} {b:>10.1f} {d / b:>7.3f}x")
            print()

    if args.csv and rows:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
