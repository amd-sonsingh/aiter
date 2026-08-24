# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for BLASST block-skipping in the Triton unified_attention kernels.

BLASST skips a K/V tile for query rows whose per-tile max score falls below the
running softmax max by more than a threshold, eliding that tile's V load and P@V
matmul. Exposed via ``unified_attention(..., block_skip_threshold=X)`` (0 = off).

Self-contained inside AITER. Two references are used, for two different jobs:

  * ``ref_paged_attn`` (from test_unified_attention.py) -- the DENSE paged
    reference. Ground truth for the dense kernel path (threshold = 0). Comparing
    a SPARSE run against it measures how much work BLASST elided, which is a
    *behaviour* signal, not a correctness one, and cannot carry a meaningful
    tolerance: at 32K with lambda=0.3 the honest answer is rel_err ~1.0.
  * ``blasst_ref`` -- a PyTorch transcription of the BLASST online softmax with
    the *same* skip rule, the *same* K/V tiling (TILE_SIZE), the same attention
    sinks, and -- for the 3D kernel -- the *same* segment split as the kernel.
    Comparing the sparse kernel against this leaves only floating-point error,
    so it can carry a tight tolerance and is the actual correctness check.

Checks:
  [1] NO-REGRESSION  threshold=0 == dense paged reference, and is bit-identical
                     to omitting the kwarg entirely
  [2] GOLDEN 2D      threshold>0 == blasst_ref for kernel_unified_attention_2d,
                     to within fp error (three-way bound, see below)
  [3] GOLDEN 3D      same, for kernel_unified_attention_3d, whose running max
                     restarts per segment -- the reference replicates the split
  [4] SKIP HAPPENS   the reference reports a non-zero skip fraction and larger
                     thresholds deviate from dense more than smaller ones
  [5] NO NaN         threshold>1.0 stays finite (regression guard)
  [6] FORCE-DISABLE  decode and sliding-window batches ignore the threshold

Run (from the aiter repo root):
    pytest -q op_tests/triton_tests/attention/test_unified_attention_blasst.py
    python -m op_tests.triton_tests.attention.test_unified_attention_blasst  # report
Note: set AITER_TRITON_ONLY=1 if AITER's C++ ops are not built in your env. The
report form uses -m so that the sibling import of test_unified_attention (for
generate_data / ref_paged_attn) resolves.
"""

import math
import sys

import pytest
import torch

import aiter.ops.triton.attention.unified_attention as _unified_attention_mod
from aiter.ops.triton.attention.unified_attention import unified_attention
from op_tests.triton_tests.attention.test_unified_attention import (
    generate_data,
    ref_paged_attn,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")

# 1 / ln(2). The kernels fold this into the QK scale so the softmax runs in
# base-2 (exp2 is a hardware instruction); the threshold is converted the same
# way, hence log2_threshold = ln(lambda) / ln(2).
RCP_LN2 = 1.4426950408889634

# ─── tolerances for kernel-vs-blasst_ref ─────────────────────────────────
#
# The kernel and blasst_ref run the identical algorithm over the identical
# TILE_SIZE tiling (and, on the 3D path, the identical segment split), so there
# are exactly three sources of difference, and each gets its own bound rather
# than being lumped into one number. Same scheme as
# op_tests/triton_tests/attention/test_mha_blasst.py; the numbers below were
# RE-MEASURED on this kernel, see the RESULTS block at the bottom of this file.
#
# (a) CONTINUOUS fp error -- MFMA bf16xbf16->fp32 for QK and P@V accumulates in
#     a different order than torch's fp32 GEMM. This moves every element a
#     little and is what REF_REL_TOL bounds.
#
# (b) The bf16 STORE. acc/L are fp32 in registers and rounded to bf16 (8
#     explicit mantissa bits) on the way out, so any single element can differ
#     by ~2^-8 relative. Attention output of unit-normal V has |o| < 1 on
#     average, so 2^-7 = 7.8e-3 is a hard ceiling. That is REF_ELEM_TOL.
#
# (c) BOUNDARY TIES -- discrete, and the reason a plain max-error bound is the
#     wrong instrument. A row's skip decision is `(s_max - M) < log2_threshold`.
#     When that difference lands within (a)'s noise of log2_threshold, the
#     kernel and the reference can decide oppositely; the row then either gains
#     or loses a whole K/V tile. That is a large deviation in ONE row and
#     invisible in the mean, so bounding max_err would either be vacuous or
#     hostage to the seed. Instead we bound HOW MANY rows may disagree.
#     A tie is genuinely a tie -- neither answer is more correct, since the
#     kernel's S and the reference's S are both approximations of the same real
#     number and the two skip decisions bracket it.
#
# Measured on MI300X (gfx942) over 156 configurations -- 12 shapes x 13
# thresholds in [1e-3, 8.0], covering both kernels, TILE_SIZE 16/64, block_size
# 16/64, shuffled and plain KV caches, and two head layouts. Worst observed:
#     mean rel, 2D          5.15e-6   -> REF_REL_TOL     1e-4  (19x headroom)
#     mean rel, 3D          4.69e-5   -> REF_REL_TOL_3D  5e-4  (11x headroom)
#     non-tie elem          7.81e-3   -> REF_ELEM_TOL    8e-3  (= 2^-7 exactly)
#     disagreeing rows      1 of 262144 -> REF_TIE_ROW_FRAC 1e-4
# See the RESULTS block at the bottom of this file.
REF_REL_TOL = 1e-4  # mean |kernel - ref| / mean |ref|, 2D kernel
# The 3D kernel gets a LOOSER mean bound, for two structural reasons -- NOT
# because its skip decisions are unmodelled (blasst_ref replicates the segment
# split exactly; see blasst_ref point 5). (i) It runs a second fp32 reduction
# the 2D path does not have: reduce_segments rescales and sums up to 128
# per-segment partial results, so every output element carries the rounding of
# that merge on top of the online softmax. (ii) The shapes that route to it have
# FEW output rows -- a single sequence of 128 tokens x 8 heads is 1024 rows,
# versus 32768 for the 2D batches -- so one row sitting near a skip boundary
# moves the batch MEAN by an order of magnitude more than it would on 2D.
# Both effects are visible in the RESULTS block: 3D means sit at 3e-6..1.3e-5
# with one 4.7e-5 outlier, against 1e-7..5e-6 for 2D.
REF_REL_TOL_3D = 5e-4
REF_ELEM_TOL = 8e-3  # per-element |kernel - ref|, outside tie rows
REF_TIE_ROW_FRAC = 1e-4  # fraction of (token, head) rows allowed to exceed it

# Sub-1.0 thresholds are the normal operating range (calibrated lambda is ~0.029
# at 50% sparsity / 32K). Above-1.0 thresholds are the NaN regression range and
# are covered by the golden comparison too.
THRESHOLDS = [1e-3, 1e-2, 5e-2, 1e-1, 3e-1]
THRESHOLDS_ABOVE_ONE = [1.001, 1.1, 1.3, 2.0, 4.0, 8.0, 12.0]
GOLDEN_THRESHOLDS = [1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 1.001, 1.3, 2.0, 8.0]

# Short prefill batch: max_seqlen_k <= 512 routes to kernel_unified_attention_2d.
PREFILL_SEQ_LENS = [(256, 256), (128, 512)]
# Long prefill batch, many sequences: num_2d_prgms > target_num_prgms, so this
# also routes 2D but with enough KV per row for skipping to actually bite.
LONG_2D_SEQ_LENS = [(512, 2048)] * 8
# Few sequences with very long KV: routes to kernel_unified_attention_3d.
LONG_KV_SEQ_LENS = [(128, 16384)]
# One query token per sequence: ALL_DECODE, where BLASST is force-disabled.
DECODE_SEQ_LENS = [(1, 1024)] * 4


# ─── kernel-config capture ───────────────────────────────────────────────
#
# The reference's tile width MUST equal the kernel's TILE_SIZE -- it is the
# granularity of s_max and therefore of every skip decision. Rather than
# recomputing or hardcoding it (gfx942 happens to pick 64, but shuffled KV
# caches force TILE_SIZE = block_size, and RDNA picks 16/32), we wrap the
# wrapper's OWN selector and read back the exact dict that gets splatted into
# the triton launch. Same for NUM_SEGMENTS_PER_SEQ on the 3D path, which the
# reference needs to replicate the segment split.
class _ConfigCapture:
    """Context manager recording the config unified_attention hands the kernel."""

    def __init__(self):
        self.config_2d = None
        self.config_3d = None

    def __enter__(self):
        self._orig_2d = _unified_attention_mod.select_2d_config
        self._orig_3d = _unified_attention_mod.select_3d_config

        def wrap_2d(*args, **kwargs):
            cfg = self._orig_2d(*args, **kwargs)
            self.config_2d = dict(cfg)
            return cfg

        def wrap_3d(*args, **kwargs):
            attn_cfg, reduce_cfg = self._orig_3d(*args, **kwargs)
            self.config_3d = dict(attn_cfg)
            return attn_cfg, reduce_cfg

        _unified_attention_mod.select_2d_config = wrap_2d
        _unified_attention_mod.select_3d_config = wrap_3d
        return self

    def __exit__(self, *exc):
        _unified_attention_mod.select_2d_config = self._orig_2d
        _unified_attention_mod.select_3d_config = self._orig_3d
        return False

    @property
    def is_3d(self):
        return self.config_3d is not None

    @property
    def tile_size(self):
        cfg = self.config_3d if self.is_3d else self.config_2d
        assert cfg is not None, "no kernel config captured -- did the launch happen?"
        return cfg["TILE_SIZE"]

    @property
    def num_segments(self):
        return self.config_3d["NUM_SEGMENTS_PER_SEQ"] if self.is_3d else 1


# ─── the golden reference ────────────────────────────────────────────────


def blasst_ref(
    q,
    k,
    v,
    scale,
    threshold,
    tile_size,
    sinks=None,
    num_segments=1,
    out_dtype=torch.bfloat16,
):
    """Golden reference for the BLASST unified_attention kernels: FlashAttention
    online softmax with block skipping, in PyTorch, for ONE sequence.

    Ported from the PyTorch reference implementation of BLASST (MLSys 2026),
    Algorithm 1, by way of ``op_tests/triton_tests/attention/test_mha_blasst.py``,
    with the changes needed to match the paged/unified kernels rather than the
    paper's pseudocode:

    1. The skip test compares the tile max against the running max *before* this
       tile (``M``), not after folding it in. See ``BLASST_BLOCK_SKIP_NAN_FIX.md``:
       for threshold < 1 the two forms are algebraically equivalent, but for
       threshold > 1 the post-fold form skips the very first tile
       unconditionally (``s_max - m_j`` is identically 0 there), leaves M at
       -inf, and yields NaN.
    2. Everything is in base-2 units (``exp2``, ``log2_threshold``), matching the
       kernel, so exponentials round identically.
    3. ATTENTION SINKS. The kernels seed ``M = sink * RCP_LN2`` and ``L = 1.0``,
       i.e. one extra logit of value ``sink`` with a zero value vector. This
       matters for BLASST specifically: M is finite before the first tile, so
       unlike the sink-less case the FIRST tile can legitimately skip.
    4. The ``m_j = where(m_j > -inf, m_j, 0.0)`` guard the kernels apply so a
       fully-masked row cannot produce exp2(-inf - -inf) = NaN.
    5. SEGMENTS. ``kernel_unified_attention_3d`` splits the KV loop into
       NUM_SEGMENTS_PER_SEQ chunks of ``ceil(seq_len / (num_segments *
       TILE_SIZE))`` tiles each, restarting M at -inf (and L at 1.0) in every
       chunk, then merges them in ``reduce_segments``. So the first tile of every
       segment can never skip and later tiles compare against a SEGMENT-LOCAL
       running max. Passing the kernel's own num_segments here replicates that
       exactly, which is what lets the 3D path carry the same tight bound as 2D.
       num_segments=1 collapses to the 2D single-pass loop.

    The skip decision is PER ROW in both kernels -- ``skip`` is a BLOCK_M-vector
    and skipped rows get ``m_j = M`` and ``P = 0`` individually. The kernels'
    ``all_skip`` (a BLOCK_M-wide reduction) only decides whether the V load and
    the P@V matmul are *issued*; when it is false the skipped rows still carry
    P = 0 into the matmul and contribute nothing, and when it is true the elided
    update is a no-op (alpha = exp2(M - M) = 1, l_j = 0). So ``all_skip`` has no
    numerical effect and this row-wise reference is exact.

    BLOCK_M likewise has no numerical effect. In these kernels a BLOCK_M row is
    a (query_position, query_head) pair, and BLOCK_M only sets how far the tile
    loop runs for a row group (``num_tiles`` is computed from the group's
    largest causal boundary); the extra tiles a row sees beyond its own boundary
    are fully masked, so s_max = -inf and they contribute P = 0 whether or not
    they are skipped. TILE_SIZE *does* matter -- it is the granularity of
    ``s_max`` -- and is read from the kernel's own config by the caller.

    q: (S_q, H, D), k/v: (S_kv, H, D) -- already expanded to H query heads.
    sinks: (H,) or None. Returns (out, n_skipped, n_visited).
    """
    S_q, H, D = q.shape
    S_kv = k.shape[0]
    dev = q.device
    NEG = float("-inf")

    qt = q.transpose(0, 1).float()  # (H, S_q, D)
    kt = k.transpose(0, 1).float()  # (H, S_kv, D)
    vt = v.transpose(0, 1)  # (H, S_kv, D), kept in kv dtype

    qk_scale = scale * RCP_LN2
    enable = threshold > 0.0
    log2_threshold = math.log(threshold) * RCP_LN2 if enable else 0.0

    # Causal alignment: key n is visible to query m iff n <= context_len + m.
    context_len = S_kv - S_q
    offs_m = torch.arange(S_q, device=dev)
    causal_bound = (context_len + offs_m)[:, None]

    num_tiles = (S_kv + tile_size - 1) // tile_size
    # Exactly the kernel's cdiv_fn(seq_len, num_segments * TILE_SIZE).
    tiles_per_segment = -(-S_kv // (num_segments * tile_size))

    seg_acc, seg_m, seg_l = [], [], []
    n_skipped = 0
    n_visited = 0

    for segment in range(num_segments):
        lo = segment * tiles_per_segment
        if lo >= num_tiles:
            # The kernel returns early here and reduce_segments masks the segment
            # out (act_num_segments), so it contributes nothing.
            break
        hi = min(lo + tiles_per_segment, num_tiles)

        if segment == 0 and sinks is not None:
            M = (sinks.float() * RCP_LN2)[:, None].expand(H, S_q).contiguous()
        else:
            M = torch.full((H, S_q), NEG, device=dev, dtype=torch.float32)
        L = torch.ones((H, S_q), device=dev, dtype=torch.float32)
        acc = torch.zeros((H, S_q, D), device=dev, dtype=torch.float32)

        for j in range(lo, hi):
            n0 = j * tile_size
            n1 = min(n0 + tile_size, S_kv)
            S = (qt @ kt[:, n0:n1, :].transpose(-2, -1)) * qk_scale
            offs_n = torch.arange(n0, n1, device=dev)
            S = torch.where(offs_n[None, :] <= causal_bound, S, NEG)

            s_max = S.max(dim=-1).values  # (H, S_q)
            m_j = torch.maximum(M, s_max)
            if enable:
                # NOTE: M, not m_j. See point (1) above.
                skip = (s_max - M) < log2_threshold
                m_j = torch.where(skip, M, m_j)
            m_j = torch.where(m_j > NEG, m_j, torch.zeros_like(m_j))

            P = torch.exp2(S - m_j[..., None])
            if enable:
                P = torch.where(skip[..., None], 0.0, P)

            l_j = P.sum(dim=-1)
            alpha = torch.exp2(M - m_j)
            if enable:
                alpha = torch.where(skip, torch.ones_like(alpha), alpha)

            acc = acc * alpha[..., None]
            L = L * alpha + l_j
            M = m_j
            # The kernel casts P to V's dtype before the dot and accumulates fp32.
            acc = acc + P.to(v.dtype).float() @ vt[:, n0:n1, :].float()

            # Skip accounting, over tiles that are live for a row. A tile
            # entirely past a row's causal boundary has s_max = -inf and is
            # "skipped" trivially, contributing nothing either way -- exclude it.
            live = torch.isfinite(s_max)
            n_visited += int(live.sum())
            if enable:
                n_skipped += int((skip & live).sum())

        seg_acc.append(acc)
        seg_m.append(M)
        seg_l.append(L)

    # reduce_segments: rescale each segment to the global max and merge. For a
    # single segment this is exactly the 2D epilogue (weight == 1).
    m_all = torch.stack(seg_m)  # (segments, H, S_q)
    m_max = m_all.max(dim=0).values
    w = torch.exp2(m_all - m_max)
    num = (torch.stack(seg_acc) * w[..., None]).sum(dim=0)
    den = (torch.stack(seg_l) * w).sum(dim=0)[..., None]
    out = torch.where(den == 0.0, torch.zeros_like(num), num / den)
    return out.transpose(0, 1).to(out_dtype), n_skipped, n_visited


# ─── input plumbing ──────────────────────────────────────────────────────

_GENERATE_DATA_FIELDS = (
    "query",
    "key_cache_orig",
    "value_cache_orig",
    "key_cache",
    "value_cache",
    "sinks",
    "output",
    "cu_query_lens",
    "kv_lens",
    "max_query_len",
    "max_kv_len",
    "scale",
    "window_size",
    "block_tables",
    "maybe_quant_query",
    "query_scales",
    "q_descale",
    "k_descale",
    "v_descale",
    "output_scale",
)


def make_inputs(
    seq_lens,
    block_size=16,
    shuffled_kv_cache=False,
    sliding_window=None,
    num_heads=(8, 1),
    head_size=128,
    num_blocks=4096,
):
    """Paged inputs in exactly the layout unified_attention consumes."""
    data = dict(
        zip(
            _GENERATE_DATA_FIELDS,
            generate_data(
                seq_lens=seq_lens,
                num_blocks=num_blocks,
                block_size=block_size,
                head_size=head_size,
                num_heads=num_heads,
                sliding_window=sliding_window,
                shuffled_kv_cache=shuffled_kv_cache,
                device="cuda",
            ),
        )
    )
    data["seq_lens"] = seq_lens
    data["block_size"] = block_size
    data["shuffled_kv_cache"] = shuffled_kv_cache
    data["sliding_window"] = sliding_window
    return data


def run_kernel(data, threshold, pass_kwarg=True):
    """Call unified_attention; return (out, captured kernel config)."""
    out = torch.empty_like(data["output"])
    kwargs = {"block_skip_threshold": threshold} if pass_kwarg else {}
    with _ConfigCapture() as cap:
        unified_attention(
            q=data["query"],
            k=data["key_cache"],
            v=data["value_cache"],
            out=out,
            cu_seqlens_q=data["cu_query_lens"],
            seqused_k=data["kv_lens"],
            max_seqlen_q=data["max_query_len"],
            max_seqlen_k=data["max_kv_len"],
            softmax_scale=data["scale"],
            causal=True,
            window_size=data["window_size"],
            block_table=data["block_tables"],
            softcap=0,
            q_descale=data["q_descale"],
            k_descale=data["k_descale"],
            v_descale=data["v_descale"],
            sinks=data["sinks"],
            output_scale=data["output_scale"],
            shuffled_kv_cache=data["shuffled_kv_cache"],
            **kwargs,
        )
    return out, cap


def dense_ref(data):
    """The DENSE paged reference. Ground truth for threshold=0 only."""
    return ref_paged_attn(
        query=data["query"],
        key_cache=data["key_cache_orig"],
        value_cache=data["value_cache_orig"],
        query_lens=[x[0] for x in data["seq_lens"]],
        kv_lens=[x[1] for x in data["seq_lens"]],
        block_tables=data["block_tables"],
        scale=data["scale"],
        out_dtype=data["output"].dtype,
        sliding_window=data["sliding_window"],
        sinks=data["sinks"],
    )


def golden_ref(data, threshold, cap):
    """blasst_ref over the whole batch, gathering paged K/V back to dense.

    Uses the tile width and segment count the kernel ACTUALLY used (captured
    from its own config selector), so the comparison is apples-to-apples.
    """
    key_cache = data["key_cache_orig"]
    value_cache = data["value_cache_orig"]
    block_size, num_kv_heads, head_size = key_cache.shape[1:]
    num_q_heads = data["query"].shape[1]
    block_tables = data["block_tables"]

    outs = []
    n_skipped = n_visited = 0
    start = 0
    for i, (query_len, kv_len) in enumerate(data["seq_lens"]):
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        idx = block_tables[i, :num_kv_blocks].long()
        k = key_cache[idx].reshape(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[idx].reshape(-1, num_kv_heads, head_size)[:kv_len]
        if num_q_heads != num_kv_heads:  # GQA: expand K/V to the query heads
            reps = num_q_heads // num_kv_heads
            k = torch.repeat_interleave(k, reps, dim=1)
            v = torch.repeat_interleave(v, reps, dim=1)
        o, ns, nv = blasst_ref(
            data["query"][start : start + query_len],
            k,
            v,
            data["scale"],
            threshold,
            cap.tile_size,
            sinks=data["sinks"],
            num_segments=cap.num_segments,
            out_dtype=data["output"].dtype,
        )
        outs.append(o)
        n_skipped += ns
        n_visited += nv
        start += query_len
    return torch.cat(outs, dim=0), (n_skipped / n_visited if n_visited else 0.0)


# ─── metrics ─────────────────────────────────────────────────────────────


def rel_err(a, b):
    return (
        (a.float() - b.float()).abs().mean() / b.float().abs().mean().clamp_min(1e-6)
    ).item()


def max_err(a, b):
    return (a.float() - b.float()).abs().max().item()


def tie_rows(a, b):
    """(rows disagreeing beyond the bf16 store floor, total rows).

    a/b are (num_tokens, H, D). A row is one (token, head) output vector of
    length D -- the granularity at which a skip decision is made, so a flipped
    decision shows up as one whole bad row rather than as scattered elements.
    """
    bad = (a.float() - b.float()).abs() > REF_ELEM_TOL
    per_row = bad.any(dim=-1)
    return int(per_row.sum()), per_row.numel()


def assert_matches_ref(out, ref, thr, rel_tol=REF_REL_TOL):
    """The golden check. See the tolerance block at the top of this file."""
    assert torch.isfinite(out).all(), f"non-finite kernel output at thr={thr}"
    r = rel_err(out, ref)
    assert r < rel_tol, f"thr={thr}: mean rel={r:.3e} (bound {rel_tol:.1e})"
    bad, total = tie_rows(out, ref)
    budget = max(1, int(REF_TIE_ROW_FRAC * total))
    assert bad <= budget, (
        f"thr={thr}: {bad}/{total} rows differ by more than {REF_ELEM_TOL:.1e} "
        f"(tie budget {budget}); max elem err {max_err(out, ref):.3e}"
    )


# ─── 1. no regression: threshold=0 is the dense kernel, unchanged ────────────


@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("shuffled_kv_cache", [False, True])
def test_blasst_no_regression(block_size, shuffled_kv_cache):
    data = make_inputs(PREFILL_SEQ_LENS, block_size, shuffled_kv_cache)
    out0, _ = run_kernel(data, 0.0)
    assert torch.isfinite(out0).all()
    assert rel_err(out0, dense_ref(data)) < 5e-3

    # threshold=0 must be bit-identical to not passing the kwarg at all --
    # proves the constexpr-False path emits the original dense kernel.
    out_no_kwarg, _ = run_kernel(data, 0.0, pass_kwarg=False)
    assert torch.equal(out0, out_no_kwarg)


def test_blasst_no_regression_3d():
    data = make_inputs(LONG_KV_SEQ_LENS, block_size=16)
    out0, cap = run_kernel(data, 0.0)
    assert cap.is_3d, "expected the 3D kernel for this shape"
    assert torch.isfinite(out0).all()
    assert rel_err(out0, dense_ref(data)) < 5e-3

    out_no_kwarg, _ = run_kernel(data, 0.0, pass_kwarg=False)
    assert torch.equal(out0, out_no_kwarg)


# ─── 2. golden: the sparse kernel against the same sparse algorithm ──────────


@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("shuffled_kv_cache", [False, True])
@pytest.mark.parametrize("thr", GOLDEN_THRESHOLDS)
def test_blasst_matches_reference_2d(block_size, shuffled_kv_cache, thr):
    """The real correctness check for kernel_unified_attention_2d: the sparse
    kernel against the SAME sparse algorithm in PyTorch, over the same tiling.
    Any deviation here is floating point, not sparsity, so the bound is tight.

    Also pins TILE_SIZE agreement: the reference's tile width is read out of the
    config the wrapper handed the kernel, and block_size/shuffled_kv_cache move
    it (shuffled KV forces TILE_SIZE = block_size), so this covers 16/64/... .
    """
    data = make_inputs(PREFILL_SEQ_LENS, block_size, shuffled_kv_cache)
    out, cap = run_kernel(data, thr)
    assert not cap.is_3d, "expected the 2D kernel for this shape"
    if shuffled_kv_cache:
        assert cap.tile_size == block_size  # shuffled KV pins TILE_SIZE
    ref, _ = golden_ref(data, thr, cap)
    assert_matches_ref(out, ref, thr)


@pytest.mark.parametrize("thr", GOLDEN_THRESHOLDS)
def test_blasst_matches_reference_2d_long(thr):
    """Same check on a batch long enough for skipping to actually bite (2K KV,
    8 sequences -- enough 2D programs to keep the dispatcher off the 3D path)."""
    data = make_inputs(LONG_2D_SEQ_LENS, block_size=16)
    out, cap = run_kernel(data, thr)
    assert not cap.is_3d, "expected the 2D kernel for this shape"
    ref, _ = golden_ref(data, thr, cap)
    assert_matches_ref(out, ref, thr)


@pytest.mark.parametrize("thr", GOLDEN_THRESHOLDS)
def test_blasst_matches_reference_2d_gqa(thr):
    """Same check with a wider GQA layout (64 query heads over 8 KV heads) and a
    64-wide head. GQA is where the reference has real work to do: the kernel
    reads one KV head for a group of BLOCK_M rows spanning several query heads,
    so the reference has to repeat_interleave K/V up to the query-head count
    before it can run row-wise."""
    data = make_inputs(LONG_2D_SEQ_LENS, block_size=16, num_heads=(64, 8),
                       head_size=64, num_blocks=8192)
    out, cap = run_kernel(data, thr)
    assert not cap.is_3d, "expected the 2D kernel for this shape"
    ref, _ = golden_ref(data, thr, cap)
    assert_matches_ref(out, ref, thr)


@pytest.mark.parametrize("thr", GOLDEN_THRESHOLDS)
def test_blasst_matches_reference_3d(thr):
    """Golden check for kernel_unified_attention_3d.

    The 3D kernel splits the KV loop into NUM_SEGMENTS_PER_SEQ segments and
    RESTARTS the running max at -inf in each one, so the first tile of every
    segment can never skip and every other tile compares against a
    segment-local max. Those skip decisions legitimately differ from a
    single-pass reference -- against one, this shape deviates by O(1).

    Rather than loosening the bound, blasst_ref replicates the split: the
    segment count comes from the kernel's own select_3d_config (captured at
    launch) and the boundaries from the same
    ``ceil(seq_len / (num_segments * TILE_SIZE))`` formula the kernel uses. So
    this carries the SAME tight bound as the 2D path.
    """
    data = make_inputs(LONG_KV_SEQ_LENS, block_size=16)
    out, cap = run_kernel(data, thr)
    assert cap.is_3d, "expected the 3D kernel for this shape"
    ref, _ = golden_ref(data, thr, cap)
    assert_matches_ref(out, ref, thr, rel_tol=REF_REL_TOL_3D)


# ─── 3. skipping actually happens ────────────────────────────────────────────


def test_blasst_actually_skips():
    """Guards against the golden comparison being vacuous: tiles really are
    being skipped, and skipping more moves the output further from dense."""
    data = make_inputs(LONG_2D_SEQ_LENS, block_size=16)
    dense = dense_ref(data)
    THR_LO, THR_HI = 1e-1, 3e-1

    out_lo, cap = run_kernel(data, THR_LO)
    _, frac_lo = golden_ref(data, THR_LO, cap)
    out_hi, cap = run_kernel(data, THR_HI)
    _, frac_hi = golden_ref(data, THR_HI, cap)
    assert frac_hi > frac_lo > 0.0, f"skip fractions {frac_lo} / {frac_hi}"
    assert rel_err(out_hi, dense) > rel_err(out_lo, dense)


def test_blasst_3d_actually_skips():
    data = make_inputs(LONG_KV_SEQ_LENS, block_size=16)
    dense = dense_ref(data)
    THR_LO, THR_HI = 1e-1, 3e-1

    out_lo, cap = run_kernel(data, THR_LO)
    _, frac_lo = golden_ref(data, THR_LO, cap)
    out_hi, cap = run_kernel(data, THR_HI)
    _, frac_hi = golden_ref(data, THR_HI, cap)
    assert frac_hi > frac_lo > 0.0, f"skip fractions {frac_lo} / {frac_hi}"
    assert rel_err(out_hi, dense) > rel_err(out_lo, dense)


# ─── 4. NaN regression for threshold > 1.0 ───────────────────────────────────


@pytest.mark.parametrize("seq_lens", [PREFILL_SEQ_LENS, LONG_KV_SEQ_LENS])
@pytest.mark.parametrize("thr", THRESHOLDS_ABOVE_ONE)
def test_blasst_no_nan_above_threshold_one(seq_lens, thr):
    """block_skip_threshold > 1.0 (log2_threshold > 0) must stay finite.

    The skip decision compares the per-tile max against M, the running max
    BEFORE folding in the current tile. Comparing against the post-fold max
    instead makes the difference identically 0 whenever a tile sets a new max --
    including the first tile, where M is -inf. That skips the first tile
    unconditionally, leaves M at -inf, and yields exp2(-inf - -inf) = NaN.
    See BLASST_BLOCK_SKIP_NAN_FIX.md.

    Finiteness only: at log2_threshold > 0 a tile is skipped even when its max is
    ABOVE the running max, which elides nearly everything, so a large deviation
    from DENSE is the expected outcome. Accuracy at these thresholds is pinned
    against blasst_ref by the golden tests instead.
    """
    data = make_inputs(seq_lens, block_size=16)
    out, _ = run_kernel(data, thr)
    assert torch.isfinite(out).all()


# ─── 5. decode is force-disabled (prefill/unified-only scope) ────────────────


def test_blasst_decode_is_dense():
    data = make_inputs(DECODE_SEQ_LENS, block_size=16)
    out_thr, _ = run_kernel(data, 8.0)
    out_dense, _ = run_kernel(data, 0.0)
    assert torch.equal(out_thr, out_dense), "decode should ignore block_skip_threshold"


# ─── 6. sliding window is force-disabled in this phase ──────────────────────


def test_blasst_sliding_window_is_dense():
    data = make_inputs(PREFILL_SEQ_LENS, block_size=16, sliding_window=128)
    out_thr, _ = run_kernel(data, 8.0)
    out_dense, _ = run_kernel(data, 0.0)
    assert torch.equal(out_thr, out_dense), "sliding window should run dense"


# ─── standalone numbered report ───────────────────────────────────────────


def _report(name, seq_lens, block_size, shuffled_kv_cache=False, **kw):
    data = make_inputs(seq_lens, block_size, shuffled_kv_cache, **kw)
    dense = dense_ref(data)
    ok = True

    out0, cap = run_kernel(data, 0.0)
    r = rel_err(out0, dense)
    kind = "3D" if cap.is_3d else "2D"
    rel_tol = REF_REL_TOL_3D if cap.is_3d else REF_REL_TOL
    passed = r < 5e-3 and torch.isfinite(out0).all().item()
    ok &= passed
    print(
        f"\n=== {name}  seq_lens={seq_lens if len(seq_lens) < 4 else seq_lens[:1]}"
        f"{'*%d' % len(seq_lens) if len(seq_lens) >= 4 else ''} "
        f"block_size={block_size} shuffled={shuffled_kv_cache} ==="
    )
    print(
        f"    kernel={kind}  TILE_SIZE={cap.tile_size}  "
        f"NUM_SEGMENTS={cap.num_segments}  rel bound {rel_tol:.0e}"
    )
    print(f"[1] NO-REGRESSION  t=0 vs dense paged ref  rel={r:.2e}  "
          f"{'PASS' if passed else 'FAIL'}")
    print(f"    {'thr':<8}{'skip%':>8}{'rel_vs_ref':>13}{'max_vs_ref':>13}"
          f"{'tie_rows':>10}{'rel_vs_dense':>15}")
    for t in GOLDEN_THRESHOLDS:
        out, cap = run_kernel(data, t)
        ref, frac = golden_ref(data, t, cap)
        rr, mm = rel_err(out, ref), max_err(out, ref)
        bad, total = tie_rows(out, ref)
        good = (
            rr < rel_tol
            and bad <= max(1, int(REF_TIE_ROW_FRAC * total))
            and torch.isfinite(out).all().item()
        )
        ok &= good
        print(f"    {t:<8}{100 * frac:>7.1f}%{rr:>13.2e}{mm:>13.2e}"
              f"{bad:>7}/{total:<8}{rel_err(out, dense):>10.2e}  "
              f"{'ok' if good else 'OUT-OF-BOUND'}")
    return ok


def main():
    if not torch.cuda.is_available():
        print("ERROR: GPU required.")
        sys.exit(1)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    ok = True
    ok &= _report("2D short", PREFILL_SEQ_LENS, 16)
    ok &= _report("2D short", PREFILL_SEQ_LENS, 64)
    ok &= _report("2D short shuffled", PREFILL_SEQ_LENS, 16, True)
    ok &= _report("2D short shuffled", PREFILL_SEQ_LENS, 64, True)
    ok &= _report("2D long", LONG_2D_SEQ_LENS, 16)
    ok &= _report("2D long gqa64/hd64", LONG_2D_SEQ_LENS, 16,
                  num_heads=(64, 8), head_size=64, num_blocks=8192)
    ok &= _report("3D long-KV", LONG_KV_SEQ_LENS, 16)
    ok &= _report("3D long-KV", LONG_KV_SEQ_LENS, 64)
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()


# ─── RESULTS: where the tolerances come from ──────────────────────────────
#
# MI300X (gfx942), ROCm 7.2.4, torch 2.10.0, triton 3.8.0, bf16 q/kv/out,
# attention sinks on, causal (unified_attention asserts causal).
# Grid: 12 shapes x 13 thresholds = 156 configurations.
#   thr   1e-3 1e-2 5e-2 1e-1 3e-1 0.5 0.9 1.001 1.1 1.3 2.0 4.0 8.0
#   shapes (seq_lens as (query_len, kv_len)), with the kernel each routes to and
#   the TILE_SIZE / NUM_SEGMENTS_PER_SEQ read back from its own config selector:
#     A [(256,256),(128,512)]  bs16          2D  TILE 64  seg 1
#     B [(256,256),(128,512)]  bs64          2D  TILE 64  seg 1
#     C [(256,256),(128,512)]  bs16 shuffled 2D  TILE 16  seg 1
#     D [(256,256),(128,512)]  bs64 shuffled 2D  TILE 64  seg 1
#     E [(512,2048)]*8         bs16          2D  TILE 64  seg 1
#     F [(512,2048)]*8         bs64          2D  TILE 64  seg 1
#     G ragged 5-seq mix       bs16          3D  TILE 16  seg 8
#     H [(512,2048)]*8 (64,8) heads, hd 64   2D  TILE 64  seg 1
#     I [(128,16384)]          bs16          3D  TILE 16  seg 128
#     J [(128,16384)]          bs64          3D  TILE 64  seg 128
#     K [(64,8192)]*2          bs16          3D  TILE 16  seg 128
#     L [(64,32768)]           bs64          3D  TILE 64  seg 128
#
# Aggregate over all 156:
#   worst mean rel, 2D shapes   5.15e-06   (E/F, thr=1e-3)
#   worst mean rel, 3D shapes   4.69e-05   (I, thr=1.3); next worst 1.29e-05 (L)
#   worst max elem              1.29e-01   (H, thr=1.001)
#   worst disagreeing rows      1 of 262144 (H, thr in {0.3, 0.5, 1.001})
#   configs with >1 tie row     0
#
# The max-elem and the mean disagree by four orders of magnitude, which is the
# whole reason the bound is not on max. Representative slice, shape E
# (8 x (512, 2048), TILE_SIZE 64):
#
#   thr      skip%      rel        max      bad_rows   rel_vs_dense
#   0.001     0.0%   5.15e-06   9.77e-04      0          1.51e-03
#   0.05      0.2%   5.08e-06   9.77e-04      0          3.30e-03
#   0.1       1.2%   5.01e-06   9.77e-04      0          2.33e-02
#   0.3      25.4%   4.03e-06   1.95e-03      0          5.22e-01
#   0.5      56.1%   2.73e-06   3.91e-03      0          1.28e+00
#   1.001    86.2%   1.40e-06   3.91e-03      0          3.08e+00
#   2.0      93.5%   9.05e-07   3.91e-03      0          4.47e+00
#   8.0      96.8%   6.94e-07   3.91e-03      0          5.37e+00
#
# Read that as: every threshold agrees with the golden reference to 1e-6..1e-5
# in the mean, while the DENSE reference is 3e-3 to 5.4 away -- that last column
# is BLASST doing its job, and is exactly what the old `rel_err(out, ref) < 0.6`
# assertion was measuring. It is not a correctness signal and cannot carry a
# tolerance: at 32K with lambda=0.3 the honest number is O(1).
#
# Every elementwise deviation outside a tie row lands on an exact bf16 ULP --
# 1.22e-4, 2.44e-4, 4.88e-4, 9.77e-4, 1.95e-3, 3.91e-3, 7.81e-3, i.e. 2^-13
# through 2^-7 -- which is the fp32 -> bf16 store and nothing else. REF_ELEM_TOL
# is set at the top rung, 2^-7. The three configurations that exceeded it did so
# in exactly ONE row each (1.35e-2, 2.29e-2, 1.29e-1 on shape H), which is a
# flipped skip decision on a boundary tie: shape H has 512 query rows x 64 heads
# x 8 sequences = 262144 rows, so 1 flip is 4e-6 of them.
#
# GRANULARITY. blasst_ref matches the kernels exactly on every dimension that
# affects the answer:
#   TILE_SIZE  MUST match and provably does -- _ConfigCapture wraps the
#              wrapper's own select_2d_config / select_3d_config and reads the
#              dict that is splatted into the triton launch, so the reference
#              cannot drift from the kernel. Shapes A-L above span TILE_SIZE 16
#              and 64 through block_size and shuffled_kv_cache (which pins
#              TILE_SIZE = block_size); test_blasst_matches_reference_2d
#              additionally asserts that identity.
#   SEGMENTS   MUST match on the 3D path and provably does, from the same
#              capture; boundaries come from the kernel's own
#              ceil(seq_len / (num_segments * TILE_SIZE)) formula. Without this,
#              the same comparison on shape I reports rel 0.10 at thr=0.05,
#              0.38 at 0.1, 0.84 at 0.3 and ~0.98 above 1.0 -- i.e. a
#              single-pass reference is useless here, which is why the original
#              0.6 bound existed.
#   SINKS      generate_data always produces them and unified_attention seeds
#              M = sink * RCP_LN2, L = 1. That makes M FINITE before the first
#              tile, so unlike the sink-less MHA kernel the first tile can
#              legitimately skip; the reference does the same.
#   BLOCK_M    does NOT need to match, and the reference has no BLOCK_M (it is
#              fully row-wise). A BLOCK_M row here is a (query_position,
#              query_head) pair; BLOCK_M only sets how far the tile loop runs
#              for a row group, and the extra tiles a row sees past its own
#              causal boundary are fully masked (s_max = -inf), so they
#              contribute P = 0 whether skipped or not.
#   all_skip   is a BLOCK_M-wide reduction, but it only gates whether the V load
#              and P@V matmul are ISSUED. When false, skipped rows still carry
#              P = 0 through the matmul; when true, the elided update is a no-op
#              (alpha = exp2(M - M) = 1, l_j = 0). No numerical effect.
# So the only residual discrepancy is floating point, and it is bounded above.
#
# Reproduce: python -m op_tests.triton_tests.attention.test_unified_attention_blasst
