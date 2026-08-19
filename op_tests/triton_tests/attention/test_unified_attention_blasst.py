# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for BLASST block-skipping in the Triton unified_attention kernels.

BLASST skips a K/V tile for query rows whose per-tile max score falls below the
running softmax max by more than a threshold, eliding that tile's V load and P@V
matmul. Exposed via ``unified_attention(..., block_skip_threshold=X)`` (0 = off).

These mirror op_tests/triton_tests/attention/test_mha_blasst.py, plus three
invariants specific to the paged/unified path (decode force-disable, the 3D
segmented kernel, and sliding-window force-disable).

Reference is ``ref_paged_attn`` from test_unified_attention.py, and inputs come
from that file's ``generate_data`` so these tests exercise the same paged layout
the production path uses.

Run:
    AITER_TRITON_ONLY=1 pytest -q op_tests/triton_tests/attention/test_unified_attention_blasst.py
"""

import pytest
import torch

from aiter.ops.triton.attention.unified_attention import unified_attention
from op_tests.triton_tests.attention.test_unified_attention import (
    generate_data,
    ref_paged_attn,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")

# Sub-1.0 thresholds: the normal operating range (calibrated lambda is ~0.029 at
# 50% sparsity / 32K). Above-1.0 thresholds are the NaN regression range.
THRESHOLDS = [1e-3, 1e-2, 5e-2, 1e-1, 3e-1]
THRESHOLDS_ABOVE_ONE = [1.001, 1.1, 1.3, 2.0, 4.0, 8.0, 12.0]

# A prefill-shaped batch (routes 2D) and a decode-shaped batch (routes ALL_DECODE).
PREFILL_SEQ_LENS = [(256, 256), (128, 512)]
DECODE_SEQ_LENS = [(1, 1024)] * 4


def _run(seq_lens, threshold, block_size=16, shuffled_kv_cache=False,
         sliding_window=None, num_heads=(8, 1), head_size=128, num_blocks=4096,
         pass_kwarg=True):
    """Run unified_attention on generated paged inputs; return (out, ref)."""
    (query, key_cache_orig, value_cache_orig, key_cache, value_cache, sinks,
     output, cu_query_lens, kv_lens, max_query_len, max_kv_len, scale,
     window_size, block_tables, _mq, _qs, q_descale, k_descale, v_descale,
     output_scale) = generate_data(
        seq_lens=seq_lens,
        num_blocks=num_blocks,
        block_size=block_size,
        head_size=head_size,
        num_heads=num_heads,
        sliding_window=sliding_window,
        shuffled_kv_cache=shuffled_kv_cache,
        device="cuda",
    )

    kwargs = {} if not pass_kwarg else {"block_skip_threshold": threshold}
    unified_attention(
        q=query, k=key_cache, v=value_cache, out=output,
        cu_seqlens_q=cu_query_lens, seqused_k=kv_lens,
        max_seqlen_q=max_query_len, max_seqlen_k=max_kv_len,
        softmax_scale=scale, causal=True, window_size=window_size,
        block_table=block_tables, softcap=0,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
        sinks=sinks, output_scale=output_scale,
        shuffled_kv_cache=shuffled_kv_cache,
        **kwargs,
    )

    ref = ref_paged_attn(
        query=query, key_cache=key_cache_orig, value_cache=value_cache_orig,
        query_lens=[x[0] for x in seq_lens], kv_lens=[x[1] for x in seq_lens],
        block_tables=block_tables, scale=scale, out_dtype=output.dtype,
        sliding_window=sliding_window,
    )
    return output.clone(), ref


def rel_err(a, b):
    return (
        (a.float() - b.float()).abs().mean() / b.float().abs().mean().clamp_min(1e-6)
    ).item()


# ─── 1. no regression: threshold=0 is the dense kernel, unchanged ────────────

@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("shuffled_kv_cache", [False, True])
def test_blasst_no_regression(block_size, shuffled_kv_cache):
    out0, ref = _run(PREFILL_SEQ_LENS, 0.0, block_size, shuffled_kv_cache)
    assert torch.isfinite(out0).all()
    assert rel_err(out0, ref) < 5e-2

    # threshold=0 must be bit-identical to not passing the kwarg at all --
    # proves the constexpr-False path emits the original dense kernel.
    out_no_kwarg, _ = _run(PREFILL_SEQ_LENS, 0.0, block_size, shuffled_kv_cache,
                           pass_kwarg=False)
    assert torch.equal(out0, out_no_kwarg)


# ─── 2. bounded degradation, always finite ───────────────────────────────────

@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("shuffled_kv_cache", [False, True])
@pytest.mark.parametrize("thr", THRESHOLDS)
def test_blasst_degrades_gracefully(block_size, shuffled_kv_cache, thr):
    out, ref = _run(PREFILL_SEQ_LENS, thr, block_size, shuffled_kv_cache)
    assert torch.isfinite(out).all()
    assert rel_err(out, ref) < 0.6


# ─── 3. skipping actually happens ────────────────────────────────────────────

def test_blasst_actually_skips():
    ref = _run(PREFILL_SEQ_LENS, 0.0)[1]
    r_lo = rel_err(_run(PREFILL_SEQ_LENS, THRESHOLDS[0])[0], ref)
    r_hi = rel_err(_run(PREFILL_SEQ_LENS, THRESHOLDS[-1])[0], ref)
    assert r_hi > r_lo, f"no additional deviation at high threshold: {r_lo} -> {r_hi}"


# ─── 4. NaN regression for threshold > 1.0 ───────────────────────────────────

@pytest.mark.parametrize("thr", THRESHOLDS_ABOVE_ONE)
def test_blasst_no_nan_above_threshold_one(thr):
    """block_skip_threshold > 1.0 (log2_threshold > 0) must stay finite.

    The skip decision compares the per-tile max against M, the running max
    BEFORE folding in the current tile. Comparing against the post-fold max
    instead makes the difference identically 0 whenever a tile sets a new max --
    including the first tile, where M is -inf. That skips the first tile
    unconditionally, leaves M at -inf, and yields exp2(-inf - -inf) = NaN.
    See BLASST_BLOCK_SKIP_NAN_FIX.md.
    """
    out, _ = _run(PREFILL_SEQ_LENS, thr)
    assert torch.isfinite(out).all()


# ─── 5. decode is force-disabled (prefill/unified-only scope) ────────────────

def test_blasst_decode_is_dense():
    out_thr, _ = _run(DECODE_SEQ_LENS, 8.0)
    out_dense, _ = _run(DECODE_SEQ_LENS, 0.0)
    assert torch.equal(out_thr, out_dense), "decode should ignore block_skip_threshold"


# ─── 6. sliding window is force-disabled in this phase ──────────────────────

def test_blasst_sliding_window_is_dense():
    out_thr, _ = _run(PREFILL_SEQ_LENS, 8.0, sliding_window=128)
    out_dense, _ = _run(PREFILL_SEQ_LENS, 0.0, sliding_window=128)
    assert torch.equal(out_thr, out_dense), "sliding window should run dense"


# ─── 7. the 3D (segmented) kernel path ──────────────────────────────────────
# Few sequences with long KV pushes the dispatcher to the 3D split-K kernel.
# Skips there are segment-local (M resets per segment), so rates are lower --
# the invariants still hold.

LONG_KV_SEQ_LENS = [(128, 16384)]


def test_blasst_3d_no_regression():
    out0, ref = _run(LONG_KV_SEQ_LENS, 0.0, block_size=16)
    assert torch.isfinite(out0).all()
    assert rel_err(out0, ref) < 5e-2


@pytest.mark.parametrize("thr", THRESHOLDS)
def test_blasst_3d_degrades_gracefully(thr):
    """Sub-1.0 thresholds are the real operating range: bounded degradation."""
    out, ref = _run(LONG_KV_SEQ_LENS, thr, block_size=16)
    assert torch.isfinite(out).all()
    assert rel_err(out, ref) < 0.6


@pytest.mark.parametrize("thr", THRESHOLDS_ABOVE_ONE)
def test_blasst_3d_no_nan_above_threshold_one(thr):
    """Thresholds > 1.0 must stay FINITE, but are not accuracy-bounded.

    log2_threshold > 0 means a tile is skipped even when its max is up to
    log2(lambda) ABOVE the running max, which skips almost everything -- a large
    deviation from dense is the expected outcome, not a bug. These thresholds
    exist only to pin the NaN regression (BLASST_BLOCK_SKIP_NAN_FIX.md); the
    2D path asserts the same way.
    """
    out, _ = _run(LONG_KV_SEQ_LENS, thr, block_size=16)
    assert torch.isfinite(out).all()
