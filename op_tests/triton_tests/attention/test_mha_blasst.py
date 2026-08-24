# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness test for BLASST block-skipping in the Triton MHA kernel.

BLASST (dynamic BLocked Attention Sparsity via Softmax Thresholding) skips a
K/V block for query rows whose per-block max score falls below the running
softmax max by more than a threshold, eliding that block's V load and P@V
matmul. Exposed via ``flash_attn_func(..., block_skip_threshold=X)`` (0 = off).
See ``BLASST_BLOCK_SKIP.md`` at the repo root.

Self-contained inside AITER. Two references are used, for two different jobs:

  * ``dense_ref``  -- PyTorch SDPA. Ground truth for the *dense* kernel path
    (threshold = 0). Comparing a SPARSE run against it measures how much work
    BLASST elided, which is a *behaviour* signal, not a correctness one.
  * ``blasst_ref`` -- a PyTorch transcription of the BLASST online softmax with
    the *same* skip rule and the *same* K/V tiling as the kernel. Comparing the
    sparse kernel against this leaves only floating-point error, so it can carry
    a tight tolerance and is the actual correctness check.

For SPEEDUP measurement use ``op_tests/op_benchmarks/triton/bench_mha_blasst.py``
-- it captures REAL Qwen3-8B attention patterns, which is required for a
meaningful speedup number (random tensors rarely produce whole skippable blocks).

Checks (bf16, causal + non-causal):
  [1] NO-REGRESSION  threshold=0 == dense SDPA
  [2] GOLDEN         threshold>0 == blasst_ref, to within fp error: a tight
                     bound on the MEAN, plus a bound on how many output rows
                     may disagree at all (see the tolerance block below --
                     rows disagree only when a skip decision lands on a
                     floating-point tie, and the count, not the magnitude, is
                     the meaningful quantity)
  [3] SKIP HAPPENS   the reference reports a non-zero skip fraction, and larger
                     thresholds deviate from dense more than smaller ones
  [4] NO NaN         threshold>1.0 stays finite (regression guard)

Run:
    pytest op_tests/triton_tests/attention/test_mha_blasst.py
    python op_tests/triton_tests/attention/test_mha_blasst.py   # numbered report
Note: set AITER_TRITON_ONLY=1 if AITER's C++ ops are not built in your env.
"""

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton._triton_kernels.attention.mha import _get_config
from aiter.ops.triton.attention.mha import flash_attn_func

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")

# 1 / ln(2). The kernel folds this into the QK scale so the softmax runs in
# base-2 (exp2 is a hardware instruction); the threshold is converted the same
# way, hence log2_threshold = ln(lambda) / ln(2).
RCP_LN2 = 1.4426950408889634

# ─── tolerances for kernel-vs-blasst_ref ─────────────────────────────────
#
# The kernel and blasst_ref run the identical algorithm over the identical
# BLOCK_N tiling, so there are exactly three sources of difference, and each
# gets its own bound rather than being lumped into one number.
#
# (a) CONTINUOUS fp error -- MFMA bf16xbf16->fp32 for QK and P@V accumulates in
#     a different order than torch's fp32 GEMM. This moves every element a
#     little and is what REF_REL_TOL bounds.
#
# (b) The bf16 STORE. acc/l_i are fp32 in registers and rounded to bf16 (8
#     explicit mantissa bits) on the way out, so any single element can differ
#     by ~2^-8 relative. Attention output of unit-normal V has |o| < 1 on
#     average, so 2^-7 = 7.8e-3 is a hard ceiling. That is REF_ELEM_TOL.
#     Every measured elementwise deviation that is not (c) lands on an exact
#     bf16 ULP: 4.9e-4, 9.8e-4, 2.0e-3, or 3.9e-3.
#
# (c) BOUNDARY TIES -- discrete, and the reason a plain max-error bound is the
#     wrong instrument. A row's skip decision is `(qk_max - m_i) < log2_thr`.
#     When that difference lands within (a)'s noise of log2_thr, the kernel and
#     the reference can decide oppositely; the row then either gains or loses a
#     whole K/V block. That is a large deviation in ONE row and invisible in the
#     mean, so bounding max_err would either be vacuous or hostage to the seed.
#     Instead we bound HOW MANY rows may disagree: REF_TIE_ROW_FRAC.
#     A tie is genuinely a tie -- neither answer is more correct, since the
#     kernel's qk and the reference's qk are both approximations of the same
#     real number and the two skip decisions bracket it.
#
# Measured on MI300X (gfx942, BLOCK_M=128 / BLOCK_N=64) over the full grid
# {(1,4096,8,128), (2,2048,4,128), (1,1024,8,64), (1,3000,4,128)} x 2 seeds
# x {causal, non-causal} x 12 thresholds in [1e-3, 8.0] = 192 configurations.
# See the RESULTS block at the bottom of this file. Worst observed:
#     mean rel        1.40e-5   -> REF_REL_TOL 1e-4   (7x headroom)
#     non-tie elem    3.91e-3   -> REF_ELEM_TOL 8e-3  (2x, and = 2^-7 exactly)
#     disagreeing rows      1   -> REF_TIE_ROW_FRAC 1e-4 (3 rows of 32768)
REF_REL_TOL = 1e-4  # mean |kernel - ref| / mean |ref|
REF_ELEM_TOL = 8e-3  # per-element |kernel - ref|, outside tie rows
REF_TIE_ROW_FRAC = 1e-4  # fraction of rows allowed to exceed REF_ELEM_TOL


def dense_ref(q, k, v, causal):
    """Dense attention reference. q/k/v: (B, S, H, D); fp32 SDPA math."""
    qt, kt, vt = (x.transpose(1, 2).float() for x in (q, k, v))
    o = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
    return o.transpose(1, 2).to(q.dtype)


def blasst_ref(q, k, v, causal, threshold, block_n):
    """Golden reference for the BLASST kernel: FlashAttention online softmax
    with block skipping, in PyTorch.

    Ported from the PyTorch reference implementation of BLASST (MLSys 2026),
    Algorithm 1, with two deliberate changes so that it matches this kernel
    exactly rather than the paper's pseudocode:

    1. The skip test compares the block max against the running max *before*
       this block (``m_i``), not after folding it in (``m_ij``). See
       ``BLASST_BLOCK_SKIP_NAN_FIX.md``: for threshold < 1 the two forms are
       algebraically equivalent, but for threshold > 1 the post-fold form skips
       the very first block unconditionally (``qk_max - m_ij`` is identically 0
       there), leaves ``m_i`` at -inf, and yields NaN.
    2. Everything is in base-2 units (``exp2``, ``log2_threshold``), matching
       the kernel, so exponentials round identically.

    The skip decision is PER ROW in the kernel -- ``skip`` is a BLOCK_M-vector
    and skipped rows get ``m_ij = m_i`` and ``p = 0`` individually. The kernel's
    ``all_skip`` (a BLOCK_M-wide reduction) only decides whether the V load and
    the P@V matmul are *issued*; when it is false the skipped rows still carry
    p = 0 into the matmul and contribute nothing, and when it is true the elided
    update is a no-op (alpha = exp2(m_i - m_i) = 1, l_ij = 0). So ``all_skip``
    has no numerical effect and this row-wise reference is exact.

    BLOCK_M likewise has no numerical effect: it only sets how far the kernel's
    K loop runs for a given row group under causality, and the extra tiles a row
    sees beyond its own causal boundary are fully masked (qk_max = -inf), so
    they are skipped / contribute p = 0 either way. BLOCK_N *does* matter -- it
    is the granularity of ``qk_max`` -- and is read from the kernel's own
    ``_get_config`` by the caller.

    q/k/v: (B, S, H, D). Returns (out, skip_fraction).
    """
    B, S_q, H, D = q.shape
    S_k = k.shape[1]
    qt = q.transpose(1, 2).float()  # (B, H, S_q, D)
    kt = k.transpose(1, 2).float()
    vt = v.transpose(1, 2)

    qk_scale = (1.0 / math.sqrt(D)) * RCP_LN2
    enable = threshold > 0.0
    log2_threshold = math.log(threshold) * RCP_LN2 if enable else 0.0

    m_i = torch.full((B, H, S_q), float("-inf"), device=q.device, dtype=torch.float32)
    l_i = torch.zeros_like(m_i)
    acc = torch.zeros(B, H, S_q, D, device=q.device, dtype=torch.float32)

    offs_m = torch.arange(S_q, device=q.device)
    n_skipped = 0
    n_visited = 0

    for start_n in range(0, S_k, block_n):
        end_n = min(start_n + block_n, S_k)
        kb = kt[:, :, start_n:end_n, :]
        qk = (qt @ kb.transpose(-2, -1)) * qk_scale
        if causal:
            offs_n = torch.arange(start_n, end_n, device=q.device) + (S_q - S_k)
            qk = torch.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

        qk_max = qk.max(dim=-1).values          # (B, H, S_q)
        m_ij = torch.maximum(m_i, qk_max)
        if enable:
            # NOTE: m_i, not m_ij. See point (1) above.
            skip = (qk_max - m_i) < log2_threshold
            m_ij = torch.where(skip, m_i, m_ij)

        p = torch.exp2(qk - m_ij[..., None])
        if enable:
            p = torch.where(skip[..., None], 0.0, p)

        l_ij = p.sum(dim=-1)
        alpha = torch.exp2(m_i - m_ij)
        acc = acc * alpha[..., None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        # The kernel casts p to V's dtype before the dot and accumulates in fp32.
        acc = acc + p.to(v.dtype).float() @ vt[:, :, start_n:end_n, :].float()

        # Skip accounting, over tiles that are live for a row. A tile entirely
        # past a row's causal boundary has qk_max = -inf and is "skipped"
        # trivially -- the kernel does not even visit it -- so exclude those.
        live = torch.isfinite(qk_max)
        n_visited += int(live.sum())
        if enable:
            n_skipped += int((skip & live).sum())

    out = (acc / l_i[..., None]).transpose(1, 2).to(q.dtype)
    return out, (n_skipped / n_visited if n_visited else 0.0)


def kernel_block_n(q, v):
    """The BLOCK_N the kernel will actually use for this call."""
    return _get_config(False, q.dtype, has_pe=False, head_dim_v=v.shape[-1])["BLOCK_N"]


def rel_err(a, b):
    return (
        (a.float() - b.float()).abs().mean() / b.float().abs().mean().clamp_min(1e-6)
    ).item()


def max_err(a, b):
    return (a.float() - b.float()).abs().max().item()


def tie_rows(a, b):
    """(rows disagreeing beyond the bf16 store floor, total rows).

    a/b are (B, S, H, D). A row is one (b, s, h) output vector of length D --
    the granularity at which a skip decision is made, so a flipped decision
    shows up as one whole bad row rather than as scattered elements.
    """
    bad = (a.float() - b.float()).abs() > REF_ELEM_TOL
    per_row = bad.any(dim=-1)
    return int(per_row.sum()), per_row.numel()


def assert_matches_ref(out, ref, thr):
    """The golden check. See the tolerance block at the top of this file."""
    assert torch.isfinite(out).all(), f"non-finite kernel output at thr={thr}"
    r = rel_err(out, ref)
    assert r < REF_REL_TOL, f"thr={thr}: mean rel={r:.3e} (bound {REF_REL_TOL:.1e})"
    bad, total = tie_rows(out, ref)
    budget = max(1, int(REF_TIE_ROW_FRAC * total))
    assert bad <= budget, (
        f"thr={thr}: {bad}/{total} rows differ by more than {REF_ELEM_TOL:.1e} "
        f"(tie budget {budget}); max elem err {max_err(out, ref):.3e}"
    )


def _qkv(B, S, H, D, seed=0):
    torch.manual_seed(seed)
    g = lambda: torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    return g(), g(), g()


# All thresholds the golden comparison covers, including the >1.0 region that
# used to produce all-NaN output.
THRESHOLDS = [1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 1.001, 1.3, 2.0, 8.0]


# ─── pytest ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("B,S,H,D", [(1, 4096, 8, 128)])
def test_blasst_no_regression(B, S, H, D, causal):
    q, k, v = _qkv(B, S, H, D)
    dense = dense_ref(q, k, v, causal)
    out0 = flash_attn_func(q, k, v, causal=causal, block_skip_threshold=0.0)
    assert torch.isfinite(out0).all()
    assert rel_err(out0, dense) < 5e-3


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("thr", THRESHOLDS)
def test_blasst_matches_reference(causal, thr):
    """The real correctness check: the sparse kernel against the SAME sparse
    algorithm in PyTorch, over the same BLOCK_N tiling. Any deviation here is
    floating point, not sparsity, so the bound is tight.
    """
    q, k, v = _qkv(1, 4096, 8, 128)
    ref, _ = blasst_ref(q, k, v, causal, thr, kernel_block_n(q, v))
    out = flash_attn_func(q, k, v, causal=causal, block_skip_threshold=thr)
    assert_matches_ref(out, ref, thr)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("thr", [1e-3, 1e-2, 5e-2, 1e-1, 3e-1])
def test_blasst_degrades_gracefully(causal, thr):
    """Deviation from dense grows with the threshold -- that is BLASST doing its
    job, not an error -- but the kernel must still track the sparse reference to
    within floating-point error at every threshold.

    The sparse run may only move AWAY from dense, never towards it. The
    comparison carries 0.1% slack because enabling block skipping changes the
    compiled kernel even when it skips nothing: the P@V dot moves inside an
    `scf.if`, which changes the warp layout and hence the MFMA accumulation
    order, so the two runs are not bit-identical. At thr=1e-3 nothing is
    skipped and the two rel-vs-dense values agree to 7 significant figures
    (1.4207333e-3 vs 1.4207363e-3) with the sparse one very slightly lower --
    pure reassociation noise, ~2e-6 relative.
    """
    q, k, v = _qkv(1, 4096, 8, 128)
    ref, _ = blasst_ref(q, k, v, causal, thr, kernel_block_n(q, v))
    out = flash_attn_func(q, k, v, causal=causal, block_skip_threshold=thr)
    dense = dense_ref(q, k, v, causal)
    r0 = rel_err(
        flash_attn_func(q, k, v, causal=causal, block_skip_threshold=0.0), dense
    )
    r = rel_err(out, dense)
    assert r >= r0 * (1 - 1e-3), f"thr={thr}: moved towards dense, {r:.8e} < {r0:.8e}"
    # ... while staying pinned to the sparse reference.
    assert_matches_ref(out, ref, thr)


@pytest.mark.parametrize("causal", [True, False])
def test_blasst_actually_skips(causal):
    """Guards against the golden comparison being vacuous: blocks really are
    being skipped, and skipping more moves the output further from dense.

    The low anchor is 0.1, not 1e-3. On i.i.d. normal Q/K the per-block score
    maxima are tightly clustered -- over a BLOCK_N=64 tile the max of 64 scores
    barely varies from tile to tile -- so no tile is ever 1000x below the
    running max and the measured skip fraction at 1e-3 is exactly 0.0 (both
    causal and not, every seed and shape tried). That is BLASST behaving
    correctly on unstructured data, not a bug; real attention is skewed enough
    for small thresholds to bite, which is what the Qwen3-8B benchmark in
    op_benchmarks/triton/bench_mha_blasst.py exercises. 0.1 is the smallest
    threshold that skips a measurable amount here (~1.5%).
    """
    q, k, v = _qkv(1, 4096, 8, 128)
    block_n = kernel_block_n(q, v)
    dense = dense_ref(q, k, v, causal)
    THR_LO, THR_HI = 1e-1, 3e-1

    _, frac_lo = blasst_ref(q, k, v, causal, THR_LO, block_n)
    _, frac_hi = blasst_ref(q, k, v, causal, THR_HI, block_n)
    assert frac_hi > frac_lo > 0.0, f"skip fractions {frac_lo} / {frac_hi}"

    r_lo = rel_err(
        flash_attn_func(q, k, v, causal=causal, block_skip_threshold=THR_LO), dense
    )
    r_hi = rel_err(
        flash_attn_func(q, k, v, causal=causal, block_skip_threshold=THR_HI), dense
    )
    assert r_hi > r_lo


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("thr", [1.001, 1.1, 1.3, 2.0, 4.0, 8.0, 12.0])
def test_blasst_no_nan_above_threshold_one(causal, thr):
    """block_skip_threshold > 1.0 (log2_threshold > 0) previously produced
    100% NaN output: the skip decision compared qk_max against m_ij (the
    running max AFTER folding in the current block) instead of m_i (the
    running max BEFORE it), so qk_max - m_ij was identically 0 whenever a
    block set a new running max -- including the first block, where m_i
    starts at -inf. That made the first block get skipped unconditionally,
    leaving m_i at -inf and producing exp2(-inf - (-inf)) = NaN downstream.
    """
    q, k, v = _qkv(1, 4096, 8, 128)
    out = flash_attn_func(q, k, v, causal=causal, block_skip_threshold=thr)
    assert torch.isfinite(out).all()


# ─── standalone numbered report ───────────────────────────────────────────

def main():
    if not torch.cuda.is_available():
        print("ERROR: GPU required.")
        sys.exit(1)

    dtype = torch.bfloat16
    B, S, H, D = 1, 4096, 8, 128
    causal = os.environ.get("BLASST_CAUSAL", "1") == "1"
    q, k, v = _qkv(B, S, H, D)
    block_n = kernel_block_n(q, v)
    print(f"Config: B={B} S={S} H={H} D={D} dtype={dtype} causal={causal}")
    print(f"Device: {torch.cuda.get_device_name(0)}   kernel BLOCK_N={block_n}\n")

    dense = dense_ref(q, k, v, causal)
    ok = True

    # [1] no-regression
    out0 = flash_attn_func(q, k, v, causal=causal, block_skip_threshold=0.0)
    r, m = rel_err(out0, dense), max_err(out0, dense)
    passed = r < 5e-3 and torch.isfinite(out0).all().item()
    ok &= passed
    print(f"[1] NO-REGRESSION  t=0 vs dense SDPA   rel={r:.2e} max={m:.2e}  "
          f"{'PASS' if passed else 'FAIL'}")

    # [2] golden reference. `tie` counts rows differing by more than the bf16
    # store floor -- flipped skip decisions, budget printed alongside.
    print(f"\n[2] vs BLASST GOLDEN REFERENCE (bounds rel<{REF_REL_TOL:.0e}, "
          f"<={max(1, int(REF_TIE_ROW_FRAC * B * S * H))} tie rows of {B * S * H}); "
          f"rel_vs_dense is BLASST working, not error:")
    print(f"      {'thr':<8}{'skip%':>8}{'rel_vs_ref':>13}{'max_vs_ref':>13}"
          f"{'tie_rows':>10}{'rel_vs_dense':>15}")
    rels_dense = {}
    for t in THRESHOLDS:
        ref, frac = blasst_ref(q, k, v, causal, t, block_n)
        out = flash_attn_func(q, k, v, causal=causal, block_skip_threshold=t)
        rr, mm = rel_err(out, ref), max_err(out, ref)
        bad, total = tie_rows(out, ref)
        rels_dense[t] = rel_err(out, dense)
        good = (rr < REF_REL_TOL and bad <= max(1, int(REF_TIE_ROW_FRAC * total))
                and torch.isfinite(out).all().item())
        ok &= good
        print(f"      {t:<8}{100 * frac:>7.1f}%{rr:>13.2e}{mm:>13.2e}{bad:>10}"
              f"{rels_dense[t]:>15.2e}  {'ok' if good else 'OUT-OF-BOUND'}")

    # [3] skip happens
    passed = rels_dense[3e-1] > rels_dense[1e-1]
    ok &= passed
    print(f"\n[3] SKIP HAPPENS   rel_vs_dense(t=0.3) > rel_vs_dense(t=0.1)  "
          f"{'PASS' if passed else 'FAIL'}")

    print("\n(For speedup: op_tests/op_benchmarks/triton/bench_mha_blasst.py)")
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()


# ─── RESULTS: where the tolerances come from ──────────────────────────────
#
# MI300X (gfx942), ROCm 7.2.4, torch 2.10.0, triton 3.8.0, bf16.
# Kernel config for all of these: BLOCK_M=128, BLOCK_N=64, PRELOAD_V=False.
# Grid: 4 shapes x 2 seeds x {causal, non-causal} x 12 thresholds = 192 runs.
#   shapes  (B,S,H,D) = (1,4096,8,128) (2,2048,4,128) (1,1024,8,64) (1,3000,4,128)
#   thr     1e-3 1e-2 5e-2 1e-1 3e-1 0.5 0.9 1.001 1.3 2.0 4.0 8.0
#
# Aggregate:
#   worst mean rel        1.40e-05   at (1,4096,8,128) seed0 non-causal thr=0.3
#   worst max elem        1.43e-01   at (1,4096,8,128) seed0 causal     thr=1.001
#   worst tie rows                1  (of 32768)  -- same config
#   configs with >1 tie row       0
#
# The max-elem and the mean disagree by four orders of magnitude, which is the
# whole reason the bound is not on max. Representative slice, (1,4096,8,128)
# seed 0, causal; `ties` = decisions the reference places within 1e-3 (log2) of
# the boundary, i.e. an upper bound on how many the kernel could flip:
#
#   thr      skip%      rel        max     bad_rows   ties
#   0.001     0.0%   4.50e-06   1.95e-03      0          0
#   0.01      0.0%   4.50e-06   1.95e-03      0          0
#   0.05      0.2%   4.49e-06   1.95e-03      0          7
#   0.1       1.5%   4.49e-06   1.95e-03      0         75
#   0.3      30.9%   3.58e-06   1.95e-03      0        857
#   0.5      61.8%   2.65e-06   1.95e-03      0        843
#   0.9      86.0%   1.70e-06   3.91e-03      0        355
#   1.001    88.4%   1.16e-05   1.43e-01      1        298
#   1.3      92.0%   1.33e-06   3.91e-03      0        195
#   2.0      94.4%   1.18e-06   3.91e-03      0        107
#   8.0      96.8%   9.62e-07   3.91e-03      0         13
#
# Read that as: every threshold agrees with the reference to 1e-6..1e-5 in the
# mean; every elementwise deviation is an exact bf16 ULP (1.95e-3 = 2^-9,
# 3.91e-3 = 2^-8) except in one row at thr=1.001, where one skip decision landed
# on the wrong side of the boundary and that row lost a K/V block. Across all
# 192 configurations bad_rows never exceeded 1 and never exceeded `ties`, so the
# outliers are entirely accounted for by boundary ties and nothing else.
#
# thr=1.001 is the worst case by construction: log2(1.001) = 1.4e-3, so the test
# is "skip unless this block beats the running max by 0.0014" -- the decision
# boundary sits right where the running max is being set, which is both the
# densest part of the (qk_max - m_i) distribution and the point where a flip
# costs the most (the flipped block is a top-contributing one, not a negligible
# one). Hence 1.43e-1 in a single row while the mean stays at 1.16e-5.
#
# GRANULARITY. blasst_ref matches the kernel exactly on the dimension that
# affects the answer:
#   BLOCK_N  MUST match and does -- kernel_block_n() reads _get_config(), the
#            same call the wrapper makes. It sets the granularity of qk_max and
#            therefore of every skip decision.
#   BLOCK_M  does NOT need to match, and the reference has no BLOCK_M (it is
#            fully row-wise). The kernel's skip vector is per-row; BLOCK_M only
#            sets how far the K loop runs for a row group under causality, and
#            the extra tiles a row sees past its own causal boundary are fully
#            masked (qk_max = -inf), so they contribute p = 0 either way.
#   all_skip is a BLOCK_M-wide reduction, but it only gates whether the V load
#            and P@V matmul are ISSUED. When false, skipped rows still carry
#            p = 0 through the matmul; when true, the elided update is a no-op
#            (alpha = exp2(m_i - m_i) = 1, l_ij = 0). No numerical effect.
# So the only residual discrepancy is floating point, and it is bounded above.
#
# Reproduce: python op_tests/triton_tests/attention/test_mha_blasst.py
