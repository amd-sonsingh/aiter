# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness test for BLASST block-skipping in the Triton MHA kernel.

BLASST (dynamic BLocked Attention Sparsity via Softmax Thresholding) skips a
K/V block for query rows whose per-block max score falls below the running
softmax max by more than a threshold, eliding that block's V load and P@V
matmul. Exposed via ``flash_attn_func(..., block_skip_threshold=X)`` (0 = off).
See ``BLASST_BLOCK_SKIP.md`` at the repo root.

Self-contained inside AITER: the dense reference is PyTorch SDPA (the ground
truth). For SPEEDUP measurement use
``op_tests/op_benchmarks/triton/bench_mha_blasst.py`` — it captures REAL
Qwen3-8B attention patterns, which is required for a meaningful speedup number
(random tensors rarely produce whole skippable blocks).

Checks (bf16, causal + non-causal):
  [1] NO-REGRESSION  threshold=0 == dense SDPA
  [2] DEGRADATION    threshold>0 vs dense stays bounded, grows with threshold
  [3] SKIP HAPPENS   larger threshold deviates from dense more than smaller

Run:
    pytest op_tests/triton_tests/attention/test_mha_blasst.py
    python op_tests/triton_tests/attention/test_mha_blasst.py   # numbered report
Note: set AITER_TRITON_ONLY=1 if AITER's C++ ops are not built in your env.
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.attention.mha import flash_attn_func

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def dense_ref(q, k, v, causal):
    """Dense attention reference. q/k/v: (B, S, H, D); fp32 SDPA math."""
    qt, kt, vt = (x.transpose(1, 2).float() for x in (q, k, v))
    o = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
    return o.transpose(1, 2).to(q.dtype)


def rel_err(a, b):
    return (
        (a.float() - b.float()).abs().mean() / b.float().abs().mean().clamp_min(1e-6)
    ).item()


def max_err(a, b):
    return (a.float() - b.float()).abs().max().item()


def _qkv(B, S, H, D, seed=0):
    torch.manual_seed(seed)
    g = lambda: torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    return g(), g(), g()


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
@pytest.mark.parametrize("thr", [1e-3, 1e-2, 5e-2, 1e-1, 3e-1])
def test_blasst_degrades_gracefully(causal, thr):
    q, k, v = _qkv(1, 4096, 8, 128)
    dense = dense_ref(q, k, v, causal)
    out = flash_attn_func(q, k, v, causal=causal, block_skip_threshold=thr)
    assert torch.isfinite(out).all()
    assert rel_err(out, dense) < 0.6


@pytest.mark.parametrize("causal", [True, False])
def test_blasst_actually_skips(causal):
    q, k, v = _qkv(1, 4096, 8, 128)
    dense = dense_ref(q, k, v, causal)
    r_lo = rel_err(flash_attn_func(q, k, v, causal=causal, block_skip_threshold=1e-3), dense)
    r_hi = rel_err(flash_attn_func(q, k, v, causal=causal, block_skip_threshold=3e-1), dense)
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


# ─── standalone numbered report (mirrors scripts/test_aiter_blasst.py) ─────

def main():
    if not torch.cuda.is_available():
        print("ERROR: GPU required.")
        sys.exit(1)

    dev, dtype = "cuda", torch.bfloat16
    B, S, H, D = 1, 4096, 8, 128
    causal = os.environ.get("BLASST_CAUSAL", "1") == "1"
    print(f"Config: B={B} S={S} H={H} D={D} dtype={dtype} causal={causal}")
    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    q, k, v = _qkv(B, S, H, D)
    dense = dense_ref(q, k, v, causal)
    ok = True

    # [1] no-regression
    out0 = flash_attn_func(q, k, v, causal=causal, block_skip_threshold=0.0)
    r, m = rel_err(out0, dense), max_err(out0, dense)
    passed = r < 5e-3 and torch.isfinite(out0).all().item()
    ok &= passed
    print(f"[1] NO-REGRESSION  t=0 vs dense SDPA   rel={r:.2e} max={m:.2e}  "
          f"{'PASS' if passed else 'FAIL'}")

    # [2] degradation
    print("\n[2] DEGRADATION vs dense (should grow with threshold, stay bounded):")
    thresholds = [1e-3, 1e-2, 5e-2, 1e-1, 3e-1]
    rels = {}
    for t in thresholds:
        out = flash_attn_func(q, k, v, causal=causal, block_skip_threshold=t)
        rels[t] = rel_err(out, dense)
        bounded = rels[t] < 0.6 and torch.isfinite(out).all().item()
        ok &= bounded
        print(f"      t={t:<6} rel_vs_dense={rels[t]:.3e}  "
              f"{'ok' if bounded else 'OUT-OF-BOUND'}")

    # [3] skip happens
    passed = rels[thresholds[-1]] > rels[thresholds[0]]
    ok &= passed
    print(f"\n[3] SKIP HAPPENS   rel(t={thresholds[-1]}) > rel(t={thresholds[0]})  "
          f"{'PASS' if passed else 'FAIL'}")

    print("\n(For speedup: op_tests/op_benchmarks/triton/bench_mha_blasst.py)")
    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
