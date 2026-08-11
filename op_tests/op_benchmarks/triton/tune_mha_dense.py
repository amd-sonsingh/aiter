# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune the DENSE MHA baseline for the current Triton stack.

The BLASST speedup ratio (dense / BLASST) is only honest if the DENSE baseline
is well-tuned on the stack you measure on. The shipped gfx942 MHA config was
tuned for an older Triton; on a newer stack (e.g. ROCm 7.2.4 / Triton 3.7) the
same config can leave the dense kernel slow, inflating the apparent speedup.

Dense-kernel timing is DATA-INDEPENDENT (it does the full QK/softmax/PV over
every block regardless of values), so this uses synthetic Q/K/V of the model's
shape — no model download/load needed. It sweeps `flash_attn_func`'s `config`
(BLOCK_M/N, num_warps, num_stages, waves_per_eu, PRELOAD_V) for the dense kernel
(block_skip_threshold=0), finds the fastest, and prints the speedup of the tuned
dense over the shipped-default dense. Combine the tuned-dense time with the real
BLASST time (from bench_mha_blasst.py, which needs real patterns) to get the
honest BLASST speedup = tuned_dense / BLASST.

Env: BLASST_CAUSAL (default 0), HEADS (default 32), KV_HEADS (unused; Q shape
     already GQA-expanded), HEAD_DIM (default 128), SEQ_LEN (default 32768),
     BLASST_REF_MS (optional real BLASST ms to print honest speedup).
Run (inside the target container):
     AITER_TRITON_ONLY=1 python op_tests/op_benchmarks/triton/tune_mha_dense.py
"""

import itertools
import os
import time

import torch

from aiter.ops.triton.attention.mha import flash_attn_func

CAUSAL = os.environ.get("BLASST_CAUSAL", "0") == "1"
HEADS = int(os.environ.get("HEADS", "32"))          # Qwen3-8B: 32 query heads
HEAD_DIM = int(os.environ.get("HEAD_DIM", "128"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "32768"))
REF_MS = float(os.environ.get("BLASST_REF_MS", "0"))  # real BLASST ms, optional

GRID = {
    "BLOCK_M": [64, 128, 256],
    "BLOCK_N": [64, 128],
    "num_warps": [4, 8],
    "num_stages": [1, 2],
    "waves_per_eu": [2],
    "PRELOAD_V": [True, False],
    "num_ctas": [1],
}


def bench(fn, warmup=3, repeat=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / repeat * 1000.0


def main():
    if not torch.cuda.is_available():
        print("GPU required")
        return
    import triton
    print(f"Dense-baseline tuning (synthetic q/k/v)")
    print(f"Device: {torch.cuda.get_device_name(0)}  triton={triton.__version__} "
          f"torch={torch.__version__}")
    print(f"Shape: B=1 H={HEADS} S={SEQ_LEN} D={HEAD_DIM} causal={CAUSAL} bf16\n")

    torch.manual_seed(0)
    # flash_attn_func expects (B, S, H, D)
    q = torch.randn(1, SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, SEQ_LEN, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)

    default_ms = bench(lambda: flash_attn_func(q, k, v, causal=CAUSAL, block_skip_threshold=0.0))
    print(f"DEFAULT dense (shipped config): {default_ms:.2f} ms\n")

    keys = list(GRID)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[GRID[k] for k in keys])]
    combos = [c for c in combos if c["BLOCK_N"] <= c["BLOCK_M"]]
    print(f"Sweeping {len(combos)} dense configs...")

    results = []
    for i, cfg in enumerate(combos):
        try:
            ms = bench(lambda cfg=cfg: flash_attn_func(
                q, k, v, causal=CAUSAL, block_skip_threshold=0.0, config=dict(cfg)))
            results.append((ms, cfg))
            tag = ""
        except Exception as e:  # noqa: BLE001
            ms = float("inf")
            tag = f"  (skip: {type(e).__name__})"
        print(f"  [{i+1:2}/{len(combos)}] {ms:8.2f} ms  "
              f"BM{cfg['BLOCK_M']} BN{cfg['BLOCK_N']} w{cfg['num_warps']} "
              f"s{cfg['num_stages']} pv{int(cfg['PRELOAD_V'])}{tag}")

    results.sort(key=lambda r: r[0])
    if not results:
        print("No valid configs.")
        return
    best_ms, best_cfg = results[0]
    print(f"\n{'='*60}")
    print(f"DEFAULT dense : {default_ms:8.2f} ms")
    print(f"BEST dense    : {best_ms:8.2f} ms  ({default_ms/best_ms:.2f}x faster than default)")
    print(f"BEST config   : {best_cfg}")
    if REF_MS > 0:
        print(f"\nReal BLASST (from bench, t=0.30): {REF_MS:.2f} ms")
        print(f"  speedup vs DEFAULT dense: {default_ms/REF_MS:.2f}x")
        print(f"  speedup vs TUNED   dense: {best_ms/REF_MS:.2f}x   <-- honest")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
