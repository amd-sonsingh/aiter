# BLASST Block-Skipping in the Triton MHA Kernel

Adds **BLASST** (dynamic **BL**ocked **A**ttention **S**parsity via **S**oftmax
**T**hresholding) block-skipping to AITER's Triton flash-attention **prefill**
kernel. A K/V block is skipped for query rows whose per-block max score falls
below the running softmax max by more than a calibrated threshold — eliding that
block's V load and P@V matmul, since it contributes negligible softmax mass.

Paper: Yuan et al., *"BLASST: Dynamic BLocked Attention Sparsity via Softmax
Thresholding"*, MLSys 2026 (arXiv:2512.12087).

---

## API

```python
from aiter.ops.triton.attention.mha import flash_attn_func

out = flash_attn_func(
    q, k, v,                    # (batch, seqlen, nheads, headdim)
    causal=True,
    block_skip_threshold=0.029, # 0.0 = disabled (dense); >0 enables BLASST
)
```

- `block_skip_threshold` is the **natural-space** softmax threshold `λ`. `0.0`
  (default) is exactly the original dense kernel — no overhead.
- Larger threshold ⇒ more blocks skipped ⇒ higher sparsity ⇒ larger deviation
  from dense.
- **Default (`default`) MHA impl only.** Not supported with `dao_ai` impl,
  dropout, or `return_attn_probs`. Forward / inference only (no backward).

### Choosing a threshold (calibration)

The threshold for a target sparsity `S` at sequence length `L` follows the
paper's Algorithm 2 fit `λ = α · exp(β · S) / L`. Example — Qwen3-8B, calibrated
α=7.4142, β=9.6915:

```
λ(50% sparsity, 32K) = 7.4142 · exp(9.6915 · 0.5) / 32768 ≈ 0.02878
```

Calibration itself is out of scope here (see the `attention-evals` repo); this
kernel just consumes the resulting `λ`.

---

## Implementation

Two files change; the logic is guarded by a compile-time `constexpr` so it
**compiles away entirely when disabled** (zero overhead for the dense path).

| File | Change |
|------|--------|
| `aiter/ops/triton/_triton_kernels/attention/mha.py` | `_attn_fwd_inner` + `_attn_fwd`: `ENABLE_BLOCK_SKIP` flag + `log2_threshold`; skip decision + gated V-load/P@V. Threaded through both inner-loop call sites (full + masked blocks). |
| `aiter/ops/triton/attention/mha.py` | `_flash_attn_forward` / `flash_attn_func`: `block_skip_threshold` arg; natural→log2 conversion; forces `PRELOAD_V=False`. |

Key details:

- **Skip decision** (per K/V block, per query row):
  `skip = (block_max − running_max) < log2_threshold`; a block's V load + P@V
  matmul are elided only when **all** rows in the block skip (`all_skip`).
  Partially-skipped rows have their probabilities zeroed so their accumulator
  updates self-cancel.
- **log2 units.** AITER runs softmax in base-2 (`exp2`, `qk` pre-scaled by
  `RCP_LN2`), so the natural-space `λ` is converted with
  `log2_threshold = ln(λ) · RCP_LN2`.
- **`PRELOAD_V=False`.** The savings come from *not loading V* for skipped
  blocks, so the wrapper forces the deferred-V-load path when skipping is on.
- **QK is always computed** (needed to decide the skip), so only the V load and
  P@V matmul are saved — the theoretical ceiling is ~2×.

---

## Correctness

Verified on AMD Instinct **MI300X (gfx942)** (also imports/runs on gfx950):

- `block_skip_threshold=0` matches dense SDPA to **rel ~1.4e-3** (bf16 noise).
- Output matches an independent PyTorch/Triton BLASST reference at every
  threshold, and deviates from dense monotonically as the threshold grows —
  confirming skipping actually happens.

Correctness test: `op_tests/triton_tests/attention/test_mha_blasst.py`
```bash
pytest op_tests/triton_tests/attention/test_mha_blasst.py
# or standalone:
python op_tests/triton_tests/attention/test_mha_blasst.py
# if C++ ops aren't built in your env:
AITER_TRITON_ONLY=1 python op_tests/triton_tests/attention/test_mha_blasst.py
```

Speedup reproduction: `op_tests/op_benchmarks/triton/bench_mha_blasst.py` — captures
**real Qwen3-8B** attention patterns (random tensors are not representative, as
they rarely produce whole skippable blocks) and times dense vs BLASST:
```bash
# non-causal; set BLASST_CAUSAL=1 for causal; BLASST_INPUT_FILE=<ruler.jsonl> for a
# real long-context prompt (else a synthetic prompt is used).
AITER_TRITON_ONLY=1 python op_tests/op_benchmarks/triton/bench_mha_blasst.py
```

---

## Performance note (read before benchmarking)

Block-skipping is **net-positive only at high sparsity** on this kernel, and can
be *slower* at low thresholds. The dense kernel is software-pipelined
(`num_stages=2`); the data-dependent skip branch defeats the pipeliner (it can't
prefetch a V block it doesn't yet know it needs), so every non-skipped block
loses load/compute overlap. You come out ahead only once enough blocks skip to
pay for that. Expect the best results on **causal prefill** at meaningful
sparsity; do not expect speedup at low sparsity or in non-causal microbenchmarks.
Recovering pipelining under the skip branch is the main open optimization.
