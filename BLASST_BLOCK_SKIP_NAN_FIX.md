# BLASST Block-Skip NaN Fix

Fixes a correctness bug in BLASST block-skipping (see `BLASST_BLOCK_SKIP.md`)
where `flash_attn_func(..., block_skip_threshold=X)` returns all-NaN output
for any `X > 1.0`.

## Bug

In `_attn_fwd_inner` (`aiter/ops/triton/_triton_kernels/attention/mha.py`),
the skip decision compared the per-block max score against the running
softmax max **after** folding in the current block (`m_ij`) instead of the
running max **before** it (`m_i`):

```python
qk_max = tl.max(qk, 1)
m_ij = tl.maximum(m_i, qk_max)
...
skip = (qk_max - m_ij) < log2_threshold   # wrong: m_ij, not m_i
```

Since `m_ij = max(m_i, qk_max)`, the quantity `qk_max - m_ij` is identically
`0` whenever the current block sets a new running max -- which is always
true for the very first K/V block of a row, where `m_i` starts at `-inf`.
`log2_threshold = ln(block_skip_threshold) / ln(2)` is `>= 0` exactly when
`block_skip_threshold >= 1.0`, so for any `block_skip_threshold > 1.0`,
`skip` evaluates to `0 < log2_threshold` = `True` unconditionally on that
first block, regardless of the actual Q/K values.

When the first block is skipped this way, `m_ij` is reset back to
`m_i = -inf` (via `m_ij = tl.where(skip, m_i, m_ij)`), and the downstream
softmax rescale computes `exp2(m_i - m_ij) = exp2(-inf - (-inf)) = NaN`,
which then propagates through the accumulator and normalizer for the rest
of the kernel.

**Net effect**: 100% NaN output for any `block_skip_threshold > 1.0`.
Values `<= 1.0` are unaffected (the comparison only diverges from correct
when `log2_threshold >= 0`).

## Fix

Compare against `m_i` (the running max *before* this block), which is the
mathematically correct form -- a block is safe to skip exactly when it does
not raise the running max relative to where it stood going in, and matches
the comment above the line describing that intent:

```python
skip = (qk_max - m_i) < log2_threshold
```

## Verification

Added `test_blasst_no_nan_above_threshold_one` to
`op_tests/triton_tests/attention/test_mha_blasst.py`, parametrized over
`block_skip_threshold in [1.001, 1.1, 1.3, 2.0, 4.0, 8.0, 12.0]` and
`causal in [True, False]` -- the region none of the existing tests reached
(they top out at `3e-1`). Fails on the pre-fix kernel (all-NaN output),
passes after.

Run:
```sh
AITER_TRITON_ONLY=1 pytest op_tests/triton_tests/attention/test_mha_blasst.py -k no_nan
```
