# MLSys 2026 Hopper-Prefill-Shape Reproduction

Reproduces the fixed benchmark shape used by the BLASST paper's own artifact
evaluation (`blasst-ae-mlsys26/hopper_prefill`) on top of this repo's Triton
MHA kernel, for direct speedup-table comparison against the paper's reported
NVIDIA Hopper numbers.

Paper: Yuan et al., *"BLASST: Dynamic BLocked Attention Sparsity via Softmax
Thresholding"*, MLSys 2026 (arXiv:2512.12087). See `BLASST_BLOCK_SKIP.md` for
the block-skipping implementation this benchmarks, and
`BLASST_BLOCK_SKIP_NAN_FIX.md` for a correctness fix required for any of the
`block_skip_threshold > 1.0` rows below to produce finite output.

---

## Shape

Matches the paper artifact's own Hopper prefill configuration:

| Param | Value |
|---|---|
| Batch size | 1 |
| Head dim | 128 |
| Query heads | 64 |
| KV heads | 4 (GQA) |
| Causal | False |
| Seqlens | 16384, 65536 |
| Q/K/V | random (`torch.randn`, bf16) |

Random Q/K/V is used here specifically to match the paper artifact's own
methodology, unlike `op_tests/op_benchmarks/triton/bench_mha_blasst.py`
(which uses real captured Qwen3-8B activations, since block-skipping depends
on the attention score distribution and random tensors rarely produce whole
skippable blocks on their own). Random-tensor sparsity curves are
sharper/steeper than real-model ones, so absolute sparsity% here is not
directly comparable in kind to `bench_mha_blasst.py`'s per-layer numbers —
both are reported as measured, not adjusted.

---

## Run

```bash
AITER_TRITON_ONLY=1 python op_tests/op_benchmarks/triton/bench_mlsys26_hopper_prefill.py
```

`AITER_TRITON_ONLY=1` skips aiter's C++/HIP ops JIT build so `import aiter`
only loads the Triton kernels this benchmark needs. No dataset or model
download is required (synthetic Q/K/V), so this script runs standalone in
any environment with a working ROCm/Triton PyTorch install — no dedicated
Docker wrapper needed.

The full sweep (both seqlens x 16 threshold points, 5 warmup + 20 timed
iterations each) takes a few minutes; the 65536-seqlen half dominates the
runtime. Consider running it as a background process with output redirected
to a log file rather than waiting on it in the foreground.

---

## Example result (AMD Instinct MI300X)

```
=== seqlen = 16k ===
   threshold    time/ms  Speedup
----------------------------------
    0(dense)     22.241   1.000x
 1e-09(tiny)     27.614   0.805x
         0.5     26.481   0.840x
         0.6     27.685   0.803x
         0.7     28.379   0.784x
         0.8     27.753   0.801x
         0.9     26.447   0.841x
           1     25.037   0.888x
         1.1     23.831   0.933x
         1.3     22.086   1.007x
         1.7     20.706   1.074x
           2     20.008   1.112x
           4     18.835   1.181x
           6     17.699   1.257x
           8     16.252   1.368x
          10     15.588   1.427x
          12     15.202   1.463x

=== seqlen = 64k ===
   threshold    time/ms  Speedup
----------------------------------
    0(dense)    338.273   1.000x
 1e-09(tiny)    439.957   0.769x
         0.5    455.480   0.743x
         0.6    431.268   0.784x
         0.7    399.023   0.848x
         0.8    365.448   0.926x
         0.9    335.830   1.007x
           1    313.676   1.078x
         1.1    295.229   1.146x
         1.3    280.124   1.208x
         1.7    267.413   1.265x
           2    262.633   1.288x
           4    253.623   1.334x
           6    250.511   1.350x
           8    243.836   1.387x
          10    239.475   1.413x
          12    235.937   1.434x
```

Measured on an AMD Instinct MI300X. Speedup is monotonically increasing with
threshold, with zero non-finite output at any point in the sweep (the script
asserts this internally). Low thresholds are net-slower than dense at both
seqlens — expected, and consistent with `BLASST_BLOCK_SKIP.md`'s
"Performance note" (block-skipping's data-dependent branch defeats the
kernel's software pipelining, so you only come out ahead once enough blocks
skip to pay for the lost overlap).

Absolute numbers depend on the node/GPU-sharing conditions at run time; this
node is shared/multi-tenant, so re-runs should expect some timing variance,
especially at the tiny-threshold and low-sparsity points (the
shortest-duration kernel invocations in the sweep).

---

## Regression test for the underlying bugfix

```bash
AITER_TRITON_ONLY=1 pytest op_tests/triton_tests/attention/test_mha_blasst.py -k no_nan
```

See `BLASST_BLOCK_SKIP_NAN_FIX.md` for what this covers.
