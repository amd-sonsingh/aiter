# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Speedup benchmark for BLASST block-skipping in the Triton MHA kernel.

Block skipping depends on the attention *score distribution*, so random tensors
are NOT representative (they rarely produce whole skippable blocks). This
benchmark captures REAL Q/K/V from a long-context Qwen3-8B forward pass, then
times dense (block_skip_threshold=0) vs BLASST (threshold>0) through the SAME
flash_attn_func kernel — an apples-to-apples comparison.

See BLASST_BLOCK_SKIP.md. Note on results: on this pipelined kernel the skip
branch defeats software pipelining, so expect net gains only at high sparsity
(and typically near break-even) — this benchmark quantifies that.

Env:
  BLASST_CAUSAL=1        causal attention (default 0 = non-causal)
  BLASST_MODEL=...       HF model id (default Qwen/Qwen3-8B)
  BLASST_INPUT_FILE=...  jsonl file with an "input" field (e.g. a RULER sample)
                         used as the long-context prompt. If unset, a long
                         synthetic prompt is generated (less representative).
  AITER_TRITON_ONLY=1    recommended so `import aiter` skips the C++ ops build.

Run:
  AITER_TRITON_ONLY=1 python op_tests/op_benchmarks/triton/bench_mha_blasst.py
"""

import json
import os
import time

import torch

from aiter.ops.triton.attention.mha import flash_attn_func

CAUSAL = os.environ.get("BLASST_CAUSAL", "0") == "1"
MODEL = os.environ.get("BLASST_MODEL", "Qwen/Qwen3-8B")


def repeat_kv(x, n_rep):
    """Expand GQA KV heads to match query heads. x: (B, H_kv, S, D)."""
    if n_rep == 1:
        return x
    B, H, S, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, S, D).reshape(B, H * n_rep, S, D)


def blasst_attn(Q, K, V, threshold):
    """flash_attn_func on (B, H, S, D) tensors. threshold=0 -> dense."""
    q, k, v = (x.transpose(1, 2).contiguous() for x in (Q, K, V))  # -> (B, S, H, D)
    o = flash_attn_func(q, k, v, causal=CAUSAL, block_skip_threshold=float(threshold))
    return o.transpose(1, 2).contiguous()


def benchmark_fn(fn, warmup=5, repeat=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / repeat * 1000.0


def get_input_text():
    """Long-context prompt: from BLASST_INPUT_FILE (RULER jsonl) or synthetic."""
    f = os.environ.get("BLASST_INPUT_FILE")
    if f and os.path.exists(f):
        with open(f) as fh:
            return json.loads(fh.readline())["input"]
    print("BLASST_INPUT_FILE not set/found; using synthetic prompt "
          "(less representative — prefer a real long-context sample).")
    para = "The quick brown fox jumps over the lazy dog. " * 40 + "\n"
    return para * 300


def capture_model_qkv(model_name, text, device="cuda"):
    """Run one forward pass and capture Q/K/V (B, H, S, D) per layer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    impl = model.config._attn_implementation
    original_fn = ALL_ATTENTION_FUNCTIONS[impl]
    captured = {}

    def hook(module, query, key, value, attention_mask, *args, **kwargs):
        captured[module.layer_idx] = {
            "Q": query.detach(), "K": key.detach(), "V": value.detach(),
            "num_kv_groups": getattr(module, "num_key_value_groups", 1),
        }
        return original_fn(module, query, key, value, attention_mask, *args, **kwargs)

    ALL_ATTENTION_FUNCTIONS[impl] = hook
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        model(ids)
    ALL_ATTENTION_FUNCTIONS[impl] = original_fn

    del model
    torch.cuda.empty_cache()
    return captured


def main():
    if not torch.cuda.is_available():
        print("ERROR: GPU required.")
        return

    print("=" * 66)
    print(f"BLASST speedup — {MODEL}, backend: aiter, causal: {CAUSAL}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("=" * 66)
    print("Same kernel: threshold=0 (dense) vs threshold>0 (BLASST)\n")

    text = get_input_text()
    print(f"Loading {MODEL} and capturing Q/K/V...")
    captured = capture_model_qkv(MODEL, text)
    seq_len = captured[0]["Q"].shape[2]
    print(f"Captured {len(captured)} layers, seq_len={seq_len}\n")

    layers = [li for li in (0, 7, 18, 35) if li in captured]
    thresholds = [0.0, 0.01, 0.05, 0.1, 0.3]

    print(f"{'Layer':>6} {'Threshold':>10} {'Dense(ms)':>10} {'BLASST(ms)':>11} {'Speedup':>8}")
    print("-" * 50)
    for li in layers:
        d = captured[li]
        Q = d["Q"]
        K = repeat_kv(d["K"], d["num_kv_groups"])
        V = repeat_kv(d["V"], d["num_kv_groups"])
        dense_ms = benchmark_fn(lambda: blasst_attn(Q, K, V, 0.0))
        for t in thresholds:
            if t == 0.0:
                print(f"{li:>6} {'0 (dense)':>10} {dense_ms:>10.2f} {dense_ms:>11.2f} {'1.00x':>8}")
            else:
                ms = benchmark_fn(lambda t=t: blasst_attn(Q, K, V, t))
                print(f"{li:>6} {t:>10.2f} {dense_ms:>10.2f} {ms:>11.2f} {dense_ms/ms:>7.2f}x")
        print()


if __name__ == "__main__":
    main()
