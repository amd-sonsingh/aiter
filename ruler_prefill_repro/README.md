# ruler_prefill_repro

Scripts and input data to reproduce our BLASST prefill benchmark results on
top of the `aiter` `blasst-block-skip` branch
(https://github.com/amd-sonsingh/aiter/tree/blasst-block-skip), using
`op_tests/op_benchmarks/triton/bench_mha_blasst.py`.

This directory is meant to live as a subdirectory directly under the root of
an `aiter` checkout (e.g. `<aiter-root>/ruler_prefill_repro`).

## Prerequisites

- A checkout of `aiter` on the `blasst-block-skip` branch, with this
  directory placed directly under its root.
- Docker with ROCm/AMDGPU device access. The scripts use
  `rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0`
  (the version-pinned tag that `rocm/pytorch:latest` resolved to at the time
  these scripts were written), so the exact ROCm/PyTorch version stays fixed
  even if `latest` moves on. `transformers` is installed into the container
  at container start since the base image doesn't include it.
- Network access to Hugging Face Hub (the scripts create `hf_cache/` under
  this directory and mount it into the container; if empty, `Qwen/Qwen3-8B`
  is downloaded into it on first run).

## Contents

- `data/ruler_qa1_ctx32768.jsonl` — RULER `qa_1` validation set at 32768
  context length (50 examples). Used as `BLASST_INPUT_FILE` so the benchmark
  captures Q/K/V from a real long-context prompt instead of the script's
  synthetic default.
- `run_causal.sh` / `run_noncausal.sh` — run the benchmark with
  `BLASST_CAUSAL=1` / `BLASST_CAUSAL=0` respectively. Both mount
  `hf_cache/` (created automatically if missing) as the container's
  Hugging Face cache, so the model is only downloaded once across runs.
- `hf_cache/` — Hugging Face cache directory (created on first run, or you
  can point it at an existing cache yourself, e.g. via a symlink).
- `causal_qa_1_32k.log` / `noncausal_qa_1_32k.log` — our results from running
  these scripts.

## How to run

From inside this directory (`<aiter-root>/ruler_prefill_repro`):

```sh
./run_causal.sh
./run_noncausal.sh
```

The scripts assume the `aiter` checkout root is the parent directory of this
script. If this directory is placed elsewhere, pass the paths explicitly:

```sh
./run_causal.sh /path/to/ruler_prefill_repro /path/to/aiter
```

Each script mounts the resolved `aiter` root into the container, sets
`BLASST_INPUT_FILE` to the bundled RULER prompt, and runs
`bench_mha_blasst.py`, which measures dense (threshold=0) vs. BLASST
(threshold>0) prefill attention time using the same Triton kernel, for
layers 0, 7, 18, and 35 of `Qwen/Qwen3-8B`.

The scripts pass `--group-add` with the host's `video`/`render` group IDs
(resolved via `getent` at run time) for GPU device access, since these
group names/GIDs are not guaranteed to exist inside the container image.

## Results summary

Speedup (BLASST vs. dense, same kernel), RULER `qa_1`, seq_len=32417:

| Layer | Causal, threshold=0.3 | Non-causal, threshold=0.3 |
|-------|------------------------|----------------------------|
| 0     | 1.18x                  | 1.23x                      |
| 7     | 1.86x                  | 2.05x                      |
| 18    | 1.62x                  | 1.57x                      |
| 35    | 1.82x                  | 1.61x                      |

See the full logs (`causal_qa_1_32k.log`, `noncausal_qa_1_32k.log`) for all
threshold values (0.01, 0.05, 0.1, 0.3).

## Notes

- This benchmark measures prefill only (a single forward pass); it does not
  exercise decode.
- The RULER input file determines the sequence length and content; a
  different file or a different RULER task/context length can be substituted
  by replacing `data/ruler_qa1_ctx32768.jsonl` (or editing
  `BLASST_INPUT_FILE` in the scripts).
