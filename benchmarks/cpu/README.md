# CPU and FLOP measurements

This folder contains the code and raw results for the supplementary CPU
feature-transform measurements used in the paper revision.

The shared diagnostic uses synthetic float32 input with shape `[1, 3, 512]`.
It reports median latency after warm-up. The TSFM rows time the frozen encoder
and its aggregation step; ROCKET, Hydra, and QUANT time an already-fitted
feature transform. The values are therefore useful for context, but they are
not an end-to-end UEA or deployment ranking.

## Linux setup

~~~bash
cd benchmarks/cpu
bash scripts/setup_cpu.sh --with-models
~~~

Example CPU runs:

~~~bash
uv run python -m tslimits_bench.cli --model moment-small --device cpu --threads 1 --warmup 20 --repetitions 100 --output results/moment-small-cpu.json
uv run python scripts/benchmark_aeon_baseline.py --model hydra --train-cases 100 --channels 3 --length 512 --classes 2 --threads 1 --warmup 20 --repetitions 100 --output results/hydra-cpu.json
~~~

For the GPU environment on Linux with an NVIDIA driver:

~~~bash
bash scripts/setup_gpu_uv.sh
~~~

## Included results

- `results/multivariate_feature_transform_summary.csv` contains the shared
  C=3, T=512 summary.
- `results/*flops.json` contains partial registered matrix-multiply and
  convolution counts for neural paths.
- `MULTIVARIATE_DIAGNOSTIC.md` explains the measurement scope and why FLOPs
  are not comparable across all methods.

Do not reinterpret the existing Windows CPU measurements as Linux results.
Re-run the commands above on the target machine and report its hardware and
software details.