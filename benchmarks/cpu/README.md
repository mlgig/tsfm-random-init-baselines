# CPU and FLOP measurements

This folder contains the code and raw results for the CPU feature-transform
checks used in the paper revision.

The common diagnostic uses synthetic float32 input with shape `[1, 3, 512]`
and reports median latency after warm-up. For TSFMs, it measures the frozen
encoder and representation aggregation. For ROCKET, Hydra, and QUANT, it
measures an already-fitted feature transform. These results give useful
context, but they are not an end-to-end UEA or deployment benchmark.

## Linux setup

~~~bash
cd benchmarks/cpu
bash scripts/setup_cpu.sh --with-models
~~~

Example CPU commands:

~~~bash
uv run python -m tslimits_bench.cli --model moment-small --device cpu --threads 1 --warmup 20 --repetitions 100 --output results/moment-small-cpu.json
uv run python scripts/benchmark_aeon_baseline.py --model hydra --train-cases 100 --channels 3 --length 512 --classes 2 --threads 1 --warmup 20 --repetitions 100 --output results/hydra-cpu.json
~~~

For the GPU environment on Linux with an NVIDIA driver:

~~~bash
bash scripts/setup_gpu_uv.sh
~~~

## Included files

- `results/multivariate_feature_transform_summary.csv` contains the common
  `C=3`, `T=512` summary.
- `results/*flops.json` contains partial registered matrix-multiply and
  convolution counts for neural models.
- `MULTIVARIATE_DIAGNOSTIC.md` explains the measurement scope and why FLOPs
  cannot be compared across every method.

Do not report the existing Windows measurements as Linux results. Run the
commands above on the target machine and record its hardware and software.
