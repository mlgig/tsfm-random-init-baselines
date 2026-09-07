# CPU microbenchmark results

> The original rows below include several single-channel paths. For the corrected common multivariate `[1,3,512]` diagnostic, scope-aligned feature-transform comparison, and registered-FLOP caveats, use [`MULTIVARIATE_DIAGNOSTIC.md`](MULTIVARIATE_DIAGNOSTIC.md). Do not use the two tables below as a cross-family speed ranking.

These measurements were run on 2026-08-25 on an Intel Core Ultra 7 265H CPU, Windows 10 (10.0.26100-SP0), Python 3.11.16, PyTorch 2.5.1+cpu, float32, and one PyTorch CPU thread. Each number uses a synthetic standard-normal input of the listed shape. The reported latency excludes model loading, network download, disk I/O, preprocessing, feature aggregation beyond the adapter, probe fitting, and classifier prediction. There were 20 warm-up calls and 100 timed calls. Peak RSS is process-wide and is therefore a useful upper bound rather than model-only memory.

| encoder | input shape | p50 (ms) | p95 (ms) | mean (ms) | throughput (samples/s) | peak RSS (MiB) |
|---|---|---:|---:|---:|---:|---:|
| NuTime (legacy BYOL) | [1, 1, 176] | 2.37 | 5.52 | 2.81 | 356.26 | 487.18 |
| Mantis-8M | [1, 1, 512] | 6.38 | 9.20 | 6.69 | 149.56 | 320.52 |
| UniTS x128 core (source UWaveGestureLibrary) | [1, 315, 3] | 36.40 | 43.24 | 36.62 | 27.31 | 409.87 |
| MantisV2 | [1, 1, 512] | 4.86 | 8.64 | 5.46 | 183.23 | 294.73 |
| MOMENT-small | [1, 1, 512] | 34.02 | 40.08 | 34.31 | 29.15 | 392.30 |
| MOMENT-base | [1, 1, 512] | 144.29 | 160.69 | 144.88 | 6.90 | 683.67 |
| MOMENT-large | [1, 1, 512] | 507.09 | 574.90 | 513.82 | 1.95 | 1577.13 |

## Appropriate use
## aeon classifier CPU baseline

ROCKET, Hydra, and QUANT were measured under the same machine, float32, one-CPU-thread, 20-warm-up/100-repetition protocol on synthetic train data `[100, 1, 512]` with two balanced classes and an independent already-fitted prediction input `[1, 1, 512]`. Prediction includes each classifier's feature transformation and final estimator prediction. Fit is separate. ROCKET's one-time Numba compilation was warmed up outside its fit time. These use `aeon==1.3.0` and `scikit-learn==1.7.2`.

| classifier | fit time (s) | fit peak RSS (MiB) | fitted B=1 p50 (ms) | p95 (ms) | mean (ms) | throughput (samples/s) | predict peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ROCKET | 1.554 | 391.94 | 16.40 | 23.25 | 17.44 | 57.34 | 342.52 |
| Hydra | 1.477 | 587.17 | 11.60 | 14.80 | 12.02 | 83.19 | 550.84 |
| QUANT | 0.772 | 322.01 | 27.61 | 32.58 | 28.58 | 34.99 | 322.09 |


These are reproducible **synthetic CPU diagnostics**, not final task-level resource results. Using CPU for all methods is the right hardware comparison, but the top table measures TSFM encoder-only forward passes while the aeon table measures the entire already-fitted classifier `predict` path. They are therefore not yet a head-to-head latency ranking: add the actual frozen-probe prediction step to the TSFM paths, or separately label the two scopes. Input lengths/channel counts also differ; NuTime was benchmarked only through its official univariate release path; and UniTS uses its released UWaveGestureLibrary source prompt/CLS path before target pooling. Do not present these as end-to-end UEA latency, memory, energy, or a latency--accuracy Pareto. For a submission-ready table, rerun the same harness on saved post-preprocessing UEA test examples under the exact experiment configuration, then report per-dataset medians/ranges, probe fit/prediction resources, and exact checkpoint/code revisions.

The individual machine-readable results are in `results/*-cpu-*.json` and contain the full protocol metadata.
