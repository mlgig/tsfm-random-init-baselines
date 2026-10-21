# Multivariate CPU and FLOP diagnostic

This controlled check uses synthetic float32 input with shape
`[B,C,T]=[1,3,512]`, one CPU thread, and the same Intel Core Ultra 7 265H
Windows host. It is an appendix diagnostic, not the paper's final resource
table. The final table must use real UEA test cases after each model's exact
experiment preprocessing, representation aggregation, and fitted-probe
prediction.

The closest comparison with matching scope is
[`results/multivariate_feature_transform_summary.csv`](results/multivariate_feature_transform_summary.csv).
TSFMs use encoder and aggregation forward passes only. ROCKET, Hydra, and
QUANT use their fitted feature-transform and scaling path only. This is closer
than comparing TSFM encoders with a full aeon `predict` call, but it still
excludes TSFM probe prediction. It is not an end-to-end ranking.

| Method | p50 ms | p95 ms | Peak RSS MiB | Registered FMA/conv GFLOPs |
|---|---:|---:|---:|---:|
| MantisV2 | 15.88 | 18.84 | 294.98 | 0.4821 |
| Mantis-8M | 18.61 | 21.35 | 321.11 | 0.8738 |
| MOMENT-small | 82.48 | 92.00 | 398.50 | 7.4003 |
| MOMENT-base | 354.91 | 389.43 | 697.28 | 33.0703 |
| MOMENT-large | 1354.18 | 1380.24 | 1592.32 | 119.5911 |
| Hydra feature transform | 14.26 | 16.66 | 544.24 | — |
| QUANT feature transform | 28.99 | 46.35 | 322.87 | — |
| ROCKET feature transform | 46.83 | 51.41 | 344.06 | — |

MOMENT-large has only 5 timed repetitions after 2 warm-ups because it is slow
on this CPU. This is a short diagnostic. All other rows in the common table
use 100 timed repetitions and 20 warm-ups, except MOMENT-base (50/10).

This check does not support the broad claim that all three aeon methods are
always faster. In this setup, Hydra is comparable to the two Mantis paths.
ROCKET and QUANT are slower than Mantis but faster than MOMENT-small, -base,
and -large. This is not an error: ROCKET uses 10,000 kernels and QUANT does
substantial quantile and rank work.

## FLOP interpretation

`registered FMA/conv GFLOPs` comes from
`torch.utils.flop_counter.FlopCounterMode` on the loaded evaluation forward
path. The built-in MOMENT/Mantis counter runs under `torch.no_grad()`. The
external UniTS path temporarily enables autograd only to avoid a PyTorch
module-hook bug; this does not change the executed arithmetic. The counter
uses two FLOPs per multiply-accumulate and counts registered `mm`, `addmm`,
`bmm`, and convolution operations. It does not count normalisation, softmax,
activation, interpolation, indexing, allocation, or non-Torch code, among
other operations. Call this column **registered FMA/conv FLOPs**, not total
FLOPs.

Do not publish one FLOP ranking across model families. ROCKET uses
Numba/NumPy, QUANT includes FFT and quantile operations without a common FLOP
formula, and the Torch counter reports zero for much of that work. As separate
and non-comparable checks, ROCKET has a 0.13485-GFLOP logical
kernel-arithmetic lower bound and Hydra has 0.02828 GFLOP of registered
convolution work. Neither value includes all transform or prediction work.
CPU p50/p95 latency, fit time, and peak RSS are the defensible measures across
families.

## NuTime and UniTS

Their native released-path checks are in
[`results/native_path_diagnostics.csv`](results/native_path_diagnostics.csv).
NuTime is valid only for the univariate `[1,1,176]` release path because the
public `C>1` path adds non-pretrained channel embeddings. UniTS's released
x128 artifact supports an authentic `[1,315,3]` UWave source-task core, but
it is source-prompt-specific and not a general target-UEA pipeline. Keep both
separate from the common `[1,3,512]` table.

## What belongs in the paper

Use the common table only as a controlled appendix scaling test. For the main
paper, run every method on the 24 common UEA datasets with native multivariate
`(C,T)` inputs. Include preprocessing, target aggregation, frozen
representation extraction, and prediction by the selected fitted probe.
Report per-dataset values and their median/IQR. Report fit time and peak RSS
separately. This supports the paper's MTSC claim without forcing an artificial
common channel adapter.
