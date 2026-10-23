# Random Initialization as a Baseline for Time Series Foundation Models

This repository contains the code for *[Random Initialization as a Baseline for Time Series Foundation Models](https://arxiv.org/abs/2610.30950)*, accepted at the NeurIPS 2026 TS-LIMITS workshop.



## Abstract

Time series foundation models rely on self-supervised objectives, such as
masked reconstruction or contrastive learning, to build reusable
representations for downstream tasks such as multivariate time series
classification (MTSC). In this work, we test how much pretrained weights
improve over a baseline that requires no pretraining: the same architecture
with randomly initialized weights. We evaluate seven publicly released
instances from four time series foundation model families on the UEA MTSC
archive[^1], with per-model exclusions where evaluation cannot be completed due
to computational constraints. For each instance, we compare the pretrained
weights against five randomly initialized variants using four downstream
classifiers on frozen representations: two linear classifiers (ridge classifier
and logistic regression) and two nonlinear classifiers (RBF SVM and random
forest), yielding 183 paired dataset-level comparisons. Pretrained variants
achieve higher mean accuracy across all seven instances, but the mean advantage
is modest, ranging from 1.43 to 5.19 percentage points, with an overall average
gain of 2.58 percentage points. Under per-instance Wilcoxon signed-rank tests,
three of seven raw p-values fall below 0.05, but after Holm correction for
multiple comparisons, only NuTime remains statistically significant. A
secondary pooled analysis across all 183 pairs yields p < 0.001, while the
corrected per-instance tests remain the primary statistical evidence. We
further compare against three established time-series-specific classifiers:
ROCKET, Hydra, and Quant, which remain competitive with the pretrained
foundation models without requiring large-scale pretraining. Together, these
results motivate a more careful assessment of the computational cost of
large-scale pretraining relative to its downstream benefit for MTSC and support
the systematic inclusion of architecture-matched random initialization and
non-pretrained baselines in future evaluations.

[^1]: [UEA Multivariate Time Series Classification Archive](https://www.timeseriesclassification.com/)

## Data

Download the UEA multivariate archive once:

```bash
bash data/download_uea.sh
export UEA_ROOT="$PWD/data/UEA"
```

The archive contains 30 datasets. Its expected layout is:

```text
$UEA_ROOT/<Dataset>/<Dataset>_TRAIN.ts
$UEA_ROOT/<Dataset>/<Dataset>_TEST.ts
```

## Run experiments

Each model folder has setup and run instructions:

- [MOMENT](MOMENT/README.md)
- [Mantis](Mantis/README.md)
- [UniTS](UniTS/README.md)
- [NuTime](NuTime/README.md)
- [ROCKET, Hydra, and QUANT](time-series-specific-models/README.md)

Run every random setting with seeds `0`, `7`, `42`, `123`, and `2026`, then run
the matching pretrained setting. Each runner writes its output to the path
given on the command line.

## Original repositories

- [MOMENT](https://github.com/moment-timeseries-foundation-model/moment)
- [Mantis](https://github.com/vfeofanov/mantis)
- [UniTS](https://github.com/mims-harvard/UniTS)
- [NuTime](https://github.com/chenguolin/NuTime)
- [aeon](https://github.com/aeon-toolkit/aeon)

## Citation


```bibtex
@misc{pinar2026randominit,
  title={Random Initialization as a Baseline for Time Series Foundation Models},
  author={Pinar Sungu Isiacik and Georgiana Ifrim},
  year={2026},
  conference={{NeurIPS} 2026 Workshop on {TS-LIMITS}},
  eprint={2610.30950},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2610.30950}
}
