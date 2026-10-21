# Random Initialization as a Baseline for Time Series Foundation Models

This repository contains the code for *Random Initialization as a Baseline for
Time Series Foundation Models*, accepted at the NeurIPS 2026 TS-LIMITS
workshop.

## Abstract

Time series foundation models are often used with pre-trained weights. This
work asks a simple question: how much do those weights help compared with the
same model built from random weights? We test MOMENT, Mantis, UniTS, and
NuTime on multivariate time series classification datasets. We use frozen
model features and the same downstream classifiers for both settings. We also
compare the models with ROCKET, Hydra, and QUANT.

## Resources

- Paper link: coming soon.
- [UEA Multivariate Time Series Classification Archive](https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip)
- [Data notes](data/README.md)

## Repository Structure

```text
MOMENT/                       MOMENT Small, Base, and Large
Mantis/                       Mantis 8M and MantisV2
UniTS/                        UniTS runner and source wrapper
NuTime/                       NuTime notebook runner
time-series-specific-models/  ROCKET, Hydra, and QUANT
data/                         UEA download script and dataset lists
common/                       Shared data and classifier code
scripts/                      Setup, validation, and result tools
benchmarks/cpu/               Separate CPU and FLOP checks
```

## Methods

### Foundation Models

- **MOMENT**: Small, Base, and Large models.
- **Mantis**: Mantis 8M and MantisV2.
- **UniTS**: runs from a fixed version of the official source code.
- **NuTime**: runs from a fixed version of the official source code.

Each model has a pre-trained run and a random-weight run. Random runs use the
five seeds from the paper: `0`, `7`, `42`, `123`, and `2026`.

### Time Series Methods

- **ROCKET**
- **Hydra**
- **QUANT**

These methods run through the aeon library.

## Data

Download the official UEA archive once:

```bash
bash data/download_uea.sh
export UEA_ROOT="$PWD/data/UEA"
```

The archive is about 846 MiB and contains 30 datasets. The expected layout is:

```text
$UEA_ROOT/<Dataset>/<Dataset>_TRAIN.ts
$UEA_ROOT/<Dataset>/<Dataset>_TEST.ts
```

Some datasets have time series with different lengths. The runners add zeros
at the end only when a model needs a fixed input length. They use the same
data path for pre-trained and random runs.

The original experiment also used a local processed-data folder that is not in
this repository. Keep new result files if you want to compare them with the
published tables.

## Usage

Each method has its own README with setup and run commands. The usual steps
are:

1. Download the UEA data.
2. Run `setup.sh` in the method folder.
3. Run the method for each random seed and once with pre-trained weights.

For example, to run MOMENT Base with random weights:

```bash
cd MOMENT
bash setup.sh
bash run.sh --data_path "$UEA_ROOT" --model MOMENT-1-base \
  --init random --seed 0 --device cuda \
  --output ../results/MOMENT-Base/random_seed0.csv
```

The [MOMENT](MOMENT/README.md), [Mantis](Mantis/README.md),
[UniTS](UniTS/README.md), [NuTime](NuTime/README.md), and
[time-series methods](time-series-specific-models/README.md) READMEs have
more details.

## Results

The original per-seed UEA result files are not stored in this repository. Once
you have new run outputs, use the result tool below to build paper-style
summary tables:

```bash
python scripts/aggregate_paper_results.py \
  --input-root /path/to/paper-input \
  --output-dir results/paper-summary
```

See [the result template](templates/paper-results/README.md) for the expected
file layout.

## References and Original Repositories

- [UEA archive](https://www.timeseriesclassification.com/)
- [MOMENT](https://github.com/moment-timeseries-foundation-model/moment)
- [Mantis](https://github.com/vfeofanov/mantis)
- [UniTS](https://github.com/mims-harvard/UniTS)
- [NuTime](https://github.com/chenguolin/NuTime)
- [aeon](https://github.com/aeon-toolkit/aeon)

## Citation

Citation details will be added when the paper is publicly available.

## License

Please check the licenses of the original model repositories before using
their code or weights.
