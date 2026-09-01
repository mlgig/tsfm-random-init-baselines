# Pretraining vs. Random Initialization in Time-Series Foundation Models

Code for the experiments in our paper. Each method has one folder and one
clear way to run it on the UEA multivariate archive.

```text
MOMENT/                       pretrained and random MOMENT runs
Mantis/                       pretrained and random Mantis runs
UniTS/                        pretrained and random UniTS runs
NuTime/                       pretrained and random NuTime runs
time-series-specific-models/  aeon ROCKET, Hydra, and QUANT runs
data/                         UEA download command and data notes
results/                      ignored outputs from local runs
```

## Data

Download the UEA multivariate `.ts` archive once:

```bash
bash data/download_uea.sh
export UEA_ROOT="$PWD/data/UEA"
```

The command uses the official
[Multivariate2018 `.ts` archive](https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip).
The [UEA dataset page](https://www.timeseriesclassification.com/dataset.php)
is the archive landing page.

## Run a model

Every model folder has `setup.sh`, `run.sh`, its exact experiment code, and a
short README. For example:

```bash
cd MOMENT
bash setup.sh
bash run.sh --data_path "$UEA_ROOT" --model MOMENT-1-base \
  --init random --seed 0 --device cuda \
  --output ../results/moment_random_seed0.csv
```

For the three time-series-specific reference methods:

```bash
cd time-series-specific-models
bash setup.sh
bash run_rocket.sh --data-root "$UEA_ROOT" --seed 42 --n-jobs 1 \
  --output-dir ../results/rocket_seed42
```

See the README in the relevant folder for Mantis, UniTS, NuTime, Hydra, and
QUANT commands.

## Random initialization

The experiment code intentionally keeps the model-specific construction
paths rather than pretending all four use an identical reset:

| Family | Random condition |
| --- | --- |
| Mantis | Fresh seeded model; no pretrained weights loaded. |
| MOMENT | Loads the model path, then applies the script's module reset. |
| UniTS | Fresh model, then applies the script's module reset. |
| NuTime | Fresh model, then applies the notebook's module reset. |

The separate `benchmarks/cpu/` directory contains the already-reported
synthetic CPU/FLOP diagnostic. It is not the UEA accuracy runner and does not
represent end-to-end deployment latency.