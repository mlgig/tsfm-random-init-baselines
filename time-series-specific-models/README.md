# Time-series-specific models

This folder runs the three non-pretrained reference methods used in the paper
with [aeon](https://www.aeon-toolkit.org/): ROCKET, Hydra, and QUANT.
Each run fits the classifier on a dataset's UEA training split and evaluates
it on its test split. The data are passed to aeon in their native multivariate
`[cases, channels, timepoints]` form; this runner does not resample, pool
channels, or add external normalisation.

```bash
cd time-series-specific-models
bash setup.sh

bash run_rocket.sh --data-root /path/to/UEA --seed 42 --n-jobs 1 \
  --output-dir ../results/rocket_seed42
bash run_hydra.sh --data-root /path/to/UEA --seed 42 --n-jobs 1 \
  --output-dir ../results/hydra_seed42
bash run_quant.sh --data-root /path/to/UEA --seed 42 --n-jobs 1 \
  --output-dir ../results/quant_seed42
```

Without `--datasets`, every valid `*_TRAIN.ts` / `*_TEST.ts` pair below the
data root is run. To reproduce a paper subset, give its names directly or put
one name per line in a text file:

```bash
bash run_rocket.sh --data-root /path/to/UEA \
  --datasets ArticularyWordRecognition BasicMotions FingerMovements \
  --output-dir ../results/rocket_subset

bash run_hydra.sh --data-root /path/to/UEA \
  --datasets-file ../data/paper_datasets.txt \
  --output-dir ../results/hydra_subset
```

Each output directory contains `results.csv` and `failures.csv`. A failed
dataset is written explicitly and causes a non-zero exit code; no dataset is
silently dropped.
