# Time-series-specific models

This folder runs the three non-pretrained reference methods in the paper with
[aeon](https://www.aeon-toolkit.org/): ROCKET, Hydra, and QUANT. Each script
fits a classifier on a UEA training split and evaluates it on the matching test
split.

For equal-length data, aeon receives its normal multivariate array format:
`[cases, channels, timepoints]`. Raw UEA datasets with unequal series lengths
are right-zero-padded to one shared length. These aeon classifiers cannot read
ragged datasets. The scripts do not resample the data, pool channels, or add
external normalization.

## Setup and run

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

Without `--datasets`, a script runs every valid `*_TRAIN.ts` / `*_TEST.ts`
pair below the data root. To run a paper subset, give dataset names directly
or store one dataset name per line in a text file:

```bash
bash run_rocket.sh --data-root /path/to/UEA \
  --datasets ArticularyWordRecognition BasicMotions FingerMovements \
  --output-dir ../results/rocket_subset

bash run_hydra.sh --data-root /path/to/UEA \
  --datasets-file ../data/paper_common_24_datasets.txt \
  --output-dir ../results/hydra_subset
```

Each output directory contains `results.csv` and `failures.csv`. A failed
dataset is written to `failures.csv` and makes the script exit with a non-zero
status. No dataset is silently skipped.

The `preprocessing` column in `results.csv` shows whether unequal-length
padding was used. This public-data path may not give the same numbers as the
old local `Multivariate_ts_length_fixed` dataset. Keep raw output files before
you compare new runs with the paper baselines.
