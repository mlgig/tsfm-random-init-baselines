# Mantis

This folder uses the public `mantis-tsfm==1.1.0` package. You do not need a
separate Mantis checkout or a `MANTIS_REPO` setting.

## Setup and run

```bash
cd Mantis
bash setup.sh
bash run.sh \
  --data_path /path/to/UEA \
  --model Mantis8M \
  --init random \
  --seed 0 \
  --output ../results/Mantis/random_seed0.csv
```

The script uses CUDA when available, otherwise it uses CPU. Use `--device
cuda` or `--device cpu` to choose a device yourself.

Use `--model MantisV2` for MantisV2. Use `--init pretrained` for the released
checkpoint. The script uses a fixed Hugging Face version by default; use
`--revision` only when you intentionally need another release. The random
setting seeds the official constructor and never loads checkpoint weights.

## Data handling

For unequal-length datasets, the script first right-zero-pads the train and
test splits to one dataset-level maximum length. It then uses the existing
linear interpolation step to reach `--target_len` (default: `512`). Random and
pretrained runs use the same steps.

Run the random setting with all paper seeds: `0`, `7`, `42`, `123`, and
`2026`.

## Source code

- [Official Mantis repository](https://github.com/vfeofanov/mantis)
