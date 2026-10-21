# MOMENT

This folder evaluates frozen features from MOMENT Small, Base, and Large on
the UEA multivariate archive. You need Python 3.11 and access to Hugging Face
for the selected model version.

## Setup and run

```bash
cd MOMENT
bash setup.sh
bash run.sh \
  --data_path /path/to/UEA \
  --model MOMENT-1-base \
  --init random \
  --seed 0 \
  --device cuda \
  --output ../results/MOMENT-Base/random_seed0.csv
```

Use `--init pretrained` to evaluate the released checkpoint. The script uses
fixed versions for all three model sizes. Only use `--revision` when you
want to reproduce a different release. Set `HF_HOME` if you want to use a
shared Hugging Face cache.

## Random initialization

For a random run, the script downloads only the model configuration from the
fixed release. It sets the PyTorch seed and builds a new `MOMENTPipeline`.
It does not load saved model weights. The model configuration
must still be available in the cache or downloadable from Hugging Face.

UEA datasets with unequal series lengths are right-zero-padded or truncated
to the MOMENT context length. The default `--target_len` is `512`. This is the
same for random and pretrained runs.

Run the random setting with all paper seeds: `0`, `7`, `42`, `123`, and
`2026`.

## Source code

- [Official MOMENT repository](https://github.com/moment-timeseries-foundation-model/moment)
