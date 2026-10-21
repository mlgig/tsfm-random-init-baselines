# NuTime

NuTime needs a local checkout of the official source code. From the repository
root, create it at the expected location:

```bash
mkdir -p external
git clone https://github.com/chenguolin/NuTime.git external/nutime
git -C external/nutime checkout 9acdd3d0e82df0914217ef752fa3b95ec82cd960
```

The runner checks this exact version and stops if tracked source files changed.
The paper does not give a checkpoint hash for the UEA accuracy run, so save the
exact artifact used with any regenerated results.

## Setup and run

```bash
cd NuTime
bash setup.sh
bash run.sh \
  --data-root /path/to/UEA \
  --init random \
  --seed 0 \
  --output ../results/NuTime/random_seed0.ipynb \
  --results ../results/NuTime/random_seed0.csv
```

For a pretrained run, add `--init pretrained` and
`--checkpoint /absolute/path/to/checkpoint_bias9.pth`. Random runs do not need
a checkpoint. Set `NUTIME_REPO=/another/checkout` to use another checkout at
the same fixed version.

`--output` saves the executed notebook. `--results` saves a full-precision CSV
with one row per dataset. Before running the notebook, the configurator adds
absolute paths for the data, result, and source checkout.

## Feature extraction

The notebook uses the released `WindowNormEncoder + WinT` model path. It
linearly resizes every univariate channel to 176 points, then removes the BYOL
classifier and projector to extract 128-D CLS features. It does not
z-normalize samples before `WindowNormEncoder`, because that encoder stores
the window mean and standard deviation. It concatenates channel features in
the same way for every dataset and for both initialization settings. The
shared four-probe evaluation then uses these features.

Run the random setting with all paper seeds: `0`, `7`, `42`, `123`, and
`2026`.

## Source code

- [Official NuTime repository](https://github.com/chenguolin/NuTime)
