# UniTS

UniTS needs a local checkout of the official source code because its model
classes are not included in this repository. From the repository root, create
the checkout used by the wrapper:

```bash
mkdir -p external
git clone https://github.com/mims-harvard/UniTS.git external/units
git -C external/units checkout f8e5dfc71f6799164f30b090240a74ee70da01ca
```

The wrapper and launcher check this exact version and stop if tracked source
files changed. The paper does not name an exact UEA task YAML or checkpoint
SHA. Save the generated YAML and checkpoint identifier with any regenerated
results.

## Setup and run

```bash
cd UniTS
bash setup.sh
.venv/bin/python generate_uea_yaml.py \
  --data-root "$UEA_ROOT" \
  --output ../results/UniTS/uea_tasks.yaml
bash run.sh \
  --yaml ../results/UniTS/uea_tasks.yaml \
  --init random \
  --seed 0 \
  --out ../results/UniTS/random_seed0.csv
```

`generate_uea_yaml.py` reads the official raw TRAIN and TEST files. It records
the channel count, shared maximum series length, and number of training
classes for each collection. By default, it creates a recipe for the 27
collections used for UniTS. It leaves out DuckDuckGeese, InsectWingbeat, and
PEMS-SF. Use `--include-paper-unavailable` to generate a recipe with all 30
collections.

For a pretrained run, add `--init pretrained` and
`--ckpt /absolute/path/to/checkpoint.pth`. The wrapper resolves `--yaml`,
`--ckpt`, and `--out` before it changes into the external checkout. Therefore,
relative paths are read from the directory where you run `run.sh`. Set
`UNITS_REPO=/another/checkout` to use another checkout at the same fixed
version.

The setup script installs the dependencies declared by the official source.
For a random run, the wrapper seeds and builds a new official model without
loading a checkpoint. It evaluates the four shared probes in
[`../common/downstream.py`](../common/downstream.py). Run the random setting
with seeds `0`, `7`, `42`, `123`, and `2026`.

## Source code

- [Official UniTS repository](https://github.com/mims-harvard/UniTS)
