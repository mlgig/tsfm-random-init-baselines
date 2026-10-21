# UEA multivariate data

Download the aeon `.ts` version of the UEA multivariate archive:

```bash
bash data/download_uea.sh
export UEA_ROOT="$PWD/data/UEA"
```

You can also download the archive directly:

- [Multivariate2018_ts.zip](https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip)

The archive is about 846 MiB. The downloader can continue an interrupted
download, checks the ZIP before extraction, and verifies that it extracted 30
matching `TRAIN` / `TEST` pairs with the paper dataset names. It will
not overwrite a non-empty incomplete destination.

## Expected layout

The experiment scripts expect:

```text
$UEA_ROOT/<Dataset>/<Dataset>_TRAIN.ts
$UEA_ROOT/<Dataset>/<Dataset>_TEST.ts
```

The archive is ignored by Git. See the [UEA dataset page](https://www.timeseriesclassification.com/dataset.php)
for source information.

The public archive contains datasets with different series lengths. Some
runners need fixed-size arrays, so they add zeros at the end when needed. See
the main README for the difference between this public-data path and the old
local `Multivariate_ts_length_fixed` directory.

UniTS also needs a task YAML. Create the public-archive recipe with:

```bash
UniTS/.venv/bin/python UniTS/generate_uea_yaml.py \
  --data-root "$UEA_ROOT" \
  --output results/UniTS/uea_tasks.yaml
```

By default, the recipe contains the 27 UniTS-eligible paper datasets.
