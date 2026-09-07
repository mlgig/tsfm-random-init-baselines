# UniTS

Clone the official source once:

```bash
git clone https://github.com/mims-harvard/UniTS.git ../external/units
```

The UniTS run needs the UEA YAML file used in the experiment and, for
pretrained runs, the exact checkpoint file.

```bash
cd UniTS
bash setup.sh
bash run.sh \
  --yaml /path/to/uea_zeroshot.yaml \
  --init random \
  --seed 0 \
  --out ../results/units_random_seed0.csv
```

For pretrained results, add `--ckpt /path/to/checkpoint.pth` and use
`--init pretrained`. Set `UNITS_REPO` if the clone is elsewhere.

Official code: <https://github.com/mims-harvard/UniTS>
