# MOMENT

This folder runs the MOMENT experiment script. It accepts the same
arguments as `moment_random.py`.

```bash
cd MOMENT
bash setup.sh
bash run.sh \
  --data_path /path/to/UEA \
  --model MOMENT-1-base \
  --init random \
  --seed 0 \
  --device cuda \
  --output ../results/moment_random_seed0.csv
```

For pretrained results, use `--init pretrained`. The script obtains MOMENT
weights through `momentfm`; set `HF_HOME` before running if you want a shared
Hugging Face cache.

Official code: <https://github.com/moment-timeseries-foundation-model/moment>
