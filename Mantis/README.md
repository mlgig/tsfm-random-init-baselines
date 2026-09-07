# Mantis

Clone the official Mantis source once, next to the model folders:

```bash
git clone https://github.com/vfeofanov/mantis.git ../external/mantis
```

Then run the experiment through the Linux path wrapper:

```bash
cd Mantis
bash setup.sh
bash run.sh \
  --data_path /path/to/UEA \
  --model Mantis8M \
  --init random \
  --seed 0 \
  --device cuda \
  --output ../results/mantis_random_seed0.csv
```

Use `--init pretrained` for the released checkpoint path used by the script.
Set `MANTIS_REPO=/path/to/mantis` if the clone is elsewhere.

Official code: <https://github.com/vfeofanov/mantis>
