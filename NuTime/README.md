# NuTime

Clone the NuTime repository used by this experiment:

```bash
git clone https://github.com/chenguolin/NuTime.git ../external/nutime
cd ../external/nutime
git checkout 9acdd3d0e82df0914217ef752fa3b95ec82cd960
```

Then configure and execute the notebook from this folder:

```bash
cd NuTime
bash setup.sh
bash run.sh \
  --data-root /path/to/UEA \
  --checkpoint ../external/nutime/ckpt/checkpoint_bias9.pth \
  --init random \
  --seed 0 \
  --output ../results/nutime_random_seed0.ipynb
```

Set `NUTIME_REPO` if the clone is elsewhere. The notebook retains its
experiment-specific forward patch and channel aggregation.

Official code: <https://github.com/chenguolin/NuTime>
