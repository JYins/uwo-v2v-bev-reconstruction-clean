# Final U-Net Baseline

This is the official best baseline for the project.

The final promoted version is not just one lucky seed. It is the 3-seed U-Net rerun using the shared loss selected by Optuna:

```text
local_runs/training_unet_optuna_seed42
local_runs/training_unet_optuna_seed43
local_runs/training_unet_optuna_seed44
```

The clean copied metrics are in:

```text
results/seed42/
results/seed43/
results/seed44/
results/summary/unet_optuna_3seed_summary.json
```

## Why This Is Final

- It uses the locked shared reconstruction loss from `configs/shared_loss_optuna.json`.
- It was rerun with seeds `42`, `43`, and `44`.
- It has the strongest final masked Occ-IoU and masked RMSE.
- Later focal-loss and layer-preserving probes did not clearly beat it, so they stay in `old_versions/unet/`.

Final 3-seed result:

```text
masked Occ-IoU   = 0.1494 +/- 0.0019
masked RMSE      = 0.05595 +/- 0.00008
fused full PSNR  = 33.66 +/- 0.01 dB
```

## Files

```text
train.py                  U-Net training entry
dataset.py                BEV dataset loader
unet.py                   U-Net model
tune_unet_optuna.py       loss-weight Optuna search
configs/                  final shared loss config
scripts/run_3seeds.ps1    exact final rerun script
TRAINING_DETAILS.md       detailed hyperparameters and rationale
```

## Run

From the repo root:

```powershell
$env:V2V_PYTHON = "D:\Anaconda3\envs\v2v4real\python.exe"
.\01_unet_final\scripts\run_3seeds.ps1
```

If `V2V_PYTHON` is not set, the script uses `python`.

Local outputs go into:

```text
01_unet_final/local_runs/
```

That folder is ignored by Git.
