# Final Pix2Pix Baseline

This is the final GAN comparison for the project.

The confirmed final run is:

```text
local_runs/training_pix2pix_full_seed42
```

It uses the same shared reconstruction loss as U-Net, plus a small adversarial term:

```text
L_pix2pix = L_shared + 0.1 * L_adv
```

## Why This Is Final

- The early `lambda_adv=1.0` run was too adversarial and underperformed.
- The full seed42 run with `lambda_adv=0.1` is the meaningful Pix2Pix comparison.
- The Optuna adversarial search in `configs/pix2pix_adv_best.json` selected `0.1`, including checks against larger values such as `1.0` and `2.0`.
- It does not beat U-Net, so it is a comparison baseline rather than the final main model.

Final seed42 test result:

```text
masked Occ-IoU       = 0.1317
masked precision     = 0.2701
masked recall        = 0.2044
masked F1            = 0.2327
masked RMSE          = 0.06139
fused full PSNR      = 32.86 dB
fused full Occ-IoU   = 0.8215
```

The copied figures and metric summary for this exact run are in:

```text
results/METRICS_SUMMARY.md
results/figures/prediction_samples/
results/figures/thresholded_samples/
results/figures/training_curves_pix2pix_seed42.png
results/figures/channel_split_pix2pix.png
```

## Files

```text
train_pix2pix.py               Pix2Pix training entry
train.py                       shared loss/evaluation helpers copied from U-Net code
dataset.py                     BEV dataset loader
unet.py                        generator backbone
tune_pix2pix_adv.py            adversarial-weight Optuna search
configs/shared_loss_optuna.json
configs/pix2pix_adv_best.json
scripts/run_seed42.ps1
TRAINING_DETAILS.md
results/METRICS_SUMMARY.md
results/figures/
```

## Run

```powershell
$env:V2V_PYTHON = "D:\Anaconda3\envs\v2v4real\python.exe"
.\02_pix2pix_final\scripts\run_seed42.ps1
```

Local outputs go into:

```text
02_pix2pix_final/local_runs/
```
