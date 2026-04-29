# Final Diffusion v3 Baseline

This is the corrected diffusion baseline.

The confirmed final run is:

```text
local_runs/training_diffusion_full_seed42_v3
```

It was trained on Narval A100 for 120 epochs with timestep conditioning and DDIM-style sampled validation/test. This matters because earlier diffusion attempts were not fair final comparisons.

## Why This Is Final

- It fixes the old missing timestep-conditioning issue.
- It evaluates with reverse sampling instead of the older proxy evaluation.
- It completed a fresh 120-epoch run.
- It still performs badly, so it is a valid negative baseline rather than a competitive model.

Final seed42 test result:

```text
masked Occ-IoU       = 0.0467
masked precision     = 0.0467
masked recall        = 1.0000
masked F1            = 0.0893
masked RMSE          = 0.3810
fused full PSNR      = 17.00 dB
fused full Occ-IoU   = 0.2554
```

## Files

```text
train_diffusion.py             diffusion-style training entry
train.py                       shared loss/evaluation helpers copied from U-Net code
dataset.py                     BEV dataset loader
unet.py                        denoiser backbone with timestep support
configs/shared_loss_optuna.json
scripts/run_final_seed42.ps1
TRAINING_DETAILS.md
```

## Run

```powershell
$env:V2V_PYTHON = "D:\Anaconda3\envs\v2v4real\python.exe"
.\03_diffusion_final\scripts\run_final_seed42.ps1
```

Local outputs go into:

```text
03_diffusion_final/local_runs/
```

For this model, a GPU is not optional in practice.
