# Confirmed Final Runs

This file is here so the final repo does not blur finished results with exploratory runs.

## 1. U-Net Final

Evidence:

- Local run folders: `01_unet_final/local_runs/training_unet_optuna_seed42`, `01_unet_final/local_runs/training_unet_optuna_seed43`, `01_unet_final/local_runs/training_unet_optuna_seed44`
- Clean copied metrics: `01_unet_final/results/`
- Shared loss config: `01_unet_final/configs/shared_loss_optuna.json`
- Summary: `01_unet_final/results/summary/unet_optuna_3seed_summary.json`

Final setup:

```text
epochs = 80
batch_size = 12
features = 16,32,64,128
lr = 1e-4
seeds = 42, 43, 44
```

Status: official best baseline.

## 2. Pix2Pix Final

Evidence:

- Local run folder: `02_pix2pix_final/local_runs/training_pix2pix_full_seed42`
- Clean copied metrics: `02_pix2pix_final/results/seed42/`
- Adversarial search result: `02_pix2pix_final/configs/pix2pix_adv_best.json`

Final setup:

```text
epochs = 40
batch_size = 6
g_lr = 2e-4
d_lr = 2e-4
lambda_adv = 0.1
seed = 42
```

Status: valid GAN comparison, weaker than U-Net.

## 3. Diffusion v3 Final

Evidence:

- Local run folder: `03_diffusion_final/local_runs/training_diffusion_full_seed42_v3`
- Clean copied metrics: `03_diffusion_final/results/seed42/`

Final setup:

```text
epochs = 120
batch_size = 8
lr = 5e-5
min_lr = 5e-6
timesteps = 1000
sample_steps = 25
warmup_epochs = 2
grad_clip = 1.0
seed = 42
```

Status: corrected negative baseline.
