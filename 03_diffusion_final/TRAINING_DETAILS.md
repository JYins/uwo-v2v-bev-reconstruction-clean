# Diffusion v3 Training Details

## Task

The final diffusion model is conditional BEV reconstruction:

```text
condition = masked ego BEV + neighbor BEV + mask
target    = clean ego BEV
```

During training, noise is added to the target BEV and the model predicts the noise. The estimated clean sample `x0_hat` also receives the shared reconstruction loss.

## What Was Fixed Compared With Old Diffusion Runs

The old diffusion runs were not strong enough to use as final evidence because:

- the denoiser did not properly know the diffusion timestep;
- validation was closer to a proxy `x0_hat` check than full sampled inference;
- some runs were smoke tests or timed out before a fair result.

Final v3 fixes this with:

- timestep conditioning;
- DDIM-style sampled validation/test;
- resume-safe checkpointing;
- LR schedule and gradient clipping;
- full 120-epoch Narval A100 run.

## Final Hyperparameters

| Parameter | Value | Why |
| --- | ---: | --- |
| epochs | 120 | Fresh full Narval run. |
| batch_size | 8 | Used by the confirmed run. |
| lr | 5e-5 | Lower than U-Net because diffusion training was less stable. |
| min_lr | 5e-6 | Cosine-style decay floor in the script. |
| timesteps | 1000 | Standard DDPM-style training horizon. |
| sample_steps | 25 | Faster DDIM-style evaluation. |
| val_every | 10 | Sampling evaluation is expensive. |
| warmup_epochs | 2 | Avoids sharp early LR jump. |
| grad_clip | 1.0 | Added for stability. |
| seed | 42 | Confirmed final diffusion seed. |
| amp | true | Used for the Narval A100 run. |

Shared reconstruction loss is the same locked config:

```text
L_diffusion = L_noise + L_shared
```

with:

```text
L_shared =
  0.8082426104 * masked_weighted_L1
+ 0.1917573896 * masked_MSE
+ 0.2784099353 * masked_occ_BCE
```

## Exact Command

```powershell
python .\03_diffusion_final\train_diffusion.py `
  --dataset_root .\dataset_prepared `
  --training_root .\03_diffusion_final\local_runs\training_diffusion_full_seed42_v3 `
  --epochs 120 `
  --batch_size 8 `
  --lr 5e-5 `
  --min_lr 5e-6 `
  --timesteps 1000 `
  --sample_steps 25 `
  --val_every 10 `
  --warmup_epochs 2 `
  --grad_clip 1.0 `
  --num_workers 0 `
  --print_every 100 `
  --save_every 10 `
  --shared_config .\03_diffusion_final\configs\shared_loss_optuna.json `
  --seed 42 `
  --amp
```

## Final Diagnosis

Diffusion v3 predicts too much of the masked region as occupied. That gives recall `1.0`, but precision collapses. The result is useful because it is a corrected negative baseline, not because it is competitive.
