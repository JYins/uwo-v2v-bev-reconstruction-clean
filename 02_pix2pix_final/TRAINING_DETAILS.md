# Pix2Pix Training Details

## Task

Same BEV reconstruction task as U-Net:

```text
input  = masked_ego(8ch) + neighbor_bev(8ch)
target = clean ego_bev(8ch)
```

Pix2Pix asks whether an adversarial term can make the BEV output sharper and more realistic.

## Architecture

- Generator: same `UNet(in_channels=16, out_channels=8, features=[16,32,64,128])`.
- Discriminator: small PatchGAN-style discriminator over `condition + BEV`.
- Discriminator input channels: `16 + 8 = 24`.
- GAN loss: hinge-style discriminator loss.

## Final Hyperparameters

| Parameter | Value | Why |
| --- | ---: | --- |
| epochs | 40 | Final full Pix2Pix run length. |
| batch_size | 6 | Used by the confirmed full run and adversarial search. |
| g_lr | 2e-4 | Standard GAN learning rate, stable in this script. |
| d_lr | 2e-4 | Matches generator LR. |
| Adam betas | 0.5, 0.999 | Common Pix2Pix/GAN choice. |
| lambda_adv | 0.1 | Selected by adversarial-weight Optuna search. |
| seed | 42 | Confirmed final Pix2Pix seed. |
| amp | true | Used in the final run. |

Shared reconstruction loss is the same as U-Net:

```text
L_shared =
  0.8082426104 * masked_weighted_L1
+ 0.1917573896 * masked_MSE
+ 0.2784099353 * masked_occ_BCE
```

Occupancy internals:

```text
occ_weight = 2.7593891858
occ_pos_weight = 8
occ_threshold = 0.07
occ_logit_temp = 0.03
```

Model objective:

```text
L_generator = L_shared + 0.1 * L_adv
```

## Exact Command

```powershell
python .\02_pix2pix_final\train_pix2pix.py `
  --dataset_root .\dataset_prepared `
  --training_root .\02_pix2pix_final\local_runs\training_pix2pix_full_seed42 `
  --epochs 40 `
  --batch_size 6 `
  --g_lr 0.0002 `
  --d_lr 0.0002 `
  --lambda_adv 0.1 `
  --num_workers 0 `
  --print_every 100 `
  --save_every 10 `
  --shared_config .\02_pix2pix_final\configs\shared_loss_optuna.json `
  --seed 42 `
  --amp
```

## Why The Adversarial Weight Is Small

Large adversarial weights made the model care too much about fooling the discriminator and not enough about reconstructing the masked sector accurately. The search result selected `lambda_adv=0.1`, which keeps the GAN term as a gentle sharpness pressure instead of the main training signal.

## Final Results

Seed 42 test:

```text
masked Occ-IoU       = 0.1317
masked precision     = 0.2701
masked recall        = 0.2044
masked F1            = 0.2327
masked RMSE          = 0.061386
fused full PSNR      = 32.86 dB
fused full Occ-IoU   = 0.8215
```

See `results/METRICS_SUMMARY.md` and `results/figures/` for the copied result tables and images.

## Known Limitation

Pix2Pix is more conservative than U-Net: its precision is higher, but recall is lower. It does not beat the U-Net final baseline.
