# One-Page Loss Summary

## 1. Shared task loss (fixed across models)

```text
L_shared =
  0.8082 * masked_weighted_L1
  + 0.1918 * masked_MSE
  + 0.2784 * masked_occ_BCE
```

Fixed hyperparameters:

- `occ_weight = 2.7594`
- `occ_pos_weight = 8`
- `occ_threshold = 0.07`
- `occ_logit_temp = 0.03`

Why:

- `masked_weighted_L1`: main reconstruction term; keeps structure sharper than pure MSE
- `masked_MSE`: stabilizes numeric reconstruction
- `masked_occ_BCE`: directly teaches occupied / empty recovery in the masked region

This shared loss was selected by Optuna on the U-Net baseline, then reused as the common auxiliary supervision for later models.

## 2. Final loss for each model

| Model | Final loss | Why |
|---|---|---|
| U-Net | `L_unet = L_shared` | Pure reconstruction baseline |
| Pix2Pix | `L_pix2pix = 0.1 * L_adv + L_shared` | Keep GAN sharpness, but do not let adversarial loss dominate the masked recovery task |
| Diffusion | `L_diffusion = L_noise + L_shared(x0_hat, target, mask)` | Keep standard noise-prediction objective, then add the same reconstruction target |

Notes:

- `L_adv`: hinge-GAN generator loss
- `L_noise`: diffusion noise-prediction loss
- `x0_hat`: clean BEV estimated from noisy sample and predicted noise

## 3. What is fixed and what is not fixed

| Fixed across models | Model-specific |
|---|---|
| Same dataset split | Core model loss (`L_adv` or `L_noise`) |
| Same 8-channel BEV representation | Batch size |
| Same masked sector rule | Training length |
| Same input/output setting | Model architecture details |
| Same shared loss parameters | Extra optimization settings |
| Same main metrics | |

Fixed input / target:

- Input: `masked_ego (8ch) + neighbor_bev (8ch)`
- Target: `clean ego_bev (8ch)`

Fixed main metrics:

- `masked MAE`
- `masked RMSE`
- `masked PSNR`
- `masked Occ-IoU`
- `fused full PSNR`
- `fused full Occ-IoU`

## 4. Current result snapshot

| Model | Current result summary |
|---|---|
| U-Net (3 seeds) | Strongest baseline; `masked Occ-IoU = 0.1494 +- 0.0023`, `masked RMSE = 0.05595 +- 0.00010` |
| Pix2Pix (seed42) | Works and gives useful comparison, but does not beat U-Net on masked main metrics |
| Diffusion (2-epoch smoke) | Engineering pipeline is now working, but performance is still very early and not ready for conclusion |

## 5. Main decision

The experiment design is:

1. keep the reconstruction task fixed
2. keep the shared occupancy-aware loss fixed
3. allow each model to keep its own core training objective
4. compare all models with the same masked and fused-full metrics

This makes the comparison fair, while still keeping each model training in a natural way.
