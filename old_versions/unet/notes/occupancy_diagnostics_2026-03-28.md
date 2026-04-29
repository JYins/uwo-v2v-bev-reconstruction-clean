# Occupancy Diagnostics Update

## Main read

- U-Net still has the best masked IoU, but its `precision` is lower than its `recall`, which matches the false-positive / empty-area haze problem.
- Pix2Pix is more conservative: higher `precision`, lower `recall`.
- Diffusion smoke is still too early to judge as a model comparison result.

## Refreshed test metrics

| Model | Masked IoU | Precision | Recall | F1 | Masked RMSE | Fused PSNR |
|---|---:|---:|---:|---:|---:|---:|
| U-Net seed42 | 0.1500 | 0.2002 | 0.3745 | 0.2609 | 0.055882 | 33.67 |
| Pix2Pix seed42 | 0.1317 | 0.2701 | 0.2044 | 0.2327 | 0.061386 | 32.86 |
| Diffusion smoke seed42 | 0.0445 | 0.0466 | 0.5035 | 0.0852 | 0.455776 | 15.44 |

## Threshold sweep for visualization

| Model | Selected render threshold | Best validation F1 | Note |
|---|---:|---:|---|
| U-Net seed42 | 0.07 | 0.2982 | Same as current occupancy threshold |
| Pix2Pix seed42 | 0.03 | 0.2960 | Slightly lower threshold gives cleaner display |

## Output files

- U-Net refreshed metrics: `D:\MEng_Project\training_unet_optuna_seed42\results\test_metrics_refresh.json`
- Pix2Pix refreshed metrics: `D:\MEng_Project\training_pix2pix_full_seed42\results\test_metrics_refresh.json`
- Diffusion refreshed metrics: `D:\MEng_Project\training_diffusion_smoke_v2_seed42\results\test_metrics_refresh.json`
- U-Net threshold panels: `D:\MEng_Project\reports\thresholded_unet_seed42`
- Pix2Pix threshold panels: `D:\MEng_Project\reports\thresholded_pix2pix_seed42`

## Next move

- Run U-Net focal-loss ablation for `15 epochs` on `seed42`
- Compare against current U-Net seed42 on:
  - masked IoU
  - precision / recall / F1
  - masked RMSE
  - thresholded qualitative panels
