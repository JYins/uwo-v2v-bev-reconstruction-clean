# U-Net Training Details

## Task

Input is masked ego BEV plus neighbor BEV:

```text
masked_ego(8ch) + neighbor_bev(8ch) -> 16 channels
```

Target is clean ego BEV:

```text
ego_bev -> 8 channels
```

The loss and main metrics focus on the masked region because that is the reconstruction task.

## Architecture

The model is the local `UNet` in `unet.py`.

Final features:

```text
features = [16, 32, 64, 128]
input channels = 16
output channels = 8
```

The model is kept small on purpose. The project is about whether neighbor BEV can help recover the hidden sector, not about hiding the result behind a huge network.

## Final Hyperparameters

| Parameter | Value | Why |
| --- | ---: | --- |
| epochs | 80 | Long enough for the final U-Net baseline to settle. |
| batch_size | 12 | Fits the local GPU memory for 500x500 BEV tensors. |
| lr | 1e-4 | Stable Adam learning rate used across the U-Net runs. |
| weight_decay | 1e-5 | Small regularization, kept from the baseline training script. |
| features | 16,32,64,128 | Keeps the model readable and not too heavy. |
| seeds | 42,43,44 | Used for final 3-seed reporting. |
| mask_variant | sector75 | Original sector-mask setting used for the official comparison. |
| amp | true | Speeds up CUDA training without changing the experimental target. |

Shared loss from `configs/shared_loss_optuna.json`:

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

Where these came from:

- Optuna search: `tune_unet_optuna.py`
- Search evidence: `configs/shared_loss_optuna.json`
- Selected trial: `21`
- Selection metric: validation masked Occ-IoU, with masked RMSE as tie-break style evidence in the run summaries.

## Exact Command

The script form is:

```powershell
.\01_unet_final\scripts\run_3seeds.ps1
```

Equivalent command for one seed:

```powershell
python .\01_unet_final\train.py `
  --dataset_root .\dataset_prepared `
  --training_root .\01_unet_final\local_runs\training_unet_optuna_seed42 `
  --epochs 80 `
  --batch_size 12 `
  --features 16,32,64,128 `
  --num_workers 0 `
  --print_every 100 `
  --save_every 10 `
  --shared_config .\01_unet_final\configs\shared_loss_optuna.json `
  --seed 42
```

## Final Results

3-seed summary:

```text
masked Occ-IoU   = 0.1494 +/- 0.0023
masked precision = 0.2009 +/- 0.0046
masked recall    = 0.3684 +/- 0.0056
masked F1        = 0.2600 +/- 0.0035
masked RMSE      = 0.05595 +/- 0.00010
fused full PSNR  = 33.66 +/- 0.02 dB
```

Seed 42 test:

```text
masked Occ-IoU       = 0.1500
masked precision     = 0.2002
masked recall        = 0.3745
masked F1            = 0.2609
masked RMSE          = 0.05588
fused full PSNR      = 33.67 dB
fused full Occ-IoU   = 0.7502
```

See `results/METRICS_SUMMARY.md` for the full seed42/43/44 table.

## Known Limitation

U-Net is the best final baseline, but it still has false positives and weak layer preservation in some height channels, especially `h0`. This is why the report includes thresholded visualizations and per-channel split views.
