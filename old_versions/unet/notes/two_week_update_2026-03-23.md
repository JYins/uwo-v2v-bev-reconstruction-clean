# Two-Week Project Update

This note is my compact summary for what I finished in the last two weeks.

## Main story

The main progress is that the project moved from "pipeline looks okay but masked
region is not really learned" to "U-Net baseline can actually recover occupancy
structure in the masked sector".

The big turning point was adding occupancy-aware supervision.

## What I finished

### 1. Data / pipeline side

- confirmed the BEV preparation pipeline is stable
- fixed the experiment around the same 8-channel BEV setting
- kept the same sector-mask setup for fair comparison

### 2. U-Net baseline

I ran the old 80-epoch U-Net and found the key problem:

- full-image metrics looked okay
- but masked-region occupancy recovery was basically not working
- `masked Occ-IoU = 0`

That meant the model was being too conservative in the hole region.

### 3. Occupancy-focused fix

Then I added `masked_occ_BCE` and did a short probe run.

That run showed the right signal immediately:

- masked Occ-IoU became positive
- visual prediction also showed real repair in the masked sector

So this part was very useful, because it told me the pipeline was not broken.
The loss design was the main issue.

### 4. Balanced 80-epoch U-Net

I then ran a balanced full experiment:

```text
L = 0.7 * masked_weighted_L1
  + 0.3 * masked_MSE
  + 0.35 * masked_occ_BCE
```

Key test result:

- `masked RMSE = 0.055974`
- `masked PSNR = 25.04 dB`
- `masked Occ-IoU = 0.1277`
- `fused full PSNR = 33.66 dB`

This was already a strong baseline, because the masked region started to become
meaningful instead of dead.

### 5. Optuna tuning

After supervisor feedback, I added:

- seed control
- Optuna tuning
- Bayesian-style TPE sampler
- Median pruner

Search was done on the shared loss only, with fixed `seed=42`.

Final selected shared loss:

```text
loss_l1_weight = 0.8082426103528428
loss_mse_weight = 0.19175738964715716
occ_bce_weight = 0.27840993525472485
occ_weight = 2.7593891857986734
occ_pos_weight = 8
occ_threshold = 0.07
occ_logit_temp = 0.03
```

Selected trial result:

- `val masked Occ-IoU = 0.1652`
- `best epoch = 12`
- `best val masked RMSE = 0.04729`
- `best val fused full PSNR = 35.12 dB`

### 6. Formal rerun

Formal multi-seed rerun is started with `seed 42 / 43 / 44`.

Finished formal result so far:

#### U-Net + Optuna shared loss (`seed=42`)

- `Best epoch = 21`
- `Test masked MAE = 0.008347`
- `Test masked RMSE = 0.055882`
- `Test masked PSNR = 25.05 dB`
- `Test masked Occ-IoU = 0.1500`
- `Test fused full PSNR = 33.67 dB`
- `Test fused full IoU = 0.7502`

This is the cleanest finished formal result at the moment.

## Code changes in this period

- added occupancy-aware shared loss logic in `train.py`
- expanded evaluation to `masked + fused-full + raw-full`
- added seed control for reproducibility
- added `train_pix2pix.py`
- added `train_diffusion.py`
- added `tune_unet_optuna.py`
- added `summarize_seed_results.py`
- added PowerShell run scripts for repeatable experiments

## Current experiment logic

The logic is now:

1. search one good shared loss on U-Net
2. rerun U-Net with `42 / 43 / 44`
3. use the same shared auxiliary loss for Pix2Pix / Diffusion
4. compare all three under the same data split, same mask, same metrics
5. move to `BEV -> Cloud`

## What I will do next

1. finish U-Net seeds `43` and `44`
2. summarize `mean +- std`
3. start Pix2Pix
4. start Diffusion
5. prepare `BEV -> Cloud` baseline

## Quick conclusion

The biggest improvement in these two weeks is not only one better number.  
The bigger thing is that the project question is now much clearer:

- the masked region is actually being learned
- the loss design is more justified
- the experiment protocol is more reproducible
- the next model comparison stage is ready
