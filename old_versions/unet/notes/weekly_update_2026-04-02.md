# Weekly Update - 2026-04-02

## 1. Checklist of what we completed this week

- Finalized the official U-Net baseline using the Optuna-selected shared loss and 3 seeds (`42/43/44`).
- Added occupancy diagnostics beyond IoU:
  - masked precision
  - masked recall
  - masked F1
- Added thresholded visualization panels for U-Net and Pix2Pix to make false positives easier to inspect visually.
- Added 8-channel split visualization to diagnose the layer-collapse issue.
- Ran a valid focal-loss ablation on U-Net and confirmed it did **not** beat the current baseline.
- Ran Pix2Pix full training with `lambda_adv = 0.1`.
- Ran Pix2Pix Optuna tuning for adversarial weight and confirmed `lambda_adv = 0.1` is still the best setting.
- Ran the first cloud Diffusion training on Narval and confirmed the pipeline could train, but that run was **not a valid final comparison** because:
  - it timed out at epoch 31
  - it still used proxy evaluation
  - it had no timestep conditioning
- Fixed the Diffusion pipeline:
  - added timestep conditioning
  - replaced proxy evaluation with real DDIM-style reverse sampling
  - added resume-safe checkpointing
  - relaunched the corrected seed-42 Narval job

## 2. What we learned this week

### U-Net

The U-Net baseline is still the strongest and most stable model overall for the current task.

Official 3-seed U-Net result:

| Metric | Result |
|---|---:|
| masked Occ-IoU | `0.1494 +- 0.0023` |
| masked RMSE | `0.05595 +- 0.00010` |
| masked MAE | `0.00846 +- 0.00017` |
| masked PSNR | `25.04 +- 0.02 dB` |
| fused full PSNR | `33.66 +- 0.02 dB` |
| fused full Occ-IoU | `0.7522 +- 0.0045` |

Extra occupancy diagnostics from seed 42:

| Metric | U-Net seed42 |
|---|---:|
| masked Occ-IoU | `0.1500` |
| masked precision | `0.2002` |
| masked recall | `0.3745` |
| masked F1 | `0.2609` |

Interpretation:
- U-Net is still the best overall reconstruction baseline.
- It is relatively recall-oriented: it finds more occupied structure, but it also introduces some false positives in empty regions.

### Additional issue found in U-Net: layer information loss

Besides false positives, we also confirmed a second issue in the U-Net outputs: some layer-wise BEV structure becomes weaker or partially disappears in the reconstructed region.

What we found:
- at first, part of this effect was caused by the old visualization, which was too biased toward the first 4 channels
- after fixing the visualization and adding 8-channel split panels, we confirmed there is also a real model-side issue
- the current supervision protects overall occupancy recovery better than per-layer geometry recovery
- in practice, this means the model often learns **whether something exists**, but not always **which layer/bin structure should be preserved**

Current conclusion:
- this issue is real, but it does not invalidate the current U-Net baseline as the best overall model
- it becomes the next technical improvement target if we continue refining U-Net later
- for now, we keep the current baseline and explicitly report the layer-collapse issue as a limitation

### Focal-loss ablation on U-Net

We tested a proper focal-loss ablation on U-Net. It did **not** improve the baseline.

Conclusion:
- focal loss did not reduce false positives enough
- focal loss did not improve the main masked metrics
- therefore, the official U-Net baseline remains unchanged

### Pix2Pix

The best Pix2Pix setup remains `lambda_adv = 0.1`.

Optuna search result:

| Item | Result |
|---|---:|
| best lambda_adv | `0.1` |
| best validation masked Occ-IoU | `0.1451` |
| best validation epoch | `8` |

Full Pix2Pix seed-42 test result:

| Metric | Pix2Pix seed42 |
|---|---:|
| masked Occ-IoU | `0.1317` |
| masked precision | `0.2701` |
| masked recall | `0.2044` |
| masked F1 | `0.2327` |
| masked RMSE | `0.06139` |
| masked PSNR | `24.24 dB` |
| fused full PSNR | `32.86 dB` |
| fused full Occ-IoU | `0.8215` |

Interpretation:
- Pix2Pix is more precision-oriented than U-Net.
- It produces fewer false positives, but it misses more occupied structure.
- For the main masked reconstruction objective, it is still weaker than the official U-Net baseline.

### Diffusion

We had two distinct diffusion stages this week.

#### Stage A - old cloud run

The first Narval Diffusion run:
- reached epoch 31
- timed out at 24 hours
- used `seed = 42`

But this run is **not a valid final comparison** because the implementation still had two scientific issues:
- no timestep conditioning
- proxy validation/test instead of real reverse sampling

So we treat this run only as a pipeline debugging run, not as an official model result.

#### Stage B - corrected diffusion pipeline

We then fixed the pipeline:
- added timestep embedding to the denoiser
- changed evaluation to DDIM-style reverse sampling
- added resume-safe checkpointing

The corrected diffusion job was relaunched on Narval:

| Item | Value |
|---|---|
| job id | `58756505` |
| status at latest check | `PD` (queued) |
| seed | `42` |
| output root | `/home/syin94/scratch/MEng_Project/runs/training_diffusion_full_seed42_v2` |

Important note:
- There is **not yet a valid official diffusion result** for comparison.
- The corrected run is now the one we should wait for.

## 3. Horizontal model comparison for the current report

For the report this week, I would present the comparison like this:

| Model | Status | Main conclusion |
|---|---|---|
| U-Net | Official baseline | Best overall masked reconstruction result; strongest current baseline |
| Pix2Pix | Valid single-seed comparison | More conservative, higher precision tendency, but weaker than U-Net on masked recovery |
| Diffusion | In progress / corrected rerun pending | Previous run invalid for final comparison; corrected version has now been submitted |

If the supervisor wants a numbers table right now, use:

| Model | Seed(s) | masked Occ-IoU | masked RMSE | fused full PSNR | Note |
|---|---|---:|---:|---:|---|
| U-Net | `42/43/44` | `0.1494 +- 0.0023` | `0.05595 +- 0.00010` | `33.66 +- 0.02` | Official baseline |
| Pix2Pix | `42` | `0.1317` | `0.06139` | `32.86` | Valid comparison run |
| Diffusion | `42` | `N/A yet` | `N/A yet` | `N/A yet` | Corrected run submitted; result pending |

## 4. Figure and output locations

### U-Net

- 3-seed summary:
  - `D:\MEng_Project\reports\unet_optuna_3seed_summary.json`
- thresholded U-Net visualization:
  - `D:\MEng_Project\reports\thresholded_unet_seed42_v2`
- 8-channel split diagnosis:
  - `D:\MEng_Project\reports\channel_splits_unet_seed42`

### Pix2Pix

- full training summary:
  - `D:\MEng_Project\training_pix2pix_full_seed42\results\pix2pix_summary.json`
- Optuna best adversarial weight:
  - `D:\MEng_Project\optuna_pix2pix_adv_v2\best_params.json`
- prediction panels:
  - `D:\MEng_Project\reports\predictions_pix2pix_seed42`
- thresholded Pix2Pix panels:
  - `D:\MEng_Project\reports\thresholded_pix2pix_seed42`

### Diffusion

- corrected local smoke summary:
  - `D:\MEng_Project\training_diffusion_smoke_v3_seed42\results\diffusion_summary.json`
- Narval corrected job output:
  - `/home/syin94/scratch/MEng_Project/runs/training_diffusion_full_seed42_v2`
- Narval sbatch script:
  - `/home/syin94/scratch/MEng_Project/runs/run_diffusion_seed42_v2.sbatch`

## 5. English spoken update for supervisor

Here is a spoken-English version:

> This week, I mainly focused on making the model comparison cleaner and more scientifically reliable.
>
> First, I finalized the official U-Net baseline. I used the Optuna-selected shared loss and completed the formal 3-seed experiment with seeds 42, 43, and 44. The final U-Net result is still the strongest baseline, with masked Occupancy IoU around 0.149 and masked RMSE around 0.056. So at this stage, U-Net is still the best-performing model for the masked BEV reconstruction task.
>
> Second, I improved the evaluation protocol. I added occupancy precision, recall, and F1, because IoU alone was not enough to explain the model behavior. These extra metrics helped show an important pattern: U-Net is more recall-oriented, so it tends to recover more occupied structure, but it also introduces some false positives in empty regions. Pix2Pix is more precision-oriented, so it is more conservative, but it misses more occupied structure.
>
> I also identified another issue in the U-Net outputs. After fixing the visualization and inspecting the 8-channel split view, I confirmed that the model sometimes loses part of the layer-wise BEV structure in the reconstructed region. So the current U-Net learns occupancy existence better than detailed per-layer geometry. I am treating this as a known limitation of the current baseline rather than hiding it.
>
> Third, I tested whether focal loss could improve the U-Net baseline. I ran a proper focal-loss ablation, but it did not beat the current baseline, so I kept the original U-Net configuration as the official result.
>
> Fourth, I completed the Pix2Pix tuning work. I ran Optuna for the adversarial weight, and the best result still came from lambda_adv equals 0.1. So the earlier manual finding was confirmed by the search. The full Pix2Pix run is valid, but it is still weaker than U-Net on the main masked reconstruction metrics.
>
> Finally, I revisited Diffusion. The first cloud run on Narval was useful for debugging, but it was not yet a valid final comparison, because it timed out at epoch 31, and more importantly, the implementation still used proxy evaluation and did not include timestep conditioning. So this week I corrected the diffusion pipeline by adding timestep conditioning, changing validation and test to real DDIM-style reverse sampling, and adding resume-safe checkpointing. The corrected seed-42 Narval job has now been resubmitted, and this is the run we should wait for before making a fair comparison against U-Net and Pix2Pix.
>
> So the main conclusion for this week is that the U-Net baseline is now solid, Pix2Pix has been properly tuned and remains weaker than U-Net, and Diffusion is currently in the corrected rerun stage. Next week, the main priority is to finish the corrected Diffusion run on Narval and then make the final three-model comparison under the same valid protocol.

## 6. Next-week plan

- Wait for the corrected Narval Diffusion run to start and complete.
- If needed, resume it across multiple jobs until 80 epochs finish.
- Once the corrected Diffusion result is available, build the final fair comparison table:
  - U-Net
  - Pix2Pix
  - Diffusion
- If Diffusion is still clearly weaker after the corrected run, close it out as a negative result and focus the thesis around the stronger U-Net baseline plus the Pix2Pix comparison.
