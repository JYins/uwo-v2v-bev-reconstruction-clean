# V2V BEV Reconstruction

This is my MEng research workspace for cooperative BEV reconstruction from V2V LiDAR.

The current repo is cleaned for supervisor review: the first-level folders are the real runnable parts of the project, and the older smoke runs / probes / failed attempts are separated under `old_versions/`. I kept the experiment trail because it matters: the final result only makes sense after seeing what did not work.

## Quick Start For Review

- Open `teacher_overview.html` first for the visual summary.
- Read `01_unet_final/` for the official best baseline.
- Read `02_pix2pix_final/` for the GAN comparison.
- Read `03_diffusion_final/` for the corrected diffusion negative baseline.
- Read `DATASET_ACCESS.md` before trying to run anything, because `dataset_prepared/` is about 151.7 GB and is not committed.

## Final Versions Only

| Model | Final run | Why this is the final version |
| --- | --- | --- |
| U-Net | `01_unet_final/local_runs/training_unet_optuna_seed42/43/44`, 80 epochs | Official 3-seed result using the Optuna-selected shared loss. Best final model family. |
| Pix2Pix | `02_pix2pix_final/local_runs/training_pix2pix_full_seed42`, 40 epochs, `lambda_adv=0.1` | Valid GAN comparison. Pix2Pix Optuna confirmed that a small adversarial weight works best. |
| Diffusion v3 | `03_diffusion_final/local_runs/training_diffusion_full_seed42_v3`, 120 epochs | Corrected diffusion run with timestep conditioning and DDIM-style sampled evaluation. It is a real negative baseline, not an invalid early run. |

Final metric summary:

| Model | Seeds | Masked Occ-IoU | Masked RMSE | Fused Full PSNR | Status |
| --- | --- | ---: | ---: | ---: | --- |
| U-Net | 42 / 43 / 44 | 0.1494 +/- 0.0019 | 0.05595 +/- 0.00008 | 33.66 +/- 0.01 | Best official baseline |
| Pix2Pix | 42 | 0.1317 | 0.06139 | 32.86 | Valid but weaker GAN baseline |
| Diffusion v3 | 42 | 0.0467 | 0.38101 | 17.00 | Correct negative baseline |

## Repo Layout

```text
00_data_pipeline/      raw V2V4Real -> prepared 8-channel BEV tensors
01_unet_final/         final U-Net baseline and exact 3-seed setup
02_pix2pix_final/      final Pix2Pix comparison and adversarial tuning evidence
03_diffusion_final/    final corrected diffusion v3 baseline
results/               final figures, all report images, metrics, visualization scripts
docs/                  final report HTML/markdown, proposal/literature page, report builders
old_versions/          smoke runs, probes, ablations, invalid attempts, old reports
DATASET_ACCESS.md      dataset sharing and expected local structure
teacher_overview.html  first page to open for a supervisor meeting
```

## Core Task

Input:

```text
masked ego BEV (8 channels) + neighbor BEV (8 channels) -> 16 channels
```

Target:

```text
clean ego BEV -> 8 channels
```

Fixed BEV representation:

```text
Range:       x,y in [-40m, 40m]
Resolution: 0.16 m / pixel
Grid:       500 x 500
Height bins:
  [-3.0, -1.5), [-1.5, 0.0), [0.0, 1.0), [1.0, 2.0)
Channels:
  0-3 density, 4-7 max height
```

The main metric is masked `Occ-IoU`, because this project is about reconstructing the hidden sector, not making the full image look smooth by copying visible pixels.

## Running

Set the Python environment if needed:

```powershell
$env:V2V_PYTHON = "D:\Anaconda3\envs\v2v4real\python.exe"
```

Then run one of:

```powershell
.\01_unet_final\scripts\run_3seeds.ps1
.\02_pix2pix_final\scripts\run_seed42.ps1
.\03_diffusion_final\scripts\run_final_seed42.ps1
```

Local training outputs go into each model folder's `local_runs/`, which is intentionally ignored by Git.

## Notes For My Supervisor

I did not hide the weaker runs. They are in `old_versions/` with notes on what each attempt taught me. The final promoted versions are only the confirmed runs listed above.

The code is still research code: simple scripts, explicit arguments, and comments where the design choice is not obvious. I kept it this way because it is easier to inspect and correct than a heavy package wrapper.
