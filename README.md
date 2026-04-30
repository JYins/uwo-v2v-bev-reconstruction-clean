# V2V BEV Reconstruction

```text
uwo-v2v-bev-reconstruction-clean/
|-- teacher_overview.html          visual summary for the meeting
|-- DATASET_ACCESS.md              processed dataset location and Drive notes
|-- DATASET_SANITY_CHECK.md        processed dataset count and tensor-shape check
|-- 00_data_pipeline/              raw V2V4Real -> prepared BEV tensors
|-- 01_unet_final/                 final 3-seed U-Net baseline
|   |-- README.md
|   |-- TRAINING_DETAILS.md
|   |-- results/
|   |   |-- METRICS_SUMMARY.md
|   |   `-- figures/
|-- 02_pix2pix_final/              final Pix2Pix comparison
|   |-- README.md
|   |-- TRAINING_DETAILS.md
|   |-- results/
|   |   |-- METRICS_SUMMARY.md
|   |   `-- figures/
|-- 03_diffusion_final/            final corrected diffusion v3 baseline
|   |-- README.md
|   |-- TRAINING_DETAILS.md
|   |-- results/
|   |   |-- METRICS_SUMMARY.md
|   |   `-- figures/
|-- results/                       shared final tables, figures, and old report image archive
|-- docs/                          final report HTML/markdown and proposal/literature page
`-- old_versions/                  exploratory runs and runs not promoted as final
```

This is a cleaned MEng research workspace for cooperative BEV reconstruction from V2V LiDAR. The first-level model folders are the confirmed final versions for review, and each one includes its own code, exact run command, metrics, and figures.

For a quick pass, please start with `teacher_overview.html`, then review the three final model folders in order: `01_unet_final/`, `02_pix2pix_final/`, and `03_diffusion_final/`.

## Dataset And Literature Notes

Dataset:

- Source dataset: V2V4Real cooperative perception data.
- Processed training data: `dataset_prepared/`, converted into 8-channel 500 x 500 BEV tensors.
- Input to models: masked ego BEV plus neighbor BEV, 16 channels total.
- Target: clean ego BEV, 8 channels.
- Splits used in the final experiments: 7105 train samples, 748 validation samples, and 1993 test samples.
- Processed dataset size: 39429 files, about 151.73 GB.
- Google Drive link: https://drive.google.com/drive/folders/1nQuA6qQhK8wcshwI5CHk5BQDv4R4eAPs?usp=sharing
- Sanity check: `DATASET_SANITY_CHECK.md`

Literature and proposal notes:

- Main HTML page: `docs/research_proposal_and_paper_reading.html`
- Source markdown: `docs/source_markdown/research_proposal_generative_v2v_3d_reconstruction.md`
- The notes connect the original longer direction, generative V2V 3D reconstruction, with the completed practical experiment: masked multi-height BEV reconstruction.
- The reading notes cover the motivation for V2V communication efficiency, BEV representation, cooperative perception, and why U-Net / Pix2Pix / diffusion were used as the main model families.

## Main Result

The final U-Net baseline is still the strongest model. Pix2Pix is a useful GAN comparison but loses recall. Diffusion v3 is corrected enough to be a fair negative baseline, but it predicts too much occupied space in the masked region.

| Model | Seeds | Masked Occ-IoU | Precision | Recall | F1 | Masked RMSE | Fused PSNR | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| U-Net | 42 / 43 / 44 | 0.1494 +/- 0.0023 | 0.2009 +/- 0.0046 | 0.3684 +/- 0.0056 | 0.2600 +/- 0.0035 | 0.055945 +/- 0.000102 | 33.66 +/- 0.02 | Best final model family |
| Pix2Pix | 42 | 0.1317 | 0.2701 | 0.2044 | 0.2327 | 0.061386 | 32.86 | Higher precision, lower recall |
| Diffusion v3 | 42 | 0.0467 | 0.0467 | 1.0000 | 0.0893 | 0.381013 | 17.00 | Corrected but not competitive |

Detailed metric files:

```text
01_unet_final/results/METRICS_SUMMARY.md
02_pix2pix_final/results/METRICS_SUMMARY.md
03_diffusion_final/results/METRICS_SUMMARY.md
results/metrics/final_metrics_summary.csv
```

## Confirmed Final Runs

| Model | Confirmed run | Why it is promoted |
| --- | --- | --- |
| U-Net | `training_unet_optuna_seed42/43/44`, 80 epochs | Same Optuna-selected shared loss, rerun across three seeds, best masked Occ-IoU/F1 balance. |
| Pix2Pix | `training_pix2pix_full_seed42`, 40 epochs, `lambda_adv=0.1` | The adversarial-weight search selected the small GAN term; useful comparison, but not better than U-Net. |
| Diffusion v3 | `training_diffusion_full_seed42_v3`, 120 epochs on Narval A100 | Corrected timestep conditioning and DDIM-style sampled evaluation, with LR schedule and gradient clipping. |

Older smoke tests, probes, ablations, and invalid diffusion attempts are under `old_versions/`. They are kept there so the experiment path is visible, but they are separated from the final code for review.

## Task Setup

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

The main metric is masked `Occ-IoU`, with precision/recall/F1 reported beside it. The masked region is the real reconstruction target, so full-image metrics are used as supporting evidence rather than the main conclusion.

## Running

The processed dataset is too large for Git. See `DATASET_ACCESS.md` before running anything.

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

Training outputs go into each model folder's `local_runs/`. That folder is ignored by Git so checkpoints and large local artifacts do not end up in the repository.

## Project Notes

The repo is split into final code, final results, supporting docs, and old versions:

- `01_unet_final/`, `02_pix2pix_final/`, and `03_diffusion_final/` are the model folders for review or editing.
- Each final model folder has its own `TRAINING_DETAILS.md`, `results/METRICS_SUMMARY.md`, and `results/figures/`.
- `results/` at the root keeps the cross-model summary and the full figure archive from the report.
- `docs/` has the final HTML report and the proposal/literature-review page.
- `old_versions/` keeps earlier experiments with notes on why they were not promoted.

The code is kept as readable research scripts rather than a large package. The goal is for the training choices, hyperparameters, and failure cases to be easy to inspect and correct.
