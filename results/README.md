# Results

This folder keeps the supervisor-facing outputs that are small enough to live in Git.

```text
figures/final_results/       final report figures
figures/all_report_images/   flat copy of every old report image
metrics/                     cross-model JSON/CSV metric summaries
FIGURE_MANIFEST.md           image index
```

Each final model folder now also keeps its own results, so it can be read independently:

```text
01_unet_final/results/METRICS_SUMMARY.md
01_unet_final/results/figures/
02_pix2pix_final/results/METRICS_SUMMARY.md
02_pix2pix_final/results/figures/
03_diffusion_final/results/METRICS_SUMMARY.md
03_diffusion_final/results/figures/
```

Checkpoints are not stored here. They are local artifacts and are ignored.
