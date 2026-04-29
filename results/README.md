# Results

This folder keeps the supervisor-facing outputs that are small enough to live in Git.

```text
figures/final_results/       final report figures
figures/all_report_images/   flat copy of every old report image
metrics/                     final JSON/CSV metric summaries
FIGURE_MANIFEST.md           image index
```

The model folders also keep their own copied metrics:

```text
01_unet_final/results/
02_pix2pix_final/results/
03_diffusion_final/results/
```

Checkpoints are not stored here. They are local artifacts and are ignored.
