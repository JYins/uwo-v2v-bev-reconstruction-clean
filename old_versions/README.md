# Old Versions

This folder is intentionally part of the cleaned repo.

These folders keep the smoke runs, probes, ablations, and invalid early attempts separated from the final code so the review can distinguish:

- what is actually final;
- what was tried and rejected.

Layout:

```text
unet/       old U-Net baselines, occupancy probes, focal/layer-preserving attempts
pix2pix/    Pix2Pix smoke runs and non-final adversarial settings
diffusion/  early diffusion attempts before the corrected v3 baseline
misc/       old sync/output helpers that are not part of the final runnable path
reports_archive/ old report tree after final figures were copied to results/
```

The final versions live at the repo root:

```text
01_unet_final/
02_pix2pix_final/
03_diffusion_final/
```
