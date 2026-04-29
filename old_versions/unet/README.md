# Old U-Net Versions

This folder is for U-Net experiments that are useful history but not the final promoted result.

What belongs here:

- original 80-epoch U-Net baseline;
- balanced and occupancy-probe runs;
- seed sanity checks;
- focal-loss ablations;
- layer-preserving probes;
- old U-Net Optuna smoke/debug outputs.

Main lesson:

```text
Plain reconstruction loss looked okay on full-image metrics but did not recover the masked occupancy well.
The occupancy-aware shared loss fixed the zero/near-zero masked Occ-IoU problem.
Focal and layer-preserving probes were reasonable ideas, but did not clearly beat the final Optuna U-Net.
```

Final U-Net result is in:

```text
01_unet_final/
```
