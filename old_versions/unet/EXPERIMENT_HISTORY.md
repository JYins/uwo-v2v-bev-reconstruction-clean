# U-Net Experiment History

These are not the final promoted runs, but they explain why the final U-Net looks the way it does.

## Local Run Folders

The full folders are kept locally under:

```text
old_versions/unet/local_runs/
```

They are ignored by Git because many contain checkpoints and generated outputs.

Important groups:

```text
training/                              first U-Net baseline
training_balanced80/                   balanced occupancy-aware 80 epoch run
training_occprobe/                     short occupancy BCE probe
training_seedcheck_a,b/                seed sanity checks
training_unet_focal15_seed42*/         focal occupancy ablations
training_unet_layerprobe_*/            layer-preserving probes
optuna_unet*/                          old/debug Optuna runs
```

## What I Learned

- The plain baseline could look acceptable on full-image metrics while masked Occ-IoU stayed too weak.
- Adding an occupancy term was the important turning point.
- Optuna gave a cleaner justification for the shared loss weights.
- Focal and layer-preserving probes were reasonable, but did not clearly beat the Optuna U-Net.

## Notes Kept In Git

```text
notes/loss_formula_onepage_2026-03-26.md
notes/occupancy_diagnostics_2026-03-28.md
notes/two_week_update_2026-03-23.md
notes/weekly_update_2026-04-02.md
notes/seedcheck_summary.json
```

Final U-Net:

```text
../../01_unet_final/
```
