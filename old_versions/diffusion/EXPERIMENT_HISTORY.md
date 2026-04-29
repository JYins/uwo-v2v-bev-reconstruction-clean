# Diffusion Experiment History

The full old diffusion folders are kept locally under:

```text
old_versions/diffusion/local_runs/
```

They are ignored by Git because they contain checkpoints and generated logs.

Important groups:

```text
training_diffusion_smoke_seed42/
training_diffusion_smoke_v2_seed42/
training_diffusion_smoke_v3_seed42/
training_diffusion_smoke_v4_seed42/
training_diffusion_full_seed42/
training_diffusion_full_seed42_v2/
diffusion_seed42*.out / *.err
```

## What I Learned

- The first diffusion attempts were smoke/proxy experiments, not final evidence.
- Missing or weak timestep handling made the early results hard to interpret.
- The corrected v3 run was needed before calling diffusion a real negative baseline.
- Even after correction, diffusion over-predicted occupancy and stayed much weaker than U-Net.

Final diffusion:

```text
../../03_diffusion_final/
```
