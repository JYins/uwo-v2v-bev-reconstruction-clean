# Pix2Pix Experiment History

The full old Pix2Pix folders are kept locally under:

```text
old_versions/pix2pix/local_runs/
```

They are ignored by Git because they contain local training outputs.

Important groups:

```text
training_pix2pix_smoke_seed42/
training_pix2pix_smoke_v2_seed42/
optuna_pix2pix_adv/
```

## What I Learned

- A GAN term is not automatically better for this task.
- `lambda_adv=1.0` was too strong and pushed the model away from accurate masked reconstruction.
- The final search selected `lambda_adv=0.1`, so the adversarial term is only a small extra pressure.

Final Pix2Pix:

```text
../../02_pix2pix_final/
```
