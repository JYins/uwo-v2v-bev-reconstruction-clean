# Data Pipeline

This folder contains the code that turns V2V4Real point clouds into the prepared 8-channel BEV tensors used by all final models.

The important current script is:

```text
code/prepare_dataset.py
```

The older exploratory script is kept too:

```text
code/multi_height_bin_bev.py
```

I keep it because it records the early BEV design decisions and visualization work. For a clean rerun, use `prepare_dataset.py`.

## Output Format

The final training code expects:

```text
dataset_prepared/
  train_*/
    <scene>/
      ego_bev/*.npy
      masked_ego/*.npy
      neighbor_bev/*.npy
      sector_mask.npy
  val_*/
  test_*/
```

Each `.npy` BEV has shape:

```text
500 x 500 x 8
```

Channel meaning:

```text
0-3 density channels for four height bins
4-7 max-height channels for the same bins
```

## Fixed BEV Choices

```text
Range:       x,y in [-40m, 40m]
Resolution: 0.16 m / pixel
Grid:       500 x 500
Height:     [-3m, 2m], split into 4 bins
```

These settings are intentionally fixed across U-Net, Pix2Pix, and Diffusion so the comparison is fair.

## Run

From the repo root:

```powershell
python .\00_data_pipeline\code\prepare_dataset.py --project_root . --output_root .\dataset_prepared
```

The processed dataset is not committed. See `DATASET_ACCESS.md`.
