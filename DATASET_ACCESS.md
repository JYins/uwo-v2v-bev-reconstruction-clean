# Dataset Access

The processed dataset is too large for Git.

Local size checked on 2026-04-29:

```text
dataset_prepared/  151.7 GB
V2V4Real/            2.4 GB
```

## Google Drive Copy

On 2026-04-29, the processed dataset was copied into the local Google Drive Desktop folder:

```text
Google Drive Desktop folder / uwo-v2v-bev-reconstruction-dataset / dataset_prepared
```

Local copy check:

```text
dataset_prepared/  39429 files, 151.73 GB
```

Shared Google Drive folder:

```text
Google Drive dataset_prepared link: https://drive.google.com/drive/folders/1nQuA6qQhK8wcshwI5CHk5BQDv4R4eAPs?usp=sharing
Google Drive raw V2V4Real link: optional
```

Upload helper after `rclone` is configured:

```powershell
.\tools\upload_dataset_to_drive_rclone.ps1 -Remote gdrive -DriveFolder uwo-v2v-bev-reconstruction-dataset
```

The script is resumable and does not change the Git repo. It uploads:

```text
<repo root>/dataset_prepared -> gdrive:uwo-v2v-bev-reconstruction-dataset/dataset_prepared
```

Recommended sharing target:

```text
dataset_prepared/
```

This is the exact folder needed by the training scripts.

For the checked file counts and tensor shapes, see:

```text
DATASET_SANITY_CHECK.md
```

## Expected Local Structure

The repo expects the processed dataset at:

```text
D:\MEng_Project\dataset_prepared
```

or, more generally:

```text
<repo root>/dataset_prepared
```

Expected split folders:

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

Each sample returns:

```text
input  = masked_ego(8ch) + neighbor_bev(8ch)
target = clean ego_bev(8ch)
mask   = 1 visible, 0 hidden
```

## Why It Is Not In Git

Git should contain code, configs, metrics, and figures. It should not contain:

- `dataset_prepared/`
- raw `V2V4Real/`
- model checkpoints
- local training runs

Those are ignored in `.gitignore`.

## Quick Check After Download

```powershell
python .\01_unet_final\dataset.py --dataset_root .\dataset_prepared --batch_size 2
```

This should print train/val/test split discovery and tensor shapes.
