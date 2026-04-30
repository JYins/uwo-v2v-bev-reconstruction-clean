# Dataset Sanity Check

Checked on the local processed dataset:

```text
D:\MEng_Project\dataset_prepared
```

Google Drive Desktop copy:

```text
Google Drive Desktop folder / uwo-v2v-bev-reconstruction-dataset / dataset_prepared
```

## Size Check

| Location | Files | Size |
| --- | ---: | ---: |
| Local `dataset_prepared/` | 39429 | 151.73 GB |
| Google Drive Desktop copy | 39429 | 151.73 GB |

## Split Check

| Split | Scenes | Samples |
| --- | ---: | ---: |
| train_01 | 4 | 1953 |
| train_02 | 4 | 620 |
| train_03 | 4 | 951 |
| train_04 | 4 | 438 |
| train_05 | 4 | 764 |
| train_06 | 4 | 910 |
| train_07 | 4 | 896 |
| train_08 | 4 | 573 |
| val_01 | 3 | 748 |
| test_01 | 3 | 405 |
| test_02 | 3 | 688 |
| test_03 | 3 | 900 |

Final model split totals:

```text
train = 7105 samples
val   = 748 samples
test  = 1993 samples
```

## Tensor Shape Check

Every paired BEV file checked has the expected shape:

```text
ego_bev/*.npy      500 x 500 x 8
masked_ego/*.npy   500 x 500 x 8
neighbor_bev/*.npy 500 x 500 x 8
sector_mask.npy    500 x 500
```

So each BEV tensor is 8-channel. During training:

```text
input  = masked_ego(8ch) + neighbor_bev(8ch) = 16 channels
target = ego_bev(8ch)
mask   = 1 visible, 0 hidden
```

The project loader was also checked:

```text
Input shape:  [2, 16, 500, 500]
Target shape: [2, 8, 500, 500]
Mask shape:   [2, 1, 500, 500]
Input range:  [0.0000, 1.0000]
Target range: [0.0000, 1.0000]
Mask values:  [0.0, 1.0]
```

No missing required folders, frame-id mismatches, or invalid tensor shapes were found.
