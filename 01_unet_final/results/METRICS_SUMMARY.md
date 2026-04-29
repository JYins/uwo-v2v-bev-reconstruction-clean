# U-Net Final Metrics

These are refreshed from the confirmed seed42/43/44 checkpoints with the current metric code.

| Seed | Best epoch | Occ-IoU | Precision | Recall | F1 | RMSE | Fused PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 21 | 0.1500 | 0.2002 | 0.3745 | 0.2609 | 0.055882 | 33.67 |
| 43 | 24 | 0.1469 | 0.1967 | 0.3671 | 0.2562 | 0.055892 | 33.67 |
| 44 | 21 | 0.1513 | 0.2059 | 0.3636 | 0.2629 | 0.056063 | 33.64 |

| Aggregate | Occ-IoU | Precision | Recall | F1 | RMSE | Fused PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mean +/- std | 0.1494 +/- 0.0023 | 0.2009 +/- 0.0046 | 0.3684 +/- 0.0056 | 0.2600 +/- 0.0035 | 0.055945 +/- 0.000102 | 33.66 +/- 0.02 |

Main read: U-Net has the best masked Occ-IoU and the best balance of precision/recall among the final models.
