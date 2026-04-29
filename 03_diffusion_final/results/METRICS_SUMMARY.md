# Diffusion v3 Final Metrics

| Seed | Best epoch | Occ-IoU | Precision | Recall | F1 | RMSE | Fused PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 30 | 0.0467 | 0.0467 | 1.0000 | 0.0893 | 0.381013 | 17.00 |

Main read: Diffusion v3 predicts almost everything as occupied in the masked region. Recall is 1.0, but precision is very low, so this is a corrected negative baseline.
