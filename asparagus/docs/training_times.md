PRETRAIN

| it/s | GPU(s) | Workers | Model         | Augmentations | Plugins (time) | Patch Size |
| ---- | ------ | ------- | ------------- | ------------- | -------------- | ---------- |
| 6.08 | a40    | 14      | UNet_b_lw_dec | N/A           | mem leak       | 128*3      |
| 8.55 | A100   | 14      | UNet_b_lw_dec | N/A           | N/A            | 128*3      |

SEGMENTATION

| it/s | GPU(s) | Workers | Model  | Augmentations | Plugins (time) | Patch Size |
| ---- | ------ | ------- | ------ | ------------- | -------------- | ---------- |
| 5.86 | A100   | 14      | UNet_b | N/A           | N/A            | 128*3      |
| 6.60 | A100   | 14      | UNet_b | N/A           | N/A            | 128*3      |

CLASSIFICATION

| it/s  | GPU(s) | Workers | Model         | Augmentations | Plugins (time) | Image Size |
| ----- | ------ | ------- | ------------- | ------------- | -------------- | ---------- |
| 16.10 | A100   | 14      | UNet_clsreg_b | N/A           | N/A            | 128*3      |

