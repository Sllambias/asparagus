PRETRAIN

| it/s | GPU(s) | Workers | Model         | Augmentations       | Plugins (time)    | Patch Size |
| ---- | ------ | ------- | ------------- | ------------------- | ----------------- | ---------- |
| 8.42 | A100   | 14      | UNet_b_lw_dec | None                | N/A               | 128*3      |
| 8.60 | A100   | 28      | UNet_b_lw_dec | None                | N/A               | 128*3      |
| 3.25 | A100   | 14      | UNet_b_lw_dec | all: 0.3 @ CPU      | N/A               | 128*3      |
| 6.50 | A100   | 28      | UNet_b_lw_dec | all: 0.3 @ CPU      | N/A               | 128*3      |
| 7.97 | A100   | 44      | UNet_b_lw_dec | all: 0.3 @ CPU      | N/A               | 128*3      |
| 6.35 | A100   | 14      | UNet_b_lw_dec | all: 0.3 @ GPU      | N/A               | 128*3      |
| 6.85 | A100   | 28      | UNet_b_lw_dec | all: 0.3 @ GPU      | N/A               | 128*3      |
| 7.04 | A100   | 14      | UNet_b_lw_dec | onlyspa: 0.3 @ GPU  | N/A               | 128*3      |
| 9.55 | A100   | 14      | UNet_b_lw_dec | nospa: 0.3 @ GPU    | N/A               | 128*3      |
| 6.98 | A100   | 14      | UNet_b_lw_dec | nodeform: 0.3 @ GPU | N/A               | 128*3      |
| 7.05 | A100   | 14      | UNet_b_lw_dec | norot: 0.3 @ GPU    | N/A               | 128*3      |
| 7.08 | A100   | 14      | UNet_b_lw_dec | noscale: 0.3 @ GPU  | N/A               | 128*3      |
| 6.40 | A100   | 14      | UNet_b_lw_dec | spa0: 0.3 @ GPU     | N/A               | 128*3      |
| 9.40 | A100   | 28      | UNet_b_lw_dec | spaCPU: 0.3 @ GPU   | N/A               | 128*3      |
| 8.98 | A100   | 14      | UNet_b_lw_dec | spaCPU: 0.3 @ GPU   | N/A               | 128*3      |
| 9.60 | A100   | 44      | UNet_b_lw_dec | spaCPU: 0.3 @ GPU   | N/A               | 128*3      |
| 9.60 | A100   | 44      | UNet_b_lw_dec | spaCPU: 0.3 @ GPU   | 12it/s (3600/300) | 128*3      |

SEGMENTATION

| it/s | GPU(s) | Workers | Model  | Augmentations | Plugins (time) | Patch Size |
| ---- | ------ | ------- | ------ | ------------- | -------------- | ---------- |
| 5.86 | A100   | 14      | UNet_b | N/A           | N/A            | 128*3      |
| 6.60 | A100   | 14      | UNet_b | N/A           | N/A            | 128*3      |

CLASSIFICATION

| it/s  | GPU(s) | Workers | Model         | Augmentations | Plugins (time) | Image Size |
| ----- | ------ | ------- | ------------- | ------------- | -------------- | ---------- |
| 16.10 | A100   | 14      | UNet_clsreg_b | N/A           | N/A            | 128*3      |

