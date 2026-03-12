
## Environment Variables

All environment variables should be put in a `.env` file in the project root which will be loaded automatically. Alternatively they can be set manually using `export VAR=VALUE` at the start of each session.

### Required Environment Variables:
```bash
ASPARAGUS_CONFIGS=/PATH/TO/CONFIG/DIR
ASPARAGUS_DATA=/PATH/TO/ASPARAGUS/DATA
ASPARAGUS_MODELS=/PATH/TO/ASPARAGUS/MODELS
ASPARAGUS_RESULTS=/PATH/TO/ASPARAGUS/RESULTS
ASPARAGUS_RAW_LABELS=/PATH/TO/ASPARAGUS/RAW_LABELS
ASPARAGUS_SOURCE=/PATH/TO/RAW/DATASETS
```

- `ASPARAGUS_CONFIGS` points to the directory containing the default configs found in asparagus/configs. This variable is used in the training and inference scripts.
- `ASPARAGUS_DATA` points to the directory containing preprocessed data. Preprocessing scripts must save processed data here. Data is stored as tensors: pretraining data as `[C, H, W, D]`, segmentation data as `[C+1, H, W, D]` (label is always the last channel), and classification/regression data as a list `[tensor, label]`. The 2D equivalents omit the D dimension.
- `ASPARAGUS_MODELS` points to the directory containing model checkpoints, test outputs (results only), logs and metadata.
- `ASPARAGUS_RESULTS` points to the directory containing predict outputs (output files and results).
- `ASPARAGUS_RAW_LABELS` points to the directory where preprocessing scripts save *formatted but unpreprocessed* labels. *Formatting* (e.g. remapping label values) should be applied; *preprocessing* (e.g. resampling to a target spacing) should not. These unmodified labels are used when computing final test metrics.
- `ASPARAGUS_SOURCE` points to the directory containing raw, unprocessed datasets. Preprocessing scripts use this variable to locate source data via `get_source_path()` from `asparagus_preprocessing.paths`, keeping scripts portable across users and environments.

### Optional Environment Variables 
(supports multiple colon-separated (":") paths):

```bash
ASPARAGUS_PRETRAIN_CONFIGS=/PATH1/TO/ADDITIONAL/PRETRAIN/CONFIG/DIR:PATH2/TO/ADDITIONAL/PRETRAIN/CONFIG/DIR:PATH3...
ASPARAGUS_TRAIN_CONFIGS=/PATH1/TO/ADDITIONAL/TRAIN/CONFIG/DIR:PATH2/TO/ADDITIONAL/TRAIN/CONFIG/DIR:PATH3...
ASPARAGUS_FINETUNE_CONFIGS=/PATH1/TO/ADDITIONAL/FINETUNE/CONFIG/DIR:PATH2/TO/ADDITIONAL/FINETUNECONFIG/DIR:PATH3...
```

- `ASPARAGUS_PRETRAIN_CONFIGS` points to all directories containing pretraining configs. When `asp_pretrain --config-name MyFancyPretrainConfig`is called all the given `ASPARAGUS_PRETRAIN_CONFIGS` directories are scanned for MyFancyPretrainConfig.yaml.
- `ASPARAGUS_TRAIN_CONFIGS` points to all directories containing training from scratch configs. When `asp_train_[cls,reg,seg] --config-name MyLovelyTrainConfig`is called all the given `ASPARAGUS_PRETRAIN_CONFIGS` directories are scanned for MyLovelyTrainingConfig.yaml.
- `ASPARAGUS_FINETUNE_CONFIGS` points to all directories containing finetuning configs. When `asp_finetune_[cls,reg,seg] --config-name MyAwesomeFinetuneConfig`is called all the given `ASPARAGUS_PRETRAIN_CONFIGS` directories are scanned for MyAwesomeFinetuneConfig.yaml.
