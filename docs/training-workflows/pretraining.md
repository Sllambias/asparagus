# Pretraining (Self-Supervised Learning)

## How It Works

Asparagus uses a **masked image modelling** strategy for SSL:

1. Each input image is divided into patches.
2. A random subset of patches is masked (hidden from the model).
3. The model must reconstruct the masked patches from the visible context.
4. Only the reconstruction loss on masked patches can be used (configurable via `training.rec_loss_masked_only`).

The masking ratio is set by `training.mask_ratio` (default typically ~0.75).


## Running Pretraining

### CLI (recommended)

```bash
asp_pretrain task=Task998_LauritSyn \
    +model=unet_b_lw_dec \
    data.train_split=split_75_15_10
```

### With a custom config file

```bash
asp_pretrain --config-name my_pretrain_config
```



## Key Configuration Parameters

| Parameter | Description | Default |
|---|---|---|
| `task` | Task name (must exist in `$ASPARAGUS_DATA`) | required |
| `+model` | Model architecture config to append | required |
| `data.train_split` | Split file name inside the task folder | required |
| `training.mask_ratio` | Fraction of patches to mask | ~0.75 |
| `training.epochs` | Total training epochs | configured in model |
| `training.warmup_ratio` | Fraction of epochs used for LR warmup | configured in model |
| `training.rec_loss_masked_only` | Only compute loss on masked patches | `True` |
| `training.patch_size` | 3D patch size for SSL training | configured in model |
| `run_id` | Assign a specific run ID (omit for auto) | auto-generated |



## Monitoring Training

Asparagus supports two logging backends:

### Weights & Biases
```bash
asp_pretrain task=Task998_LauritSyn +model=unet_b_lw_dec \
    logger.wandb_logging=true \
    logger.wandb_entity=my-team
```

### MLflow
```bash
asp_pretrain task=Task998_LauritSyn +model=unet_b_lw_dec \
    logger.mlflow_logging=true
```

Reconstructed images are logged every `logger.log_images_every_n_epoch` epochs so you can track reconstruction quality visually.



## Outputs

After a successful pretraining run, you will find in `$ASPARAGUS_MODELS/<run_id>/`:

```
<run_id>/
├── checkpoints/
│   └── last.ckpt       # Saved every ckpt_every_n_epoch epochs
├── hydra/
│   └── config.yaml     # Full resolved config for reproducibility
└── wandb/ or mlflow/   # Experiment logs
```



## Using the Pretrained Model

Use the `run_id` (or a direct checkpoint path) to initialise a downstream fine-tuning run:

```bash
# Finetune a segmentation model from pretraining run 4358
asp_finetune_seg task=Task004_Name \
    checkpoint_run_id=4358 \
    load_checkpoint_name=last.ckpt

# Or reference by path
asp_finetune_seg task=Task004_Name \
    checkpoint_path=$ASPARAGUS_MODELS/4358/checkpoints/last.ckpt
```

See the individual workflow pages for details:

- [Image Segmentation →](segmentation.md)
- [Classification →](classification.md)
- [Regression →](regression.md)
