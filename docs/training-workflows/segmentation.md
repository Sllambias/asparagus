# Segmentation

## Training from Scratch

Training a segmentation model with the default configuration:

 ***CLI***

```bash
asp_train_seg task=Task004_Name \
    +model=unet_b \
    data.train_split=split_75_15_10
```



### With a custom config file

```bash
asp_train_seg --config-name my_seg_config
```

### Restarting a failed run

Find the `run_id` from the output directory or Hydra logs, then rerun with `run_id=<id>`:

```bash
asp_train_seg task=Task004_Name \
    +model=unet_b \
    data.train_split=split_75_15_10 \
    run_id=123
```

### Key training parameters

| Parameter | Description |
|---|---|
| `task` | Task name (folder in `$ASPARAGUS_DATA`) |
| `+model` | Model architecture config |
| `data.train_split` | Split file name |
| `training.patch_size` | 3D patch size for patch-based training |
| `training.epochs` | Total number of training epochs |
| `training.seed` | Random seed for reproducibility |
| `run_id` | Assign a specific run ID (optional) |


## Finetuning from a Pretrained Model

Fine-tuning initialises the model encoder (and optionally the decoder) from a pretrained checkpoint.

***CLI (by run_id)***
```bash
asp_finetune_seg \
    task=Task004_Name \
    checkpoint_run_id=435850 \
    load_checkpoint_name=last.ckpt
```

***CLI (by path)***
```bash
asp_finetune_seg \
    task=Task004_Name \
    checkpoint_path=/path/to/model.ckpt
```

### Finetuning-specific parameters

| Parameter | Description |
|---|---|
| `checkpoint_run_id` | Run ID of the pretrained checkpoint |
| `checkpoint_path` | Absolute path to checkpoint (alternative to `run_id`) |
| `load_checkpoint_name` | Checkpoint filename (e.g., `last.ckpt`, `best.ckpt`) |
| `training.repeat_stem_weights` | Repeat stem weights to adapt 2D→3D or channel mismatch |

!!! note
    Fine-tuning uses a separate, typically lower learning rate (`model.finetune_lr`) and optimizer (`model.finetune_optim`) compared to training from scratch.



## Testing / Inference

Run inference on a held-out test set using a trained or fine-tuned checkpoint:

***CLI***
```bash
asp_test_seg \
    test_task=Task004_Name \
    checkpoint_run_id=1234 \
    load_checkpoint_name=last.ckpt \
    test_split=TEST_75_15_10
```


### Switching checkpoints

To test with a specific checkpoint epoch:

```bash
asp_test_seg \
    test_task=Task004_Name \
    checkpoint_run_id=1234 \
    load_checkpoint_name=epoch=4-step=25.ckpt \
    test_split=TEST_75_15_10
```

### Outputs

Predictions are saved to `$ASPARAGUS_MODELS/<run_id>/predictions/` as NIfTI files, mapped back to the original image coordinate space via reverse preprocessing.


## Evaluation at Scale

For evaluating across multiple tasks and checkpoints, use the EvalBox:

[The EvalBox →](../evaluation/evalbox.md)
