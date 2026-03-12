# Classification

Asparagus supports 2D and 3D medical image classification. This page covers the full lifecycle: training from scratch, fine-tuning from a pretrained model, running inference, and tuning hyperparameters.


## Training from Scratch

Train a classification model with the default configuration:

***CLI***
```bash
asp_train_cls task=Task004_Name \
    +model=unet_b \
    data.train_split=split_75_15_10
```

### With a custom config file

```bash
asp_train_cls --config-name my_cls_config
```

### Restarting a failed run

Find the `run_id` from the output directory or Hydra logs, then rerun with `run_id=<id>`:

```bash
asp_train_cls task=Task004_Name \
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
| `training.target_size` | Spatial size to resize inputs to |
| `training.epochs` | Total number of training epochs |
| `training.label_smoothing` | Label smoothing factor (0.0 = off) |
| `training.seed` | Random seed for reproducibility |
| `run_id` | Assign a specific run ID (optional) |

!!! info "Classification vs Segmentation"
    Classification uses a fixed `target_size` (whole-image resize) rather than a `patch_size` (patch-based). The model outputs a class score vector rather than a dense segmentation map.

### After Training

Training automatically runs inference on the test set using the **best** checkpoint (lowest validation loss) once training finishes. Predictions are saved to:

```
$ASPARAGUS_MODELS/<run_id>/predictions/<test_task>__<test_split>__best.json
```


## Finetuning from a Pretrained Model

Fine-tuning initialises the model from a pretrained checkpoint (e.g., from SSL pretraining or a related task).

***CLI (by run_id)***
```bash
asp_finetune_cls \
    task=Task004_Name \
    checkpoint_run_id=435850 \
    load_checkpoint_name=last.ckpt
```

***CLI (by path)***
```bash
asp_finetune_cls \
    task=Task004_Name \
    checkpoint_path=/path/to/model.ckpt
```


### Finetuning-specific parameters

| Parameter | Description |
|---|---|
| `checkpoint_run_id` | Run ID of the pretrained checkpoint |
| `checkpoint_path` | Absolute path to checkpoint (alternative to `run_id`) |
| `load_checkpoint_name` | Checkpoint filename (e.g., `last.ckpt`, `best.ckpt`) |
| `training.repeat_stem_weights` | Repeat stem weights for channel mismatch (e.g., 2D→3D) |
| `training.warmup_epochs` | Epochs to warm up the full model |
| `training.decoder_warmup_epochs` | Epochs to warm up only the decoder head |

!!! note
    Fine-tuning uses a separate learning rate (`model.finetune_lr`) and optimizer (`model.finetune_optim`) compared to training from scratch.


## Testing / Inference

Run inference on a held-out test set using a trained or fine-tuned checkpoint:

```bash
asp_test_cls \
    test_task=Task004_Name \
    checkpoint_run_id=1234 \
    load_checkpoint_name=best.ckpt \
    test_split=TEST_75_15_10
```



### Switching checkpoints

```bash
asp_test_cls \
    test_task=Task004_Name \
    checkpoint_run_id=1234 \
    load_checkpoint_name=epoch=9-step=50.ckpt \
    test_split=TEST_75_15_10
```

### Testing on a different dataset

```bash
asp_test_cls \
    test_task=Task005_OtherName \
    checkpoint_run_id=1234 \
    load_checkpoint_name=best.ckpt \
    test_split=TEST_75_15_10
```

### Outputs

Predictions are saved as a JSON file:

```
$ASPARAGUS_MODELS/<run_id>/predictions/<test_task>__<test_split>__<checkpoint_name>.json
```

## Evaluation at Scale

For evaluating across multiple tasks and checkpoints, use the EvalBox:

[The EvalBox →](../evaluation/evalbox.md)
