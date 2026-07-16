# Regression

Asparagus supports 2D and 3D medical image regression — predicting continuous-valued targets (e.g., age, brain volume, biomarker scores) from medical images. This page covers the full lifecycle: training from scratch, fine-tuning from a pretrained model, running inference, and tuning hyperparameters.


## Training from Scratch

Train a regression model with the default configuration:

***CLI***
```bash
asp_train_reg task=Task004_Name \
    +model=unet_b \
    data.train_split=split_75_15_10
```

***Python module***
```bash
python -m asparagus.pipeline.run.train_reg \
    task=Task004_Name \
    +model=unet_b \
    data.train_split=split_75_15_10
```

### With a custom config file

```bash
asp_train_reg --config-name my_reg_config
```

### Restarting a failed run

Find the `run_id` from the output directory or Hydra logs, then rerun with `run_id=<id>`:

```bash
asp_train_reg task=Task004_Name \
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
| `training.seed` | Random seed for reproducibility |
| `run_id` | Assign a specific run ID (optional) |

!!! info "Regression vs Classification"
    Regression and classification share the same model architecture backbone (`unet_clsreg`). The main difference is in the output head and the loss function — regression uses continuous targets and a regression loss (e.g., MSE), while classification uses discrete targets and a cross-entropy-based loss.

### After Training

Training automatically runs inference on the test set using the **best** checkpoint (lowest validation loss) once training finishes. Predictions are saved to:

```bash
$ASPARAGUS_MODELS/<run_id>/predictions/<test_task>__<test_split>__best.json
```


## Finetuning from a Pretrained Model

Fine-tuning initialises the model from a pretrained checkpoint (e.g., from SSL pretraining or a related task).

***CLI (by run_id)***
```bash
asp_finetune_reg \
    task=Task004_Name \
    checkpoint_run_id=435850 \
    load_checkpoint_name=last.ckpt
```

***CLI (by path)***
```bash
asp_finetune_reg \
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

Regression testing follows the same pattern as classification. Use `asp_test_cls` with your regression task:

***CLI***
```bash
asp_test_cls \
    test_task=Task004_Name \
    checkpoint_run_id=1234 \
    load_checkpoint_name=best.ckpt \
    test_split=TEST_75_15_10
```

!!! note "No dedicated test_reg script"
    There is currently no separate `test_reg.py` script. Because classification and regression share the same architecture and inference mechanism, `asp_test_cls` / `python -m asparagus.pipeline.run.test_cls` is used for both task types.

### Outputs

Predictions are saved as a JSON file:

```
$ASPARAGUS_MODELS/<run_id>/predictions/<test_task>__<test_split>__<checkpoint_name>.json
```


## Evaluation at Scale

For evaluating across multiple tasks and checkpoints, use the EvalBox:

[The EvalBox →](../evaluation/evalbox.md)
