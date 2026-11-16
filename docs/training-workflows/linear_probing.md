# Linear Probing

Linear probing measures how useful an encoder's features are for classification. Asparagus freezes the encoder, keeps it in evaluation mode, and trains linear classification heads on globally averaged features from its final layer. Unlike [fine-tuning](classification.md#finetuning-from-a-pretrained-model), this leaves the encoder weights unchanged.

`asp_linear_probe` uses `configs/default_linear_probe.yaml`. It trains one head per candidate learning rate in a single run, selects the head with the highest validation **macro AUROC**, and automatically evaluates that head on the test set.

## Prerequisites

- [Install Asparagus](../getting-started/install.md) and configure the [environment variables](../getting-started/environment_variables.md), including `ASPARAGUS_CONFIGS`, `ASPARAGUS_DATA`, and `ASPARAGUS_MODELS`.
- Prepare a [classification dataset and split files](../data-pipeline/data_structure.md). Each image must have a single integer class label; task metadata must include `n_classes` and `n_modalities`.
- Choose a classification model config compatible with your pretrained encoder. The example below uses `resenc_unet_b_clsreg` for a matching residual UNet encoder. Custom models must accept `late_fusion=True` and expose `_encode()` features suitable for spatial average pooling.
- Provide both a train/validation split and a held-out test split. The command runs testing automatically, so set `data.test_split` explicitly to an existing split file name, without `.json`.

## Run a Probe

Load a pretrained checkpoint by its run ID:

```bash
asp_linear_probe \
    task=Task004_Name \
    +model=resenc_unet_b_clsreg \
    checkpoint_run_id=435850 \
    load_checkpoint_name=last.ckpt \
    data.train_split=split_75_15_10 \
    data.test_split=TEST_75_15_10
```

Replace the task, checkpoint run ID, model config, and split names with your own. Match `model.dimensions` and `training.target_size` to the data and encoder; defaults are `3D` for this model and `[160,160,160]` for the probe's target size.

To load a checkpoint directly, replace `checkpoint_run_id` and `load_checkpoint_name` with a path:

```bash
asp_linear_probe \
    task=Task004_Name \
    +model=resenc_unet_b_clsreg \
    checkpoint_path=/path/to/last.ckpt \
    data.train_split=split_75_15_10 \
    data.test_split=TEST_75_15_10
```

You can also set `hf_model_id=owner/model` for an Asparagus-compatible Hugging Face checkpoint, with `hf_weight_format` configured if a different weight mapper is needed. Use only one checkpoint source: `checkpoint_run_id`, `checkpoint_path`, or `hf_model_id`. If none is supplied, the probe uses a randomly initialized, frozen encoder as a baseline.

Weights are loaded into the encoder without loading the pretrained decoder. Check the weight-loading output to confirm that the intended encoder weights were loaded.

## Configure Training

Use Hydra overrides to adjust the learning-rate candidates and training settings:

```bash
asp_linear_probe \
    task=Task004_Name \
    +model=resenc_unet_b_clsreg \
    checkpoint_run_id=435850 \
    data.train_split=split_75_15_10 \
    data.test_split=TEST_75_15_10 \
    training.probing.learning_rates='[1e-4,1e-3,1e-2]' \
    training.epochs=20 \
    training.check_val_every_n_epoch=1 \
    training.batch_size=8 \
    training.target_size='[160,160,160]' \
    training.seed=42 \
    +hardware=1gpu12cpu
```

| Parameter | Purpose | Default |
|---|---|---|
| `training.probing.learning_rates` | Candidate initial learning rates; one head per value | `[5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1]` |
| `training.epochs` | Training epochs | `15` |
| `training.check_val_every_n_epoch` | Validation interval in epochs | `2` |
| `training.batch_size` | Training and validation batch size | `4` |
| `training.accumulate_grad_batches` | Batches accumulated per optimizer step | `2` |
| `training.target_size` | Spatial crop size | `[160,160,160]` |
| `training.pretrained_target_size` | Original spatial size when positional embeddings need resizing | unset |
| `training.seed` | Random seed | randomly generated |
| `data.fold` | Fold index in the train/validation split | `0` |
| `+hardware` | Hardware config to append | `1gpu40cpu` |
| `logger.wandb_logging` | Enable Weights & Biases logging | `True` |

Use a non-empty list of distinct learning-rate candidates. Heads use SGD with momentum `0.9`, no weight decay, and cosine learning-rate decay. The probe uses `training.probing.learning_rates` rather than `model.finetune_lr` or `model.train_lr`.

Validation runs before training and at the configured interval. Each validation pass selects the highest-AUROC head at that point; testing uses the selected head's current weights after training. Set `training.check_val_every_n_epoch=1` to ensure selection also runs at the end of the final epoch.

Weights & Biases logging is enabled by default under the `LinearProbe` project. Configure your account as for other training workflows, or pass `logger.wandb_logging=false` to disable it.

To inspect the composed configuration without training, append `--cfg job` to your command. For reusable YAML configs and `--config-name`, see the [Config Reference](configs.md).

## Results

The run prints its run ID and full run directory under `$ASPARAGUS_MODELS`. Predictions are written to:

```text
<run_dir>/predictions/<test_task>__<data.test_split>__linear_probe.json
```

`test_task` defaults to `task`. The JSON contains the predicted class and label for each test sample, aggregate `AUROC_macro`, `AUPRC_macro`, and `F1_macro` metrics, and the selected `best_head` and `best_head_lr`. Per-head training and validation metrics and `val/best_head_auroc` are logged during the run.

Linear probing disables model checkpoint saving. The results describe an encoder evaluation; the trained heads are not saved for later use with `asp_test_cls`.
