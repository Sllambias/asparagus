
Asparagus is a modular machine learning framework designed for medical imaging applications. The codebase is built on PyTorch, Lightning and Hydra and handles model training, experiment management, and evaluation with prepared data. To prepare your data see [Preprocessing](data-pipeline/preprocessing.md). The functional and modules directories follow the structure of `torch.nn.functional` and `torch.nn` with the former being stateless functions and the latter being classes that combine and build upon these with the logic and conventions adopted by Asparagus from PyTorch, Lightning, and Hydra.


## Quick Navigation

| Section | Description |
|---|---|
| [Installation](getting-started/install.md) | Set up Asparagus in your environment |
| [Environment Variables](getting-started/environment_variables.md) | Configure required paths and settings |
| [Data Pipeline](data-pipeline/preprocessing.md) | Prepare data for training |
| [Training Workflows](training-workflows/pretraining.md) | Run pretraining, fine-tuning, and testing |
| [EvalBox](evaluation/evalbox.md) | Evaluate models at scale |
| [Common Issues](advanced/common_issues.md) | Troubleshooting guide |


## Getting Started

### 1. Setup

#### Package installation
For a standard local editable installation:

```bash
git clone https://github.com/Sllambias/asparagus
cd asparagus
pip install -e .
```

To install Asparagus on Gefion, please refer to [Installing Asparagus environment on Gefion](getting-started/environment_on_gefion.md).

#### Environment variables
To set up required and optional environment variables see: [Environment Variables](getting-started/environment_variables.md)

### 2. Preprocessing
See the [asparagus_preprocessing](https://github.com/Sllambias/asparagus_preprocessing) repository.

### 3. Training
Asparagus uses Hydra configs (see [Hydra](https://hydra.cc/docs/intro/)) to configure training runs. To use the default setup, 3 args are required:

- The **Task** to train on
- The **model architecture** to use
- The **data split** to use

These can be given either by creating a config that specifies them (see [Config Reference](training-workflows/configs.md#example-config-1-the-complete-config)), or at runtime using the command line as illustrated below.

When starting a new training run, Asparagus will assign the run a unique `run_id` which can be used to:

1. Restart a failed run
2. Start a training from a pretrained model (however a checkpoint can also be loaded from a path)

#### 3.1 Pretraining
Pretraining with the default pretraining config:

```bash
asp_pretrain task=PT003_LauritSyn \
    +model=unet_b_lw_dec \
    data.train_split=split_75_15_10
```

#### 3.2 Training from scratch
Training a classification/regression/segmentation model with the default training config:

```bash
asp_train_cls task=CLS004_Parrots \
    +model=unet_tiny \
    data.train_split=split_75_15_10

asp_train_reg task=REGR004_Parrots \
    +model=unet_tiny \
    data.train_split=split_75_15_10

asp_train_seg task=SEG004_Parrots \
    +model=unet_tiny \
    data.train_split=split_75_15_10
```

Training with a pre-defined config:

```bash
asp_train_cls --config-name my_dummy_config
```

Restarting a failed run — find the `run_id` and rerun with the same command plus `run_id=123`:

```bash
asp_train_cls task=CLS004_Parrots \ 
    +model=unet_tiny \ 
    data.train_split=split_75_15_10 \ 
    run_id=123
```

#### 3.3 Finetuning
Finetuning from a checkpoint with a `run_id`:

```bash
asp_finetune_cls task=CLS004_Parrots \
    checkpoint_run_id=435850 \
    load_checkpoint_name=best.ckpt

asp_finetune_seg task=SEG004_Parrots \
    checkpoint_run_id=435850 \
    load_checkpoint_name=best.ckpt

asp_finetune_reg task=REGR004_Parrots \
    checkpoint_run_id=435850 \
    load_checkpoint_name=best.ckpt
```

Or from a direct checkpoint path:

```bash
asp_finetune_cls task=CLS004_Parrots \ 
    checkpoint_path=/path/to/model.ckpt
```

### 4. Testing

```bash
asp_test_seg task=SEG004_Parrots \
    checkpoint_run_id=1234 \
    load_checkpoint_name=last.ckpt \
    test_split=TEST_75_15_10

asp_test_cls task=CLS004_Parrots \
    checkpoint_run_id=1234 \
    load_checkpoint_name=best.ckpt \
    test_split=TEST_75_15_10

asp_test_reg task=REGR004_Parrots \
    checkpoint_run_id=1234 \
    load_checkpoint_name=best.ckpt \
    test_split=TEST_75_15_10
```


## Contributing

See [Contributing](project/contributing.md) for full details on the PR workflow, branch strategy, and code style requirements.
