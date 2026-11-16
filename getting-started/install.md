# Installation

Asparagus requires **Python ≥ 3.11** and a working CUDA-capable GPU environment.

## Standard Installation

For a local editable installation (recommended for development):

```bash
git clone https://github.com/Sllambias/asparagus
cd asparagus
pip install -e .
```

## HPC / Gefion Installation

For installing on the Gefion HPC cluster, follow the dedicated guide:

[HPC Configuration (Gefion) →](environment_on_gefion.md)

## Optional Dependency Groups

### Documentation dependencies
To build or contribute to the documentation:

```bash
pip install -e ".[docs]"
```

### Gefion / DCAI dependencies
For the Gefion cluster with DCAI-specific packages (pinned versions for compatibility):

```bash
pip install -e ".[dcai]"
```

## Verifying the Installation

After installation, the following CLI entry points should be available:

```bash
# Pretraining
asp_pretrain --help

# Training
asp_train_cls --help
asp_train_seg --help
asp_train_reg --help

# Finetuning
asp_finetune_cls --help
asp_finetune_seg --help
asp_finetune_reg --help

# Testing / Inference
asp_test_cls --help
asp_test_seg --help
asp_test_reg --help

# Evaluation
asp_eval_box_prepare_data --help
asp_eval_box_run --help
asp_eval_box_collect_results --help

# Utilities
asp_getid --help
```

## Next Steps

After installation, configure the required environment variables before running any training:

[Environment Setup →](environment_variables.md)
