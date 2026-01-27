# Asparagus

Welcome to Asparagus. 

Asparagus is a modular machine learning framework designed for medical imaging applications. The codebase is built on PyTorch, Lightning and Hydra and handles model training, experiment management, and evaluation with prepared data. To prepare your data see [Preprocessing](#1-preprocessing). The functional and modules directories follow the structure of torch.nn.functional and torch.nn with the former being stateless functions and the latter being classes that combine and build upon these with the logic and conventions adopted by Asparagus from PyTorch, Lightning, and Hydra.

# Table of Contents
- [Resources](#resources)
- [Getting Started](#getting-started)
  - [0. Setup](#0-setup)
  - [1. Preprocessing](#1-preprocessing)
  - [2. Training](#3-training)
  - [3. Testing](#4-testing)
- [Contributing](#contributing)

# Resources
- [Common Issues](./docs/common_issues.md)
- [Configs](./docs/configs.md)
- [Environment Variables](./docs/environment_variables.md)
- [Evaluation Box](./docs/EvalBox.md)
- [Hacking Asparagus](./docs/hacking_asparagus.md)
- [Installation](./docs/installation.md)
- [Training Time References](./docs/training_times.md)

# Getting Started

## 0. Setup

### Package installation
For a standard local editable installation: 
```git clone https://github.com/Sllambias/asparagus
cd asparagus
pip install -e .
```

To install Asparagus on Gefion, please refer to [Installing Asparagus environment on Gefion](./docs/environment_on_gefion.md).

### Environment variables
To set up required and optional environment variables see: [Environment Variables](./docs/environment_variables.md)

## 1. Preprocessing 
See the [asparagus_preprocessing](https://github.com/Sllambias/asparagus_preprocessing) repository.

## 2. Training
Asparagus uses hydra configs (see [Hydra](https://hydra.cc/docs/intro/)) to configure training runs. To use the default setup 3 args are required.
- The Task to train on
- The model architecture to use
- The data split to use

These can be given either by creating a config that specifies them see [Example Config 1](./docs/configs.md#example-config-1-the-complete-config), or at runtime using the command line as illustrated below.

When starting a new training run, Asparagus will assign the run a unique ´run_id´ which can be used to:
1. restart a failed run
2. start a training from a pretrained model (however a checkpoint can also be loaded from a path)

### 2.1 Pretraining
- Pretraining with the default pretraining config

```asp_pretrain task=Task998_LauritSyn +model=unet_b_lw_dec data.train_split=split_75_15_10```

### 2.2 Training from scratch
- Training a classification/regression/segmentation model with the default training config

 ```asp_train_[cls,reg,seg] task=Data004_Parrots +model=unet_tiny data.train_split=split_75_15_10```

- Training a classification/regression/segmentation model with a pre-defined config

```asp_train_[cls,reg,seg] --config-name my_dummy_config```

- Restarting a failed run

To restart a job find the ```run_id``` of the job you want to restart. It will be present in the path and in the hydra output files. Then, rerun the job with the exact same command and configs and add `run_id=123` (where `123` is your run id). Asparagus will detect the existing model folder and load the most recent weights to continue from these.

### 2.3 Finetuning

Finetuning largely follows the same structure as training except 

- Finetuning a classification/regression/segmentation model using the default finetuning config from a checkpoint with a run_id 

```asp_finetune_[cls,reg,seg] task=Data004_Parrots checkpoint_run_id=435850 load_checkpoint_name=last.ckpt```

or from a checkpoint given by a path

```asp_finetune_[cls,reg,seg] task=Data004_Parrots checkpoint_path=/path/to/model.ckpt```

## 3. Testing

## Testing with hydra config

```
asp_test_seg test_task=Task997_LauritSynSeg checkpoint_run_id=1234 load_checkpoint_name=last.ckpt test_split=TEST_75_15_10
```

To change the checkpoint refer to its run_id and checkpoint name:

```
checkpoint_run_id: 532
load_checkpoint_name: epoch=4-step=25.ckpt
```

# Contributing
Contributing to the main branch of asparagus follows regular pull request standards. 
- Open pull request. 
- Pass linting, formatting and testing checks. Asparagus repositories use [ruff](https://docs.astral.sh/ruff/) for formatting and linting. To lint locally intall ruff and run ```ruff check --fix``` in the Asparagus directory (omitting --fix for a dry run) and to format locally run ```ruff format``` in the Asparagus directory. Running tests locally can be achieved with pytest. After checks 
- Request review from relevant moderators. If you're in doubt ask the repository owner.
- (if relevant) apply requested changes
- Acquire approval
- Merge
