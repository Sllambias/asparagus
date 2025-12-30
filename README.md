# Asparagus

# Table of Contents
- [Resources](#resources)
- [Getting Started](#getting-started)
  - [0. Setup](#0-setup)
  - [1. Preprocessing](#1-preprocessing)
  - [2. Data Splitting](#2-trainvaltest-splitting)
  - [3. Training](#3-training)
  - [4. Testing](#4-testing)

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

### Regular installation
To install the package see: [Installation](./docs/installation.md)
To set up required and optional environment variables see: [Environment Variables](./docs/environment_variables.md)

### DCAI/Gefion installation
To install Asparagus on Gefion, please refer to [Installing Asparagus environment on Gefion](./docs/environment_on_gefion.md).


## 1. Preprocessing 

See the [asparagus_preprocessing](https://github.com/Sllambias/asparagus_preprocessing) repository.

## 2. Training
Asparagus uses hydra configs (see [Hydra](https://hydra.cc/docs/intro/)) to set up all training runs. To use the default setup 3 args are required.
- The Task to train on
- The model architecture to use
- The data split to use

These can be given either by creating a config that specifies them see [Example Config 1](./docs/configs.md#example-config-1), or at runtime using the command line as illustrated below.


### 3.1 Run ids

When starting a new training run, Asparagus will assign the run a ´run_id´ which can be used to
1. restart a failed run
2. start a training from a pretrained model (however a checkpoint can also be loaded from a path)

### 3.2 Pretraining
- Training with the default pretraining config

```asp_pretrain task=Task998_LauritSyn +model=unet_b_lw_dec data.train_split=split_75_15_10```

### 3.3 Training (from scratch)
- Training a *segmentation* model with the default training config

 ```asp_train_seg task=Task997_LauritSynSeg +model=unet_tiny data.train_split=split_75_15_10```

- Training a *segmentation* model with a pre-defined config

```asp_train_seg --config-name seg_lauritsynseg_tiny```

- Training a *classification* model with a pre-defined config

```asp_train_cls --config-name cls_lauritsyncls_tiny```

- Training a *regression* model with a pre-defined config

```asp_train_reg --config-name reg_lauritsynreg_tiny```

- Restarting a failed run

Simply add `run_id=123` where `123` is your run id to the command used to start the job. For instance to restart the segmentation job with the pre-defined config from above, just do
```asp_train_seg --config-name seg_lauritsynseg_tiny run_id=123```

### 3.4 Finetuning

There are two ways of loading

Asparagus will create a "derived_models.log" in the folder of the pretrained model so you can always track which finetuned models and run_id's are its _children_.

- Finetuning a segmentation model using the default finetuning config

from a checkpoint with a run_id (typically trained on the same machine)

```asp_finetune_seg task=Task997_LauritSynSeg checkpoint_run_id=435850 load_checkpoint_name=last.ckpt```

or from a checkpoint given by a path

```asp_finetune_seg task=Task997_LauritSynSeg checkpoint_path=/path/to/model.ckpt```


- Finetuning a classification model from a versioned checkpoint using a pre-defined config

```asp_finetune_cls --config-name checkpoint_run_id=435850 ft_cls_lauritsyncls_tiny```
or
```asp_finetune_cls --config-name checkpoint_path=/path/to/run.ckpt ft_cls_lauritsyncls_tiny```


- Finetuning a regression model is currently not implemented.

## 4. Running a segmentation model on the test set

## Testing with hydra config

```
asp_test_seg test_task=Task997_LauritSynSeg checkpoint_run_id=1234 load_checkpoint_name=last.ckpt test_split=TEST_75_15_10
```

To change the checkpoint refer to its run_id and checkpoint name:

```
checkpoint_run_id: 532
load_checkpoint_name: epoch=4-step=25.ckpt
```


