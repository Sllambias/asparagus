# Configs
All experiments in Asparagus are configured using Hydra configs. Configs can be mixed and matched arbitrarily in configs or during runtime. Then, at runtime all the given configs are evaluated (according to the given order) and resolved to start the experiment.

## Editing configs.
### CLI editing
- To add a full config use "+": ```+FOLDER=CONFIG```. I.e. to add the resenc_unet config to your run use +model=resenc_unet
- To edit an existing parameter: ```parameter=value```. I.e. to change the model dimensions to 2D: `model.dimension=2D`

### yaml editing
- To add a full config to your experiment config, you add it under the defaults, optionally along with a prefix for its parameters (set by whatever comes after @). In the example below we import (1) the core/base.yaml config without a prefix, which means a param ```n_llamas=4``` in core/base.yaml will be accessible as ```n_llamas```in the resolved experiment config and (2) hardware/1gpu12cpu.yaml with the prefix ```hardware```, which means a param ```num_workers=12``` in hardware/1gpu12cpu will be accessible as ```hardware.num_workers``` in the resolved experiment config: 
```
defaults:
  - core/base@
  - hardware/1gpu12cpu@hardware
```
- To edit an existing parameter in your experiment config simply overwrite its value. Say we imported the configs above and wanted to reuse all variables in hardware/1gpu12cpu except the number of num_workers:
```
defaults:
  - core/base@
  - hardware/1gpu12cpu@hardware

hardware.num_workers=16
```

## Dry-run config setup
To dry-run your experiment and view the result of your config-composition add the following argument to your CLI call: ```--cfg job```

## Use pre-defined config.
CLI change config: ```--config-name name_of_config```
For detailed control of the configs see also the core/base config.

## Enable Debug prints
- Add `HYDRA_FULL_ERROR=1` to the command line args so it becomes `HYDRA_FULL_ERROR=1 asp_train_X ....` 

## Rerun previously executed jobs
```asp_[pretrain,train,etc.] --experimental-rerun $OUTPUT_DIR/config.pickle```

## Example Config 1: The Complete Config
In this example we configure a config yaml that contains everything needed:

- Add 3 predefined configs (configs/default_pretrain.yaml, configs/hardware/cpu.yaml, configs/model/unet_b_lw_dec.yaml)
- Specify the Task to Pretrain on
- Specify the data split to use for training.

```bash
# @package _global_
defaults:
  - /default_pretrain@
  - /hardware/cpu@hardware
  - /model/unet_b_lw_dec@model
  - _self_

task: Task001_MyAlpacaDataset

data:
  train_split: split_75_15_10
```

Since this pretraining config, which we will save as ```MyCompleteConfig.yaml```, already contains all the required variables (either through importing other configs or declaring them) pretraining can commence by:
```asp_pretrain --config-name MyCompleteConfig```

## Example Config 2: The Incomplete Config
In this example we configure a config yaml that contains the constant pieces of an experiment and omits the variables under investigation to let them be defined at runtime. In this example we will omit the model architecture config and the training split parameter, but it could be any (sub)config or parameter:

- Add 2 predefined configs (configs/default_pretrain.yaml, configs/hardware/cpu.yaml)
- Specify the Task to Pretrain on
- Specify the data split to use for training.

```bash
# @package _global_
defaults:
  - /default_pretrain@
  - /hardware/cpu@hardware
  - _self_

task: Task001_MyAlpacaDataset
```

Since this pretraining config, which we will save as ```MyIncompleteConfig.yaml```, contains most required variables pretraining can commence by declaring the config and the missing variables:
```asp_pretrain --config-name MyIncompleteConfig +model=primus_m data.train_split=split_99_01_00```

## Advanced use
Generally users will find what they're looking for in the default training and testing configs such as ```default_finetune_seg.yaml```, however many behind-the-scenes things are controlled by the core/base.yaml, such as how the save paths are resolved, what is ignored, which parameters are automatically given to the lightning.trainer. For advanced use familiarity with this config may be relevant. 