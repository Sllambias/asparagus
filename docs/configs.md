## Example Config 1
Here we:
- Add 4 predefined configs (configs/core/base.yaml, configs/default.yaml, configs/hardware/cpu.yaml, configs/model/unet_b_lw_dec.yaml)
- Specify the Task to Pretrain on
- Specify the data split to use for Pretraining.
```bash
# @package _global_
defaults:
  - /core/base@ 
  - /default_pretrain@
  - /hardware/cpu@hardware
  - /model/unet_b_lw_dec@model
  - _self_

task: Task998_LauritSyn

data:
  train_split: split_75_15_10
```

# Config tips.

## Dry-run config setup
CLI view config without running script: ```--cfg job```

## Use pre-defined config.
CLI change config: ```--config-name name_of_config```
For detailed control of the configs see also the core:base config.

## CLI setup
- To add an entire config use "+": ```+FOLDER=CONFIG```. I.e. to add the resenc_unet config to your run use +model=resenc_unet
- To set a single parameter: ```parameter=value```. I.e. to change the model dimensions to 2D: `model.dimension=2D`

## Enable Debug prints
- Add `HYDRA_FULL_ERROR=1` to the command line args so it becomes `HYDRA_FULL_ERROR=1 asp_train_X ....` 

## Rerun previously executed jobs
```asp_[pretrain,train,etc.] --experimental-rerun $OUTPUT_DIR/config.pickle```
