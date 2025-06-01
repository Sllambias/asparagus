# Asparagus

### Environment Variables
Required Environment Variables:
```bash
ASPARAGUS_CONFIGS=/PATH/TO/CONFIG/DIR
ASPARAGUS_SOURCE=/PATH/TO/SOURCE/DATA
ASPARAGUS_DATA=/PATH/TO/ASPARAGUS/DATA
ASPARAGUS_MODELS=/PATH/TO/ASPARAGUS/MODELS
ASPARAGUS_RESULTS=/PATH/TO/ASPARAGUS/OUTPUTS
```

Optional Environment Variables (supports multiple colon-separated (":") paths):

```bash
ASPARAGUS_PRETRAIN_CONFIGS=/PATH1/TO/ADDITIONAL/PRETRAIN/CONFIG/DIR:PATH2/TO/ADDITIONAL/PRETRAIN/CONFIG/DIR:PATH3...
ASPARAGUS_TRAIN_CONFIGS=/PATH1/TO/ADDITIONAL/TRAIN/CONFIG/DIR:PATH2/TO/ADDITIONAL/TRAIN/CONFIG/DIR:PATH3...
ASPARAGUS_PRETRAIN_CONFIGS=/PATH1/TO/ADDITIONAL/FINETUNE/CONFIG/DIR:PATH2/TO/ADDITIONAL/FINETUNECONFIG/DIR:PATH3...
```

### Task Conversion
The idea of task conversion for large pretraining datasets is to convert the data to asparagus format and carry out all preprocessing. One critical assumption here is that preprocessing for a given pretraining dataset is always identical. If this assumption is not true, Yucca may be better suited with its separation of Task Conversion and Preprocessing, which allows any given task converted dataset to be easily preprocessed in multiple ways. 

## Generate data splits
```asp_split -t TASK --fn split_80_20```

## Config tips.
CLI view config without running script: ```--cfg job```

CLI change config: ```--config-name name_of_config```
For detailed control of the configs see also the core:base config.

CLI add config: ```+FOLDER=CONFIG```
For example, the following prompt will pretrain a model on Task998 using the model config found in model/unet_b_lw_dec.yaml, adding the online segmentation plugin found in plugin/seg_Task997.yaml, with the hardware setup found in hardware/1gpu, and, finally overriding any model dimensions with "2D" (the model.dimensions is found in model/unet_b_lw_dec and defaults to 3D).
```asp_pretrain task=Task998_LauritSyn +model=unet_b_lw_dec +plugins=seg_Task997 +hardware=1gpu model.dimensions=2D```
for a folder structure that looks like this:
```
CONFIG_DIR/
├── pretrain.yaml
├── hardware/
|   ├── 1gpu.yaml
├── model/
|   ├── unet_b_lw_dec.yaml
├── plugin/
|   ├── seg_Task997.yaml
```

To create a config identical to the example above one would need to import the configs using defaults as such:
```
defaults:
  - ../hardware/1gpu@hardware
  - ../plugins/seg_Task997@plugin
  - ../model/unet_b_lw_dec@model
```

## Run Pretraining with the default pretrain config.
```asp_pretrain experiment.task=Task998_LauritSyn```

## Run Segmentation/Classification Finetuning with hydra config
```asp_finetune_seg experiment.task=Task997_LauritSynSeg experiment.pretrained_run_id=XXX```
```asp_finetune_cls experiment.task=Task996_LauritSynCls experiment.pretrained_run_id=XXX```


To change the pretrained model simply refer to its run_id and checkpoint name:
```
pretrained_run_id: 532
pretrained_checkpoint_name: epoch=4-step=25.ckpt
```

Asparagus will create a "derived_models.log" in the folder of the pretrained model so you can always track which finetuned models and run_ids are its children. 

## Run Training from scratch
```asp_train_seg experiment.task=Task997_LauritSynSeg```
```asp_train_cls experiment.task=Task996_LauritSynCls```

## Rerun previously executed jobs
```asp_[pretrain,train,etc.] --experimental-rerun $OUTPUT_DIR/config.pickle```


## Asparagus Versioning
To get RUN-ID (version) from a log/output file use:

Versioning is controlled by unique IDs. Each time you start a run Asparagus will either generate a new unique ID or resume training an identical run and re-use its existing ID. To control this behavior use the "resume_training: [True/False]".

When generating new IDs Asparagus will create IDs with higher numerical value than existing IDs for identical runs. I.e. if you have previously trained a model with run_id=532 using setup A and you want to run another identical training, without resuming run_id=532, then the new run_id will be higher than 532. This way you always know the order they were trained in.


## Hacking Asparagus
Using your own LightningModule or Datamodule or an entirely different train.py script?
No problem. Go into /configs/core/base and change the relevant path to one of your liking.

To change the LightningModule AND the default module path from where LightningModules are imported:

```
model:
  net: MyFancyLightningModule
  _model:
    _target_: my.own.local.repo.${model.net}
```

To change the train script you can simply write your MyTrain.py and call it like we would otherwise call the default scripts. (Remember to point it to the correct config or change it in the CLI.)
```
@hydra.main(
    config_path=get_config_path(),
    config_name="train",
    version_base="1.2",
)
def train(cfg: DictConfig) -> None:
    # Your Code Here

if __name__ == "__main__":
    train()
```
