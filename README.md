# Asparagus

### Environment Variables

### Task Conversion
The idea of task conversion for large pretraining datasets is to convert the data to asparagus format and carry out all preprocessing. One critical assumption here is that preprocessing for a given pretraining dataset is always identical. If this assumption is not true, Yucca may be better suited with its separation of Task Conversion and Preprocessing, which allows any given task converted dataset to be easily preprocessed in multiple ways. 

## Generate data splits
```python asparagus/pipeline/run/split.py -t TASK --fn split_80_20```

## Config tips.
CLI change config: ```--config_name name_of_config```
For detailed control of the configs see also the core:base config.

## Run Pretraining with the default pretrain config.
```python asparagus/pretrain.py experiment.task=Task998_LauritSyn```

## Run Finetuning with hydra config
```python asparagus/finetune.py experiment.task=Task997_LauritSynSeg experiment.pretrained_run_id=XXX```

To change the pretrained model simply refer to its run_id and checkpoint name:
```
pretrained_run_id: 532
pretrained_checkpoint_name: epoch=4-step=25.ckpt
```

Asparagus will create a "derived_models.log" in the folder of the pretrained model so you can always track which finetuned models and run_ids are its children. 

## Run Training from scratch
```python asparagus/train.py experiment.task=Task997_LauritSynSeg```

## Rerun previously executed jobs
```python my_app.py --experimental-rerun $OUTPUT_DIR/config.pickle```


## Asparagus Versioning
Versioning is controlled by unique IDs. Each time you start a run Asparagus will either generate a new unique ID or resume training an identical run and re-use its existing ID. To control this behavior use the "resume_training: [True/False]".

When generating new IDs Asparagus will create IDs with higher numerical value than existing IDs for identical runs. I.e. if you have previously trained a model with run_id=532 using setup A and you want to run another identical training, without resuming run_id=532, then the new run_id will be higher than 532. This way you always know the order they were trained in.


## Hacking Asparagus
Using your own LightningModule or Datamodule or an entirely different train.py script?
No problem. Go into /configs/core/base and change the relevant path to one of your liking.

To change the LightningModule AND the default module path from where LightningModules are imported:

```
model:
  net: MyFancyLightningModule

_internal_:
  net:
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