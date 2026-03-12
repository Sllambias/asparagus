# Hacking Asparagus
Using your own LightningModule or Datamodule or an entirely different train.py script?
No problem. Go into /configs/core/base and change the relevant path to one of your liking.

To change the LightningModule AND the default module path from where LightningModules are imported:

```
lightning:
  lightning_module: MyFancyLightningModule
  _lightning_module:
    _target_: my.own.local.repo.${lightning.lightning_module}
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
