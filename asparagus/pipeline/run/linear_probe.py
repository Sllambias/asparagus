import hydra
import lightning as pl
from asparagus.functional.hydra import fast_instantiate
from asparagus.modules.hydra.plugins.searchpath_plugins import FinetuneSearchpathPlugin
from asparagus.modules.lightning_modules.linear_probe_module import LinearProbeModule
from asparagus.modules.transforms.presets import CPU_clsreg_val_test_transforms_crop
from asparagus.paths import get_config_path
from asparagus.pipeline.auto_configuration.checkpoint import resolve_checkpoint
from asparagus.pipeline.auto_configuration.experiment_setup import prepare_standard_experiment
from asparagus.pipeline.auto_configuration.run_setup import (
    build_loggers,
    build_monitoring_callbacks,
    build_trainer,
    register_resolvers,
    test_output_path,
)
from dotenv import load_dotenv
from hydra.core.hydra_config import HydraConfig
from hydra.core.plugins import Plugins
from omegaconf import DictConfig, OmegaConf

load_dotenv()
register_resolvers()
Plugins.instance().register(FinetuneSearchpathPlugin)


@hydra.main(
    config_path=get_config_path(),
    config_name="default_linear_probe",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    print(f"{OmegaConf.to_yaml(cfg)}\n Version: {cfg.run_id}\n Run dir: {HydraConfig.get().run.dir}\n")
    logging_safe_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    file_store, path_store, version_store = prepare_standard_experiment(cfg)
    pl.seed_everything(seed=cfg.training.seed, workers=True)

    loggers = build_loggers(cfg, path_store, version_store, wandb_config=logging_safe_cfg)
    monitoring_callbacks = build_monitoring_callbacks(cfg)

    cpu_tr_transforms = fast_instantiate(
        cfg.transforms._cpu_tr_transforms,
        target_size=cfg.training.target_size,
    )
    cpu_val_transforms = fast_instantiate(
        cfg.transforms._cpu_val_transforms,
        target_size=cfg.training.target_size,
    )

    gpu_tr_transforms = (
        fast_instantiate(cfg.transforms._gpu_tr_transforms, ndim=len(cfg.training.target_size))
        if cfg.transforms._gpu_tr_transforms is not None
        else None
    )

    data_module = fast_instantiate(
        cfg.lightning._data_module,
        train_split=file_store.splits["train"],
        val_split=file_store.splits["val"],
        train_transforms=cpu_tr_transforms,
        val_transforms=cpu_val_transforms,
        test_samples=file_store.test,
        test_transforms=CPU_clsreg_val_test_transforms_crop(target_size=cfg.training.target_size),
        use_random_datasampler=False,
    )

    model = fast_instantiate(
        cfg.model._cls_net,
        input_channels=file_store.dataset_json["metadata"]["n_modalities"],
        output_channels=file_store.dataset_json["metadata"]["n_classes"],
        late_fusion=True,
    )

    weights = resolve_checkpoint(cfg)
    if weights is None:
        print("No checkpoint provided — using randomly initialized backbone.")

    trainer = build_trainer(
        cfg,
        path_store,
        loggers,
        monitoring_callbacks,
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
        enable_checkpointing=False,  # no need to save the model checkpoints
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
    )

    num_classes = file_store.dataset_json["metadata"]["n_classes"]
    learning_rates = list(cfg.training.probing.learning_rates)

    model_module = LinearProbeModule(
        model=model,
        learning_rates=learning_rates,
        num_classes=num_classes,
        dimensions=cfg.model.dimensions,
        loss_weight=cfg.training.get("loss_weight", None),
        train_transforms=gpu_tr_transforms,
        val_transforms=None,
        test_output_path=test_output_path(cfg, path_store, "linear_probe"),
        weights=weights,
        pretrained_target_size=cfg.training.get("pretrained_target_size", None),
        target_size=cfg.training.target_size,
    )

    data_module.setup("fit")  # otherwise the validation set is not loaded
    trainer.validate(model=model_module, datamodule=data_module)

    trainer.fit(
        model=model_module,
        datamodule=data_module,
    )

    # Test using the best head selected during final validation
    trainer.test(
        model=model_module,
        datamodule=data_module,
    )


if __name__ == "__main__":
    main()
