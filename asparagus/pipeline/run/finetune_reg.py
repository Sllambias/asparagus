import hydra
import lightning as pl
from asparagus.functional.hydra import fast_instantiate
from asparagus.modules.hydra.plugins.searchpath_plugins import FinetuneSearchpathPlugin
from asparagus.modules.transforms.presets import CPU_clsreg_val_test_transforms_crop
from asparagus.paths import get_config_path
from asparagus.pipeline.auto_configuration.checkpoint import resolve_checkpoint
from asparagus.pipeline.auto_configuration.experiment_setup import prepare_standard_experiment
from asparagus.pipeline.auto_configuration.run_setup import (
    build_checkpoint_callbacks,
    build_loggers,
    build_monitoring_callbacks,
    build_trainer,
    checkpoint_to_test,
    register_resolvers,
    test_output_path,
)
from dotenv import load_dotenv
from gardening_tools.modules.networks.components.weight_init import set_params_to_zero
from hydra.core.hydra_config import HydraConfig
from hydra.core.plugins import Plugins
from omegaconf import DictConfig, OmegaConf

load_dotenv()
register_resolvers()
Plugins.instance().register(FinetuneSearchpathPlugin)


@hydra.main(
    config_path=get_config_path(),
    config_name="default_finetune_reg",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    print(f"{OmegaConf.to_yaml(cfg)}\n Version: {cfg.run_id}\n Run dir: {HydraConfig.get().run.dir}\n")
    file_store, path_store, version_store = prepare_standard_experiment(cfg)
    weights = resolve_checkpoint(cfg)

    pl.seed_everything(seed=cfg.training.seed, workers=True)

    loggers = build_loggers(
        cfg,
        path_store,
        version_store,
        wandb_entity=cfg.logger.wandb_entity,
        log_to_stdout=cfg.logger.log_to_stdout,
    )
    best_ckpt_callback, last_ckpt_callback = build_checkpoint_callbacks(cfg, path_store)
    monitoring_callbacks = build_monitoring_callbacks(cfg)

    cpu_tr_transforms = fast_instantiate(
        cfg.transforms._cpu_tr_transforms,
        target_size=cfg.training.target_size,
    )
    cpu_val_transforms = fast_instantiate(
        cfg.transforms._cpu_val_transforms,
        target_size=cfg.training.target_size,
    )
    gpu_tr_transforms = fast_instantiate(cfg.transforms._gpu_tr_transforms, ndim=len(cfg.training.target_size))

    data_module = fast_instantiate(
        cfg.lightning._data_module,
        train_split=file_store.splits["train"],
        val_split=file_store.splits["val"],
        train_transforms=cpu_tr_transforms,
        val_transforms=cpu_val_transforms,
        test_samples=file_store.test,
        test_transforms=CPU_clsreg_val_test_transforms_crop(target_size=cfg.training.target_size),
    )

    model = fast_instantiate(
        cfg.model._cls_net,
        input_channels=file_store.dataset_json["metadata"]["n_modalities"],
        output_channels=file_store.dataset_json["metadata"]["n_classes"],
    )

    model_module = fast_instantiate(
        cfg.lightning._lightning_module,
        model=model,
        train_transforms=gpu_tr_transforms,
        val_transforms=None,
        weights=weights,
        log_image_every_n_epochs=cfg.logger.log_images_every_n_epoch,
        optimizer=cfg.model.finetune_optim,
        learning_rate=cfg.model.finetune_lr,
        warmup_epochs=cfg.training.warmup_epochs,
        weight_decay=cfg.model.weight_decay,
        load_decoder=cfg.training.load_decoder,
        test_output_path=test_output_path(cfg, path_store, cfg.test_checkpoint),
    )

    trainer = build_trainer(
        cfg,
        path_store,
        loggers,
        [last_ckpt_callback, best_ckpt_callback, *monitoring_callbacks],
        limit_train_batches=cfg.training.limit_train_batches,
        limit_val_batches=cfg.training.limit_val_batches,
    )

    trainer.fit(
        model=model_module,
        datamodule=data_module,
    )

    model_module.model.apply(set_params_to_zero)

    trainer.test(
        model=model_module,
        datamodule=data_module,
        ckpt_path=checkpoint_to_test(cfg, best_ckpt_callback, last_ckpt_callback),
    )


if __name__ == "__main__":
    main()
