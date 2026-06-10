import hydra
import os
import random
from asparagus.modules.transforms.presets import CPU_seg_test_transforms
from asparagus.paths import get_config_path
from asparagus.pipeline.auto_configuration.checkpoint import load_checkpoint_state_dict
from asparagus.pipeline.auto_configuration.versioning import pathing
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

load_dotenv()

OmegaConf.register_new_resolver("random", lambda min, max: random.randint(min, max))
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(
    config_path=get_config_path(),
    config_name="default_predict",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    paths = pathing(cfg, train=False)

    ckpt_cfg = OmegaConf.load(os.path.join(paths.ckpt_parent_folder, "hydra/config.yaml"))
    output_path = cfg.output_path

    data_module = instantiate(
        ckpt_cfg.lightning._data_module,
        batch_size=1,
        train_split=None,
        val_split=None,
        predict_samples=[cfg.input_path],
        test_transforms=CPU_seg_test_transforms(patch_size=ckpt_cfg.training.patch_size),
        num_workers=cfg.hardware.num_workers,
    )

    model = instantiate(
        ckpt_cfg.model._seg_net,
        input_channels=cfg.input_channels,
        output_channels=cfg.output_channels,
    )

    model_module = instantiate(
        ckpt_cfg.lightning._lightning_module,
        model=model,
        weights=load_checkpoint_state_dict(paths.ckpt_path),
        inference_patch_size=ckpt_cfg.training.patch_size,
        test_output_path=output_path,
    )

    trainer = instantiate(
        ckpt_cfg.lightning._trainer,
        callbacks=None,
        log_every_n_steps=250,
        logger=None,
        profiler=None,
        default_root_dir=paths.run_dir,
        enable_progress_bar=True,
    )

    trainer.predict(
        model=model_module,
        datamodule=data_module,
    )
    print(f"Test predictions saved to {output_path}")


if __name__ == "__main__":
    main()
