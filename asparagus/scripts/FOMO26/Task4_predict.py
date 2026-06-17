import os
import random
from asparagus.modules.transforms.presets import CPU_seg_test_transforms
from asparagus.pipeline.auto_configuration.checkpoint import load_checkpoint_state_dict
from dotenv import load_dotenv
from gardening_tools.functional.paths.write import save_prediction_from_logits
from hydra.utils import instantiate
from lightning import Trainer
from omegaconf import OmegaConf

load_dotenv()

OmegaConf.register_new_resolver("random", lambda min, max: random.randint(min, max))
OmegaConf.register_new_resolver("eval", eval)


MODEL_DIR = None
CHECKPOINT_NAME = None


def main(
    data: list,
    output_path: str,
    output_channels: int,
    input_channels: int,
    checkpoint_dir: str,
    checkpoint_name: str,
    accelerator: str,
) -> None:
    ckpt_cfg = OmegaConf.load(os.path.join(checkpoint_dir, "hydra/config.yaml"))
    output_path = output_path

    data_module = instantiate(
        ckpt_cfg.lightning._data_module,
        batch_size=1,
        train_split=None,
        val_split=None,
        predict_samples=data,
        predict_transforms=CPU_seg_test_transforms(patch_size=ckpt_cfg.training.patch_size),
        num_workers=0,
    )

    model = instantiate(
        ckpt_cfg.model._seg_net,
        input_channels=input_channels,
        output_channels=output_channels,
    )

    model_module = instantiate(
        ckpt_cfg.lightning._lightning_module,
        model=model,
        weights=load_checkpoint_state_dict(os.path.join(checkpoint_dir, f"checkpoints/{checkpoint_name}.ckpt")),
        inference_patch_size=ckpt_cfg.training.patch_size,
    )

    trainer = Trainer(accelerator=accelerator)

    logits, properties = trainer.predict(
        model=model_module,
        datamodule=data_module,
    )[0]

    save_prediction_from_logits(
        logits.numpy(),
        output_path,
        properties=properties,
    )
    print(f"Test predictions saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict script for FOMO26 Task 4: Trigeminal Neuralgia Segmentation")
    parser.add_argument("--t2", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="Path to save predictions")
    parser.add_argument("--input_channels", type=int, default=1, help="Number of input channels for the model")
    parser.add_argument("--output_channels", type=int, default=3, help="Number of output channels for the model")
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator to use for prediction (e.g., 'cpu', 'cuda', 'mps')",
    )
    args = parser.parse_args()
    assert MODEL_DIR is not None, "MODEL_DIR environment variable must be set to the path of the model checkpoint directory"
    assert CHECKPOINT_NAME is not None, (
        "CHECKPOINT_NAME environment variable must be set to the name of the checkpoint file (without .pt extension)"
    )

    main(
        data=[args.t2],
        output_path=args.output,
        output_channels=args.output_channels,
        input_channels=args.input_channels,
        checkpoint_dir=MODEL_DIR,
        checkpoint_name=CHECKPOINT_NAME,
        accelerator=args.accelerator,
    )
