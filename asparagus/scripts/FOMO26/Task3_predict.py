import numpy as np
import os
import random
from asparagus.modules.transforms.presets import CPU_clsreg_val_test_transforms_crop
from asparagus.pipeline.auto_configuration.checkpoint import load_checkpoint_state_dict
from dotenv import load_dotenv
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
        predict_transforms=CPU_clsreg_val_test_transforms_crop(target_size=ckpt_cfg.training.target_size),
        num_workers=0,
    )

    model = instantiate(
        ckpt_cfg.model._cls_net,
        input_channels=input_channels,
        output_channels=output_channels,
    )

    model_module = instantiate(
        ckpt_cfg.lightning._lightning_module,
        model=model,
        weights=load_checkpoint_state_dict(os.path.join(checkpoint_dir, f"checkpoints/{checkpoint_name}.ckpt")),
        test_output_path=output_path,
    )

    trainer = Trainer(accelerator=accelerator)

    output = trainer.predict(
        model=model_module,
        datamodule=data_module,
        return_predictions=True,
    )
    np.savetxt(output_path, output[0][0].cpu().numpy())

    print(f"Test predictions saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict script for FOMO26 Task 3: Brain Age Estimation")
    parser.add_argument("--t1", type=str, required=True)
    parser.add_argument("--output", type=str, required=True, help="Path to save predictions")
    parser.add_argument("--input_channels", type=int, default=1, help="Number of input channels for the model")
    parser.add_argument("--output_channels", type=int, default=1, help="Number of output channels for the model")
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
        data=[args.t1],
        output_path=args.output,
        output_channels=args.output_channels,
        input_channels=args.input_channels,
        checkpoint_dir=MODEL_DIR,
        checkpoint_name=CHECKPOINT_NAME,
        accelerator=args.accelerator,
    )
