import hydra
import os
import random
from asparagus.modules.transforms.presets import CPU_clsreg_val_test_transforms_crop
from asparagus.paths import get_config_path
from asparagus.pipeline.auto_configuration.checkpoint import load_checkpoint_state_dict
from asparagus.pipeline.auto_configuration.versioning import pathing
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, RandomSampler
from lightning import Trainer
import numpy as np
from torch.nn.functional import softmax

load_dotenv()

OmegaConf.register_new_resolver("random", lambda min, max: random.randint(min, max))
OmegaConf.register_new_resolver("eval", eval)


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

    output = softmax(output[0], dim=1)  # Get the probability of the positive class
    np.savetxt(output_path, output[:, 1].cpu().numpy())

    print(f"Test predictions saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict script for FOMO26 Task 1: Infarct Detection")
    parser.add_argument("--flair", type=str, required=True)
    parser.add_argument("--adc", type=str, required=True)
    parser.add_argument("--dwi", type=str, required=True)
    parser.add_argument("--t2s", type=str, required=False, help="Path to T2* image (optional)")
    parser.add_argument("--swi", type=str, required=False, help="Path to SWI image (optional)")
    parser.add_argument("--output", type=str, required=True, help="Path to save predictions")
    parser.add_argument("--input_channels", type=int, default=4, help="Number of input channels for the model")
    parser.add_argument("--output_channels", type=int, default=2, help="Number of output channels for the model")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="best",
        help="Name of the checkpoint file (without .pt extension)",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator to use for prediction (e.g., 'cpu', 'cuda', 'mps')",
    )
    args = parser.parse_args()

    if args.t2s is not None:
        data = [args.flair, args.adc, args.dwi, args.t2s]
    else:
        data = [args.flair, args.adc, args.dwi, args.swi]

    main(
        data=data,
        output_path=args.output,
        output_channels=args.output_channels,
        input_channels=args.input_channels,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        accelerator=args.accelerator,
    )
