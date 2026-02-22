import copy
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from abc import abstractmethod
from asparagus.functional.lr_scheduling import (
    cosine_decay_schedule,
    sawtooth_warmup_cosine_decay_schedule,
    separate_encoder_decoder_weights,
    simple_warmup_cosine_decay_schedule,
)
from asparagus.functional.visualization import (
    get_logger_compatible_image_output_target,
    log_image_output_target_to_mlflow,
    log_image_output_target_to_wandb,
)
from torch.optim import SGD, AdamW
from torchvision import transforms
from typing import Optional


class BaseModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        warmup_epochs: int = None,
        decoder_warmup_epochs: int = 0,
        cosine_period_ratio: float = 1,
        compile_mode: str = None,
        weights: dict = None,
        load_decoder: bool = True,
        optimizer: str = "SGD",
        train_transforms: Optional[transforms.Compose] = None,
        test_transforms: Optional[transforms.Compose] = None,
        val_transforms: Optional[transforms.Compose] = None,
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        repeat_stem_weights: bool = True,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.val_transforms = val_transforms

        self.loss = None
        self.train_metrics = None
        self.val_metrics = None
        self.warmup_epochs = warmup_epochs
        self.decoder_warmup_epochs = decoder_warmup_epochs
        self.ignore_index_in_metrics = 0
        self.cosine_period_ratio = cosine_period_ratio
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.momentum = momentum
        self.repeat_stem_weights = repeat_stem_weights
        assert 0 < cosine_period_ratio <= 1

        self.save_hyperparameters(ignore=["model", "weights", "train_transforms", "val_transforms", "test_transforms"])
        self.model = model

        if weights is not None:
            self.load_state_dict(weights, load_decoder=load_decoder, strict=False)

        self.model = torch.compile(model, mode=compile_mode) if compile_mode is not None else model

    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        # Separate encoder and decoder parameters for different warmup schedules
        if self.decoder_warmup_epochs > 0:
            param_groups = separate_encoder_decoder_weights(self.named_parameters())
        else:
            param_groups = self.parameters()

        if self.optimizer == "SGD":
            optimizer = SGD(
                param_groups,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                nesterov=self.nesterov,
            )
        elif self.optimizer == "AdamW":
            optimizer = AdamW(
                param_groups,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                amsgrad=False,
                betas=(0.9, 0.98),
                fused=True,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        print(f"Using optimizer {optimizer.__class__.__name__} with learning rate {self.learning_rate}")

        steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs

        # Scheduler option 1: Three-phase schedule with separate decoder/joint warmup
        if self.decoder_warmup_epochs > 0:
            scheduler = sawtooth_warmup_cosine_decay_schedule(
                optimizer,
                self.decoder_warmup_epochs,
                self.warmup_epochs,
                steps_per_epoch,
                self.cosine_period_ratio,
                self.trainer.max_epochs,
            )
        # Scheduler option 2: Two-phase schedule with joint warmup
        elif self.warmup_epochs > 0:
            scheduler = simple_warmup_cosine_decay_schedule(
                optimizer,
                self.warmup_epochs,
                steps_per_epoch,
                self.cosine_period_ratio,
                self.trainer.max_epochs,
            )
        # Scheduler option 3: Just cosine annealing
        else:
            scheduler = cosine_decay_schedule(
                optimizer,
                steps_per_epoch,
                self.cosine_period_ratio,
                self.trainer.max_epochs,
            )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,  # scheduler is updated after each batch
        }

        return [optimizer], [scheduler_config]

    def load_state_dict(self, state_dict, load_decoder=True, *args, **kwargs):
        old_params = copy.deepcopy(self.state_dict())

        target_compiled = "_orig" in next(iter(old_params.keys()))
        source_compiled = "_orig" in next(iter(state_dict.keys()))

        print(f"Target compiled: {target_compiled}, source compiled: {source_compiled}")

        if not target_compiled and source_compiled:
            print("Source state_dict is compiled, but target model is not. Removing _orig suffix from state_dict keys.")
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        # Repeat stem weights when state_dict num_channels is smaller than new_state_dict num_channels
        if hasattr(self.model, "stem_weight_name") and self.model.stem_weight_name is not None and self.repeat_stem_weights:
            prefix = "model._orig_mod." if "_orig_mod" in list(state_dict.keys())[0] else "model."
            stem_name = f"{prefix}{self.model.stem_weight_name}"
            pt_input_channels = state_dict[stem_name].shape[1]
            ft_input_channels = old_params[stem_name].shape[1]
            if pt_input_channels < ft_input_channels:
                assert pt_input_channels == 1, (
                    "Stem weights can only be repeated if the input channels in the state_dict is 1."
                )
                print(f"Repeating stem weights from {pt_input_channels} to {ft_input_channels} channels for {stem_name}.")
                state_dict[stem_name] = state_dict[stem_name].repeat(1, ft_input_channels, 1, 1, 1) / ft_input_channels

        # Filter out keys that are not in the old state dict or have different shapes
        def should_load_key(key, state_dict, old_params, load_decoder):
            # reject all decoder keys regardless of their shape
            if not load_decoder and key.startswith("model.decoder"):
                return False
            # accept all keys that are in the old state dict and have the same shape
            return (key in old_params) and (old_params[key].shape == state_dict[key].shape)

        # Filter state_dict to only include keys that should be loaded
        state_dict = {k: v for k, v in state_dict.items() if should_load_key(k, state_dict, old_params, load_decoder)}

        # Lists used to inform master of our whereabouts
        rejected_keys_new = [k for k in state_dict.keys() if k not in old_params]
        state_dict = {k: v for k, v in state_dict.items() if k in old_params}
        rejected_keys_shape = [k for k in state_dict.keys() if old_params[k].shape != state_dict[k].shape]
        rejected_keys_decoder = [k for k in state_dict.keys() if not load_decoder and k.startswith("model.decoder")]

        # Load the state dict
        kwargs["strict"] = False
        super().load_state_dict(state_dict, *args, **kwargs)

        # Check if weights were actually loaded
        new_params = self.state_dict()
        rejected_keys_data = []

        successful = 0
        unsuccessful = 0
        for param_name, p1, p2 in zip(old_params.keys(), old_params.values(), new_params.values()):
            if p1.data.ne(p2.data).sum() > 0:
                successful += 1
            else:
                unsuccessful += 1
                if param_name not in rejected_keys_new and param_name not in rejected_keys_shape:
                    rejected_keys_data.append(param_name)

        print(f"Succesfully transferred weights for {successful}/{successful + unsuccessful} layers")
        print(
            f"Rejected the following keys:\n"
            f"Not in old dict: {rejected_keys_new}.\n"
            f"Wrong shape: {rejected_keys_shape}.\n"
            f"Post check not succesful: {rejected_keys_data}."
        )
        if not load_decoder:
            print("Decoder weights were not loaded, as requested. If you want to load them, set `load_decoder=True`.")
            print(f"Rejected decoder keys: {rejected_keys_decoder}.")
        else:
            print("Warning! Also loaded the decoder. If you are finetuning, this might not be what you want.")

        assert successful > 0, "No weights were loaded. Check the state_dict and the model architecture."

    def _log_dict_of_images_to_wandb(self, imagedict: dict, log_key: str, task_type: str = ""):
        """
        Log a random image from the imagedict to wandb
        """
        batch_idx = np.random.randint(0, imagedict["input"].shape[0])
        image, output, target = get_logger_compatible_image_output_target(
            image=imagedict["input"][batch_idx],
            output=imagedict["output"][batch_idx],
            target=imagedict["target"][batch_idx],
            task_type=task_type,
        )
        for logger in self.trainer.loggers:
            if "WandbLogger" in logger.__class__.__name__:
                log_image_output_target_to_wandb(
                    logger=logger,
                    image=image,
                    output=output,
                    target=target,
                    log_key=log_key,
                    fig_title=imagedict["file"][batch_idx].split("/Task")[-1],
                    step=self.global_step,
                    task_type=task_type,
                )
            if "MLFlowLogger" in logger.__class__.__name__:
                log_image_output_target_to_mlflow(
                    logger=logger,
                    image=image,
                    output=output,
                    target=target,
                    log_key=log_key,
                    fig_title=imagedict["file"][batch_idx].split("/Task")[-1],
                    step=self.global_step,
                    task_type=task_type,
                )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training and self.train_transforms is not None:
            batch = self.train_transforms(batch)
        if (self.trainer.validating or self.trainer.sanity_checking) and self.val_transforms is not None:
            batch = self.val_transforms(batch)
        if (self.trainer.testing or self.trainer.predicting) and self.test_transforms is not None:
            batch = self.test_transforms(batch)
        return super().on_after_batch_transfer(batch, dataloader_idx)
