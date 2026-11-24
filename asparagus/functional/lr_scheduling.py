import lightning as L
import re
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from typing import Any, Dict, List


def get_optimizer_steps_per_epoch(trainer: L.Trainer) -> int:
    """Number of optimizer steps in one training epoch."""
    # Without max_epochs the run is probably set by max_steps, so count from limit_train_batches
    if trainer.max_epochs <= 0:
        return trainer.limit_train_batches // trainer.accumulate_grad_batches
    return trainer.estimated_stepping_batches // trainer.max_epochs


def separate_encoder_decoder_weights(named_parameters) -> List[Dict[str, Any]]:
    """Separate the encoder and decoder weights of a model.

    Args:
        named_parameters (List[tuple]): List of named parameters from the model.
    Returns:
        List[Dict[str, Any]]: A list containing two dictionaries, one for encoder parameters and one for decoder parameters.
    """
    encoder_params = []
    decoder_params = []
    for name, param in named_parameters:
        name = name.replace("_orig_mod.", "")  # if params are compilled we laugh and do this
        if "model.encoder" in name:
            encoder_params.append(param)
        elif "model.decoder" in name:
            decoder_params.append(param)
        else:
            # Default to encoder params for any other parameters (e.g., stem)
            encoder_params.append(param)

    # All hail the almighty assert which saved my ass. Twice.
    assert len(encoder_params) > 0 and len(decoder_params) > 0, "Encoder or decoder parameters not found."
    return [
        {"params": encoder_params, "name": "encoder"},
        {"params": decoder_params, "name": "decoder"},
    ]


def vit_layer_decay_param_groups(named_parameters, learning_rate: float, layer_decay: float) -> List[Dict[str, Any]]:
    """Param groups of a ViT encoder and a decoder, where each layer trains at layer_decay times the rate of the layer above.

    Layer 0 is the patch and position embeddings and the cls and register tokens, layer i + 1 is block i, and the final
    norm and the decoder train at learning_rate. Frozen parameters are left out.

    Args:
        named_parameters: Named parameters of a module whose model has a ViT encoder (vit.blocks) and a decoder.
        learning_rate: Learning rate of the final norm and the decoder.
        layer_decay: Factor between the learning rates of a layer and the layer above it.
    Returns:
        A group per encoder layer, named encoder_layer_<layer>, then a group named decoder.
    """
    embedding_params = []
    block_params = {}
    norm_params = []
    decoder_params = []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        name = name.replace("_orig_mod.", "")
        block = re.search(r"\.vit\.blocks\.(\d+)\.", name)
        if "model.decoder" in name:
            decoder_params.append(param)
        elif block is not None:
            block_params.setdefault(int(block.group(1)), []).append(param)
        elif ".vit.norm." in name:
            norm_params.append(param)
        elif "patch_embedding" in name or "cls_token" in name or "register_tokens" in name:
            embedding_params.append(param)
        else:
            raise ValueError(f"{name} is in neither a ViT layer nor the decoder")

    num_blocks = len(block_params)
    assert sorted(block_params) == list(range(num_blocks)), f"Found ViT blocks {sorted(block_params)}"
    assert len(decoder_params) > 0, "Decoder parameters not found."
    layers = [embedding_params] + [block_params[block] for block in range(num_blocks)] + [norm_params]
    param_groups = []
    for layer, params in enumerate(layers):
        param_groups.append(
            {
                "params": params,
                "name": f"encoder_layer_{layer}",
                "lr": learning_rate * layer_decay ** (num_blocks + 1 - layer),
            }
        )
    param_groups.append({"params": decoder_params, "name": "decoder", "lr": learning_rate})
    return param_groups


def merge_param_groups(param_groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge parameter groups whose settings (everything except params and name) are equal.
    merged_names lists the names of the groups that went into each merged group."""
    merged = {}
    for group in param_groups:
        settings = {key: value for key, value in group.items() if key not in ["params", "name", "merged_names"]}
        settings_key = tuple(sorted(settings.items()))
        if settings_key not in merged:
            merged[settings_key] = {"name": f"group_{len(merged)}", "params": [], "merged_names": [], **settings}
        merged[settings_key]["params"].extend(group["params"])
        merged[settings_key]["merged_names"].append(group.get("name"))
    return list(merged.values())


def sawtooth_warmup_cosine_decay_schedule(
    optimizer,
    decoder_warmup_epochs,
    warmup_epochs,
    steps_per_epoch,
    cosine_period_ratio,  # cosine_half_period is from max to min
    max_epochs,
):
    """
    Phase 1: Decoder warmup, encoder frozen
    Phase 2: Both encoder and decoder warmup
    Phase 3: Cosine annealing for both
    """
    assert max_epochs > 0 and steps_per_epoch > 0, "max_epochs and steps_per_epoch must be greater than 0"
    print(f"Using separate warmup: decoder for {decoder_warmup_epochs} epochs, then both for {warmup_epochs} epochs")

    decoder_warmup_steps = int(decoder_warmup_epochs * steps_per_epoch)
    encoder_decoder_warmup_steps = int(warmup_epochs * steps_per_epoch)
    total_warmup_steps = decoder_warmup_steps + encoder_decoder_warmup_steps
    cosine_steps = int(cosine_period_ratio * (max_epochs * steps_per_epoch - total_warmup_steps))

    def encoder_phase1_lambda(_step):
        return 0.0  # Encoder frozen during phase 1

    def decoder_phase1_lambda(step):
        return 0.999 * step / decoder_warmup_steps + 0.001

    # LambdaLR takes one lambda per param group, in the order of the groups
    phase1_lambdas = []
    for group in optimizer.param_groups:
        assert group["name"] == "decoder" or group["name"].startswith("encoder"), (
            f"Param group {group['name']} is neither an encoder nor the decoder group."
        )
        if group["name"] == "decoder":
            phase1_lambdas.append(decoder_phase1_lambda)
        else:
            phase1_lambdas.append(encoder_phase1_lambda)

    phase1_scheduler = LambdaLR(optimizer, lr_lambda=phase1_lambdas)
    phase2_scheduler = LinearLR(optimizer, start_factor=1.0 / 1000, total_iters=encoder_decoder_warmup_steps)
    phase3_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps)

    return SequentialLR(
        optimizer,
        schedulers=[phase1_scheduler, phase2_scheduler, phase3_scheduler],
        milestones=[decoder_warmup_steps, total_warmup_steps],
    )


def simple_warmup_cosine_decay_schedule(
    optimizer, warmup_epochs, steps_per_epoch, cosine_period_ratio, max_epochs=-1, max_steps=-1
):
    """
    Phase 1: Warmup for both encoder and decoder
    Phase 2: Cosine annealing for both
    """
    assert warmup_epochs >= 0, "Warmup epochs must be greater than or equal to 0."
    assert cosine_period_ratio > 0, "Cosine period ratio must be greater than 0."
    assert steps_per_epoch > 0, "Steps per epoch must be greater than 0."
    assert max_epochs > 0 or max_steps > 0, "Either max_epochs or max_steps must be greater than 0."

    # cosine_half_period is from max to min
    if max_epochs > 0:
        max_steps = max_epochs * steps_per_epoch

    total_warmup_steps = int(warmup_epochs * steps_per_epoch)
    cosine_steps = int(cosine_period_ratio * (max_steps - total_warmup_steps))
    assert max_steps > total_warmup_steps, "Max steps must be greater than total warmup steps for cosine decay schedule."

    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1.0 / 1000,
        total_iters=total_warmup_steps,
    )

    print(f"Using warmup for {warmup_epochs} epochs ({total_warmup_steps} steps)")
    print(f"Cosine decay for {cosine_steps} steps after warmup")
    assert total_warmup_steps > 0, "Warmup steps must be greater than 0 for warmup schedule."
    assert cosine_steps > 0, "Cosine steps must be greater than 0 for warmup cosine decay schedule."

    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[total_warmup_steps],
    )


def cosine_decay_schedule(optimizer, steps_per_epoch, cosine_period_ratio, max_epochs=-1, max_steps=-1):
    """
    Phase 1: Cosine annealing for both encoder and decoder
    """
    # cosine_half_period is from max to min
    if max_epochs > 0:
        max_steps = max_epochs * steps_per_epoch
    cosine_steps = int(cosine_period_ratio * max_steps)
    assert cosine_steps > 0, "Cosine steps must be greater than 0 for cosine decay schedule."
    return CosineAnnealingLR(optimizer, T_max=cosine_steps)
