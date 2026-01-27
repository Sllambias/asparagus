from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, SequentialLR
from typing import Any, Dict, List


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
    print(f"Using separate warmup: decoder for {decoder_warmup_epochs} epochs, then both for {warmup_epochs} epochs")

    decoder_warmup_steps = int(decoder_warmup_epochs * steps_per_epoch)
    encoder_decoder_warmup_steps = int(warmup_epochs * steps_per_epoch)
    total_warmup_steps = decoder_warmup_steps + encoder_decoder_warmup_steps
    cosine_steps = int(cosine_period_ratio * (max_epochs * steps_per_epoch - total_warmup_steps))

    def encoder_phase1_lambda(_step):
        return 0.0  # Encoder frozen during phase 1

    def decoder_phase1_lambda(step):
        return 0.999 * step / decoder_warmup_steps + 0.001

    # LambdaLR depends on the order of the param groups
    assert optimizer.param_groups[0]["name"] == "encoder" and optimizer.param_groups[1]["name"] == "decoder", (
        "Param groups are not in the expected order."
    )

    phase1_scheduler = LambdaLR(optimizer, lr_lambda=[encoder_phase1_lambda, decoder_phase1_lambda])
    phase2_scheduler = LinearLR(optimizer, start_factor=1.0 / 1000, total_iters=encoder_decoder_warmup_steps)
    phase3_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps)

    return SequentialLR(
        optimizer,
        schedulers=[phase1_scheduler, phase2_scheduler, phase3_scheduler],
        milestones=[decoder_warmup_steps, total_warmup_steps],
    )


def simple_warmup_cosine_decay_schedule(optimizer, warmup_epochs, steps_per_epoch, cosine_period_ratio, max_epochs):
    """
    Phase 1: Warmup for both encoder and decoder
    Phase 2: Cosine annealing for both
    """
    print(f"Using warmup for {warmup_epochs} epochs")

    total_warmup_steps = int(warmup_epochs * steps_per_epoch)
    # cosine_half_period is from max to min
    cosine_steps = int(cosine_period_ratio * (max_epochs * steps_per_epoch - total_warmup_steps))

    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1.0 / 1000,
        total_iters=total_warmup_steps,
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[total_warmup_steps],
    )


def cosine_decay_schedule(optimizer, steps_per_epoch, cosine_period_ratio, max_epochs):
    """
    Phase 1: Cosine annealing for both encoder and decoder
    """
    # cosine_half_period is from max to min
    cosine_steps = int(cosine_period_ratio * (max_epochs * steps_per_epoch))
    return CosineAnnealingLR(optimizer, T_max=cosine_steps)
