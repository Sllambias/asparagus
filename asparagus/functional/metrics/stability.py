"""
Training stability and monitoring metrics for SSL pretraining.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


def compute_on_backward(model: nn.Module, grad_clip_val: Optional[float] = None) -> Dict[str, float]:
    """Metrics computed after backward pass."""
    return compute_gradient_metrics(model, grad_clip_val)


def compute_nan_inf_metrics(
    loss: Optional[torch.Tensor] = None,
    pred: Optional[torch.Tensor] = None,
    activations: Optional[torch.Tensor] = None,
    model: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    Numerical stability monitoring for gradient explosion/vanishing.
    Any non-zero value indicates training instability requiring immediate attention.
    """
    metrics = {}

    # Check loss
    if loss is not None:
        metrics["nan_loss"] = 1.0 if torch.isnan(loss).any().item() else 0.0
        metrics["inf_loss"] = 1.0 if torch.isinf(loss).any().item() else 0.0

    # Check predictions
    if pred is not None:
        metrics["nan_predictions"] = 1.0 if torch.isnan(pred).any().item() else 0.0
        metrics["inf_predictions"] = 1.0 if torch.isinf(pred).any().item() else 0.0

    # Check activations
    if activations is not None:
        metrics["nan_activations"] = 1.0 if torch.isnan(activations).any().item() else 0.0
        metrics["inf_activations"] = 1.0 if torch.isinf(activations).any().item() else 0.0

    # Check gradients
    if model is not None:
        has_nan_grad = False
        has_inf_grad = False
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any().item():
                    has_nan_grad = True
                if torch.isinf(p.grad).any().item():
                    has_inf_grad = True
        metrics["nan_gradients"] = 1.0 if has_nan_grad else 0.0
        metrics["inf_gradients"] = 1.0 if has_inf_grad else 0.0

    return metrics


def compute_gradient_metrics(model: nn.Module, grad_clip_value: Optional[float] = None) -> Dict[str, float]:
    """
    Gradient flow diagnostics: norm, clipping frequency, and layer-wise statistics.
    High clipping frequency (>0.5) suggests learning rate or architecture issues.
    """
    total_norm = 0.0
    num_parameters = 0
    gradient_clipped = False

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            num_parameters += 1

    total_norm = total_norm**0.5

    # Check if gradients would be clipped
    if grad_clip_value is not None and total_norm > grad_clip_value:
        gradient_clipped = True

    return {
        "gradient_norm": total_norm,
        "gradient_clipping_events": 1.0 if gradient_clipped else 0.0,
        "num_params_with_grad": num_parameters,
    }
