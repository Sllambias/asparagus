"""
Feature representation quality metrics for SSL pretraining.
"""

import torch
import torch.nn as nn
from typing import Dict


def compute_train(encoder_features: torch.Tensor) -> Dict[str, float]:
    """Metrics computed every training step."""
    return compute_embedding_metrics(encoder_features)


def compute_val(encoder_features: torch.Tensor, model: nn.Module) -> Dict[str, float]:
    """Metrics computed every validation step."""
    metrics = compute_embedding_metrics(encoder_features)
    metrics |= compute_feature_covariance(encoder_features)
    metrics |= compute_whitening_diagnostics(encoder_features)
    metrics |= compute_collapse_score(encoder_features)
    metrics |= compute_participation_ratio(encoder_features)
    return metrics


def compute_embedding_metrics(encoder_features: torch.Tensor) -> Dict[str, float]:
    """L2 norm statistics of encoder representations."""
    return {
        "embedding_norm_mean": encoder_features.norm(dim=1).mean().item(),
        "embedding_norm_std": encoder_features.norm(dim=1).std().item(),
    }


def compute_feature_covariance(features: torch.Tensor) -> Dict[str, float]:
    """Eigenspectrum analysis of feature covariance matrix."""
    if features is None:
        return {}

    # Flatten spatial dimensions if present
    if features.dim() > 2:
        B = features.shape[0]
        features = features.view(B, -1)

    features_centered = features - features.mean(dim=0, keepdim=True)
    cov = torch.mm(features_centered.T, features_centered) / (features.shape[0] - 1)
    trace = torch.trace(cov).item()

    try:
        eigenvalues = torch.linalg.eigvalsh(cov)
        max_eigenval = eigenvalues.max().item()
        min_eigenval = eigenvalues.min().item()

        # Condition number: ratio of largest to smallest eigenvalue
        if min_eigenval > 1e-10:
            condition_number = max_eigenval / min_eigenval
        else:
            condition_number = float("inf")

        # Effective rank via Shannon entropy of normalized eigenvalues
        p = eigenvalues / trace
        eps = torch.finfo(p.dtype).tiny
        H = -(p * torch.log(p + eps)).sum()
        effective_rank = torch.exp(H).item()

        return {
            "feature_cov_trace": trace,
            "feature_cov_condition_number": min(condition_number, 1e6),
            "feature_cov_max_eigenval": max_eigenval,
            "feature_cov_min_eigenval": max(min_eigenval, 1e-10),
            "feature_effective_rank": effective_rank,
        }

    except Exception:
        return {
            "feature_cov_trace": trace,
            "feature_cov_condition_number": -1,
            "feature_cov_max_eigenval": -1,
            "feature_cov_min_eigenval": -1,
            "feature_effective_rank": -1,
        }


def compute_collapse_score(features: torch.Tensor, eps: float = 1e-8) -> Dict[str, float]:
    """Dimensional collapse detection via per-channel variance analysis."""
    if features is None or features.numel() == 0:
        return {}

    if features.dim() > 2:
        B = features.shape[0]
        features = features.view(B, -1)

    dim_variance = features.var(dim=0, unbiased=False)

    # Count dimensions with near-zero variance (collapsed)
    zero_var_threshold = eps
    n_collapsed = (dim_variance < zero_var_threshold).sum().item()
    n_total = dim_variance.numel()

    # Variance statistics across dimensions
    collapse_var_mean = dim_variance.mean().item()
    collapse_var_std = dim_variance.std(unbiased=False).item() if dim_variance.numel() > 1 else 0.0
    collapse_var_min = dim_variance.min().item()
    collapse_var_max = dim_variance.max().item()

    # Coefficient of variation: std/mean
    if collapse_var_mean > eps:
        collapse_var_coeff = collapse_var_std / collapse_var_mean
    else:
        collapse_var_coeff = 0.0

    # Effective dimensionality via entropy of variance distribution
    variance_sum = dim_variance.sum()
    if variance_sum > 0:
        p = dim_variance / variance_sum
        tiny_eps = torch.finfo(p.dtype).tiny
        H = -(p * torch.log(p + tiny_eps)).sum()
        effective_dims = torch.exp(H).item()
    else:
        effective_dims = 0.0

    return {
        "collapse_n_zero_var_dims": n_collapsed,
        "collapse_pct_zero_var_dims": (n_collapsed / n_total * 100) if n_total > 0 else 0,
        "collapse_var_mean": collapse_var_mean,
        "collapse_var_std": collapse_var_std,
        "collapse_var_min": collapse_var_min,
        "collapse_var_max": collapse_var_max,
        "collapse_var_coeff": collapse_var_coeff,
        "collapse_effective_dims": effective_dims,
        "collapse_dim_ratio": effective_dims / n_total if n_total > 0 else 0,
    }


def compute_participation_ratio(features: torch.Tensor, k_values: list = [10, 50, 100]) -> Dict[str, float]:
    """Compute participation ratio - energy in top-k singular values."""
    if features is None or features.numel() == 0:
        return {}

    if features.dim() > 2:
        B = features.shape[0]
        features = features.view(B, -1).float()
    else:
        features = features.float()

    features_centered = features - features.mean(dim=0, keepdim=True)

    # SVD: use randomized version for high-dimensional features
    try:
        if features_centered.shape[1] > 1000:
            U, S, V = torch.svd_lowrank(features_centered, q=min(200, min(features_centered.shape)))
        else:
            U, S, V = torch.linalg.svd(features_centered, full_matrices=False)
    except Exception:
        return {}

    total_variance = (S**2).sum().item()

    if total_variance > 0:
        metrics = {}

        # Fraction of variance in top-k components
        for k in k_values:
            if k <= len(S):
                top_k_variance = (S[:k] ** 2).sum().item()
                participation = top_k_variance / total_variance
                metrics[f"participation_ratio_top{k}"] = participation

        # Effective rank from singular value entropy
        p = S / S.sum()
        eps = torch.finfo(p.dtype).tiny
        H = -(p * torch.log(p + eps)).sum()
        effective_rank = torch.exp(H).item()

        metrics["effective_rank"] = effective_rank
        metrics["effective_rank_ratio"] = effective_rank / len(S)

        return metrics

    return {}


def compute_whitening_diagnostics(features: torch.Tensor) -> Dict[str, float]:
    """Compute whitening diagnostics - off-diagonal correlation statistics."""
    if features is None or features.numel() == 0:
        return {}

    if features.dim() > 2:
        B = features.shape[0]
        features = features.view(B, -1).float()
    else:
        features = features.float()

    # Compute correlation matrix
    features_centered = features - features.mean(dim=0, keepdim=True)
    features_std = features_centered.std(dim=0, keepdim=True, unbiased=False) + 1e-8
    features_normalized = features_centered / features_std
    corr_matrix = torch.mm(features_normalized.T, features_normalized) / (features.shape[0] - 1)

    # Extract off-diagonal elements
    n_features = corr_matrix.shape[0]
    mask = ~torch.eye(n_features, dtype=torch.bool, device=corr_matrix.device)
    off_diagonal = corr_matrix[mask]

    # Off-diagonal statistics (should be near 0 for decorrelated features)
    metrics = {}
    if off_diagonal.numel() > 0:
        whitening_offdiag_mean = off_diagonal.mean().abs().item()
        whitening_offdiag_std = off_diagonal.std(unbiased=False).item() if off_diagonal.numel() > 1 else 0.0
        whitening_offdiag_max = off_diagonal.abs().max().item()
        decorrelation_score = off_diagonal.abs().mean().item()

        metrics = {
            "whitening_offdiag_mean": whitening_offdiag_mean,
            "whitening_offdiag_std": whitening_offdiag_std,
            "whitening_offdiag_max": whitening_offdiag_max,
            "decorrelation_score": decorrelation_score,
        }

    # Diagonal elements (should be ~1 for normalized features)
    diagonal = torch.diag(corr_matrix)
    whitening_diag_mean = diagonal.mean().item()
    whitening_diag_std = diagonal.std(unbiased=False).item() if diagonal.numel() > 1 else 0.0

    metrics["whitening_diag_mean"] = whitening_diag_mean
    metrics["whitening_diag_std"] = whitening_diag_std

    return metrics
