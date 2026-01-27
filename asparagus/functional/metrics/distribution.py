"""
Statistical distribution and alignment metrics for SSL pretraining.
"""

import torch
from typing import Dict


# Frequency-based compute functions
def compute(x: torch.Tensor, pred: torch.Tensor, y: torch.Tensor, encoder_features: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Metrics computed every validation step."""
    metrics = compute_input_statistics(x, prefix="input")
    metrics |= compute_input_statistics(pred, prefix="pred")
    metrics |= compute_intensity_distribution_match(pred, y)
    metrics |= compute_alignment_uniformity(encoder_features)
    return metrics


def compute_input_statistics(x: torch.Tensor, prefix: str = "input") -> Dict[str, torch.Tensor]:
    """
    Input intensity distribution monitoring for data drift detection.
    Significant shifts indicate preprocessing issues or domain shift.
    """
    x = x.float()
    return {
        f"{prefix}_mean": x.mean(),
        f"{prefix}_std": x.std(),
        f"{prefix}_min": x.min(),
        f"{prefix}_max": x.max(),
        f"{prefix}_p10": torch.quantile(x.flatten()[::10], 0.1),
        f"{prefix}_p50": torch.quantile(x.flatten()[::10], 0.5),
        f"{prefix}_p90": torch.quantile(x.flatten()[::10], 0.9),
    }


def compute_intensity_distribution_match(pred: torch.Tensor, target: torch.Tensor, n_bins: int = 50) -> Dict[str, float]:
    """
    KL divergence and Wasserstein distance between intensity histograms.
    KL < 0.1 indicates good distribution matching; >1.0 suggests mode collapse.
    """
    metrics = {}

    # Flatten tensors and convert to float32 for histc (required for BFloat16 compatibility)
    pred_flat = pred.flatten().float()
    target_flat = target.flatten().float()

    # Compute histograms
    hist_range = (min(pred_flat.min().item(), target_flat.min().item()), max(pred_flat.max().item(), target_flat.max().item()))
    pred_hist = torch.histc(pred_flat, bins=n_bins, min=hist_range[0], max=hist_range[1])
    target_hist = torch.histc(target_flat, bins=n_bins, min=hist_range[0], max=hist_range[1])

    # Normalize to probabilities
    pred_hist = pred_hist / pred_hist.sum()
    target_hist = target_hist / target_hist.sum()

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    pred_hist = pred_hist.clamp(min=eps)
    target_hist = target_hist.clamp(min=eps)

    # Compute KL divergence: KL(P||Q) = sum(P * log(P/Q))
    kl_div = (target_hist * torch.log(target_hist / pred_hist)).sum().item()

    # Also compute reverse KL
    kl_div_reverse = (pred_hist * torch.log(pred_hist / target_hist)).sum().item()

    # Symmetric KL (Jensen-Shannon divergence related)
    kl_symmetric = (kl_div + kl_div_reverse) / 2

    metrics["intensity_kl_divergence"] = kl_div
    metrics["intensity_kl_symmetric"] = kl_symmetric

    # Also compute histogram correlation
    pred_mean = pred_hist.mean()
    target_mean = target_hist.mean()

    cov = ((pred_hist - pred_mean) * (target_hist - target_mean)).mean()
    std_pred = pred_hist.std()
    std_target = target_hist.std()

    if std_pred > 0 and std_target > 0:
        correlation = cov / (std_pred * std_target)
        metrics["intensity_histogram_correlation"] = correlation.item()
    else:
        metrics["intensity_histogram_correlation"] = 0.0

    return metrics


def compute_alignment_uniformity(
    features: torch.Tensor, temperature: float = 2.0, sample_size: int = 1000
) -> Dict[str, float]:
    """
    Wang & Isola metrics for contrastive representation quality.
    Alignment < 1.0 good; Uniformity ~ -2 to -3 optimal for hypersphere coverage.
    """
    metrics = {}

    if features is None or features.numel() == 0:
        return metrics

    # Flatten spatial dimensions if present
    if features.dim() > 2:
        B = features.shape[0]
        features = features.view(B, -1).float()  # (B, D)
    else:
        features = features.float()

    # Sample if batch is too large (for efficiency)
    if features.shape[0] > sample_size:
        indices = torch.randperm(features.shape[0])[:sample_size]
        features = features[indices]

    # L2 normalize features
    features_normalized = torch.nn.functional.normalize(features, p=2, dim=1)

    # For alignment, we need positive pairs - in SSL pretraining context,
    # we assume consecutive samples might be augmented versions (simplified)
    # In practice, this would need proper positive pair tracking
    if features.shape[0] >= 2:
        # Simple heuristic: treat consecutive pairs as positives
        features1 = features_normalized[0::2]  # Even indices
        features2 = features_normalized[1::2]  # Odd indices

        if features1.shape[0] > 0 and features2.shape[0] > 0:
            # Alignment: average distance between positive pairs
            alignment = (features1 - features2).norm(dim=1).pow(2).mean()
            metrics["alignment_loss"] = alignment.item()

    # Uniformity: log of average pairwise Gaussian potential
    # This measures how uniformly distributed features are
    if features.shape[0] > 1:
        # Compute pairwise distances
        sq_pdist = torch.pdist(features_normalized, p=2).pow(2)

        # Gaussian potential with temperature
        uniformity = sq_pdist.mul(-temperature).exp().mean().log()
        metrics["uniformity_loss"] = uniformity.item()

        # Additional uniformity statistics
        metrics["uniformity_mean_dist"] = sq_pdist.mean().item()
        # Only compute std if we have more than 1 distance
        if sq_pdist.numel() > 1:
            metrics["uniformity_std_dist"] = sq_pdist.std().item()
        else:
            metrics["uniformity_std_dist"] = 0.0
        metrics["uniformity_min_dist"] = sq_pdist.min().item()
        metrics["uniformity_max_dist"] = sq_pdist.max().item()

    # Compute cosine similarity statistics
    if features.shape[0] > 1:
        # Compute cosine similarity matrix
        cos_sim = torch.mm(features_normalized, features_normalized.t())

        # Get upper triangular part (excluding diagonal)
        mask = torch.triu(torch.ones_like(cos_sim, dtype=torch.bool), diagonal=1)
        cos_sim_values = cos_sim[mask]

        if cos_sim_values.numel() > 0:
            metrics["cosine_sim_mean"] = cos_sim_values.mean().item()
            # Only compute std if we have more than 1 similarity value
            if cos_sim_values.numel() > 1:
                metrics["cosine_sim_std"] = cos_sim_values.std().item()
            else:
                metrics["cosine_sim_std"] = 0.0
            metrics["cosine_sim_max"] = cos_sim_values.max().item()
            metrics["cosine_sim_min"] = cos_sim_values.min().item()

    return metrics
