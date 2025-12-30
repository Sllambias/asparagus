"""
Reconstruction quality metrics for SSL pretraining.
"""

import torch
from typing import Dict, Optional


def compute(pred: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]) -> Dict[str, float]:
    """Metrics computed every validation step."""
    metrics = compute_ssim_3d(pred, y, mask, n_slices=3)
    metrics |= compute_edge_aware_error(pred, y, mask)
    metrics |= compute_frequency_domain_error(pred, y, mask)
    return metrics


def compute_ssim_3d(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, window_size: int = 11, n_slices: int = 5
) -> Dict[str, torch.Tensor]:
    """
    Structural Similarity Index for 3D volumes via slice sampling.
    SSIM > 0.8 indicates good structural preservation in reconstruction.
    """
    metrics = {}

    # Ensure 5D tensors (B, C, D, H, W)
    if pred.dim() == 4:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
        if mask is not None:
            mask = mask.unsqueeze(1)

    B, C, D, H, W = pred.shape

    # Helper function for 2D SSIM computation
    def ssim_2d(img1, img2, window_size=11):
        """Compute SSIM for 2D images."""
        C1 = 0.01**2
        C2 = 0.03**2

        # Create Gaussian window
        sigma = 1.5
        gauss = torch.exp(-(torch.arange(window_size).float() ** 2) / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        window = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        window = window.unsqueeze(0).unsqueeze(0)
        window = window.to(img1.device)

        # Compute means
        mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size // 2)
        mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size // 2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Compute variances and covariance
        sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=window_size // 2) - mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=window_size // 2) - mu2_sq
        sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=window_size // 2) - mu1_mu2

        # SSIM formula
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    # Sample slices along each axis
    ssim_values = []

    # Axial slices (along depth)
    slice_indices = torch.linspace(0, D - 1, n_slices).long()
    for idx in slice_indices:
        slice_pred = pred[:, :, idx, :, :].squeeze(1)  # (B, H, W)
        slice_target = target[:, :, idx, :, :].squeeze(1)
        if slice_pred.dim() == 3:
            slice_pred = slice_pred.unsqueeze(1)
            slice_target = slice_target.unsqueeze(1)
        ssim_val = ssim_2d(slice_pred, slice_target, window_size)
        ssim_values.append(ssim_val)

    # Compute mean SSIM across all sampled slices
    metrics["ssim_3d"] = torch.stack(ssim_values).mean()

    # Also compute SSIM for masked region only if mask is provided
    if mask is not None:
        # Apply mask and compute SSIM on masked region
        pred_masked = pred * mask
        target_masked = target * mask

        ssim_masked_values = []
        for idx in slice_indices:
            slice_pred = pred_masked[:, :, idx, :, :].squeeze(1)
            slice_target = target_masked[:, :, idx, :, :].squeeze(1)
            slice_mask = mask[:, :, idx, :, :].squeeze(1)

            if slice_mask.sum() > 0:  # Only compute if there are masked pixels
                if slice_pred.dim() == 3:
                    slice_pred = slice_pred.unsqueeze(1)
                    slice_target = slice_target.unsqueeze(1)
                ssim_val = ssim_2d(slice_pred, slice_target, window_size)
                ssim_masked_values.append(ssim_val)

        if ssim_masked_values:
            metrics["ssim_3d_masked"] = torch.stack(ssim_masked_values).mean()
        else:
            metrics["ssim_3d_masked"] = torch.tensor(0.0)

    return metrics


def compute_edge_aware_error(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Sobel gradient-based edge preservation assessment.
    High edge error indicates blurring or loss of fine structure.
    """
    metrics = {}

    # Sobel kernels for 3D
    sobel_x = torch.tensor(
        [[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]], [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
        dtype=torch.float32,
    )

    sobel_y = torch.tensor(
        [[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]], [[-2, -4, -2], [0, 0, 0], [2, 4, 2]], [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]],
        dtype=torch.float32,
    )

    sobel_z = torch.tensor(
        [[[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[1, 2, 1], [2, 4, 2], [1, 2, 1]]]],
        dtype=torch.float32,
    )

    # Move kernels to device
    device = pred.device
    sobel_x = sobel_x.to(device).unsqueeze(0)
    sobel_y = sobel_y.to(device).unsqueeze(0)
    sobel_z = sobel_z.to(device).unsqueeze(0)

    # Ensure 5D tensors (B, C, D, H, W)
    if pred.dim() == 4:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
        if mask is not None:
            mask = mask.unsqueeze(1)

    # Compute gradients for prediction and target
    pred_grad_x = torch.nn.functional.conv3d(pred, sobel_x, padding=1)
    pred_grad_y = torch.nn.functional.conv3d(pred, sobel_y, padding=1)
    pred_grad_z = torch.nn.functional.conv3d(pred, sobel_z, padding=1)

    target_grad_x = torch.nn.functional.conv3d(target, sobel_x, padding=1)
    target_grad_y = torch.nn.functional.conv3d(target, sobel_y, padding=1)
    target_grad_z = torch.nn.functional.conv3d(target, sobel_z, padding=1)

    # Compute gradient magnitude
    pred_grad_mag = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + pred_grad_z**2 + 1e-8)
    target_grad_mag = torch.sqrt(target_grad_x**2 + target_grad_y**2 + target_grad_z**2 + 1e-8)

    # Compute edge-aware error
    edge_error = torch.nn.functional.mse_loss(pred_grad_mag, target_grad_mag, reduction="none")

    if mask is not None:
        # Compute error only on masked regions
        masked_edge_error = edge_error * (~mask).float()
        metrics["edge_error_masked"] = (masked_edge_error.sum() / (~mask).float().sum().clamp(min=1)).item()

        # Also compute on visible regions for comparison
        visible_edge_error = edge_error * mask.float()
        metrics["edge_error_visible"] = (visible_edge_error.sum() / mask.float().sum().clamp(min=1)).item()

    # Overall edge error
    metrics["edge_error_total"] = edge_error.mean().item()

    return metrics


def compute_frequency_domain_error(
    pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Fourier spectrum analysis for high-frequency detail preservation.
    Low/mid/high band errors reveal frequency-specific reconstruction issues.
        mask: Optional mask tensor

    Returns:
        Dictionary with frequency domain error metrics
    """
    metrics = {}

    if pred is None or target is None:
        return metrics

    # Apply mask if provided
    if mask is not None:
        pred = pred * mask
        target = target * mask

    # Convert to float32 for FFT (BFloat16 not supported)
    pred = pred.float()
    target = target.float()

    # Compute FFT (use rfft for real inputs to save memory)
    # We use 2D/3D FFT depending on input dimensions
    if pred.dim() == 5:  # 3D volumes (B, C, D, H, W)
        pred_fft = torch.fft.rfftn(pred, dim=(2, 3, 4))
        target_fft = torch.fft.rfftn(target, dim=(2, 3, 4))
    elif pred.dim() == 4:  # 2D images (B, C, H, W)
        pred_fft = torch.fft.rfftn(pred, dim=(2, 3))
        target_fft = torch.fft.rfftn(target, dim=(2, 3))
    else:
        return metrics

    # Compute magnitude (absolute value)
    pred_mag = torch.abs(pred_fft)
    target_mag = torch.abs(target_fft)

    # Compute MSE in frequency domain
    freq_mse = torch.mean((pred_mag - target_mag) ** 2)

    # Compute low and high frequency errors separately
    # Low frequency = center of FFT, high frequency = edges
    freq_shape = pred_mag.shape[-3:]  # Get spatial FFT dimensions

    # Define low-frequency region (center 25% of frequencies)
    low_freq_mask = torch.zeros_like(pred_mag, dtype=torch.bool)
    for i, size in enumerate(freq_shape):
        center = size // 4
        if i == 0 and pred.dim() == 5:  # Depth dimension for 3D
            low_freq_mask[..., :center, :, :] = True
        elif i == 1 or (i == 0 and pred.dim() == 4):  # Height dimension
            if pred.dim() == 5:
                low_freq_mask[..., :, :center, :] = True
            else:
                low_freq_mask[..., :center, :] = True
        elif i == 2 or (i == 1 and pred.dim() == 4):  # Width dimension
            if pred.dim() == 5:
                low_freq_mask[..., :, :, :center] = True
            else:
                low_freq_mask[..., :, :center] = True

    low_freq_error = (
        torch.mean((pred_mag[low_freq_mask] - target_mag[low_freq_mask]) ** 2) if low_freq_mask.any() else freq_mse
    )
    high_freq_error = (
        torch.mean((pred_mag[~low_freq_mask] - target_mag[~low_freq_mask]) ** 2) if (~low_freq_mask).any() else freq_mse
    )

    metrics["freq_domain_mse"] = freq_mse.item()
    metrics["freq_domain_low_mse"] = low_freq_error.item()
    metrics["freq_domain_high_mse"] = high_freq_error.item()
    metrics["freq_domain_high_low_ratio"] = (high_freq_error / (low_freq_error + 1e-8)).item()

    return metrics
