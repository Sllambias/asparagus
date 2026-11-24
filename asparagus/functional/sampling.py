import numpy as np
from typing import List, Optional


def class_sampling_weights(
    labels: np.ndarray,
    class_factors: Optional[List[float]] = None,
    class_balance: float = 0.0,
) -> np.ndarray:
    """Weight of each sample, so that class c is drawn in proportion to class_factors[c] * count_c ** (1 - class_balance)."""
    labels = labels.astype(np.int64)
    counts = np.bincount(labels)
    if class_factors is None:
        class_factors = [1.0] * len(counts)
    factors = np.asarray(class_factors, dtype=np.float64)
    assert len(factors) >= len(counts), f"class_factors has {len(factors)} values, but the labels go up to {len(counts) - 1}"
    return factors[labels] / counts[labels] ** class_balance
