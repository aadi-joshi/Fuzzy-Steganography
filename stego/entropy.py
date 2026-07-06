"""
Local Entropy and Edge Feature Extraction
==========================================
Computes pixel-level feature maps used by the fuzzy adaptive embedding system:

1. **Local Shannon entropy** — windowed entropy measuring local texture complexity.
2. **Sobel edge magnitude** — gradient magnitude indicating edge strength.

Both outputs are normalised float64 arrays of the same spatial dimensions as the
input image, enabling direct element-wise use in the fuzzy inference engine.

All operations are vectorized using NumPy and SciPy; no Python-level pixel loops.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import generic_filter, sobel


# ---------------------------------------------------------------------------
# Local Shannon Entropy
# ---------------------------------------------------------------------------
def _shannon_entropy_window(window: np.ndarray) -> float:
    """Shannon entropy of a flat uint8 window (used as generic_filter func)."""
    counts = np.bincount(window.astype(np.intp), minlength=256)
    probs = counts[counts > 0] / window.size
    return float(-np.sum(probs * np.log2(probs)))


def local_entropy(
    image: np.ndarray,
    window_size: int = 7,
) -> np.ndarray:
    """
    Compute per-pixel local Shannon entropy over a sliding window.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, uint8, shape (H, W).
    window_size : int
        Side length of the square window (must be odd, ≥ 3).

    Returns
    -------
    np.ndarray
        Entropy map, float64, shape (H, W), values in [0, 8].
    """
    if image.ndim != 2:
        raise ValueError("Input must be a 2-D grayscale image.")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer ≥ 3.")

    return generic_filter(
        image.astype(np.float64),
        _shannon_entropy_window,
        size=window_size,
        mode="reflect",
    )


def local_entropy_fast(
    image: np.ndarray,
    window_size: int = 7,
) -> np.ndarray:
    """
    Fast approximation of local entropy using histogram binning with
    sliding-window sums (uniform filter).

    This avoids the per-pixel Python callback of ``generic_filter`` and is
    significantly faster on large images, at a small accuracy trade-off.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, uint8, shape (H, W).
    window_size : int
        Side length of the square window (must be odd, ≥ 3).

    Returns
    -------
    np.ndarray
        Approximate entropy map, float64, shape (H, W).
    """
    from scipy.ndimage import uniform_filter

    if image.ndim != 2:
        raise ValueError("Input must be a 2-D grayscale image.")

    img = image.astype(np.float64)
    h, w = img.shape

    # Number of quantisation bins (full 256 is exact but slow; 64 is a
    # practical compromise between speed and fidelity).
    n_bins = 64
    bin_edges = np.linspace(0, 256, n_bins + 1)
    quantised = np.digitize(img, bin_edges) - 1
    quantised = np.clip(quantised, 0, n_bins - 1)

    # One-hot per bin → local histogram via uniform_filter
    area = window_size * window_size
    entropy_map = np.zeros((h, w), dtype=np.float64)

    for b in range(n_bins):
        indicator = (quantised == b).astype(np.float64)
        prob = uniform_filter(indicator, size=window_size, mode="reflect")
        mask = prob > 0
        entropy_map[mask] -= prob[mask] * np.log2(prob[mask])

    return entropy_map


# ---------------------------------------------------------------------------
# Sobel Edge Magnitude
# ---------------------------------------------------------------------------
def sobel_edge_magnitude(
    image: np.ndarray,
    normalise: bool = True,
) -> np.ndarray:
    """
    Compute the Sobel gradient magnitude of a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape (H, W). Will be cast to float64.
    normalise : bool
        If True, the output is scaled to [0, 1].

    Returns
    -------
    np.ndarray
        Edge magnitude map, float64, shape (H, W).
    """
    if image.ndim != 2:
        raise ValueError("Input must be a 2-D grayscale image.")

    img = image.astype(np.float64)
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    magnitude = np.hypot(gx, gy)

    if normalise:
        mag_max = magnitude.max()
        if mag_max > 0:
            magnitude /= mag_max

    return magnitude


# ---------------------------------------------------------------------------
# Combined feature extraction
# ---------------------------------------------------------------------------
def extract_features(
    image: np.ndarray,
    window_size: int = 7,
    fast_entropy: bool = True,
    strip_lsb: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract both feature maps in one call.

    Parameters
    ----------
    image : np.ndarray
        Grayscale uint8 image, shape (H, W).
    window_size : int
        Entropy window size.
    fast_entropy : bool
        Use the fast approximation for local entropy.
    strip_lsb : int
        Number of lower bits to mask before computing features.
        Default 3 ensures features are invariant to up-to-3-bit
        LSB embedding, so the depth map is identical on cover and stego.

    Returns
    -------
    entropy_map : np.ndarray, shape (H, W)
    edge_map : np.ndarray, shape (H, W)
    """
    # Strip lower bits from each channel BEFORE grayscale conversion.
    # This ensures features are invariant to up-to-3-bit LSB embedding,
    # so the depth map is identical for cover and stego images.
    img = image.copy()
    if strip_lsb > 0:
        mask = np.uint8((0xFF << strip_lsb) & 0xFF)
        img = img & mask

    if img.ndim == 3:
        # Convert to grayscale (luminance)
        gray = np.dot(img[..., :3].astype(np.float64), [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        gray = img

    entropy_fn = local_entropy_fast if fast_entropy else local_entropy
    ent = entropy_fn(gray, window_size=window_size)
    edge = sobel_edge_magnitude(gray)
    return ent, edge
