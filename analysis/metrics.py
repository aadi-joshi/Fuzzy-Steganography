"""
Image Quality & Embedding Distortion Metrics
=============================================
Provides publication-standard image quality metrics for evaluating
steganographic distortion:

- **MSE** — Mean Squared Error
- **PSNR** — Peak Signal-to-Noise Ratio (dB)
- **SSIM** — Structural Similarity Index (Wang et al., 2004)
- **KL Divergence** — Kullback–Leibler divergence between histograms
- **Distortion per bit** — MSE normalised by embedded payload size

All functions accept NumPy arrays and are fully vectorized.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Mean Squared Error
# ---------------------------------------------------------------------------
def mse(cover: np.ndarray, stego: np.ndarray) -> float:
    """
    Compute the Mean Squared Error between *cover* and *stego*.

    .. math::
        \\text{MSE} = \\frac{1}{N} \\sum_{i=1}^{N} (x_i - y_i)^2

    Parameters
    ----------
    cover, stego : np.ndarray
        Images of the same shape (any dtype — cast to float64 internally).

    Returns
    -------
    float
    """
    diff = cover.astype(np.float64) - stego.astype(np.float64)
    return float(np.mean(diff ** 2))


# ---------------------------------------------------------------------------
# Peak Signal-to-Noise Ratio
# ---------------------------------------------------------------------------
def psnr(cover: np.ndarray, stego: np.ndarray, peak: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio.

    .. math::
        \\text{PSNR} = 10 \\log_{10}\\!\\left(\\frac{\\text{peak}^2}{\\text{MSE}}\\right)

    Returns ``float('inf')`` when MSE = 0 (identical images).

    Parameters
    ----------
    cover, stego : np.ndarray
    peak : float
        Maximum possible pixel value (255 for uint8).

    Returns
    -------
    float
        PSNR in decibels.
    """
    m = mse(cover, stego)
    if m == 0.0:
        return float("inf")
    return float(10.0 * np.log10(peak ** 2 / m))


# ---------------------------------------------------------------------------
# Structural Similarity Index (SSIM)
# ---------------------------------------------------------------------------
def ssim(
    cover: np.ndarray,
    stego: np.ndarray,
    window_size: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
    L: float = 255.0,
) -> float:
    """
    Structural Similarity Index (mean over spatial dimensions).

    Implements the simplified SSIM with a uniform window (no Gaussian weighting)
    for transparency and reproducibility.

    .. math::
        \\text{SSIM}(x, y) = \\frac{(2\\mu_x \\mu_y + C_1)(2\\sigma_{xy} + C_2)}
        {(\\mu_x^2 + \\mu_y^2 + C_1)(\\sigma_x^2 + \\sigma_y^2 + C_2)}

    Parameters
    ----------
    cover, stego : np.ndarray
        Images of the same shape.
    window_size : int
        Side length of the uniform averaging window.
    k1, k2 : float
        Stabilisation constants.
    L : float
        Dynamic range of pixel values.

    Returns
    -------
    float
        Mean SSIM in [−1, 1] (higher is better; 1 = identical).
    """
    from scipy.ndimage import uniform_filter

    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    x = cover.astype(np.float64)
    y = stego.astype(np.float64)

    # If colour, compute per-channel and average
    if x.ndim == 3:
        vals = [
            ssim(x[..., ch], y[..., ch], window_size, k1, k2, L)
            for ch in range(x.shape[2])
        ]
        return float(np.mean(vals))

    mu_x = uniform_filter(x, size=window_size, mode="reflect")
    mu_y = uniform_filter(y, size=window_size, mode="reflect")

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = uniform_filter(x ** 2, size=window_size, mode="reflect") - mu_x_sq
    sigma_y_sq = uniform_filter(y ** 2, size=window_size, mode="reflect") - mu_y_sq
    sigma_xy = uniform_filter(x * y, size=window_size, mode="reflect") - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


# ---------------------------------------------------------------------------
# KL Divergence (histogram-based)
# ---------------------------------------------------------------------------
def histogram_kl_divergence(
    cover: np.ndarray,
    stego: np.ndarray,
    bins: int = 256,
) -> float:
    """
    Kullback–Leibler divergence between the pixel-intensity histograms
    of *cover* and *stego*.

    .. math::
        D_{\\text{KL}}(P \\| Q) = \\sum_{i} P(i) \\ln\\!\\left(\\frac{P(i)}{Q(i)}\\right)

    A small epsilon is added to avoid log(0).

    Parameters
    ----------
    cover, stego : np.ndarray
    bins : int

    Returns
    -------
    float
        KL divergence (≥ 0; lower means less detectable).
    """
    eps = 1e-12
    hist_c, _ = np.histogram(cover.ravel(), bins=bins, range=(0, 256), density=True)
    hist_s, _ = np.histogram(stego.ravel(), bins=bins, range=(0, 256), density=True)

    hist_c = hist_c + eps
    hist_s = hist_s + eps

    return float(np.sum(hist_c * np.log(hist_c / hist_s)))


# ---------------------------------------------------------------------------
# Distortion per embedded bit
# ---------------------------------------------------------------------------
def distortion_per_bit(
    cover: np.ndarray,
    stego: np.ndarray,
    payload_bits: int,
) -> float:
    """
    Embedding distortion normalised by the number of embedded bits.

    .. math::
        D_b = \\frac{\\text{MSE}}{n_{\\text{bits}}}

    Parameters
    ----------
    cover, stego : np.ndarray
    payload_bits : int
        Total number of payload bits embedded.

    Returns
    -------
    float
    """
    if payload_bits <= 0:
        return 0.0
    return mse(cover, stego) / payload_bits


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once
# ---------------------------------------------------------------------------
def compute_all(
    cover: np.ndarray,
    stego: np.ndarray,
    payload_bits: int = 0,
) -> dict:
    """
    Return a dictionary of all quality metrics.

    Returns
    -------
    dict
        Keys: ``mse``, ``psnr``, ``ssim``, ``kl_divergence``, ``distortion_per_bit``.
    """
    return {
        "mse": mse(cover, stego),
        "psnr": psnr(cover, stego),
        "ssim": ssim(cover, stego),
        "kl_divergence": histogram_kl_divergence(cover, stego),
        "distortion_per_bit": distortion_per_bit(cover, stego, payload_bits),
    }
