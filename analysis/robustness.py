"""
Robustness Testing Module
=========================
Tests steganographic payload survivability under common image processing
attacks:

1. **JPEG compression** at configurable quality levels (e.g., 70, 50).
2. **Gaussian noise** addition at configurable sigma.
3. **Centre cropping** at a configurable percentage.

Each test applies the distortion, attempts extraction, and reports the
bit-level success rate.
"""

from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Distortion functions
# ---------------------------------------------------------------------------
def apply_jpeg_compression(
    image: np.ndarray,
    quality: int = 70,
) -> np.ndarray:
    """
    Apply JPEG compression and decompression to an image.

    Parameters
    ----------
    image : np.ndarray, uint8
    quality : int
        JPEG quality factor (1–95).

    Returns
    -------
    np.ndarray, uint8
    """
    pil_img = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = np.array(Image.open(buffer))
    # Ensure same shape (JPEG may drop alpha)
    if compressed.ndim == 2 and image.ndim == 3:
        compressed = np.stack([compressed] * image.shape[2], axis=-1)
    return compressed.astype(np.uint8)


def apply_gaussian_noise(
    image: np.ndarray,
    sigma: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """
    Add Gaussian noise to an image (σ relative to [0, 255]).

    Parameters
    ----------
    image : np.ndarray, uint8
    sigma : float
        Standard deviation of noise relative to full scale (0.01 = ±2.55).

    Returns
    -------
    np.ndarray, uint8
    """
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, sigma * 255, image.shape)
    noisy = np.clip(image.astype(np.float64) + noise, 0, 255)
    return noisy.astype(np.uint8)


def apply_center_crop(
    image: np.ndarray,
    crop_fraction: float = 0.10,
) -> np.ndarray:
    """
    Crop *crop_fraction* from each border and zero-pad back to original size.

    This simulates partial image loss; the border pixels are set to zero.

    Parameters
    ----------
    image : np.ndarray, uint8
    crop_fraction : float
        Fraction of each dimension to crop from each side (0.10 = 10%).

    Returns
    -------
    np.ndarray, uint8
        Same shape as input; border pixels zeroed.
    """
    h, w = image.shape[:2]
    dh = int(h * crop_fraction)
    dw = int(w * crop_fraction)

    result = np.zeros_like(image)
    result[dh:h - dh, dw:w - dw] = image[dh:h - dh, dw:w - dw]
    return result


# ---------------------------------------------------------------------------
# Extraction success evaluation
# ---------------------------------------------------------------------------
def _bit_accuracy(original: bytes, extracted: bytes) -> float:
    """Compute bit-level accuracy between two byte sequences."""
    orig_bits = np.unpackbits(np.frombuffer(original, dtype=np.uint8))
    extr_bits = np.unpackbits(np.frombuffer(extracted, dtype=np.uint8))

    min_len = min(len(orig_bits), len(extr_bits))
    if min_len == 0:
        return 0.0

    matches = np.sum(orig_bits[:min_len] == extr_bits[:min_len])
    return float(matches / len(orig_bits))


@dataclass
class RobustnessResult:
    """Container for a single robustness test outcome."""
    attack_name: str
    attack_params: dict
    extraction_success: bool
    bit_accuracy: float
    payload_match: bool  # exact byte match

    def to_dict(self) -> dict:
        return {
            "attack": self.attack_name,
            **{f"param_{k}": v for k, v in self.attack_params.items()},
            "extraction_success": self.extraction_success,
            "bit_accuracy": self.bit_accuracy,
            "payload_match": self.payload_match,
        }


# ---------------------------------------------------------------------------
# Robustness test runner
# ---------------------------------------------------------------------------
def test_robustness(
    stego: np.ndarray,
    original_payload: bytes,
    extract_fn: Callable[[np.ndarray], bytes],
    jpeg_qualities: List[int] = (70, 50),
    noise_sigmas: List[float] = (0.01, 0.05),
    crop_fraction: float = 0.10,
    seed: int = 42,
) -> List[RobustnessResult]:
    """
    Run a battery of robustness tests on a stego image.

    Parameters
    ----------
    stego : np.ndarray
        Stego image, uint8.
    original_payload : bytes
        The original secret payload for comparison.
    extract_fn : callable
        ``extract_fn(image) → bytes`` — the extraction function matching
        the embedding method used.
    jpeg_qualities : list of int
    noise_sigmas : list of float
    crop_fraction : float
    seed : int

    Returns
    -------
    list of RobustnessResult
    """
    results: List[RobustnessResult] = []

    # --- JPEG compression ---
    for q in jpeg_qualities:
        try:
            attacked = apply_jpeg_compression(stego, quality=q)
            extracted = extract_fn(attacked)
            success = True
        except Exception:
            extracted = b""
            success = False

        match = extracted == original_payload
        acc = _bit_accuracy(original_payload, extracted) if success else 0.0
        results.append(RobustnessResult(
            attack_name="jpeg_compression",
            attack_params={"quality": q},
            extraction_success=success,
            bit_accuracy=acc,
            payload_match=match,
        ))

    # --- Gaussian noise ---
    for sigma in noise_sigmas:
        try:
            attacked = apply_gaussian_noise(stego, sigma=sigma, seed=seed)
            extracted = extract_fn(attacked)
            success = True
        except Exception:
            extracted = b""
            success = False

        match = extracted == original_payload
        acc = _bit_accuracy(original_payload, extracted) if success else 0.0
        results.append(RobustnessResult(
            attack_name="gaussian_noise",
            attack_params={"sigma": sigma},
            extraction_success=success,
            bit_accuracy=acc,
            payload_match=match,
        ))

    # --- Cropping ---
    try:
        attacked = apply_center_crop(stego, crop_fraction=crop_fraction)
        extracted = extract_fn(attacked)
        success = True
    except Exception:
        extracted = b""
        success = False

    match = extracted == original_payload
    acc = _bit_accuracy(original_payload, extracted) if success else 0.0
    results.append(RobustnessResult(
        attack_name="center_crop",
        attack_params={"crop_fraction": crop_fraction},
        extraction_success=success,
        bit_accuracy=acc,
        payload_match=match,
    ))

    return results
