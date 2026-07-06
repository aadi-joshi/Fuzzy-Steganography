"""
Dataset Generator
=================
Generates a diverse synthetic dataset of 256×256 RGB images for
steganographic evaluation, or loads BOSSbase / BOWS-2 if available.

Image categories (evenly distributed):
    1. Smooth — gradients, blurred regions (low entropy)
    2. Noise — uniform/Gaussian random (high entropy)
    3. Natural — 1/f (pink) noise approximating natural image statistics
    4. Textured — sinusoidal/Gabor patterns (medium entropy)
    5. Mixed — block mosaics combining different textures

Each image is saved as lossless PNG.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("dataset")


# ---------------------------------------------------------------------------
# Individual image generators
# ---------------------------------------------------------------------------
def generate_smooth(size: Tuple[int, int], rng: np.random.RandomState) -> np.ndarray:
    """Generate a smooth gradient image with mild noise."""
    h, w = size
    # Random gradient direction
    angle = rng.uniform(0, 2 * np.pi)
    base_val = rng.randint(40, 200)
    amplitude = rng.randint(30, 80)

    y, x = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    gradient = np.cos(angle) * x + np.sin(angle) * y
    gradient = (gradient - gradient.min()) / (np.ptp(gradient) + 1e-10)
    img = base_val + amplitude * gradient
    # Add very mild noise
    img += rng.normal(0, 2, (h, w))
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_noise(size: Tuple[int, int], rng: np.random.RandomState) -> np.ndarray:
    """Generate a high-entropy noise image."""
    h, w = size
    lo = rng.randint(0, 60)
    hi = rng.randint(180, 256)
    return rng.randint(lo, hi, (h, w), dtype=np.uint8)


def generate_natural(size: Tuple[int, int], rng: np.random.RandomState) -> np.ndarray:
    """Generate a natural-like image using 1/f (pink) noise in frequency domain."""
    h, w = size
    # Create random phase
    phase = rng.uniform(0, 2 * np.pi, (h, w))
    # Create 1/f magnitude spectrum
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    freq_mag = np.sqrt(fx ** 2 + fy ** 2)
    freq_mag[0, 0] = 1.0  # avoid division by zero
    # 1/f^beta, beta ∈ [0.8, 1.3] for natural images
    beta = rng.uniform(0.8, 1.3)
    magnitude = 1.0 / (freq_mag ** beta)
    magnitude[0, 0] = 0  # zero DC
    # Combine
    spectrum = magnitude * np.exp(1j * phase)
    img = np.real(np.fft.ifft2(spectrum))
    # Normalise to [lo, hi]
    lo = rng.randint(20, 80)
    hi = rng.randint(170, 240)
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-10:
        img = (img - img_min) / (img_max - img_min) * (hi - lo) + lo
    else:
        img = np.full_like(img, (lo + hi) / 2)
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_textured(size: Tuple[int, int], rng: np.random.RandomState) -> np.ndarray:
    """Generate a textured image using superimposed sinusoids."""
    h, w = size
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    img = np.zeros((h, w), dtype=np.float64)
    n_components = rng.randint(3, 8)
    for _ in range(n_components):
        freq_x = rng.uniform(0.01, 0.15)
        freq_y = rng.uniform(0.01, 0.15)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(15, 50)
        img += amp * np.sin(2 * np.pi * (freq_x * x + freq_y * y) + phase)

    base = rng.randint(80, 160)
    img += base
    # Add mild noise
    img += rng.normal(0, 5, (h, w))
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_mixed(size: Tuple[int, int], rng: np.random.RandomState) -> np.ndarray:
    """Generate a mosaic image with blocks of different textures."""
    h, w = size
    img = np.zeros((h, w), dtype=np.uint8)

    block_size = rng.choice([32, 64, 128])
    generators = [generate_smooth, generate_noise, generate_natural, generate_textured]

    for r in range(0, h, block_size):
        for c in range(0, w, block_size):
            gen = generators[rng.randint(len(generators))]
            bh = min(block_size, h - r)
            bw = min(block_size, w - c)
            block = gen((bh, bw), rng)
            img[r:r + bh, c:c + bw] = block

    return img


# ---------------------------------------------------------------------------
# Full dataset generation
# ---------------------------------------------------------------------------
def generate_full_dataset(
    n_images: int = 1000,
    size: Tuple[int, int] = (256, 256),
    output_dir: str = "data/covers_v2",
    seed: int = 42,
) -> List[str]:
    """
    Generate a diverse synthetic dataset.

    Creates n_images PNG files evenly distributed across 5 categories.
    Images are stored as RGB (grayscale expanded to 3 channels).
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    generators = [
        ("smooth", generate_smooth),
        ("noise", generate_noise),
        ("natural", generate_natural),
        ("textured", generate_textured),
        ("mixed", generate_mixed),
    ]
    per_cat = n_images // len(generators)
    remainder = n_images - per_cat * len(generators)

    paths = []
    idx = 0
    for cat_idx, (cat_name, gen_fn) in enumerate(generators):
        count = per_cat + (1 if cat_idx < remainder else 0)
        for j in range(count):
            gray = gen_fn(size, rng)
            # Convert grayscale to RGB
            rgb = np.stack([gray, gray, gray], axis=-1)
            fname = f"{cat_name}_{idx:05d}.png"
            fpath = os.path.join(output_dir, fname)
            Image.fromarray(rgb).save(fpath, compress_level=1)
            paths.append(fpath)
            idx += 1

    logger.info(f"Generated {len(paths)} synthetic images in {output_dir}")
    return paths


# ---------------------------------------------------------------------------
# Load existing dataset (BOSSbase / BOWS-2)
# ---------------------------------------------------------------------------
def discover_images(directory: str) -> List[str]:
    """Find all supported image files in a directory."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".pgm", ".ppm"}
    paths = []
    for entry in sorted(os.listdir(directory)):
        if Path(entry).suffix.lower() in exts:
            paths.append(os.path.join(directory, entry))
    return paths


def load_bossbase(
    directory: str,
    max_images: Optional[int] = None,
) -> Optional[List[str]]:
    """
    Attempt to load BOSSbase 1.01 or BOWS-2 from a directory.

    Returns list of file paths if found, None otherwise.
    """
    if not os.path.isdir(directory):
        return None

    paths = discover_images(directory)
    if len(paths) < 100:
        return None

    if max_images is not None:
        paths = paths[:max_images]

    logger.info(f"Loaded {len(paths)} images from {directory}")
    return paths


def prepare_dataset(
    n_images: int = 1000,
    size: Tuple[int, int] = (256, 256),
    output_dir: str = "data/covers_v2",
    bossbase_dir: Optional[str] = None,
    seed: int = 42,
) -> List[str]:
    """
    Prepare the experiment dataset.

    If bossbase_dir is set and contains images, use those.
    Otherwise, generate a synthetic dataset.
    """
    # Try BOSSbase first
    if bossbase_dir:
        paths = load_bossbase(bossbase_dir, max_images=n_images)
        if paths:
            return paths
        logger.warning(f"BOSSbase not found at {bossbase_dir}, falling back to synthetic.")

    # Check if dataset already generated
    if os.path.isdir(output_dir):
        existing = discover_images(output_dir)
        if len(existing) >= n_images:
            logger.info(f"Using existing dataset: {len(existing)} images in {output_dir}")
            return existing[:n_images]

    # Generate synthetic
    return generate_full_dataset(n_images, size, output_dir, seed)


def load_image_rgb(path: str) -> np.ndarray:
    """Load any image as RGB uint8 array."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
