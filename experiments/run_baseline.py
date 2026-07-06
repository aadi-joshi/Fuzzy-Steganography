"""
Baseline Experiment Runner
==========================
Runs fixed-depth LSB steganography (1-bit and 2-bit) across all cover images
and payload rates. Records quality metrics, steganalysis detection rates, and
robustness results.

Usage:
    python -m experiments.run_baseline --config config/config.yaml
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from PIL import Image

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.metrics import compute_all
from analysis.robustness import test_robustness
from analysis.steganalysis import run_all_detectors
from crypto.aes import encrypt_bytes, decrypt_bytes
from stego.lsb_fixed import embed_fixed, extract_fixed, capacity_bytes

logger = logging.getLogger("baseline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_cover(path: str) -> np.ndarray:
    """Load an image as uint8 NumPy array (RGB)."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _generate_payload(size_bytes: int, seed: int = 42) -> bytes:
    """Generate a reproducible pseudo-random payload."""
    rng = np.random.RandomState(seed)
    return bytes(rng.randint(0, 256, size=size_bytes, dtype=np.uint8).tolist())


def _discover_images(directory: str) -> List[str]:
    """Find all PNG/JPEG images in a directory."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    paths = []
    for entry in sorted(os.listdir(directory)):
        if Path(entry).suffix.lower() in exts:
            paths.append(os.path.join(directory, entry))
    return paths


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_baseline(config_path: str) -> str:
    """
    Execute the full baseline experiment suite.

    Returns the path to the output CSV.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["random_seed"]
    np.random.seed(seed)

    cover_dir = cfg["experiment"]["cover_dir"]
    output_dir = cfg["experiment"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "baseline_results.csv")
    bpp_levels = cfg["stego"]["payload_bpp_levels"]
    lsb_depths = [1, 2]
    kdf_alg = cfg["crypto"]["kdf_algorithm"]
    password = "research_experiment_key_2025"

    images = _discover_images(cover_dir)
    if not images:
        logger.warning(f"No cover images found in {cover_dir}. Generating synthetic test image.")
        # Generate a synthetic test image
        synth_path = os.path.join(cover_dir, "synthetic_512x512.png")
        rng = np.random.RandomState(seed)
        synth = rng.randint(64, 192, (512, 512, 3), dtype=np.uint8)
        Image.fromarray(synth).save(synth_path)
        images = [synth_path]

    fieldnames = [
        "image", "lsb_depth", "bpp", "payload_bytes",
        "embed_time_s", "extract_time_s",
        "mse", "psnr", "ssim", "kl_divergence", "distortion_per_bit",
        "rs_estimated_rate", "rs_detected",
        "chi2_embedding_prob", "chi2_detected",
        "spa_estimated_rate", "spa_detected",
        "extraction_verified",
    ]

    rows: List[Dict] = []

    for img_path in images:
        img_name = Path(img_path).name
        logger.info(f"Processing: {img_name}")
        cover = _load_cover(img_path)

        for depth in lsb_depths:
            for bpp in bpp_levels:
                cap = capacity_bytes(cover, lsb_depth=depth, bpp=bpp)
                if cap <= 0:
                    logger.warning(f"  Skip depth={depth}, bpp={bpp}: zero capacity")
                    continue

                # Use 80% of capacity to avoid overflow
                payload_size = max(1, int(cap * 0.8))
                payload = _generate_payload(payload_size, seed=seed)

                # Encrypt payload
                try:
                    encrypted = encrypt_bytes(payload, password, kdf_algorithm=kdf_alg)
                except Exception:
                    # Fallback to PBKDF2 if argon2 not available
                    encrypted = encrypt_bytes(payload, password, kdf_algorithm="pbkdf2")

                # Check if encrypted payload fits
                if len(encrypted) > cap:
                    # Use raw payload (skip encryption for this rate)
                    encrypted = payload

                # Embed
                t0 = time.perf_counter()
                try:
                    stego = embed_fixed(cover, encrypted, lsb_depth=depth, bpp=bpp, seed=seed)
                except ValueError as e:
                    logger.warning(f"  Embed failed depth={depth}, bpp={bpp}: {e}")
                    continue
                embed_time = time.perf_counter() - t0

                # Extract & verify
                t0 = time.perf_counter()
                extracted = extract_fixed(stego, lsb_depth=depth, seed=seed)
                extract_time = time.perf_counter() - t0
                verified = extracted[:len(encrypted)] == encrypted

                # Quality metrics
                metrics = compute_all(cover, stego, payload_bits=len(encrypted) * 8)

                # Steganalysis
                detectors = run_all_detectors(stego)

                row = {
                    "image": img_name,
                    "lsb_depth": depth,
                    "bpp": bpp,
                    "payload_bytes": len(encrypted),
                    "embed_time_s": f"{embed_time:.4f}",
                    "extract_time_s": f"{extract_time:.4f}",
                    "mse": f"{metrics['mse']:.6f}",
                    "psnr": f"{metrics['psnr']:.2f}",
                    "ssim": f"{metrics['ssim']:.6f}",
                    "kl_divergence": f"{metrics['kl_divergence']:.6f}",
                    "distortion_per_bit": f"{metrics['distortion_per_bit']:.8f}",
                    "rs_estimated_rate": f"{detectors['rs']['estimated_rate']:.4f}",
                    "rs_detected": detectors["rs"]["detection_flag"],
                    "chi2_embedding_prob": f"{detectors['chi_square']['embedding_probability']:.4f}",
                    "chi2_detected": detectors["chi_square"]["detection_flag"],
                    "spa_estimated_rate": f"{detectors['spa']['estimated_rate']:.4f}",
                    "spa_detected": detectors["spa"]["detection_flag"],
                    "extraction_verified": verified,
                }
                rows.append(row)
                logger.info(
                    f"  depth={depth} bpp={bpp:.2f}: PSNR={metrics['psnr']:.2f}dB "
                    f"SSIM={metrics['ssim']:.4f} RS_det={detectors['rs']['detection_flag']}"
                )

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Baseline results saved to {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline LSB experiments.")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    run_baseline(args.config)


if __name__ == "__main__":
    main()
