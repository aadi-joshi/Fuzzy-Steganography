"""
Adaptive Fuzzy Experiment Runner
================================
Runs the fuzzy adaptive LSB steganography across all cover images and payload
rates. Records quality metrics, steganalysis detection rates, and robustness
results.

Usage:
    python -m experiments.run_adaptive --config config/config.yaml
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.metrics import compute_all
from analysis.robustness import test_robustness
from analysis.steganalysis import run_all_detectors
from crypto.aes import encrypt_bytes
from stego.fuzzy import FuzzyDepthController
from stego.lsb_adaptive import (
    adaptive_capacity_bytes,
    compute_depth_map,
    embed_adaptive,
    extract_adaptive,
)

logger = logging.getLogger("adaptive")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_cover(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _generate_payload(size_bytes: int, seed: int = 42) -> bytes:
    rng = np.random.RandomState(seed)
    return bytes(rng.randint(0, 256, size=size_bytes, dtype=np.uint8).tolist())


def _discover_images(directory: str) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    paths = []
    for entry in sorted(os.listdir(directory)):
        if Path(entry).suffix.lower() in exts:
            paths.append(os.path.join(directory, entry))
    return paths


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_adaptive(config_path: str) -> str:
    """Execute the full adaptive fuzzy experiment suite."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["random_seed"]
    np.random.seed(seed)

    cover_dir = cfg["experiment"]["cover_dir"]
    output_dir = cfg["experiment"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "adaptive_results.csv")
    bpp_levels = cfg["stego"]["payload_bpp_levels"]
    window_size = cfg["stego"]["fuzzy"]["entropy_window_size"]
    kdf_alg = cfg["crypto"]["kdf_algorithm"]
    password = "research_experiment_key_2025"

    # Build fuzzy controller
    controller = FuzzyDepthController.from_config(cfg["stego"]["fuzzy"])

    images = _discover_images(cover_dir)
    if not images:
        logger.warning(f"No cover images found in {cover_dir}. Generating synthetic test image.")
        synth_path = os.path.join(cover_dir, "synthetic_512x512.png")
        rng = np.random.RandomState(seed)
        # Create a more realistic synthetic image with texture
        synth = rng.randint(64, 192, (512, 512, 3), dtype=np.uint8)
        # Add some structure
        for i in range(0, 512, 64):
            synth[i:i+2, :, :] = 200
            synth[:, i:i+2, :] = 200
        Image.fromarray(synth).save(synth_path)
        images = [synth_path]

    # Robustness test config
    jpeg_qualities = cfg["evaluation"]["jpeg_qualities"]
    noise_sigmas = cfg["evaluation"]["gaussian_noise_sigma"]
    crop_pct = cfg["evaluation"]["crop_percentage"]

    fieldnames = [
        "image", "bpp", "pressure", "payload_bytes",
        "adaptive_capacity_bytes", "mean_depth", "depth_std",
        "embed_time_s", "extract_time_s",
        "mse", "psnr", "ssim", "kl_divergence", "distortion_per_bit",
        "rs_estimated_rate", "rs_detected",
        "chi2_embedding_prob", "chi2_detected",
        "spa_estimated_rate", "spa_detected",
        "extraction_verified",
        # Robustness columns
        "rob_jpeg70_bit_acc", "rob_jpeg50_bit_acc",
        "rob_noise001_bit_acc", "rob_noise005_bit_acc",
        "rob_crop10_bit_acc",
    ]

    rows: List[Dict] = []

    for img_path in images:
        img_name = Path(img_path).name
        logger.info(f"Processing: {img_name}")
        cover = _load_cover(img_path)

        for bpp in bpp_levels:
            # Determine pressure heuristic: bpp / max_bpp
            max_bpp = cfg["stego"]["max_lsb_depth"] * (cover.shape[2] if cover.ndim == 3 else 1)
            pressure = min(1.0, bpp / max_bpp)

            # Compute depth map for capacity estimation
            depth_map = compute_depth_map(cover, controller, pressure, window_size)
            n_channels = cover.shape[2] if cover.ndim == 3 else 1
            ada_cap = adaptive_capacity_bytes(depth_map, n_channels)

            if ada_cap <= 0:
                logger.warning(f"  Skip bpp={bpp}: zero adaptive capacity")
                continue

            # Optionally clamp by bpp — leave margin for header + encryption overhead
            h, w = cover.shape[:2]
            bpp_budget_bytes = max(1, int(h * w * bpp / 8))
            # Use 50% of the smaller of adaptive capacity or bpp budget
            # to ensure header + crypto overhead fits comfortably
            payload_size = min(int(ada_cap * 0.45), int(bpp_budget_bytes * 0.45))
            if payload_size <= 0:
                continue

            payload = _generate_payload(payload_size, seed=seed)

            # Encrypt payload
            try:
                encrypted = encrypt_bytes(payload, password, kdf_algorithm=kdf_alg)
            except Exception:
                encrypted = encrypt_bytes(payload, password, kdf_algorithm="pbkdf2")

            # If encrypted is too big, use raw payload
            if len(encrypted) > ada_cap:
                encrypted = payload

            # Embed
            t0 = time.perf_counter()
            try:
                stego, used_depth_map = embed_adaptive(
                    cover, encrypted, controller,
                    bpp=bpp, pressure=pressure,
                    window_size=window_size, seed=seed,
                )
            except ValueError as e:
                logger.warning(f"  Embed failed bpp={bpp}: {e}")
                continue
            embed_time = time.perf_counter() - t0

            # Extract & verify
            t0 = time.perf_counter()
            try:
                extracted = extract_adaptive(
                    stego, controller,
                    pressure=pressure, window_size=window_size, seed=seed,
                )
                extract_time = time.perf_counter() - t0
                verified = extracted[:len(encrypted)] == encrypted
            except Exception as e:
                logger.warning(f"  Extract failed: {e}")
                extract_time = 0
                verified = False

            # Quality metrics
            metrics = compute_all(cover, stego, payload_bits=len(encrypted) * 8)

            # Steganalysis
            detectors = run_all_detectors(stego)

            # Robustness tests
            def _extract_fn(img):
                return extract_adaptive(
                    img, controller,
                    pressure=pressure, window_size=window_size, seed=seed,
                )

            rob_results = test_robustness(
                stego, encrypted, _extract_fn,
                jpeg_qualities=jpeg_qualities,
                noise_sigmas=noise_sigmas,
                crop_fraction=crop_pct,
                seed=seed,
            )

            # Parse robustness into flat columns
            rob_dict = {}
            for r in rob_results:
                if r.attack_name == "jpeg_compression":
                    q = r.attack_params["quality"]
                    rob_dict[f"rob_jpeg{q}_bit_acc"] = f"{r.bit_accuracy:.4f}"
                elif r.attack_name == "gaussian_noise":
                    s = r.attack_params["sigma"]
                    key = f"rob_noise{str(s).replace('.', '')}_bit_acc"
                    rob_dict[key] = f"{r.bit_accuracy:.4f}"
                elif r.attack_name == "center_crop":
                    rob_dict["rob_crop10_bit_acc"] = f"{r.bit_accuracy:.4f}"

            row = {
                "image": img_name,
                "bpp": bpp,
                "pressure": f"{pressure:.3f}",
                "payload_bytes": len(encrypted),
                "adaptive_capacity_bytes": ada_cap,
                "mean_depth": f"{used_depth_map.mean():.3f}",
                "depth_std": f"{used_depth_map.std():.3f}",
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
                **rob_dict,
            }
            rows.append(row)
            logger.info(
                f"  bpp={bpp:.2f}: PSNR={metrics['psnr']:.2f}dB "
                f"SSIM={metrics['ssim']:.4f} depth_μ={used_depth_map.mean():.2f}"
            )

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Adaptive results saved to {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive fuzzy LSB experiments.")
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

    run_adaptive(args.config)


if __name__ == "__main__":
    main()
