"""
Depth Map Synchronization Analysis
===================================
Quantifies feature stability between cover and stego images to validate
that the LSB-stripping strategy produces identical depth maps.

Measured quantities:
    - MAE between entropy/edge maps (cover vs. stego)
    - Percentage of pixels where depth map differs
    - Bit extraction failure rate due to depth mismatch
    - Histograms of depth differences
    - Robustness under mild additive noise
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Feature map comparison
# ---------------------------------------------------------------------------
def feature_maps_comparison(
    cover: np.ndarray,
    stego: np.ndarray,
    window_size: int = 7,
    strip_lsb: int = 3,
) -> Dict[str, float]:
    """Compare feature maps (entropy, edge) computed from cover and stego."""
    from stego.entropy import extract_features

    ent_c, edge_c = extract_features(cover, window_size=window_size, strip_lsb=strip_lsb)
    ent_s, edge_s = extract_features(stego, window_size=window_size, strip_lsb=strip_lsb)

    return {
        "entropy_mae": float(np.mean(np.abs(ent_c - ent_s))),
        "entropy_max_diff": float(np.max(np.abs(ent_c - ent_s))),
        "entropy_rmse": float(np.sqrt(np.mean((ent_c - ent_s) ** 2))),
        "edge_mae": float(np.mean(np.abs(edge_c - edge_s))),
        "edge_max_diff": float(np.max(np.abs(edge_c - edge_s))),
        "edge_rmse": float(np.sqrt(np.mean((edge_c - edge_s) ** 2))),
    }


def feature_maps_comparison_no_strip(
    cover: np.ndarray,
    stego: np.ndarray,
    window_size: int = 7,
) -> Dict[str, float]:
    """Compare feature maps WITHOUT LSB stripping (to show its necessity)."""
    from stego.entropy import extract_features

    ent_c, edge_c = extract_features(cover, window_size=window_size, strip_lsb=0)
    ent_s, edge_s = extract_features(stego, window_size=window_size, strip_lsb=0)

    return {
        "entropy_mae_nostrip": float(np.mean(np.abs(ent_c - ent_s))),
        "entropy_max_diff_nostrip": float(np.max(np.abs(ent_c - ent_s))),
        "edge_mae_nostrip": float(np.mean(np.abs(edge_c - edge_s))),
        "edge_max_diff_nostrip": float(np.max(np.abs(edge_c - edge_s))),
    }


# ---------------------------------------------------------------------------
# Depth map comparison
# ---------------------------------------------------------------------------
def depth_map_comparison(
    cover: np.ndarray,
    stego: np.ndarray,
    controller,
    pressure: float = 0.5,
    window_size: int = 7,
) -> Dict:
    """
    Compare depth maps computed from cover vs. stego images.

    Returns
    -------
    dict with depth_mae, pct_pixels_differ, depth_diff_histogram, etc.
    """
    from stego.lsb_adaptive import compute_depth_map

    dm_cover = compute_depth_map(cover, controller, pressure, window_size)
    dm_stego = compute_depth_map(stego, controller, pressure, window_size)

    diff = dm_cover.astype(np.int32) - dm_stego.astype(np.int32)
    n_pixels = diff.size

    mae = float(np.mean(np.abs(diff)))
    pct_differ = float(np.sum(diff != 0) / n_pixels * 100)

    unique, counts = np.unique(diff, return_counts=True)
    hist = {int(u): int(c) for u, c in zip(unique, counts)}

    return {
        "depth_mae": mae,
        "pct_pixels_differ": pct_differ,
        "n_pixels": n_pixels,
        "depth_diff_histogram": hist,
        "mean_cover_depth": float(dm_cover.mean()),
        "mean_stego_depth": float(dm_stego.mean()),
    }


# ---------------------------------------------------------------------------
# Noise robustness of depth synchronization
# ---------------------------------------------------------------------------
def sync_under_noise(
    cover: np.ndarray,
    controller,
    pressure: float = 0.5,
    window_size: int = 7,
    noise_sigma: float = 0.005,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Measure depth map synchronization when mild Gaussian noise is added
    to the stego image (simulating channel noise).
    """
    from stego.lsb_adaptive import compute_depth_map

    dm_cover = compute_depth_map(cover, controller, pressure, window_size)

    rng = np.random.RandomState(seed)
    noise = rng.normal(0, noise_sigma * 255, cover.shape)
    noisy = np.clip(cover.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    dm_noisy = compute_depth_map(noisy, controller, pressure, window_size)
    diff = dm_cover.astype(np.int32) - dm_noisy.astype(np.int32)

    return {
        "noise_sigma": noise_sigma,
        "depth_mae_noisy": float(np.mean(np.abs(diff))),
        "pct_pixels_differ_noisy": float(np.sum(diff != 0) / diff.size * 100),
    }


# ---------------------------------------------------------------------------
# Batch synchronization analysis
# ---------------------------------------------------------------------------
def batch_sync_analysis(
    covers: List[np.ndarray],
    stegos: List[np.ndarray],
    controller,
    pressures: List[float],
    window_size: int = 7,
    noise_sigmas: Optional[List[float]] = None,
    seed: int = 42,
) -> List[Dict]:
    """
    Run sync analysis for a batch of cover–stego pairs.

    Parameters
    ----------
    covers, stegos : lists of np.ndarray (same length)
    controller : FuzzyDepthController
    pressures : per-image pressures
    noise_sigmas : optional noise levels for robustness test

    Returns
    -------
    List of result dicts, one per image.
    """
    if noise_sigmas is None:
        noise_sigmas = [0.001, 0.005, 0.01]

    results = []
    for i, (cover, stego) in enumerate(zip(covers, stegos)):
        pressure = pressures[i] if i < len(pressures) else pressures[-1]

        row: Dict = {"image_idx": i}

        # Feature maps WITH strip (should be identical)
        feat = feature_maps_comparison(cover, stego, window_size, strip_lsb=3)
        row.update(feat)

        # Feature maps WITHOUT strip (shows LSB impact)
        feat_ns = feature_maps_comparison_no_strip(cover, stego, window_size)
        row.update(feat_ns)

        # Depth map comparison
        dm = depth_map_comparison(cover, stego, controller, pressure, window_size)
        row.update(dm)

        # Noise robustness
        for sigma in noise_sigmas:
            nr = sync_under_noise(
                cover, controller, pressure, window_size,
                noise_sigma=sigma, seed=seed,
            )
            row[f"depth_mae_noise_{sigma}"] = nr["depth_mae_noisy"]
            row[f"pct_differ_noise_{sigma}"] = nr["pct_pixels_differ_noisy"]

        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Extraction failure analysis
# ---------------------------------------------------------------------------
def extraction_failure_analysis(
    covers: List[np.ndarray],
    controller,
    bpp_levels: List[float],
    window_size: int = 7,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Measure bit extraction failure rate across images and bpp levels.

    A 'failure' means extracted != embedded (depth map mismatch caused errors).
    """
    from stego.lsb_adaptive import (
        adaptive_capacity_bytes,
        compute_depth_map,
        embed_adaptive,
        extract_adaptive,
    )

    results = {}

    for bpp in bpp_levels:
        n_total = 0
        n_failures = 0
        for cover in covers:
            h, w = cover.shape[:2]
            n_ch = cover.shape[2] if cover.ndim == 3 else 1
            max_bpp = 3 * n_ch
            pressure = min(1.0, bpp / max_bpp)

            depth_map = compute_depth_map(cover, controller, pressure, window_size)
            ada_cap = adaptive_capacity_bytes(depth_map, n_ch)
            payload_size = max(1, int(ada_cap * 0.25))

            rng = np.random.RandomState(seed)
            payload = bytes(rng.randint(0, 256, payload_size, dtype=np.uint8))

            try:
                stego, _ = embed_adaptive(
                    cover, payload, controller, bpp=bpp,
                    pressure=pressure, window_size=window_size, seed=seed,
                )
                extracted = extract_adaptive(
                    stego, controller, pressure=pressure,
                    window_size=window_size, seed=seed,
                )
                if extracted[:len(payload)] != payload:
                    n_failures += 1
            except Exception:
                n_failures += 1
            n_total += 1

        results[f"bpp_{bpp:.2f}_total"] = n_total
        results[f"bpp_{bpp:.2f}_failures"] = n_failures
        results[f"bpp_{bpp:.2f}_failure_rate"] = n_failures / max(n_total, 1)

    total = sum(v for k, v in results.items() if k.endswith("_total"))
    failures = sum(v for k, v in results.items() if k.endswith("_failures") and not k.endswith("_failure_rate"))
    results["overall_failure_rate"] = failures / max(total, 1)
    return results
