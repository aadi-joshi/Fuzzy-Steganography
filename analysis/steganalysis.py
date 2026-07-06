"""
Steganalysis Detectors
======================
Implements classical steganalysis attacks for evaluating the security of
LSB-based steganographic schemes:

1. **RS Analysis** (Fridrich, Goljan & Du, 2001)
   Estimates the embedding rate by comparing regular/singular group counts.

2. **Chi-Square (χ²) Attack** (Westfeld & Pfitzmann, 1999)
   Detects LSB embedding by testing the uniformity of Pairs of Values (PoVs).

3. **Sample Pair Analysis (SPA)** (Dumitrescu, Wu & Wang, 2003)
   Estimates the hidden message length using sample pair statistics.

All implementations are vectorized with NumPy.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import stats


# ============================================================================
# RS ANALYSIS
# ============================================================================
def _apply_mask_flip(group: np.ndarray, mask: np.ndarray, positive: bool) -> np.ndarray:
    """
    Apply a flipping operation to pixel values based on a mask.

    For positive mask (+1): flip LSB  (x → x^1 if mask bit is 1)
    For negative mask (−1): flip via x → x^1 then negate least significant
    """
    result = group.copy()
    for i in range(len(mask)):
        if mask[i] == 1:
            if positive:
                result[:, i] = result[:, i] ^ 1  # F1: flip LSB
            else:
                result[:, i] = result[:, i] ^ 1
                # F−1: flip then adjust neighbour parity
                result[:, i] = np.where(
                    result[:, i] % 2 == 0,
                    result[:, i] + 1,
                    result[:, i] - 1,
                )
    return result


def _discrimination_function(group: np.ndarray) -> np.ndarray:
    """
    Compute the discrimination value for each group of pixels.
    Uses the sum of absolute differences between adjacent pixels.
    """
    return np.sum(np.abs(np.diff(group.astype(np.float64), axis=1)), axis=1)


def rs_analysis(
    image: np.ndarray,
    group_width: int = 4,
) -> Dict[str, float]:
    """
    Perform RS steganalysis on a grayscale image.

    Estimates the hidden message length (embedding rate) by partitioning
    pixels into groups and measuring the relative counts of Regular (R),
    Singular (S), and Unusable (U) groups under positive and negative masks.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, uint8, shape (H, W).
    group_width : int
        Number of pixels per group (default 4).

    Returns
    -------
    dict
        Keys: ``R_m``, ``S_m``, ``R_neg_m``, ``S_neg_m``,
              ``estimated_rate``, ``detection_flag``.
    """
    if image.ndim == 3:
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        gray = image.copy()

    flat = gray.ravel()
    # Trim to multiple of group_width
    n = (len(flat) // group_width) * group_width
    flat = flat[:n]
    groups = flat.reshape(-1, group_width)

    mask_pos = np.array([0, 1, 1, 0])[:group_width]
    if len(mask_pos) < group_width:
        mask_pos = np.tile(mask_pos, (group_width // len(mask_pos) + 1))[:group_width]

    # Compute discrimination for original groups
    d_original = _discrimination_function(groups)

    # Apply positive mask (F1)
    groups_f1 = groups.copy()
    for i in range(group_width):
        if mask_pos[i] == 1:
            groups_f1[:, i] = groups_f1[:, i] ^ 1

    d_f1 = _discrimination_function(groups_f1)

    # Apply negative mask (F−1)
    groups_fn1 = groups.copy()
    for i in range(group_width):
        if mask_pos[i] == 1:
            even_mask = (groups_fn1[:, i] % 2 == 0)
            groups_fn1[:, i] = np.where(even_mask,
                                        groups_fn1[:, i] - 1,
                                        groups_fn1[:, i] + 1)

    d_fn1 = _discrimination_function(groups_fn1)

    n_groups = len(groups)

    # Positive mask statistics
    R_m = np.sum(d_f1 > d_original) / n_groups
    S_m = np.sum(d_f1 < d_original) / n_groups

    # Negative mask statistics
    R_neg_m = np.sum(d_fn1 > d_original) / n_groups
    S_neg_m = np.sum(d_fn1 < d_original) / n_groups

    # Estimate embedding rate using the RS quadratic equation
    # p = (R_m - S_m) / (R_neg_m - S_neg_m) approximately
    denom = R_neg_m - S_neg_m
    if abs(denom) > 1e-10:
        # Simplified RS estimator
        d0 = R_m - S_m
        d1 = R_neg_m - S_neg_m
        # Solve: 2(d1 + d0) * p^2 + (d1 - 3*d0) * p + d0 - d1 = 0
        a_coeff = 2 * (d1 + d0)
        b_coeff = d1 - 3 * d0
        c_coeff = d0 - d1

        if abs(a_coeff) > 1e-10:
            discriminant = b_coeff ** 2 - 4 * a_coeff * c_coeff
            if discriminant >= 0:
                p1 = (-b_coeff + np.sqrt(discriminant)) / (2 * a_coeff)
                p2 = (-b_coeff - np.sqrt(discriminant)) / (2 * a_coeff)
                # Choose the root in [0, 1]
                valid = [p for p in [p1, p2] if 0 <= p <= 1]
                est_rate = min(valid) if valid else 0.0
            else:
                est_rate = 0.0
        else:
            est_rate = 0.0
    else:
        est_rate = 0.0

    detection_flag = est_rate > 0.05  # threshold for "embedding detected"

    return {
        "R_m": R_m,
        "S_m": S_m,
        "R_neg_m": R_neg_m,
        "S_neg_m": S_neg_m,
        "estimated_rate": float(np.clip(est_rate, 0, 1)),
        "detection_flag": bool(detection_flag),
    }


# ============================================================================
# CHI-SQUARE ATTACK
# ============================================================================
def chi_square_attack(
    image: np.ndarray,
    block_size: int = 128,
) -> Dict[str, float]:
    """
    Chi-square (χ²) steganalysis attack on an image.

    Tests whether Pairs of Values (PoVs) (2i, 2i+1) have equal frequencies,
    which is the expected effect of LSB embedding.

    Parameters
    ----------
    image : np.ndarray
        Image (grayscale or colour), uint8.
    block_size : int
        Number of pixels per analysis block (for local detection).

    Returns
    -------
    dict
        Keys: ``chi2_statistic``, ``p_value``, ``embedding_probability``,
              ``detection_flag``.
    """
    flat = image.ravel().astype(np.int32)

    # Global histogram
    hist = np.bincount(flat, minlength=256).astype(np.float64)

    # Pairs of Values: (0,1), (2,3), ..., (254,255)
    chi2_stat = 0.0
    n_pairs = 0

    for i in range(0, 256, 2):
        expected = (hist[i] + hist[i + 1]) / 2.0
        if expected > 5:  # only use pairs with adequate counts
            chi2_stat += ((hist[i] - expected) ** 2 + (hist[i + 1] - expected) ** 2) / expected
            n_pairs += 1

    # Degrees of freedom = number of pairs used
    if n_pairs > 0:
        p_value = float(1.0 - stats.chi2.cdf(chi2_stat, df=n_pairs))
    else:
        p_value = 1.0

    # High p-value → PoV frequencies are uniform → embedding likely
    embedding_prob = 1.0 - p_value

    return {
        "chi2_statistic": float(chi2_stat),
        "p_value": p_value,
        "embedding_probability": float(np.clip(embedding_prob, 0, 1)),
        "detection_flag": bool(embedding_prob > 0.5),
    }


# ============================================================================
# SAMPLE PAIR ANALYSIS (SPA)
# ============================================================================
def sample_pair_analysis(
    image: np.ndarray,
) -> Dict[str, float]:
    """
    Sample Pair Analysis (SPA) steganalysis.

    Estimates the hidden message length by analysing horizontally adjacent
    pixel pairs and their LSB relationships.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, uint8, shape (H, W).

    Returns
    -------
    dict
        Keys: ``estimated_rate``, ``P``, ``Q``, ``detection_flag``.
    """
    if image.ndim == 3:
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        gray = image.copy()

    h, w = gray.shape
    # Horizontal neighbour pairs
    left = gray[:, :-1].ravel().astype(np.int32)
    right = gray[:, 1:].ravel().astype(np.int32)

    n = len(left)

    # Classify pairs
    # Close pairs: |left - right| ≤ 1
    diff = np.abs(left - right)

    # Pairs where both are even or both are odd
    same_parity = (left % 2) == (right % 2)
    diff_parity = ~same_parity

    # For clean images, we expect specific relationships
    # P: pairs where flipping LSB of left increases |diff|
    # Q: pairs where flipping LSB of left decreases |diff|

    left_flipped = left ^ 1
    diff_flipped = np.abs(left_flipped - right)

    P = np.sum(diff_flipped > diff) / n  # "constructive" pairs
    Q = np.sum(diff_flipped < diff) / n  # "destructive" pairs

    # In a clean image: P ≈ Q
    # After embedding at rate p: the relationship shifts
    # Estimate: p ≈ (P - Q) / (P + Q) when embedding is present
    if (P + Q) > 1e-10:
        est_rate = float(np.clip(abs(P - Q) / (P + Q), 0, 1))
    else:
        est_rate = 0.0

    return {
        "estimated_rate": est_rate,
        "P": float(P),
        "Q": float(Q),
        "detection_flag": bool(est_rate > 0.05),
    }


# ============================================================================
# Convenience: run all detectors
# ============================================================================
def run_all_detectors(
    image: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Run RS analysis, chi-square attack, and SPA on a single image.

    Returns
    -------
    dict
        Keys: ``rs``, ``chi_square``, ``spa``.
    """
    return {
        "rs": rs_analysis(image),
        "chi_square": chi_square_attack(image),
        "spa": sample_pair_analysis(image),
    }
