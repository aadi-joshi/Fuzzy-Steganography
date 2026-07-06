"""
Feature-Based Deep Steganalysis Module
=======================================
Implements a lightweight Spatial Rich Model (SRM-lite) feature extractor
with Fisher Linear Discriminant (FLD) classification for LSB steganalysis.

This approach follows the methodology of Fridrich & Kodovsky (2012) but
uses a reduced filter bank (6 high-pass filters → 42 histogram features)
for computational feasibility without GPU.

Evaluation protocol:
    - 5-fold cross-validation (image-level folds, not sample-level)
    - AUC (Area Under ROC Curve)
    - TPR at 5% FPR (security-relevant metric)
    - Full ROC curve data for plotting

Limitation Statement:
    This module implements a *simplified* feature-based approach rather
    than full CNN-based steganalysis (e.g., SRNet, Yedroudj-Net).  CNN
    evaluations require GPU resources.  This feature-based detector still
    captures the primary statistical artefacts of LSB replacement and
    provides meaningful detection rates for Q1 experimental validation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import convolve2d


# ---------------------------------------------------------------------------
# High-pass filter bank (SRM-lite)
# ---------------------------------------------------------------------------
# 1st-order residuals
_F1_H = np.array([[-1.0, 1.0]])                          # horizontal
_F1_V = np.array([[-1.0], [1.0]])                         # vertical
# 2nd-order residuals
_F2_H = np.array([[1.0, -2.0, 1.0]])                     # horizontal
_F2_V = np.array([[1.0], [-2.0], [1.0]])                  # vertical
# 3rd-order residuals
_F3_H = np.array([[-1.0, 3.0, -3.0, 1.0]])               # horizontal
_F3_V = np.array([[-1.0], [3.0], [-3.0], [1.0]])          # vertical
# SPAM-like
_SPAM_H = np.array([[-1.0, 2.0, -1.0]])                   # 2nd-order (SPAM)
_SPAM_D = np.array([[-1.0, 0, 0], [0, 2.0, 0], [0, 0, -1.0]])  # diagonal
# Edge filters
_EDGE_3x3 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
_EDGE_5d = np.array([
    [-1, 2, -2, 2, -1],
    [2, -6, 8, -6, 2],
    [-2, 8, -12, 8, -2],
    [2, -6, 8, -6, 2],
    [-1, 2, -2, 2, -1],
], dtype=np.float64) / 12.0

FILTER_BANK = {
    "f1_h": _F1_H,
    "f1_v": _F1_V,
    "f2_h": _F2_H,
    "f2_v": _F2_V,
    "f3_h": _F3_H,
    "f3_v": _F3_V,
    "spam_h": _SPAM_H,
    "spam_d": _SPAM_D,
    "edge_3x3": _EDGE_3x3,
    "edge_5d": _EDGE_5d,
}

# Truncation threshold
_DEFAULT_T = 4
_N_FILTERS = len(FILTER_BANK)
_N_BINS = 2 * _DEFAULT_T + 1  # bins per histogram
FEATURE_DIM = _N_FILTERS * _N_BINS  # total features per image


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale float64."""
    if image.ndim == 3:
        return np.dot(image[..., :3].astype(np.float64),
                       [0.2989, 0.5870, 0.1140])
    return image.astype(np.float64)


def extract_srm_features(
    image: np.ndarray,
    T: int = _DEFAULT_T,
) -> np.ndarray:
    """
    Extract SRM-lite features from a single image.

    Parameters
    ----------
    image : np.ndarray
        Input image (grayscale or RGB), uint8.
    T : int
        Truncation threshold for residuals (default 4).

    Returns
    -------
    np.ndarray, shape (FEATURE_DIM,)
        Concatenated normalised histograms of filtered residuals.
    """
    gray = _to_grayscale(image)

    features = []
    for name, kernel in FILTER_BANK.items():
        residual = convolve2d(gray, kernel, mode="same", boundary="symm")
        # Truncate to [-T, T] and quantize to integers
        residual_q = np.clip(np.round(residual), -T, T).astype(np.int32) + T
        # Histogram over [0, 2T]
        hist = np.bincount(residual_q.ravel(), minlength=2 * T + 1)[:2 * T + 1]
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total > 0:
            hist /= total
        features.append(hist)

    return np.concatenate(features)


def extract_features_batch(
    images: List[np.ndarray],
    T: int = _DEFAULT_T,
) -> np.ndarray:
    """Extract SRM features for a batch of images → (N, FEATURE_DIM)."""
    return np.array([extract_srm_features(img, T) for img in images])


# ---------------------------------------------------------------------------
# Fisher Linear Discriminant (FLD)
# ---------------------------------------------------------------------------
def fisher_lda_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    reg: float = 1e-4,
) -> Tuple[np.ndarray, float]:
    """
    Train Fisher Linear Discriminant.

    Parameters
    ----------
    X_train : (N, D) feature matrix.
    y_train : (N,) binary labels {0, 1}.
    reg : float
        Tikhonov regularisation for within-class scatter.

    Returns
    -------
    w : (D,) projection direction.
    threshold : float, decision boundary on the projected axis.
    """
    mask0 = y_train == 0
    mask1 = y_train == 1
    X0 = X_train[mask0]
    X1 = X_train[mask1]

    if len(X0) < 2 or len(X1) < 2:
        return np.zeros(X_train.shape[1]), 0.0

    mu0 = np.mean(X0, axis=0)
    mu1 = np.mean(X1, axis=0)

    # Within-class scatter
    C0 = np.cov(X0.T) if X0.shape[0] > 1 else np.zeros((X0.shape[1], X0.shape[1]))
    C1 = np.cov(X1.T) if X1.shape[0] > 1 else np.zeros((X1.shape[1], X1.shape[1]))
    S_w = C0 * len(X0) + C1 * len(X1)
    S_w += np.eye(S_w.shape[0]) * reg  # regularisation

    w = np.linalg.solve(S_w, mu1 - mu0)
    norm = np.linalg.norm(w)
    if norm > 1e-10:
        w = w / norm

    threshold = float(np.dot((mu0 + mu1) / 2.0, w))
    return w, threshold


def fisher_lda_score(
    X: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Project samples onto Fisher direction. Higher → more stego-like."""
    return X @ w


# ---------------------------------------------------------------------------
# ROC computation
# ---------------------------------------------------------------------------
def compute_roc_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_thresholds: int = 500,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve from binary labels and continuous scores.

    Returns
    -------
    fpr : (M,) array of false positive rates.
    tpr : (M,) array of true positive rates.
    auc : float, area under the ROC curve.
    """
    y = np.asarray(y_true).ravel()
    s = np.asarray(scores).ravel()

    thresholds = np.linspace(s.min() - 1e-6, s.max() + 1e-6, n_thresholds)
    fpr = np.zeros(n_thresholds)
    tpr = np.zeros(n_thresholds)

    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)

    if n_pos == 0 or n_neg == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.5

    for i, t in enumerate(thresholds):
        pred_pos = s >= t
        tp = np.sum(pred_pos & (y == 1))
        fp = np.sum(pred_pos & (y == 0))
        tpr[i] = tp / n_pos
        fpr[i] = fp / n_neg

    # Sort by ascending FPR
    idx = np.argsort(fpr)
    fpr = fpr[idx]
    tpr = tpr[idx]

    # Deduplicate
    unique_mask = np.concatenate([[True], np.diff(fpr) > 0])
    fpr = fpr[unique_mask]
    tpr = tpr[unique_mask]

    # Ensure endpoints
    if fpr[0] > 0:
        fpr = np.concatenate([[0.0], fpr])
        tpr = np.concatenate([[0.0], tpr])
    if fpr[-1] < 1.0:
        fpr = np.concatenate([fpr, [1.0]])
        tpr = np.concatenate([tpr, [1.0]])

    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def tpr_at_fpr(
    fpr: np.ndarray,
    tpr: np.ndarray,
    target_fpr: float = 0.05,
) -> float:
    """Interpolate TPR at a given FPR threshold."""
    if len(fpr) < 2:
        return 0.0
    return float(np.interp(target_fpr, fpr, tpr))


# ---------------------------------------------------------------------------
# Cross-validated evaluation
# ---------------------------------------------------------------------------
def cross_validate_steganalysis(
    X_clean: np.ndarray,
    X_stego: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Run k-fold cross-validation for Fisher LDA steganalysis.

    Both X_clean and X_stego have one row per image.
    Folds split by image index (same image cannot be in both train and test).

    Returns
    -------
    dict with mean_auc, std_auc, mean_tpr_at_5fpr, fold_results, etc.
    """
    n = min(len(X_clean), len(X_stego))
    if n < n_folds:
        return {
            "mean_auc": 0.5, "std_auc": 0.0,
            "mean_tpr_at_5fpr": 0.0, "n_images": n,
            "fold_results": [],
        }

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    fold_size = n // n_folds
    fold_aucs = []
    fold_tpr5 = []
    all_fpr = []
    all_tpr = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        # Build train/test sets (clean=0, stego=1)
        X_train = np.vstack([X_clean[train_idx], X_stego[train_idx]])
        y_train = np.concatenate([
            np.zeros(len(train_idx)),
            np.ones(len(train_idx)),
        ])

        X_test = np.vstack([X_clean[test_idx], X_stego[test_idx]])
        y_test = np.concatenate([
            np.zeros(len(test_idx)),
            np.ones(len(test_idx)),
        ])

        # Train
        w, _ = fisher_lda_train(X_train, y_train)

        # Score
        scores = fisher_lda_score(X_test, w)

        # ROC
        fpr_arr, tpr_arr, auc = compute_roc_curve(y_test, scores)
        t5 = tpr_at_fpr(fpr_arr, tpr_arr, 0.05)

        fold_aucs.append(auc)
        fold_tpr5.append(t5)
        all_fpr.append(fpr_arr)
        all_tpr.append(tpr_arr)

    return {
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "ci95_auc_lo": float(np.mean(fold_aucs) - 1.96 * np.std(fold_aucs) / np.sqrt(n_folds)),
        "ci95_auc_hi": float(np.mean(fold_aucs) + 1.96 * np.std(fold_aucs) / np.sqrt(n_folds)),
        "mean_tpr_at_5fpr": float(np.mean(fold_tpr5)),
        "std_tpr_at_5fpr": float(np.std(fold_tpr5)),
        "n_images": n,
        "n_folds": n_folds,
        "fold_aucs": fold_aucs,
        "fold_tpr5": fold_tpr5,
        "all_fpr": all_fpr,
        "all_tpr": all_tpr,
    }


def mean_roc_curve(
    all_fpr: List[np.ndarray],
    all_tpr: List[np.ndarray],
    n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average ROC curves across folds. Returns (mean_fpr, mean_tpr, std_tpr)."""
    base_fpr = np.linspace(0, 1, n_points)
    interp_tprs = []
    for f, t in zip(all_fpr, all_tpr):
        interp_tprs.append(np.interp(base_fpr, f, t))
    interp_tprs = np.array(interp_tprs)
    mean_tpr = np.mean(interp_tprs, axis=0)
    std_tpr = np.std(interp_tprs, axis=0)
    return base_fpr, mean_tpr, std_tpr
