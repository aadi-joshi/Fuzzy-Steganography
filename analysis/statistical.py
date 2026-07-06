"""
Statistical Analysis Module
============================
Provides paired t-tests, Cohen's d, 95% confidence intervals,
and statistical power analysis for comparing steganographic methods.

All comparisons follow:
    H0: There is no significant difference between methods A and B.
    H1: There is a significant difference between methods A and B.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from scipy.stats import nct


# ---------------------------------------------------------------------------
# Core statistical functions
# ---------------------------------------------------------------------------
def paired_ttest(
    a: np.ndarray,
    b: np.ndarray,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Paired t-test between two matched sample arrays.

    Parameters
    ----------
    a, b : array-like
        Matched observations.
    alternative : str
        'two-sided', 'less', or 'greater'.

    Returns
    -------
    dict with t_statistic, p_value, df, n.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if len(a) != len(b):
        raise ValueError("Arrays must have same length.")
    if len(a) < 3:
        return {"t_statistic": 0.0, "p_value": 1.0, "df": 0, "n": len(a)}

    t_stat, p_val = stats.ttest_rel(a, b, alternative=alternative)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "df": len(a) - 1,
        "n": len(a),
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cohen's d for paired samples (standardised mean difference).

    Interpretation:
        |d| < 0.2  → negligible
        0.2 ≤ |d| < 0.5  → small
        0.5 ≤ |d| < 0.8  → medium
        |d| ≥ 0.8  → large
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    diff = a - b
    sd = np.std(diff, ddof=1)
    if sd < 1e-15:
        return 0.0
    return float(np.mean(diff) / sd)


def confidence_interval_95(data: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute 95% confidence interval for the mean.

    Returns
    -------
    (mean, ci_low, ci_high)
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    n = len(data)
    if n < 2:
        m = float(data[0]) if n == 1 else 0.0
        return m, m, m
    mean = float(np.mean(data))
    sem = float(stats.sem(data))
    h = sem * stats.t.ppf(0.975, n - 1)
    return mean, mean - h, mean + h


def confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Generalised CI at any confidence level."""
    data = np.asarray(data, dtype=np.float64).ravel()
    n = len(data)
    if n < 2:
        m = float(data[0]) if n == 1 else 0.0
        return m, m, m
    mean = float(np.mean(data))
    sem = float(stats.sem(data))
    alpha = 1.0 - confidence
    h = sem * stats.t.ppf(1.0 - alpha / 2, n - 1)
    return mean, mean - h, mean + h


def statistical_power(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
) -> float:
    """
    Approximate statistical power for a paired t-test using
    the non-central t-distribution.

    Parameters
    ----------
    effect_size : float
        Cohen's d (unsigned).
    n : int
        Number of paired observations.
    alpha : float
        Significance level.

    Returns
    -------
    float in [0, 1].
    """
    if abs(effect_size) < 1e-10 or n < 3:
        return 0.0
    df = n - 1
    ncp_val = abs(effect_size) * np.sqrt(n)
    t_crit = stats.t.ppf(1.0 - alpha / 2, df)
    power = 1.0 - nct.cdf(t_crit, df, ncp_val) + nct.cdf(-t_crit, df, ncp_val)
    return float(np.clip(power, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Bonferroni correction
# ---------------------------------------------------------------------------
def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Dict[str, float]]:
    """Apply Bonferroni correction to a list of p-values."""
    m = len(p_values)
    adjusted_alpha = alpha / m
    return [
        {
            "p_original": p,
            "p_adjusted": min(p * m, 1.0),
            "alpha_corrected": adjusted_alpha,
            "significant": p < adjusted_alpha,
        }
        for p in p_values
    ]


# ---------------------------------------------------------------------------
# Full comparison between two matched arrays
# ---------------------------------------------------------------------------
def full_comparison(
    values_a: np.ndarray,
    values_b: np.ndarray,
    label_a: str = "A",
    label_b: str = "B",
    metric_name: str = "metric",
) -> Dict:
    """
    Complete statistical comparison between two matched arrays.

    Returns a dict containing descriptive stats, t-test, Cohen's d,
    95% CI, and power for both methods.
    """
    a = np.asarray(values_a, dtype=np.float64).ravel()
    b = np.asarray(values_b, dtype=np.float64).ravel()
    if len(a) != len(b):
        raise ValueError("Arrays must have same length.")

    mean_a, ci_a_lo, ci_a_hi = confidence_interval_95(a)
    mean_b, ci_b_lo, ci_b_hi = confidence_interval_95(b)
    std_a = float(np.std(a, ddof=1))
    std_b = float(np.std(b, ddof=1))

    ttest = paired_ttest(a, b)
    d = cohens_d(a, b)
    power = statistical_power(abs(d), len(a))

    return {
        "metric": metric_name,
        "label_a": label_a,
        "label_b": label_b,
        "n": len(a),
        "mean_a": mean_a,
        "std_a": std_a,
        "ci95_a_lo": ci_a_lo,
        "ci95_a_hi": ci_a_hi,
        "mean_b": mean_b,
        "std_b": std_b,
        "ci95_b_lo": ci_b_lo,
        "ci95_b_hi": ci_b_hi,
        "mean_diff": mean_a - mean_b,
        "t_statistic": ttest["t_statistic"],
        "p_value": ttest["p_value"],
        "df": ttest["df"],
        "cohens_d": d,
        "effect_magnitude": _interpret_d(d),
        "power": power,
        "significant_005": ttest["p_value"] < 0.05,
        "significant_001": ttest["p_value"] < 0.01,
    }


def _interpret_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


# ---------------------------------------------------------------------------
# Batch analysis across BPP levels
# ---------------------------------------------------------------------------
def compare_across_bpp(
    df,  # pandas DataFrame with columns: method, bpp, <metrics>
    method_a: str,
    method_b: str,
    metrics: List[str],
) -> List[Dict]:
    """
    Run full_comparison for every (bpp, metric) combination.

    Expects *df* to have one row per image per method per bpp.
    """
    import pandas as pd

    results = []
    bpp_levels = sorted(df["bpp"].unique())
    for bpp in bpp_levels:
        sub_a = df[(df["method"] == method_a) & (df["bpp"] == bpp)]
        sub_b = df[(df["method"] == method_b) & (df["bpp"] == bpp)]

        # Align by image name
        merged = pd.merge(
            sub_a, sub_b, on="image", suffixes=("_a", "_b"), how="inner"
        )
        if len(merged) < 3:
            continue

        for metric in metrics:
            col_a = f"{metric}_a"
            col_b = f"{metric}_b"
            if col_a not in merged.columns or col_b not in merged.columns:
                continue

            va = pd.to_numeric(merged[col_a], errors="coerce").dropna().values
            vb = pd.to_numeric(merged[col_b], errors="coerce").dropna().values
            n = min(len(va), len(vb))
            if n < 3:
                continue

            comp = full_comparison(
                va[:n], vb[:n],
                label_a=method_a,
                label_b=method_b,
                metric_name=metric,
            )
            comp["bpp"] = bpp
            results.append(comp)

    return results
