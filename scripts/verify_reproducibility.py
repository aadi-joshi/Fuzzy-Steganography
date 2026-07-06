#!/usr/bin/env python3
"""
Verify that committed experiment outputs match published paper metrics.

Checks key headline numbers from the README against summary CSV/JSON files.
Exit code 0 if all checks pass within tolerance, 1 otherwise.

Usage:
    python scripts/verify_reproducibility.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TOLERANCE = 0.15  # dB or absolute metric tolerance for floating comparisons


def check(name: str, actual: float, expected: float, tol: float = TOLERANCE) -> bool:
    ok = abs(actual - expected) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: expected {expected:.4f}, got {actual:.4f}")
    return ok


def verify_v2_synthetic() -> bool:
    print("\n=== Synthetic V2 (1000 images) ===")
    csv_path = REPO_ROOT / "data/outputs_v2/v2_all_results.csv"
    if not csv_path.exists():
        print(f"  [FAIL] Missing {csv_path}")
        return False

    df = pd.read_csv(csv_path)
    ok = True
    for method, expected_psnr in [
        ("Fixed-LSB-1", 70.45),
        ("Fixed-LSB-2", 66.43),
        ("Adaptive", 73.25),
    ]:
        sub = df[(df["method"] == method) & (df["bpp"] == 0.05)]
        actual = sub["psnr"].astype(float).mean()
        ok &= check(f"PSNR @ 0.05 bpp ({method})", actual, expected_psnr)

    deep = pd.read_csv(REPO_ROOT / "data/outputs_v2/v2_deep_steganalysis.csv")
    for method, expected_auc in [
        ("Fixed-LSB-1", 0.754),
        ("Adaptive", 0.660),
    ]:
        row = deep[(deep["method"] == method) & (deep["bpp"] == 0.05)]
        actual = float(row.iloc[0]["mean_auc"])
        ok &= check(f"SRM-lite AUC @ 0.05 bpp ({method})", actual, expected_auc, tol=0.02)

    sync = pd.read_csv(REPO_ROOT / "data/outputs_v2/v2_sync_analysis.csv")
    depth_mae = sync["depth_mae"].astype(float).max()
    ok &= check("Depth map MAE (max)", depth_mae, 0.0, tol=1e-6)

    return ok


def verify_real_dataset(name: str, summary_path: Path, expected_adaptive_psnr: float) -> bool:
    print(f"\n=== {name} (200 images) ===")
    if not summary_path.exists():
        print(f"  [FAIL] Missing {summary_path}")
        return False
    with open(summary_path) as f:
        summary = json.load(f)
    adaptive = summary["results"]["Adaptive"]["0.05"]["psnr_mean"]
    fixed = summary["results"]["Fixed-LSB-1"]["0.05"]["psnr_mean"]
    ok = check(f"Adaptive PSNR @ 0.05 bpp", adaptive, expected_adaptive_psnr)
    ok &= check(
        "PSNR gain over Fixed-LSB-1",
        adaptive - fixed,
        expected_adaptive_psnr - 70.45,
        tol=0.2,
    )
    return ok


def verify_environment() -> bool:
    print("\n=== Environment metadata ===")
    env_path = REPO_ROOT / "data/outputs_v2/v2_environment.json"
    if not env_path.exists():
        print(f"  [FAIL] Missing {env_path}")
        return False
    with open(env_path) as f:
        env = json.load(f)
    ok = env.get("seed") == 42 or env.get("config", {}).get("random_seed") == 42
    print(f"  [{'PASS' if ok else 'FAIL'}] Random seed = 42")
    return ok


def main() -> int:
    print("Reproducibility verification against published headline metrics")
    results = [
        verify_v2_synthetic(),
        verify_real_dataset("BOSSBase", REPO_ROOT / "data/outputs_bossbase/summary.json", 73.42),
        verify_real_dataset("BOWS2", REPO_ROOT / "data/outputs_bows2/summary.json", 73.26),
        verify_real_dataset("MIRFLICKR", REPO_ROOT / "data/outputs_mirflickr/summary.json", 73.40),
        verify_environment(),
    ]
    passed = sum(results)
    total = len(results)
    print(f"\nSummary: {passed}/{total} check groups passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
