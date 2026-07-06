"""
Comparative Analysis & Visualization
=====================================
Loads baseline and adaptive experiment CSVs, produces comparative tables and
publication-ready plots.

Usage:
    python -m experiments.compare_models --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("compare")


def _safe_import_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        logger.error("pandas is required for comparative analysis. Install: pip install pandas")
        sys.exit(1)


def _safe_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        logger.error("matplotlib is required for plotting. Install: pip install matplotlib")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_results(output_dir: str):
    """Load baseline and adaptive CSVs into DataFrames."""
    pd = _safe_import_pandas()

    baseline_path = os.path.join(output_dir, "baseline_results.csv")
    adaptive_path = os.path.join(output_dir, "adaptive_results.csv")

    dfs = {}
    if os.path.exists(baseline_path):
        dfs["baseline"] = pd.read_csv(baseline_path)
        logger.info(f"Loaded baseline: {len(dfs['baseline'])} rows")
    else:
        logger.warning(f"Baseline results not found: {baseline_path}")

    if os.path.exists(adaptive_path):
        dfs["adaptive"] = pd.read_csv(adaptive_path)
        logger.info(f"Loaded adaptive: {len(dfs['adaptive'])} rows")
    else:
        logger.warning(f"Adaptive results not found: {adaptive_path}")

    return dfs


# ---------------------------------------------------------------------------
# Comparative tables
# ---------------------------------------------------------------------------
def generate_comparison_table(dfs: dict, output_dir: str) -> Optional[str]:
    """Generate a Markdown comparison table."""
    pd = _safe_import_pandas()

    rows = []

    if "baseline" in dfs:
        df = dfs["baseline"]
        for _, grp in df.groupby(["lsb_depth", "bpp"]):
            rows.append({
                "Method": f"Fixed LSB-{int(grp['lsb_depth'].iloc[0])}",
                "bpp": grp["bpp"].iloc[0],
                "PSNR (dB)": f"{pd.to_numeric(grp['psnr'], errors='coerce').mean():.2f}",
                "SSIM": f"{pd.to_numeric(grp['ssim'], errors='coerce').mean():.4f}",
                "MSE": f"{pd.to_numeric(grp['mse'], errors='coerce').mean():.4f}",
                "KL Div.": f"{pd.to_numeric(grp['kl_divergence'], errors='coerce').mean():.6f}",
                "RS Det. Rate": f"{grp['rs_detected'].mean():.2f}",
                "χ² Det. Rate": f"{grp['chi2_detected'].mean():.2f}",
                "SPA Det. Rate": f"{grp['spa_detected'].mean():.2f}",
            })

    if "adaptive" in dfs:
        df = dfs["adaptive"]
        for bpp, grp in df.groupby("bpp"):
            rows.append({
                "Method": "Fuzzy Adaptive",
                "bpp": bpp,
                "PSNR (dB)": f"{pd.to_numeric(grp['psnr'], errors='coerce').mean():.2f}",
                "SSIM": f"{pd.to_numeric(grp['ssim'], errors='coerce').mean():.4f}",
                "MSE": f"{pd.to_numeric(grp['mse'], errors='coerce').mean():.4f}",
                "KL Div.": f"{pd.to_numeric(grp['kl_divergence'], errors='coerce').mean():.6f}",
                "RS Det. Rate": f"{grp['rs_detected'].mean():.2f}",
                "χ² Det. Rate": f"{grp['chi2_detected'].mean():.2f}",
                "SPA Det. Rate": f"{grp['spa_detected'].mean():.2f}",
            })

    if not rows:
        logger.warning("No data available for comparison.")
        return None

    table_df = pd.DataFrame(rows)
    md_path = os.path.join(output_dir, "comparison_table.md")
    with open(md_path, "w") as f:
        f.write("# Comparative Results\n\n")
        f.write(table_df.to_markdown(index=False))
        f.write("\n")

    # Also save as CSV
    csv_path = os.path.join(output_dir, "comparison_table.csv")
    table_df.to_csv(csv_path, index=False)

    logger.info(f"Comparison table saved to {md_path}")
    return md_path


# ---------------------------------------------------------------------------
# Publication-ready plots
# ---------------------------------------------------------------------------
def generate_plots(dfs: dict, output_dir: str, fmt: str = "pdf", dpi: int = 300):
    """Generate publication-quality comparative plots."""
    pd = _safe_import_pandas()
    plt = _safe_import_matplotlib()

    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Style for journal publication
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (7, 5),
        "figure.dpi": dpi,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
    })

    # --- Plot 1: PSNR vs bpp ---
    fig, ax = plt.subplots()
    if "baseline" in dfs:
        df = dfs["baseline"]
        for depth in sorted(df["lsb_depth"].unique()):
            sub = df[df["lsb_depth"] == depth]
            grouped = sub.groupby("bpp")["psnr"].apply(
                lambda x: pd.to_numeric(x, errors="coerce").mean()
            )
            ax.plot(
                grouped.index, grouped.values,
                marker="s", label=f"Fixed LSB-{int(depth)}",
                linestyle="--",
            )
    if "adaptive" in dfs:
        df = dfs["adaptive"]
        grouped = df.groupby("bpp")["psnr"].apply(
            lambda x: pd.to_numeric(x, errors="coerce").mean()
        )
        ax.plot(
            grouped.index, grouped.values,
            marker="o", label="Fuzzy Adaptive",
            linestyle="-", color="crimson",
        )
    ax.set_xlabel("Embedding Rate (bpp)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Image Quality vs. Embedding Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"psnr_vs_bpp.{fmt}"), dpi=dpi)
    plt.close(fig)
    logger.info("  Plot: psnr_vs_bpp")

    # --- Plot 2: SSIM vs bpp ---
    fig, ax = plt.subplots()
    if "baseline" in dfs:
        df = dfs["baseline"]
        for depth in sorted(df["lsb_depth"].unique()):
            sub = df[df["lsb_depth"] == depth]
            grouped = sub.groupby("bpp")["ssim"].apply(
                lambda x: pd.to_numeric(x, errors="coerce").mean()
            )
            ax.plot(
                grouped.index, grouped.values,
                marker="s", label=f"Fixed LSB-{int(depth)}",
                linestyle="--",
            )
    if "adaptive" in dfs:
        df = dfs["adaptive"]
        grouped = df.groupby("bpp")["ssim"].apply(
            lambda x: pd.to_numeric(x, errors="coerce").mean()
        )
        ax.plot(
            grouped.index, grouped.values,
            marker="o", label="Fuzzy Adaptive",
            linestyle="-", color="crimson",
        )
    ax.set_xlabel("Embedding Rate (bpp)")
    ax.set_ylabel("SSIM")
    ax.set_title("Structural Similarity vs. Embedding Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"ssim_vs_bpp.{fmt}"), dpi=dpi)
    plt.close(fig)
    logger.info("  Plot: ssim_vs_bpp")

    # --- Plot 3: RS Detection Rate ---
    fig, ax = plt.subplots()
    if "baseline" in dfs:
        df = dfs["baseline"]
        for depth in sorted(df["lsb_depth"].unique()):
            sub = df[df["lsb_depth"] == depth]
            grouped = sub.groupby("bpp")["rs_detected"].mean()
            ax.plot(
                grouped.index, grouped.values,
                marker="s", label=f"Fixed LSB-{int(depth)}",
                linestyle="--",
            )
    if "adaptive" in dfs:
        df = dfs["adaptive"]
        grouped = df.groupby("bpp")["rs_detected"].mean()
        ax.plot(
            grouped.index, grouped.values,
            marker="o", label="Fuzzy Adaptive",
            linestyle="-", color="crimson",
        )
    ax.set_xlabel("Embedding Rate (bpp)")
    ax.set_ylabel("Detection Rate")
    ax.set_title("RS Steganalysis Detection Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"rs_detection_vs_bpp.{fmt}"), dpi=dpi)
    plt.close(fig)
    logger.info("  Plot: rs_detection_vs_bpp")

    # --- Plot 4: KL Divergence ---
    fig, ax = plt.subplots()
    if "baseline" in dfs:
        df = dfs["baseline"]
        for depth in sorted(df["lsb_depth"].unique()):
            sub = df[df["lsb_depth"] == depth]
            grouped = sub.groupby("bpp")["kl_divergence"].apply(
                lambda x: pd.to_numeric(x, errors="coerce").mean()
            )
            ax.plot(
                grouped.index, grouped.values,
                marker="s", label=f"Fixed LSB-{int(depth)}",
                linestyle="--",
            )
    if "adaptive" in dfs:
        df = dfs["adaptive"]
        grouped = df.groupby("bpp")["kl_divergence"].apply(
            lambda x: pd.to_numeric(x, errors="coerce").mean()
        )
        ax.plot(
            grouped.index, grouped.values,
            marker="o", label="Fuzzy Adaptive",
            linestyle="-", color="crimson",
        )
    ax.set_xlabel("Embedding Rate (bpp)")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Histogram Divergence vs. Embedding Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"kl_divergence_vs_bpp.{fmt}"), dpi=dpi)
    plt.close(fig)
    logger.info("  Plot: kl_divergence_vs_bpp")

    # --- Plot 5: Distortion per bit ---
    fig, ax = plt.subplots()
    if "baseline" in dfs:
        df = dfs["baseline"]
        for depth in sorted(df["lsb_depth"].unique()):
            sub = df[df["lsb_depth"] == depth]
            grouped = sub.groupby("bpp")["distortion_per_bit"].apply(
                lambda x: pd.to_numeric(x, errors="coerce").mean()
            )
            ax.plot(
                grouped.index, grouped.values,
                marker="s", label=f"Fixed LSB-{int(depth)}",
                linestyle="--",
            )
    if "adaptive" in dfs:
        df = dfs["adaptive"]
        grouped = df.groupby("bpp")["distortion_per_bit"].apply(
            lambda x: pd.to_numeric(x, errors="coerce").mean()
        )
        ax.plot(
            grouped.index, grouped.values,
            marker="o", label="Fuzzy Adaptive",
            linestyle="-", color="crimson",
        )
    ax.set_xlabel("Embedding Rate (bpp)")
    ax.set_ylabel("MSE / bit")
    ax.set_title("Embedding Efficiency (Distortion per Bit)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"distortion_per_bit_vs_bpp.{fmt}"), dpi=dpi)
    plt.close(fig)
    logger.info("  Plot: distortion_per_bit_vs_bpp")

    logger.info(f"All plots saved to {plot_dir}/")
    return plot_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_comparison(config_path: str) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["experiment"]["output_dir"]
    plot_fmt = cfg["experiment"].get("plot_format", "pdf")
    plot_dpi = cfg["experiment"].get("plot_dpi", 300)

    dfs = load_results(output_dir)
    generate_comparison_table(dfs, output_dir)
    generate_plots(dfs, output_dir, fmt=plot_fmt, dpi=plot_dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs. adaptive results.")
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

    run_comparison(args.config)


if __name__ == "__main__":
    main()
