"""
Adaptive Fuzzy Logic-Based Steganographic Encryption Framework
==============================================================

Main entry point for running experiments and generating reports.

Usage examples:

    # Run the full pipeline (baseline + adaptive + comparison)
    python main.py --all

    # Run only the baseline experiment
    python main.py --baseline

    # Run only the adaptive experiment
    python main.py --adaptive

    # Run comparison & plotting (requires prior experiment runs)
    python main.py --compare

    # Specify a custom config
    python main.py --all --config path/to/config.yaml

    # Verbose logging
    python main.py --all -v
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import yaml

logger = logging.getLogger("main")


def _banner() -> str:
    return """
╔══════════════════════════════════════════════════════════════════════╗
║   Adaptive Fuzzy Logic-Based Steganographic Encryption Framework   ║
║                 Research Experiment Pipeline v1.0                   ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the requested experiment stages."""
    config_path = args.config
    if not os.path.exists(config_path):
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["experiment"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # --- Log experiment parameters ---
    logger.info("Experiment Configuration:")
    logger.info(f"  Random seed     : {cfg['random_seed']}")
    logger.info(f"  KDF algorithm   : {cfg['crypto']['kdf_algorithm']}")
    logger.info(f"  BPP levels      : {cfg['stego']['payload_bpp_levels']}")
    logger.info(f"  Cover directory  : {cfg['experiment']['cover_dir']}")
    logger.info(f"  Output directory : {output_dir}")

    t_start = time.perf_counter()

    # --- Stage 1: Baseline ---
    if args.baseline or args.all:
        logger.info("=" * 60)
        logger.info("STAGE 1: Running Baseline Experiments")
        logger.info("=" * 60)
        from experiments.run_baseline import run_baseline
        csv_path = run_baseline(config_path)
        logger.info(f"Baseline complete → {csv_path}")

    # --- Stage 2: Adaptive ---
    if args.adaptive or args.all:
        logger.info("=" * 60)
        logger.info("STAGE 2: Running Adaptive Fuzzy Experiments")
        logger.info("=" * 60)
        from experiments.run_adaptive import run_adaptive
        csv_path = run_adaptive(config_path)
        logger.info(f"Adaptive complete → {csv_path}")

    # --- Stage 3: Compare ---
    if args.compare or args.all:
        logger.info("=" * 60)
        logger.info("STAGE 3: Comparative Analysis & Plotting")
        logger.info("=" * 60)
        from experiments.compare_models import run_comparison
        run_comparison(config_path)
        logger.info("Comparison complete.")

    elapsed = time.perf_counter() - t_start
    logger.info(f"\nTotal pipeline time: {elapsed:.1f}s")
    logger.info(f"Results in: {output_dir}/")


def main() -> None:
    print(_banner())

    parser = argparse.ArgumentParser(
        description="Fuzzy Steganographic Encryption Framework — Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to YAML configuration file (default: config/config.yaml).",
    )
    parser.add_argument("--all", action="store_true", help="Run full pipeline.")
    parser.add_argument("--baseline", action="store_true", help="Run baseline experiments only.")
    parser.add_argument("--adaptive", action="store_true", help="Run adaptive experiments only.")
    parser.add_argument("--compare", action="store_true", help="Run comparison & plotting only.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("experiment.log", mode="w"),
        ],
    )

    if not (args.all or args.baseline or args.adaptive or args.compare):
        logger.info("No stage selected. Use --all, --baseline, --adaptive, or --compare.")
        parser.print_help()
        sys.exit(0)

    run_pipeline(args)


if __name__ == "__main__":
    main()
