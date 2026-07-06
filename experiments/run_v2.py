"""
V2 Experiment Runner — Full Research Pipeline
===============================================
Runs all experiments for the second iteration with real data:

    Stage 1: Dataset preparation (generate or load)
    Stage 2: Main experiments (Baseline + Adaptive, all images × bpp)
    Stage 3: Statistical analysis (t-tests, Cohen's d, CI, power)
    Stage 4: Depth map synchronization analysis
    Stage 5: Feature-based deep steganalysis (SRM-lite + Fisher LDA)
    Stage 6: Ablation study (entropy-only, edge-only, no-pressure, full)
    Stage 7: Computational complexity profiling
    Stage 8: Plot generation
    Stage 9: Report generation

Usage:
    python experiments/run_v2.py --config config/config_v2.yaml
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.metrics import compute_all
from analysis.steganalysis import run_all_detectors
from analysis.statistical import (
    compare_across_bpp,
    confidence_interval_95,
    full_comparison,
)
from analysis.sync_analysis import batch_sync_analysis
from analysis.deep_steganalysis import (
    cross_validate_steganalysis,
    extract_features_batch,
    extract_srm_features,
    mean_roc_curve,
)
from crypto.aes import encrypt_bytes
from stego.entropy import extract_features
from stego.fuzzy import FuzzyDepthController
from stego.lsb_adaptive import (
    adaptive_capacity_bytes,
    compute_depth_map,
    embed_adaptive,
    extract_adaptive,
)
from stego.lsb_fixed import capacity_bytes, embed_fixed, extract_fixed

logger = logging.getLogger("v2")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _generate_payload(size_bytes: int, seed: int) -> bytes:
    rng = np.random.RandomState(seed)
    return bytes(rng.randint(0, 256, size_bytes, dtype=np.uint8).tolist())


def _eta_str(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def _write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------

def collect_environment() -> Dict[str, Any]:
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ===================================================================
# MAIN V2 RUNNER
# ===================================================================

class V2Runner:
    """Orchestrates the complete V2 experiment pipeline."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.seed = self.cfg["random_seed"]
        np.random.seed(self.seed)
        self.output_dir = self.cfg["experiment"]["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "plots"), exist_ok=True)
        self.bpp_levels = self.cfg["stego"]["payload_bpp_levels"]
        self.window_size = self.cfg["stego"]["fuzzy"]["entropy_window_size"]
        self.controller = FuzzyDepthController.from_config(self.cfg["stego"]["fuzzy"])
        self.password = "research_experiment_key_2025"
        self.kdf_alg = self.cfg["crypto"]["kdf_algorithm"]
        self.image_paths: List[str] = []
        self.plot_fmt = self.cfg["experiment"].get("plot_format", "pdf")
        self.plot_dpi = self.cfg["experiment"].get("plot_dpi", 300)

    # ---------------------------------------------------------------
    # Stage 1: Dataset
    # ---------------------------------------------------------------
    def stage_dataset(self) -> List[str]:
        logger.info("=" * 60)
        logger.info("STAGE 1: Dataset Preparation")
        logger.info("=" * 60)
        from experiments.generate_dataset import prepare_dataset

        ds_cfg = self.cfg.get("dataset", {})
        n = ds_cfg.get("n_images", 1000)
        sz = tuple(ds_cfg.get("image_size", [256, 256]))
        out = ds_cfg.get("output_dir", "data/covers_v2")
        boss = ds_cfg.get("bossbase_dir", None)

        self.image_paths = prepare_dataset(n, sz, out, boss, self.seed)
        logger.info(f"Dataset ready: {len(self.image_paths)} images")
        return self.image_paths

    # ---------------------------------------------------------------
    # Stage 2: Main experiments (Baseline + Adaptive)
    # ---------------------------------------------------------------
    def stage_main_experiments(self) -> str:
        logger.info("=" * 60)
        logger.info("STAGE 2: Main Experiments (Baseline + Adaptive)")
        logger.info("=" * 60)

        csv_path = os.path.join(self.output_dir, "v2_all_results.csv")
        fieldnames = [
            "image", "method", "bpp", "payload_bytes",
            "embed_time_s", "extract_time_s", "extraction_verified",
            "mse", "psnr", "ssim", "kl_divergence", "distortion_per_bit",
            "rs_estimated_rate", "rs_detected",
            "chi2_embedding_prob", "chi2_detected",
            "spa_estimated_rate", "spa_detected",
            "mean_depth", "depth_std",
        ]

        rows: List[Dict] = []
        # Also collect SRM features for deep steganalysis
        clean_features: List[np.ndarray] = []
        stego_features: Dict[Tuple[str, float], List[np.ndarray]] = {}

        methods_bpp = []
        for d in [1, 2]:
            for bpp in self.bpp_levels:
                methods_bpp.append((f"Fixed-LSB-{d}", d, bpp))
                stego_features[(f"Fixed-LSB-{d}", bpp)] = []
        for bpp in self.bpp_levels:
            stego_features[("Adaptive", bpp)] = []

        n_total = len(self.image_paths)
        t_start = time.perf_counter()

        for i, img_path in enumerate(self.image_paths):
            if i % 100 == 0 and i > 0:
                elapsed = time.perf_counter() - t_start
                eta = (elapsed / i) * (n_total - i)
                logger.info(f"  Progress: {i}/{n_total} ({_eta_str(eta)} remaining)")

            img_name = Path(img_path).name
            cover = _load_image(img_path)
            h, w = cover.shape[:2]
            n_ch = cover.shape[2] if cover.ndim == 3 else 1

            # SRM features for clean image
            clean_feat = extract_srm_features(cover)
            clean_features.append(clean_feat)

            # --- Baseline experiments ---
            for method_name, depth, bpp in methods_bpp:
                cap = capacity_bytes(cover, lsb_depth=depth, bpp=bpp)
                if cap <= 0:
                    continue
                payload_size = max(1, int(cap * 0.7))
                payload = _generate_payload(payload_size, self.seed)

                t0 = time.perf_counter()
                try:
                    stego = embed_fixed(cover, payload, lsb_depth=depth,
                                        bpp=bpp, seed=self.seed)
                except ValueError:
                    continue
                embed_t = time.perf_counter() - t0

                t0 = time.perf_counter()
                extracted = extract_fixed(stego, lsb_depth=depth, seed=self.seed)
                extract_t = time.perf_counter() - t0
                verified = extracted[:len(payload)] == payload

                metrics = compute_all(cover, stego, len(payload) * 8)
                det = run_all_detectors(stego)

                stego_feat = extract_srm_features(stego)
                stego_features[(method_name, bpp)].append(stego_feat)

                rows.append({
                    "image": img_name, "method": method_name, "bpp": bpp,
                    "payload_bytes": len(payload),
                    "embed_time_s": f"{embed_t:.6f}",
                    "extract_time_s": f"{extract_t:.6f}",
                    "extraction_verified": verified,
                    "mse": f"{metrics['mse']:.6f}",
                    "psnr": f"{metrics['psnr']:.4f}",
                    "ssim": f"{metrics['ssim']:.6f}",
                    "kl_divergence": f"{metrics['kl_divergence']:.8f}",
                    "distortion_per_bit": f"{metrics['distortion_per_bit']:.10f}",
                    "rs_estimated_rate": f"{det['rs']['estimated_rate']:.6f}",
                    "rs_detected": det["rs"]["detection_flag"],
                    "chi2_embedding_prob": f"{det['chi_square']['embedding_probability']:.6f}",
                    "chi2_detected": det["chi_square"]["detection_flag"],
                    "spa_estimated_rate": f"{det['spa']['estimated_rate']:.6f}",
                    "spa_detected": det["spa"]["detection_flag"],
                    "mean_depth": "", "depth_std": "",
                })

            # --- Adaptive experiments ---
            # Compute features once
            ent_map, edge_map = extract_features(cover, self.window_size)
            max_bpp = 3 * n_ch

            for bpp in self.bpp_levels:
                pressure = min(1.0, bpp / max_bpp)
                depth_map = self.controller.infer(ent_map, edge_map, pressure)
                ada_cap = adaptive_capacity_bytes(depth_map, n_ch)
                if ada_cap <= 0:
                    continue

                bpp_budget = max(1, int(h * w * bpp / 8))
                payload_size = min(int(ada_cap * 0.35), int(bpp_budget * 0.35))
                if payload_size <= 0:
                    continue
                payload = _generate_payload(payload_size, self.seed)

                t0 = time.perf_counter()
                try:
                    stego, used_dm = embed_adaptive(
                        cover, payload, self.controller,
                        bpp=bpp, pressure=pressure,
                        window_size=self.window_size, seed=self.seed,
                        depth_map=depth_map,
                    )
                except ValueError:
                    continue
                embed_t = time.perf_counter() - t0

                t0 = time.perf_counter()
                try:
                    extracted = extract_adaptive(
                        stego, self.controller,
                        pressure=pressure, window_size=self.window_size,
                        seed=self.seed, depth_map=depth_map,
                    )
                    extract_t = time.perf_counter() - t0
                    verified = extracted[:len(payload)] == payload
                except Exception:
                    extract_t = 0
                    verified = False

                metrics = compute_all(cover, stego, len(payload) * 8)
                det = run_all_detectors(stego)

                stego_feat = extract_srm_features(stego)
                stego_features[("Adaptive", bpp)].append(stego_feat)

                rows.append({
                    "image": img_name, "method": "Adaptive", "bpp": bpp,
                    "payload_bytes": len(payload),
                    "embed_time_s": f"{embed_t:.6f}",
                    "extract_time_s": f"{extract_t:.6f}",
                    "extraction_verified": verified,
                    "mse": f"{metrics['mse']:.6f}",
                    "psnr": f"{metrics['psnr']:.4f}",
                    "ssim": f"{metrics['ssim']:.6f}",
                    "kl_divergence": f"{metrics['kl_divergence']:.8f}",
                    "distortion_per_bit": f"{metrics['distortion_per_bit']:.10f}",
                    "rs_estimated_rate": f"{det['rs']['estimated_rate']:.6f}",
                    "rs_detected": det["rs"]["detection_flag"],
                    "chi2_embedding_prob": f"{det['chi_square']['embedding_probability']:.6f}",
                    "chi2_detected": det["chi_square"]["detection_flag"],
                    "spa_estimated_rate": f"{det['spa']['estimated_rate']:.6f}",
                    "spa_detected": det["spa"]["detection_flag"],
                    "mean_depth": f"{used_dm.mean():.4f}",
                    "depth_std": f"{used_dm.std():.4f}",
                })

        _write_csv(csv_path, rows, fieldnames)
        elapsed = time.perf_counter() - t_start
        logger.info(f"Main experiments complete: {len(rows)} rows in {_eta_str(elapsed)}")
        logger.info(f"  Saved to {csv_path}")

        # Save SRM features for deep steganalysis
        feat_path = os.path.join(self.output_dir, "srm_features.npz")
        save_dict = {"clean": np.array(clean_features)}
        for (method, bpp), feats in stego_features.items():
            if feats:
                key = f"stego_{method}_{bpp}".replace("-", "_").replace(".", "p")
                save_dict[key] = np.array(feats)
        np.savez_compressed(feat_path, **save_dict)
        logger.info(f"  SRM features saved to {feat_path}")

        return csv_path

    # ---------------------------------------------------------------
    # Stage 3: Statistical analysis
    # ---------------------------------------------------------------
    def stage_statistical(self) -> str:
        logger.info("=" * 60)
        logger.info("STAGE 3: Statistical Analysis")
        logger.info("=" * 60)
        import pandas as pd

        csv_in = os.path.join(self.output_dir, "v2_all_results.csv")
        csv_out = os.path.join(self.output_dir, "v2_statistical_tests.csv")
        df = pd.read_csv(csv_in)

        # Convert numeric columns
        for col in ["psnr", "ssim", "mse", "kl_divergence", "distortion_per_bit",
                     "rs_estimated_rate", "chi2_embedding_prob", "spa_estimated_rate"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        metrics = ["psnr", "ssim", "mse", "kl_divergence", "distortion_per_bit",
                    "rs_estimated_rate"]

        comparisons = [
            ("Fixed-LSB-1", "Adaptive"),
            ("Fixed-LSB-2", "Adaptive"),
            ("Fixed-LSB-1", "Fixed-LSB-2"),
        ]

        all_results = []
        for method_a, method_b in comparisons:
            results = compare_across_bpp(df, method_a, method_b, metrics)
            for r in results:
                r["comparison"] = f"{method_a} vs {method_b}"
            all_results.extend(results)

        if all_results:
            fieldnames = list(all_results[0].keys())
            _write_csv(csv_out, all_results, fieldnames)
            logger.info(f"Statistical analysis: {len(all_results)} comparisons saved to {csv_out}")
        else:
            logger.warning("No statistical comparisons produced.")

        return csv_out

    # ---------------------------------------------------------------
    # Stage 4: Synchronization analysis
    # ---------------------------------------------------------------
    def stage_sync(self) -> str:
        logger.info("=" * 60)
        logger.info("STAGE 4: Depth Map Synchronization Analysis")
        logger.info("=" * 60)

        n_sync = self.cfg.get("evaluation", {}).get("n_images_sync", 200)
        sync_paths = self.image_paths[:n_sync]
        bpp_test = [0.1, 0.2, 0.4]

        csv_out = os.path.join(self.output_dir, "v2_sync_analysis.csv")
        rows = []

        for bpp in bpp_test:
            logger.info(f"  Sync analysis at bpp={bpp}")
            covers = []
            stegos = []
            pressures = []

            for path in sync_paths:
                cover = _load_image(path)
                h, w = cover.shape[:2]
                n_ch = cover.shape[2] if cover.ndim == 3 else 1
                max_bpp = 3 * n_ch
                pressure = min(1.0, bpp / max_bpp)

                depth_map = compute_depth_map(cover, self.controller, pressure, self.window_size)
                ada_cap = adaptive_capacity_bytes(depth_map, n_ch)
                bpp_budget = max(1, int(h * w * bpp / 8))
                payload_size = min(int(ada_cap * 0.25), int(bpp_budget * 0.25))
                if payload_size <= 0:
                    continue
                payload = _generate_payload(payload_size, self.seed)

                try:
                    stego, _ = embed_adaptive(
                        cover, payload, self.controller,
                        bpp=bpp, pressure=pressure,
                        window_size=self.window_size, seed=self.seed,
                        depth_map=depth_map,
                    )
                    covers.append(cover)
                    stegos.append(stego)
                    pressures.append(pressure)
                except ValueError:
                    continue

            results = batch_sync_analysis(
                covers, stegos, self.controller, pressures,
                self.window_size, seed=self.seed,
            )
            for r in results:
                r["bpp"] = bpp
                # Flatten depth_diff_histogram
                if "depth_diff_histogram" in r:
                    r["depth_diff_histogram"] = str(r["depth_diff_histogram"])
                rows.append(r)

        if rows:
            fieldnames = list(rows[0].keys())
            _write_csv(csv_out, rows, fieldnames)
            logger.info(f"Sync analysis: {len(rows)} rows saved to {csv_out}")

        return csv_out

    # ---------------------------------------------------------------
    # Stage 5: Deep steganalysis
    # ---------------------------------------------------------------
    def stage_deep_steganalysis(self) -> str:
        logger.info("=" * 60)
        logger.info("STAGE 5: Feature-Based Deep Steganalysis")
        logger.info("=" * 60)

        feat_path = os.path.join(self.output_dir, "srm_features.npz")
        csv_out = os.path.join(self.output_dir, "v2_deep_steganalysis.csv")

        if not os.path.exists(feat_path):
            logger.warning("SRM features not found. Run stage 2 first.")
            return csv_out

        data = np.load(feat_path, allow_pickle=True)
        X_clean = data["clean"]
        n_folds = self.cfg.get("evaluation", {}).get("deep_steganalysis_folds", 5)

        rows = []
        roc_data = {}

        methods = ["Fixed_LSB_1", "Fixed_LSB_2", "Adaptive"]

        for method in methods:
            for bpp in self.bpp_levels:
                key = f"stego_{method}_{bpp}".replace(".", "p")
                if key not in data:
                    continue
                X_stego = data[key]
                n = min(len(X_clean), len(X_stego))
                if n < 20:
                    continue

                result = cross_validate_steganalysis(
                    X_clean[:n], X_stego[:n],
                    n_folds=n_folds, seed=self.seed,
                )

                method_label = method.replace("_", "-")
                logger.info(
                    f"  {method_label} bpp={bpp}: AUC={result['mean_auc']:.4f}±{result['std_auc']:.4f} "
                    f"TPR@5%FPR={result['mean_tpr_at_5fpr']:.4f}"
                )

                rows.append({
                    "method": method_label,
                    "bpp": bpp,
                    "n_images": n,
                    "n_folds": n_folds,
                    "mean_auc": f"{result['mean_auc']:.6f}",
                    "std_auc": f"{result['std_auc']:.6f}",
                    "ci95_auc_lo": f"{result['ci95_auc_lo']:.6f}",
                    "ci95_auc_hi": f"{result['ci95_auc_hi']:.6f}",
                    "mean_tpr_at_5fpr": f"{result['mean_tpr_at_5fpr']:.6f}",
                    "std_tpr_at_5fpr": f"{result['std_tpr_at_5fpr']:.6f}",
                })

                roc_data[(method_label, bpp)] = result

        if rows:
            fieldnames = list(rows[0].keys())
            _write_csv(csv_out, rows, fieldnames)
            logger.info(f"Deep steganalysis: {len(rows)} rows saved to {csv_out}")

        # Save ROC data for plotting
        roc_save = {}
        for (m, b), res in roc_data.items():
            base_fpr, mean_tpr, std_tpr = mean_roc_curve(res["all_fpr"], res["all_tpr"])
            k = f"{m}_{b}".replace("-", "_").replace(".", "p")
            roc_save[f"{k}_fpr"] = base_fpr
            roc_save[f"{k}_tpr"] = mean_tpr
            roc_save[f"{k}_tpr_std"] = std_tpr
        if roc_save:
            np.savez_compressed(
                os.path.join(self.output_dir, "roc_curves.npz"), **roc_save
            )

        return csv_out

    # ---------------------------------------------------------------
    # Stage 6: Ablation study
    # ---------------------------------------------------------------
    def stage_ablation(self) -> str:
        logger.info("=" * 60)
        logger.info("STAGE 6: Ablation Study")
        logger.info("=" * 60)

        n_abl = self.cfg.get("evaluation", {}).get("n_images_ablation", 100)
        abl_paths = self.image_paths[:n_abl]

        csv_out = os.path.join(self.output_dir, "v2_ablation_results.csv")
        fieldnames = [
            "image", "ablation", "bpp", "payload_bytes",
            "mse", "psnr", "ssim", "kl_divergence",
            "rs_detected", "chi2_detected", "spa_detected",
            "mean_depth", "extraction_verified",
        ]
        rows = []

        ablation_configs = {
            "full_system": {"entropy": True, "edge": True, "pressure": True},
            "entropy_only": {"entropy": True, "edge": False, "pressure": False},
            "edge_only": {"entropy": False, "edge": True, "pressure": False},
            "no_pressure": {"entropy": True, "edge": True, "pressure": False},
        }

        t_start = time.perf_counter()

        for i, img_path in enumerate(abl_paths):
            if i % 25 == 0 and i > 0:
                elapsed = time.perf_counter() - t_start
                eta = (elapsed / i) * (len(abl_paths) - i)
                logger.info(f"  Ablation progress: {i}/{len(abl_paths)} ({_eta_str(eta)} remaining)")

            img_name = Path(img_path).name
            cover = _load_image(img_path)
            h, w = cover.shape[:2]
            n_ch = cover.shape[2] if cover.ndim == 3 else 1
            max_bpp = 3 * n_ch

            # Compute real features
            ent_map, edge_map = extract_features(cover, self.window_size)

            for abl_name, cfg_flags in ablation_configs.items():
                # Modify inputs based on ablation
                ent_in = ent_map if cfg_flags["entropy"] else np.full_like(ent_map, 4.0)
                edge_in = edge_map if cfg_flags["edge"] else np.full_like(edge_map, 0.5)

                for bpp in self.bpp_levels:
                    if cfg_flags["pressure"]:
                        pressure = min(1.0, bpp / max_bpp)
                    else:
                        pressure = 0.0  # no pressure signal

                    depth_map = self.controller.infer(ent_in, edge_in, pressure)
                    ada_cap = adaptive_capacity_bytes(depth_map, n_ch)
                    if ada_cap <= 0:
                        continue

                    bpp_budget = max(1, int(h * w * bpp / 8))
                    payload_size = min(int(ada_cap * 0.30), int(bpp_budget * 0.30))
                    if payload_size <= 0:
                        continue
                    payload = _generate_payload(payload_size, self.seed)

                    try:
                        stego, used_dm = embed_adaptive(
                            cover, payload, self.controller,
                            bpp=bpp, pressure=pressure,
                            window_size=self.window_size, seed=self.seed,
                            depth_map=depth_map,
                        )
                    except ValueError:
                        continue

                    # Verify extraction (use same modified inputs for depth map)
                    try:
                        extracted = extract_adaptive(
                            stego, self.controller,
                            pressure=pressure, window_size=self.window_size,
                            seed=self.seed, depth_map=depth_map,
                        )
                        verified = extracted[:len(payload)] == payload
                    except Exception:
                        verified = False

                    metrics = compute_all(cover, stego, len(payload) * 8)
                    det = run_all_detectors(stego)

                    rows.append({
                        "image": img_name, "ablation": abl_name, "bpp": bpp,
                        "payload_bytes": len(payload),
                        "mse": f"{metrics['mse']:.6f}",
                        "psnr": f"{metrics['psnr']:.4f}",
                        "ssim": f"{metrics['ssim']:.6f}",
                        "kl_divergence": f"{metrics['kl_divergence']:.8f}",
                        "rs_detected": det["rs"]["detection_flag"],
                        "chi2_detected": det["chi_square"]["detection_flag"],
                        "spa_detected": det["spa"]["detection_flag"],
                        "mean_depth": f"{used_dm.mean():.4f}",
                        "extraction_verified": verified,
                    })

        _write_csv(csv_out, rows, fieldnames)
        elapsed = time.perf_counter() - t_start
        logger.info(f"Ablation study: {len(rows)} rows in {_eta_str(elapsed)}")
        return csv_out

    # ---------------------------------------------------------------
    # Stage 7: Computational complexity
    # ---------------------------------------------------------------
    def stage_complexity(self) -> str:
        logger.info("=" * 60)
        logger.info("STAGE 7: Computational Complexity Analysis")
        logger.info("=" * 60)

        n_comp = self.cfg.get("evaluation", {}).get("n_images_complexity", 50)
        comp_paths = self.image_paths[:n_comp]

        csv_out = os.path.join(self.output_dir, "v2_complexity.csv")
        fieldnames = [
            "image", "method", "bpp",
            "feature_extract_s", "fuzzy_infer_s",
            "embed_s", "extract_s", "total_s",
            "peak_memory_kb",
        ]
        rows = []
        bpp = 0.2  # single representative bpp

        for i, img_path in enumerate(comp_paths):
            img_name = Path(img_path).name
            cover = _load_image(img_path)
            h, w = cover.shape[:2]
            n_ch = cover.shape[2]
            max_bpp = 3 * n_ch
            pressure = min(1.0, bpp / max_bpp)

            # --- Fixed LSB-1 ---
            cap = capacity_bytes(cover, lsb_depth=1, bpp=bpp)
            if cap > 0:
                payload = _generate_payload(max(1, int(cap * 0.7)), self.seed)
                tracemalloc.start()
                t0 = time.perf_counter()
                stego = embed_fixed(cover, payload, lsb_depth=1, bpp=bpp, seed=self.seed)
                t_embed = time.perf_counter() - t0
                t0 = time.perf_counter()
                _ = extract_fixed(stego, lsb_depth=1, seed=self.seed)
                t_extract = time.perf_counter() - t0
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                rows.append({
                    "image": img_name, "method": "Fixed-LSB-1", "bpp": bpp,
                    "feature_extract_s": "0.0", "fuzzy_infer_s": "0.0",
                    "embed_s": f"{t_embed:.6f}", "extract_s": f"{t_extract:.6f}",
                    "total_s": f"{t_embed + t_extract:.6f}",
                    "peak_memory_kb": f"{peak / 1024:.1f}",
                })

            # --- Fixed LSB-2 ---
            cap = capacity_bytes(cover, lsb_depth=2, bpp=bpp)
            if cap > 0:
                payload = _generate_payload(max(1, int(cap * 0.7)), self.seed)
                tracemalloc.start()
                t0 = time.perf_counter()
                stego = embed_fixed(cover, payload, lsb_depth=2, bpp=bpp, seed=self.seed)
                t_embed = time.perf_counter() - t0
                t0 = time.perf_counter()
                _ = extract_fixed(stego, lsb_depth=2, seed=self.seed)
                t_extract = time.perf_counter() - t0
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                rows.append({
                    "image": img_name, "method": "Fixed-LSB-2", "bpp": bpp,
                    "feature_extract_s": "0.0", "fuzzy_infer_s": "0.0",
                    "embed_s": f"{t_embed:.6f}", "extract_s": f"{t_extract:.6f}",
                    "total_s": f"{t_embed + t_extract:.6f}",
                    "peak_memory_kb": f"{peak / 1024:.1f}",
                })

            # --- Adaptive ---
            tracemalloc.start()
            t0 = time.perf_counter()
            ent_map, edge_map = extract_features(cover, self.window_size)
            t_feat = time.perf_counter() - t0

            t0 = time.perf_counter()
            depth_map = self.controller.infer(ent_map, edge_map, pressure)
            t_fuzzy = time.perf_counter() - t0

            ada_cap = adaptive_capacity_bytes(depth_map, n_ch)
            if ada_cap > 0:
                bpp_budget = max(1, int(h * w * bpp / 8))
                ps = min(int(ada_cap * 0.30), int(bpp_budget * 0.30))
                if ps > 0:
                    payload = _generate_payload(ps, self.seed)
                    t0 = time.perf_counter()
                    try:
                        stego, _ = embed_adaptive(
                            cover, payload, self.controller,
                            bpp=bpp, pressure=pressure,
                            window_size=self.window_size, seed=self.seed,
                            depth_map=depth_map,
                        )
                        t_embed = time.perf_counter() - t0
                        t0 = time.perf_counter()
                        _ = extract_adaptive(
                            stego, self.controller,
                            pressure=pressure, window_size=self.window_size,
                            seed=self.seed, depth_map=depth_map,
                        )
                        t_extract = time.perf_counter() - t0
                    except Exception:
                        t_embed = 0
                        t_extract = 0
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    rows.append({
                        "image": img_name, "method": "Adaptive", "bpp": bpp,
                        "feature_extract_s": f"{t_feat:.6f}",
                        "fuzzy_infer_s": f"{t_fuzzy:.6f}",
                        "embed_s": f"{t_embed:.6f}",
                        "extract_s": f"{t_extract:.6f}",
                        "total_s": f"{t_feat + t_fuzzy + t_embed + t_extract:.6f}",
                        "peak_memory_kb": f"{peak / 1024:.1f}",
                    })
                else:
                    tracemalloc.stop()
            else:
                tracemalloc.stop()

        _write_csv(csv_out, rows, fieldnames)
        logger.info(f"Complexity analysis: {len(rows)} rows saved to {csv_out}")
        return csv_out

    # ---------------------------------------------------------------
    # Stage 8: Plot generation
    # ---------------------------------------------------------------
    def stage_plots(self):
        logger.info("=" * 60)
        logger.info("STAGE 8: Plot Generation")
        logger.info("=" * 60)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        plt.rcParams.update({
            "font.size": 11, "font.family": "serif",
            "axes.labelsize": 12, "axes.titlesize": 13,
            "legend.fontsize": 9, "xtick.labelsize": 10,
            "ytick.labelsize": 10, "figure.figsize": (7, 5),
            "figure.dpi": self.plot_dpi, "lines.linewidth": 1.5,
            "lines.markersize": 6,
        })

        plot_dir = os.path.join(self.output_dir, "plots")
        fmt = self.plot_fmt

        # Load main results
        csv_main = os.path.join(self.output_dir, "v2_all_results.csv")
        if os.path.exists(csv_main):
            df = pd.read_csv(csv_main)
            for col in ["psnr", "ssim", "mse", "kl_divergence",
                         "distortion_per_bit", "rs_estimated_rate"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            self._plot_metric_vs_bpp(df, "psnr", "PSNR (dB)", "Image Quality (PSNR) vs. Embedding Rate",
                                     plot_dir, fmt, plt)
            self._plot_metric_vs_bpp(df, "ssim", "SSIM", "Structural Similarity vs. Embedding Rate",
                                     plot_dir, fmt, plt)
            self._plot_metric_vs_bpp(df, "mse", "MSE", "Distortion (MSE) vs. Embedding Rate",
                                     plot_dir, fmt, plt)
            self._plot_metric_vs_bpp(df, "kl_divergence", "KL Divergence",
                                     "Histogram Divergence vs. Embedding Rate",
                                     plot_dir, fmt, plt)
            self._plot_detection_rate(df, plot_dir, fmt, plt)

        # ROC curves
        roc_path = os.path.join(self.output_dir, "roc_curves.npz")
        if os.path.exists(roc_path):
            self._plot_roc_curves(roc_path, plot_dir, fmt, plt)

        # Ablation plot
        csv_abl = os.path.join(self.output_dir, "v2_ablation_results.csv")
        if os.path.exists(csv_abl):
            self._plot_ablation(csv_abl, plot_dir, fmt, plt)

        # Complexity plot
        csv_comp = os.path.join(self.output_dir, "v2_complexity.csv")
        if os.path.exists(csv_comp):
            self._plot_complexity(csv_comp, plot_dir, fmt, plt)

        logger.info(f"All plots saved to {plot_dir}/")

    def _plot_metric_vs_bpp(self, df, metric, ylabel, title, plot_dir, fmt, plt):
        fig, ax = plt.subplots()
        colors = {"Fixed-LSB-1": "#2196F3", "Fixed-LSB-2": "#FF9800", "Adaptive": "#E91E63"}
        markers = {"Fixed-LSB-1": "s", "Fixed-LSB-2": "D", "Adaptive": "o"}

        for method in ["Fixed-LSB-1", "Fixed-LSB-2", "Adaptive"]:
            sub = df[df["method"] == method]
            if sub.empty:
                continue
            grouped = sub.groupby("bpp")[metric]
            means = grouped.mean()
            stds = grouped.std()
            ax.errorbar(
                means.index, means.values, yerr=1.96 * stds.values / np.sqrt(grouped.count().values),
                marker=markers.get(method, "o"),
                label=method,
                color=colors.get(method, None),
                linestyle="--" if "Fixed" in method else "-",
                capsize=3,
            )
        ax.set_xlabel("Embedding Rate (bpp)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{metric}_vs_bpp.{fmt}"), dpi=self.plot_dpi)
        plt.close(fig)
        logger.info(f"  Plot: {metric}_vs_bpp")

    def _plot_detection_rate(self, df, plot_dir, fmt, plt):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, det_col, det_name in zip(
            axes,
            ["rs_detected", "chi2_detected", "spa_detected"],
            ["RS Analysis", "Chi-Square", "SPA"],
        ):
            for method in ["Fixed-LSB-1", "Fixed-LSB-2", "Adaptive"]:
                sub = df[df["method"] == method]
                if sub.empty:
                    continue
                grouped = sub.groupby("bpp")[det_col].mean()
                ax.plot(grouped.index, grouped.values,
                        marker="o" if method == "Adaptive" else "s",
                        label=method,
                        linestyle="-" if method == "Adaptive" else "--")
            ax.set_xlabel("Embedding Rate (bpp)")
            ax.set_ylabel("Detection Rate")
            ax.set_title(det_name)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"detection_rates.{fmt}"), dpi=self.plot_dpi)
        plt.close(fig)
        logger.info("  Plot: detection_rates")

    def _plot_roc_curves(self, roc_path, plot_dir, fmt, plt):
        data = np.load(roc_path)
        # One plot per bpp level
        for bpp in self.bpp_levels:
            fig, ax = plt.subplots()
            for method in ["Fixed-LSB-1", "Fixed-LSB-2", "Adaptive"]:
                k = f"{method}_{bpp}".replace("-", "_").replace(".", "p")
                fpr_key = f"{k}_fpr"
                tpr_key = f"{k}_tpr"
                if fpr_key in data and tpr_key in data:
                    fpr = data[fpr_key]
                    tpr = data[tpr_key]
                    auc = float(np.trapz(tpr, fpr))
                    ax.plot(fpr, tpr, label=f"{method} (AUC={auc:.3f})")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curves at bpp={bpp}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            bpp_str = f"{bpp}".replace(".", "p")
            fig.savefig(os.path.join(plot_dir, f"roc_bpp_{bpp_str}.{fmt}"), dpi=self.plot_dpi)
            plt.close(fig)
        logger.info("  Plot: ROC curves (per bpp)")

    def _plot_ablation(self, csv_path, plot_dir, fmt, plt):
        import pandas as pd
        df = pd.read_csv(csv_path)
        df["psnr"] = pd.to_numeric(df["psnr"], errors="coerce")
        df["rs_detected"] = df["rs_detected"].astype(bool)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ablations = ["full_system", "entropy_only", "edge_only", "no_pressure"]
        colors = ["#E91E63", "#2196F3", "#FF9800", "#4CAF50"]
        for abl, color in zip(ablations, colors):
            sub = df[df["ablation"] == abl]
            g = sub.groupby("bpp")["psnr"]
            ax1.plot(g.mean().index, g.mean().values, marker="o",
                     label=abl.replace("_", " ").title(), color=color)
        ax1.set_xlabel("Embedding Rate (bpp)")
        ax1.set_ylabel("PSNR (dB)")
        ax1.set_title("Ablation: Image Quality")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        for abl, color in zip(ablations, colors):
            sub = df[df["ablation"] == abl]
            g = sub.groupby("bpp")["rs_detected"].mean()
            ax2.plot(g.index, g.values, marker="o",
                     label=abl.replace("_", " ").title(), color=color)
        ax2.set_xlabel("Embedding Rate (bpp)")
        ax2.set_ylabel("RS Detection Rate")
        ax2.set_title("Ablation: Detection Rate")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)

        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"ablation.{fmt}"), dpi=self.plot_dpi)
        plt.close(fig)
        logger.info("  Plot: ablation")

    def _plot_complexity(self, csv_path, plot_dir, fmt, plt):
        import pandas as pd
        df = pd.read_csv(csv_path)
        df["total_s"] = pd.to_numeric(df["total_s"], errors="coerce")
        df["peak_memory_kb"] = pd.to_numeric(df["peak_memory_kb"], errors="coerce")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        methods = df["method"].unique()
        means_t = [df[df["method"] == m]["total_s"].mean() for m in methods]
        stds_t = [df[df["method"] == m]["total_s"].std() for m in methods]
        ax1.bar(methods, means_t, yerr=stds_t, capsize=5,
                color=["#2196F3", "#FF9800", "#E91E63"][:len(methods)])
        ax1.set_ylabel("Time (seconds)")
        ax1.set_title("Embed+Extract Time per Image")
        ax1.grid(True, alpha=0.3, axis="y")

        means_m = [df[df["method"] == m]["peak_memory_kb"].mean() for m in methods]
        stds_m = [df[df["method"] == m]["peak_memory_kb"].std() for m in methods]
        ax2.bar(methods, means_m, yerr=stds_m, capsize=5,
                color=["#2196F3", "#FF9800", "#E91E63"][:len(methods)])
        ax2.set_ylabel("Peak Memory (KB)")
        ax2.set_title("Memory Usage per Image")
        ax2.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"complexity.{fmt}"), dpi=self.plot_dpi)
        plt.close(fig)
        logger.info("  Plot: complexity")

    # ---------------------------------------------------------------
    # Stage 9: Save environment + summary
    # ---------------------------------------------------------------
    def stage_environment(self):
        env = collect_environment()
        env["config"] = self.cfg
        env["n_images"] = len(self.image_paths)
        env["seed"] = self.seed
        env_path = os.path.join(self.output_dir, "v2_environment.json")
        with open(env_path, "w") as f:
            json.dump(env, f, indent=2, default=str)
        logger.info(f"Environment info saved to {env_path}")

    # ---------------------------------------------------------------
    # Run all stages
    # ---------------------------------------------------------------
    def run_all(self):
        t_total = time.perf_counter()

        self.stage_dataset()
        self.stage_main_experiments()
        self.stage_statistical()
        self.stage_sync()
        self.stage_deep_steganalysis()
        self.stage_ablation()
        self.stage_complexity()
        self.stage_plots()
        self.stage_environment()

        elapsed = time.perf_counter() - t_total
        logger.info("=" * 60)
        logger.info(f"V2 PIPELINE COMPLETE — Total time: {_eta_str(elapsed)}")
        logger.info(f"All results in: {self.output_dir}/")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║   Adaptive Fuzzy Steganographic Encryption Framework — V2 Runner   ║
║                    Full Research Experiment Pipeline                ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    parser = argparse.ArgumentParser(description="V2 Full Experiment Pipeline")
    parser.add_argument("--config", default="config/config_v2.yaml")
    parser.add_argument("--stage", type=int, default=0,
                        help="Run only a specific stage (1-9), 0=all")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("experiment_v2.log", mode="w"),
        ],
    )

    runner = V2Runner(args.config)

    if args.stage == 0:
        runner.run_all()
    else:
        stages = {
            1: runner.stage_dataset,
            2: runner.stage_main_experiments,
            3: runner.stage_statistical,
            4: runner.stage_sync,
            5: runner.stage_deep_steganalysis,
            6: runner.stage_ablation,
            7: runner.stage_complexity,
            8: runner.stage_plots,
            9: runner.stage_environment,
        }
        if args.stage in stages:
            if args.stage > 1:
                runner.stage_dataset()  # always need images
            stages[args.stage]()
        else:
            logger.error(f"Invalid stage: {args.stage}")


if __name__ == "__main__":
    main()
