"""
Real Dataset Experiment Runner
================================
Runs the full experiment pipeline on real image datasets (BOSSBase, BOWS2, MIRFLICKR).
Produces quality metrics, steganalysis results, statistical analysis, and plots.

Usage:
    python experiments/run_real_dataset.py --config config/config_bossbase.yaml
    python experiments/run_real_dataset.py --config config/config_bows2.yaml
    python experiments/run_real_dataset.py --config config/config_mirflickr.yaml
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.metrics import compute_all
from analysis.steganalysis import run_all_detectors
from analysis.statistical import compare_across_bpp
from analysis.deep_steganalysis import (
    cross_validate_steganalysis,
    extract_srm_features,
    mean_roc_curve,
)
from stego.entropy import extract_features
from stego.fuzzy import FuzzyDepthController
from stego.lsb_adaptive import (
    adaptive_capacity_bytes,
    compute_depth_map,
    embed_adaptive,
    extract_adaptive,
)
from stego.lsb_fixed import capacity_bytes, embed_fixed, extract_fixed

logger = logging.getLogger("real_dataset")


def _load_image(path: str, target_size=None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if target_size:
        img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def _generate_payload(size_bytes: int, seed: int) -> bytes:
    rng = np.random.RandomState(seed)
    return bytes(rng.randint(0, 256, size_bytes, dtype=np.uint8).tolist())


def _eta_str(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h > 0 else f"{m}m{s:02d}s"


def _write_csv(path: str, rows: List[Dict], fieldnames: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def discover_images(directory: str, max_images: int = None) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".pgm", ".ppm"}
    paths = []
    for entry in sorted(os.listdir(directory)):
        if Path(entry).suffix.lower() in exts:
            paths.append(os.path.join(directory, entry))
    if max_images:
        paths = paths[:max_images]
    return paths


class RealDatasetRunner:
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
        self.plot_fmt = self.cfg["experiment"].get("plot_format", "png")
        self.plot_dpi = self.cfg["experiment"].get("plot_dpi", 150)
        self.dataset_name = self.cfg["dataset"].get("dataset_name", "Unknown")
        n_images = self.cfg["dataset"].get("n_images", 200)
        dataset_dir = self.cfg["dataset"].get("dataset_dir", "")
        self.image_paths = discover_images(dataset_dir, n_images)
        logger.info(f"Dataset: {self.dataset_name} — {len(self.image_paths)} images from {dataset_dir}")

    def run_main_experiments(self) -> str:
        logger.info("=" * 60)
        logger.info(f"MAIN EXPERIMENTS: {self.dataset_name}")
        logger.info("=" * 60)

        csv_path = os.path.join(self.output_dir, "all_results.csv")
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
        clean_features: List[np.ndarray] = []
        stego_features: Dict[Tuple[str, float], List[np.ndarray]] = {}
        for bpp in self.bpp_levels:
            stego_features[("Fixed-LSB-1", bpp)] = []
            stego_features[("Fixed-LSB-2", bpp)] = []
            stego_features[("Adaptive", bpp)] = []

        n_total = len(self.image_paths)
        t_start = time.perf_counter()

        for i, img_path in enumerate(self.image_paths):
            if i % 50 == 0 and i > 0:
                elapsed = time.perf_counter() - t_start
                eta = (elapsed / i) * (n_total - i)
                logger.info(f"  Progress: {i}/{n_total} (ETA: {_eta_str(eta)})")

            img_name = Path(img_path).name
            try:
                cover = _load_image(img_path)
            except Exception as e:
                logger.warning(f"  Skip {img_name}: {e}")
                continue

            h, w = cover.shape[:2]
            n_ch = cover.shape[2] if cover.ndim == 3 else 1

            clean_feat = extract_srm_features(cover)
            clean_features.append(clean_feat)

            ent_map, edge_map = extract_features(cover, self.window_size)
            max_bpp = 3 * n_ch

            for depth in [1, 2]:
                method_name = f"Fixed-LSB-{depth}"
                for bpp in self.bpp_levels:
                    cap = capacity_bytes(cover, lsb_depth=depth, bpp=bpp)
                    if cap <= 0:
                        continue
                    payload_size = max(1, int(cap * 0.7))
                    payload = _generate_payload(payload_size, self.seed)
                    t0 = time.perf_counter()
                    try:
                        stego = embed_fixed(cover, payload, lsb_depth=depth, bpp=bpp, seed=self.seed)
                    except ValueError:
                        continue
                    embed_t = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    extracted = extract_fixed(stego, lsb_depth=depth, seed=self.seed)
                    extract_t = time.perf_counter() - t0
                    verified = extracted[:len(payload)] == payload
                    metrics = compute_all(cover, stego, len(payload) * 8)
                    det = run_all_detectors(stego)
                    stego_features[(method_name, bpp)].append(extract_srm_features(stego))
                    rows.append({
                        "image": img_name, "method": method_name, "bpp": bpp,
                        "payload_bytes": len(payload),
                        "embed_time_s": f"{embed_t:.6f}", "extract_time_s": f"{extract_t:.6f}",
                        "extraction_verified": verified,
                        "mse": f"{metrics['mse']:.6f}", "psnr": f"{metrics['psnr']:.4f}",
                        "ssim": f"{metrics['ssim']:.6f}", "kl_divergence": f"{metrics['kl_divergence']:.8f}",
                        "distortion_per_bit": f"{metrics['distortion_per_bit']:.10f}",
                        "rs_estimated_rate": f"{det['rs']['estimated_rate']:.6f}",
                        "rs_detected": det["rs"]["detection_flag"],
                        "chi2_embedding_prob": f"{det['chi_square']['embedding_probability']:.6f}",
                        "chi2_detected": det["chi_square"]["detection_flag"],
                        "spa_estimated_rate": f"{det['spa']['estimated_rate']:.6f}",
                        "spa_detected": det["spa"]["detection_flag"],
                        "mean_depth": "", "depth_std": "",
                    })

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
                        cover, payload, self.controller, bpp=bpp, pressure=pressure,
                        window_size=self.window_size, seed=self.seed, depth_map=depth_map,
                    )
                except ValueError:
                    continue
                embed_t = time.perf_counter() - t0
                t0 = time.perf_counter()
                try:
                    extracted = extract_adaptive(stego, self.controller,
                                                  pressure=pressure, window_size=self.window_size,
                                                  seed=self.seed, depth_map=depth_map)
                    extract_t = time.perf_counter() - t0
                    verified = extracted[:len(payload)] == payload
                except Exception:
                    extract_t = 0
                    verified = False
                metrics = compute_all(cover, stego, len(payload) * 8)
                det = run_all_detectors(stego)
                stego_features[("Adaptive", bpp)].append(extract_srm_features(stego))
                rows.append({
                    "image": img_name, "method": "Adaptive", "bpp": bpp,
                    "payload_bytes": len(payload),
                    "embed_time_s": f"{embed_t:.6f}", "extract_time_s": f"{extract_t:.6f}",
                    "extraction_verified": verified,
                    "mse": f"{metrics['mse']:.6f}", "psnr": f"{metrics['psnr']:.4f}",
                    "ssim": f"{metrics['ssim']:.6f}", "kl_divergence": f"{metrics['kl_divergence']:.8f}",
                    "distortion_per_bit": f"{metrics['distortion_per_bit']:.10f}",
                    "rs_estimated_rate": f"{det['rs']['estimated_rate']:.6f}",
                    "rs_detected": det["rs"]["detection_flag"],
                    "chi2_embedding_prob": f"{det['chi_square']['embedding_probability']:.6f}",
                    "chi2_detected": det["chi_square"]["detection_flag"],
                    "spa_estimated_rate": f"{det['spa']['estimated_rate']:.6f}",
                    "spa_detected": det["spa"]["detection_flag"],
                    "mean_depth": f"{used_dm.mean():.4f}", "depth_std": f"{used_dm.std():.4f}",
                })

        _write_csv(csv_path, rows, fieldnames)
        elapsed = time.perf_counter() - t_start
        logger.info(f"Main experiments: {len(rows)} rows in {_eta_str(elapsed)} → {csv_path}")

        feat_path = os.path.join(self.output_dir, "srm_features.npz")
        save_dict = {"clean": np.array(clean_features)}
        for (method, bpp), feats in stego_features.items():
            if feats:
                key = f"stego_{method}_{bpp}".replace("-", "_").replace(".", "p")
                save_dict[key] = np.array(feats)
        np.savez_compressed(feat_path, **save_dict)
        return csv_path

    def run_statistical(self) -> str:
        import pandas as pd
        logger.info("Statistical Analysis...")
        csv_in = os.path.join(self.output_dir, "all_results.csv")
        csv_out = os.path.join(self.output_dir, "statistical_tests.csv")
        df = pd.read_csv(csv_in)
        for col in ["psnr", "ssim", "mse", "kl_divergence", "distortion_per_bit", "rs_estimated_rate"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        metrics = ["psnr", "ssim", "mse", "kl_divergence", "distortion_per_bit", "rs_estimated_rate"]
        comparisons = [("Fixed-LSB-1", "Adaptive"), ("Fixed-LSB-2", "Adaptive"), ("Fixed-LSB-1", "Fixed-LSB-2")]
        all_results = []
        for method_a, method_b in comparisons:
            results = compare_across_bpp(df, method_a, method_b, metrics)
            for r in results:
                r["comparison"] = f"{method_a} vs {method_b}"
            all_results.extend(results)
        if all_results:
            _write_csv(csv_out, all_results, list(all_results[0].keys()))
            logger.info(f"Statistical: {len(all_results)} comparisons → {csv_out}")
        return csv_out

    def run_deep_steganalysis(self) -> str:
        logger.info("Deep Steganalysis...")
        feat_path = os.path.join(self.output_dir, "srm_features.npz")
        csv_out = os.path.join(self.output_dir, "deep_steganalysis.csv")
        if not os.path.exists(feat_path):
            logger.warning("SRM features not found.")
            return csv_out
        data = np.load(feat_path, allow_pickle=True)
        X_clean = data["clean"]
        n_folds = self.cfg.get("evaluation", {}).get("deep_steganalysis_folds", 5)
        rows = []
        roc_data = {}
        for method in ["Fixed_LSB_1", "Fixed_LSB_2", "Adaptive"]:
            for bpp in self.bpp_levels:
                key = f"stego_{method}_{bpp}".replace(".", "p")
                if key not in data:
                    continue
                X_stego = data[key]
                n = min(len(X_clean), len(X_stego))
                if n < 20:
                    continue
                result = cross_validate_steganalysis(X_clean[:n], X_stego[:n], n_folds=n_folds, seed=self.seed)
                method_label = method.replace("_", "-")
                logger.info(f"  {method_label} bpp={bpp}: AUC={result['mean_auc']:.4f}±{result['std_auc']:.4f}")
                rows.append({
                    "method": method_label, "bpp": bpp, "n_images": n, "n_folds": n_folds,
                    "mean_auc": f"{result['mean_auc']:.6f}", "std_auc": f"{result['std_auc']:.6f}",
                    "ci95_auc_lo": f"{result['ci95_auc_lo']:.6f}", "ci95_auc_hi": f"{result['ci95_auc_hi']:.6f}",
                    "mean_tpr_at_5fpr": f"{result['mean_tpr_at_5fpr']:.6f}",
                    "std_tpr_at_5fpr": f"{result['std_tpr_at_5fpr']:.6f}",
                })
                roc_data[(method_label, bpp)] = result
        if rows:
            _write_csv(csv_out, rows, list(rows[0].keys()))
        roc_save = {}
        for (m, b), res in roc_data.items():
            base_fpr, mean_tpr, std_tpr = mean_roc_curve(res["all_fpr"], res["all_tpr"])
            k = f"{m}_{b}".replace("-", "_").replace(".", "p")
            roc_save[f"{k}_fpr"] = base_fpr
            roc_save[f"{k}_tpr"] = mean_tpr
            roc_save[f"{k}_tpr_std"] = std_tpr
        if roc_save:
            np.savez_compressed(os.path.join(self.output_dir, "roc_curves.npz"), **roc_save)
        return csv_out

    def run_complexity(self) -> str:
        logger.info("Complexity Analysis...")
        n_comp = self.cfg.get("evaluation", {}).get("n_images_complexity", 30)
        comp_paths = self.image_paths[:n_comp]
        csv_out = os.path.join(self.output_dir, "complexity.csv")
        fieldnames = ["image", "method", "bpp", "feature_extract_s", "fuzzy_infer_s",
                      "embed_s", "extract_s", "total_s", "peak_memory_kb"]
        rows = []
        bpp = 0.2
        for img_path in comp_paths:
            img_name = Path(img_path).name
            try:
                cover = _load_image(img_path)
            except Exception:
                continue
            h, w = cover.shape[:2]
            n_ch = cover.shape[2]
            pressure = min(1.0, bpp / (3 * n_ch))
            for depth in [1, 2]:
                cap = capacity_bytes(cover, lsb_depth=depth, bpp=bpp)
                if cap > 0:
                    payload = _generate_payload(max(1, int(cap * 0.7)), self.seed)
                    tracemalloc.start()
                    t0 = time.perf_counter()
                    stego = embed_fixed(cover, payload, lsb_depth=depth, bpp=bpp, seed=self.seed)
                    t_embed = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    extract_fixed(stego, lsb_depth=depth, seed=self.seed)
                    t_extract = time.perf_counter() - t0
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    rows.append({"image": img_name, "method": f"Fixed-LSB-{depth}", "bpp": bpp,
                                 "feature_extract_s": "0.0", "fuzzy_infer_s": "0.0",
                                 "embed_s": f"{t_embed:.6f}", "extract_s": f"{t_extract:.6f}",
                                 "total_s": f"{t_embed+t_extract:.6f}", "peak_memory_kb": f"{peak/1024:.1f}"})
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
                        stego, _ = embed_adaptive(cover, payload, self.controller,
                                                   bpp=bpp, pressure=pressure,
                                                   window_size=self.window_size, seed=self.seed,
                                                   depth_map=depth_map)
                        t_embed = time.perf_counter() - t0
                        t0 = time.perf_counter()
                        extract_adaptive(stego, self.controller, pressure=pressure,
                                         window_size=self.window_size, seed=self.seed, depth_map=depth_map)
                        t_extract = time.perf_counter() - t0
                    except Exception:
                        t_embed = t_extract = 0
                    _, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    rows.append({"image": img_name, "method": "Adaptive", "bpp": bpp,
                                 "feature_extract_s": f"{t_feat:.6f}", "fuzzy_infer_s": f"{t_fuzzy:.6f}",
                                 "embed_s": f"{t_embed:.6f}", "extract_s": f"{t_extract:.6f}",
                                 "total_s": f"{t_feat+t_fuzzy+t_embed+t_extract:.6f}",
                                 "peak_memory_kb": f"{peak/1024:.1f}"})
                else:
                    tracemalloc.stop()
            else:
                tracemalloc.stop()
        _write_csv(csv_out, rows, fieldnames)
        logger.info(f"Complexity: {len(rows)} rows → {csv_out}")
        return csv_out

    def run_plots(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        logger.info("Generating plots...")
        plt.rcParams.update({"font.size": 11, "font.family": "serif",
                              "axes.labelsize": 12, "axes.titlesize": 13,
                              "legend.fontsize": 9, "figure.figsize": (7, 5),
                              "figure.dpi": self.plot_dpi, "lines.linewidth": 1.5})
        plot_dir = os.path.join(self.output_dir, "plots")
        fmt = self.plot_fmt
        csv_main = os.path.join(self.output_dir, "all_results.csv")
        if os.path.exists(csv_main):
            df = pd.read_csv(csv_main)
            for col in ["psnr", "ssim", "mse", "kl_divergence", "distortion_per_bit", "rs_estimated_rate"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            colors = {"Fixed-LSB-1": "#2196F3", "Fixed-LSB-2": "#FF9800", "Adaptive": "#E91E63"}
            markers = {"Fixed-LSB-1": "s", "Fixed-LSB-2": "D", "Adaptive": "o"}
            for metric, ylabel, title in [
                ("psnr", "PSNR (dB)", f"PSNR vs Embedding Rate — {self.dataset_name}"),
                ("ssim", "SSIM", f"SSIM vs Embedding Rate — {self.dataset_name}"),
                ("mse", "MSE", f"MSE vs Embedding Rate — {self.dataset_name}"),
                ("kl_divergence", "KL Divergence", f"Histogram Divergence — {self.dataset_name}"),
            ]:
                fig, ax = plt.subplots()
                for method in ["Fixed-LSB-1", "Fixed-LSB-2", "Adaptive"]:
                    sub = df[df["method"] == method]
                    if sub.empty:
                        continue
                    g = sub.groupby("bpp")[metric]
                    means = g.mean()
                    stds = g.std()
                    ax.errorbar(means.index, means.values,
                                yerr=1.96 * stds.values / np.sqrt(g.count().values),
                                marker=markers.get(method, "o"), label=method,
                                color=colors.get(method, None),
                                linestyle="--" if "Fixed" in method else "-", capsize=3)
                ax.set_xlabel("Embedding Rate (bpp)")
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.join(plot_dir, f"{metric}_vs_bpp.{fmt}"), dpi=self.plot_dpi)
                plt.close(fig)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, det_col, det_name in zip(axes,
                ["rs_detected", "chi2_detected", "spa_detected"],
                ["RS Analysis", "Chi-Square", "SPA"]):
                for method in ["Fixed-LSB-1", "Fixed-LSB-2", "Adaptive"]:
                    sub = df[df["method"] == method]
                    if sub.empty:
                        continue
                    g = sub.groupby("bpp")[det_col].mean()
                    ax.plot(g.index, g.values,
                            marker="o" if method == "Adaptive" else "s", label=method,
                            linestyle="-" if method == "Adaptive" else "--")
                ax.set_xlabel("Embedding Rate (bpp)")
                ax.set_ylabel("Detection Rate")
                ax.set_title(f"{det_name} — {self.dataset_name}")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-0.05, 1.05)
            fig.tight_layout()
            fig.savefig(os.path.join(plot_dir, f"detection_rates.{fmt}"), dpi=self.plot_dpi)
            plt.close(fig)

        roc_path = os.path.join(self.output_dir, "roc_curves.npz")
        if os.path.exists(roc_path):
            data = np.load(roc_path)
            for bpp in self.bpp_levels:
                fig, ax = plt.subplots()
                for method in ["Fixed-LSB-1", "Fixed-LSB-2", "Adaptive"]:
                    k = f"{method}_{bpp}".replace("-", "_").replace(".", "p")
                    fpr_key, tpr_key = f"{k}_fpr", f"{k}_tpr"
                    if fpr_key in data and tpr_key in data:
                        fpr, tpr = data[fpr_key], data[tpr_key]
                        auc = float(np.trapezoid(tpr, fpr))
                        ax.plot(fpr, tpr, label=f"{method} (AUC={auc:.3f})")
                ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")
                ax.set_title(f"ROC at bpp={bpp} — {self.dataset_name}")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                bpp_str = f"{bpp}".replace(".", "p")
                fig.savefig(os.path.join(plot_dir, f"roc_bpp_{bpp_str}.{fmt}"), dpi=self.plot_dpi)
                plt.close(fig)

        logger.info(f"Plots saved to {plot_dir}/")

    def save_summary(self):
        import pandas as pd
        csv_main = os.path.join(self.output_dir, "all_results.csv")
        if not os.path.exists(csv_main):
            return
        df = pd.read_csv(csv_main)
        for col in ["psnr", "ssim", "mse", "kl_divergence", "rs_estimated_rate", "chi2_detected", "spa_detected"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        summary = {}
        for method in ["Fixed-LSB-1", "Fixed-LSB-2", "Adaptive"]:
            sub = df[df["method"] == method]
            if sub.empty:
                continue
            summary[method] = {}
            for bpp in self.bpp_levels:
                bsub = sub[sub["bpp"] == bpp]
                if bsub.empty:
                    continue
                summary[method][str(bpp)] = {
                    "psnr_mean": float(bsub["psnr"].mean()),
                    "psnr_std": float(bsub["psnr"].std()),
                    "ssim_mean": float(bsub["ssim"].mean()),
                    "mse_mean": float(bsub["mse"].mean()),
                    "kl_mean": float(bsub["kl_divergence"].mean()),
                    "rs_detection_rate": float(bsub["rs_detected"].mean()),
                    "chi2_detection_rate": float(bsub["chi2_detected"].mean()),
                    "spa_detection_rate": float(bsub["spa_detected"].mean()),
                    "n_images": int(len(bsub)),
                }
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump({"dataset": self.dataset_name, "n_images": len(self.image_paths),
                       "results": summary}, f, indent=2)
        logger.info(f"Summary saved → {summary_path}")

    def run_all(self):
        t_total = time.perf_counter()
        self.run_main_experiments()
        self.run_statistical()
        self.run_deep_steganalysis()
        self.run_complexity()
        self.run_plots()
        self.save_summary()
        elapsed = time.perf_counter() - t_total
        logger.info(f"PIPELINE COMPLETE [{self.dataset_name}] — Total: {_eta_str(elapsed)}")
        logger.info(f"Results in: {self.output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"experiment_real_{Path(args.config).stem}.log", mode="w"),
        ],
    )
    runner = RealDatasetRunner(args.config)
    runner.run_all()
