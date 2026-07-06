#!/usr/bin/env python3
"""
Download benchmark datasets for real-dataset experiments.

Each dataset is fetched from its official or archival public source, extracted
into the directory expected by the experiment configs, and trimmed to the first
200 images in sorted filename order (matching the paper protocol).

Usage:
    python scripts/download_datasets.py --dataset all
    python scripts/download_datasets.py --dataset bossbase
    python scripts/download_datasets.py --dataset bows2
    python scripts/download_datasets.py --dataset mirflickr
    python scripts/download_datasets.py --dataset all --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import List, Optional

import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".pgm", ".ppm"}

DATASETS = {
    "bossbase": {
        "name": "BOSSBase 1.01",
        "target_dir": REPO_ROOT / "data" / "BOSSbase_1.01",
        "min_images": 200,
        "n_images": 200,
        "citation": "Bas et al., BOSS: Break Our Steganographic System, 2010.",
        "sources": [
            {
                "url": "http://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip",
                "type": "zip",
                "note": "Binghamton University DDE (primary mirror)",
            },
            {
                "url": "https://agents.fel.cvut.cz/stegodata/BOSSbase/BOSSbase_1.01.zip",
                "type": "zip",
                "note": "CVUT Stego Data (alternate mirror)",
            },
        ],
    },
    "bows2": {
        "name": "BOWS2",
        "target_dir": REPO_ROOT / "data" / "BOWS2" / "cover",
        "min_images": 200,
        "n_images": 200,
        "citation": "Bas & Furon, BOWS-2 Contest, 2007-2008.",
        "sources": [
            {
                "url": "http://dud.inf.tu-dresden.de/~westfeld/rsp/bows2-1g.tar.gz",
                "type": "tar.gz",
                "note": "TU Dresden RSP archive (images 1-1000, PGM grayscale)",
            },
        ],
    },
    "mirflickr": {
        "name": "MIRFLICKR-25K",
        "target_dir": REPO_ROOT / "data" / "mirflickr",
        "min_images": 200,
        "n_images": 200,
        "citation": "Huiskes & Lew, MIRFLICKR-25K, 2008.",
        "sources": [
            {
                "url": "http://press.liacs.nl/mirflickr/mirflickr25k.zip",
                "type": "zip",
                "note": "LIACS Media Lab (official MIRFLICKR-25K release)",
            },
        ],
    },
}


def discover_images(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    paths = []
    for entry in sorted(directory.iterdir()):
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTS:
            paths.append(entry)
    return paths


def count_images_recursive(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    count = 0
    for root, _, files in os.walk(directory):
        for name in files:
            if Path(name).suffix.lower() in IMAGE_EXTS:
                count += 1
    return count


def download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as handle, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {dest.name}",
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
                    bar.update(len(chunk))


def extract_zip(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def extract_tar_gz(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest)


def flatten_images(source: Path, target: Path, limit: int) -> List[Path]:
    """Collect images from nested archives into a flat target directory."""
    target.mkdir(parents=True, exist_ok=True)
    collected: List[Path] = []

    for root, _, files in os.walk(source):
        for name in sorted(files):
            path = Path(root) / name
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            dest = target / path.name
            if dest.exists():
                stem, suffix = dest.stem, dest.suffix
                counter = 1
                while dest.exists():
                    dest = target / f"{stem}_{counter}{suffix}"
                    counter += 1
            shutil.copy2(path, dest)
            collected.append(dest)
            if len(collected) >= limit:
                return sorted(collected)[:limit]

    return sorted(collected)[:limit]


def trim_directory(directory: Path, limit: int) -> List[Path]:
    images = discover_images(directory)
    if len(images) <= limit:
        return images
    for extra in images[limit:]:
        extra.unlink()
    return images[:limit]


def resolve_bossbase_root(extracted: Path) -> Path:
    """Find the directory containing PGM cover images inside BOSSBase zip."""
    candidates = [
        extracted / "BOSSbase_1.01",
        extracted,
    ]
    for candidate in candidates:
        if count_images_recursive(candidate) >= 200:
            return candidate
    for root, _, files in os.walk(extracted):
        if sum(1 for f in files if Path(f).suffix.lower() in IMAGE_EXTS) >= 200:
            return Path(root)
    return extracted


def resolve_mirflickr_root(extracted: Path) -> Path:
    candidates = [
        extracted / "mirflickr25k",
        extracted / "mirflickr",
        extracted,
    ]
    for candidate in candidates:
        if count_images_recursive(candidate) >= 200:
            return candidate
    return extracted


def install_dataset(
    key: str,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    spec = DATASETS[key]
    target: Path = spec["target_dir"]
    limit: int = spec["n_images"]

    existing = discover_images(target) if target.is_dir() else []
    if len(existing) >= spec["min_images"] and not force:
        print(f"[skip] {spec['name']}: {len(existing)} images already in {target}")
        return True

    if dry_run:
        print(f"[dry-run] Would download {spec['name']} -> {target}")
        for src in spec["sources"]:
            print(f"  - {src['url']} ({src['note']})")
        return True

    cache_dir = REPO_ROOT / "data" / ".downloads"
    cache_dir.mkdir(parents=True, exist_ok=True)

    last_error: Optional[Exception] = None
    for source in spec["sources"]:
        url = source["url"]
        archive_type = source["type"]
        archive_name = url.split("/")[-1]
        archive_path = cache_dir / archive_name
        extract_root = cache_dir / f"{key}_extracted"

        try:
            print(f"\n=== {spec['name']} ===")
            print(f"Source: {source['note']}")
            print(f"URL:    {url}")

            if not archive_path.exists():
                download_file(url, archive_path)
            else:
                print(f"Using cached archive: {archive_path}")

            if extract_root.exists():
                shutil.rmtree(extract_root)
            extract_root.mkdir(parents=True, exist_ok=True)

            if archive_type == "zip":
                extract_zip(archive_path, extract_root)
            elif archive_type == "tar.gz":
                extract_tar_gz(archive_path, extract_root)
            else:
                raise ValueError(f"Unsupported archive type: {archive_type}")

            if key == "bossbase":
                source_root = resolve_bossbase_root(extract_root)
                if target.exists():
                    shutil.rmtree(target)
                target.mkdir(parents=True, exist_ok=True)
                images = flatten_images(source_root, target, limit)
            elif key == "bows2":
                if target.exists():
                    shutil.rmtree(target)
                target.mkdir(parents=True, exist_ok=True)
                images = flatten_images(extract_root, target, limit)
            elif key == "mirflickr":
                source_root = resolve_mirflickr_root(extract_root)
                if target.exists():
                    shutil.rmtree(target)
                target.mkdir(parents=True, exist_ok=True)
                images = flatten_images(source_root, target, limit)
            else:
                raise ValueError(f"Unknown dataset key: {key}")

            images = trim_directory(target, limit)
            print(f"Installed {len(images)} images in {target}")
            return len(images) >= spec["min_images"]

        except Exception as exc:
            last_error = exc
            print(f"Failed source {url}: {exc}", file=sys.stderr)
            if target.exists() and not discover_images(target):
                shutil.rmtree(target, ignore_errors=True)
            continue

    print(f"ERROR: Could not install {spec['name']}. Last error: {last_error}", file=sys.stderr)
    print("See data/README.md for manual download instructions.", file=sys.stderr)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download benchmark steganography datasets.")
    parser.add_argument(
        "--dataset",
        choices=["all", "bossbase", "bows2", "mirflickr"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading")
    parser.add_argument("--force", action="store_true", help="Re-download even if data exists")
    args = parser.parse_args()

    keys = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    results = {key: install_dataset(key, dry_run=args.dry_run, force=args.force) for key in keys}

    print("\nSummary:")
    for key, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {DATASETS[key]['name']}: {status}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
