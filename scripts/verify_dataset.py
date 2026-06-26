#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwatu.config import MONTH_CLASS_NAMES, TYPE_CLASS_NAMES, data_paths  # noqa: E402
from hwatu.data import build_samples, load_label_rows  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Check Hwatu dataset labels and image counts")
    parser.add_argument("--root", default=None, help="project/data root; defaults to HWATU_ROOT or repo root")
    return parser.parse_args()


def count_images_by_folder(samples):
    counts = Counter()
    for sample in samples:
        counts[sample["folder_path"]] += 1
    return counts


def main(args):
    paths = data_paths(args.root)
    print(f"labels csv: {paths['labels_csv']}")
    print(f"raw dir: {paths['raw_dir']}")

    rows = load_label_rows(paths["labels_csv"])
    samples = build_samples(paths["raw_dir"], paths["labels_csv"])
    folder_counts = count_images_by_folder(samples)

    print(f"card definitions: {len(rows)}")
    print(f"image samples: {len(samples)}")

    missing = [row["folder_path"] for row in rows if folder_counts[row["folder_path"]] == 0]
    print(f"\nempty card folders: {len(missing)}")
    for folder_path in missing:
        print(f"- {folder_path}")

    type_counts = Counter(sample["type"] for sample in samples)
    print("\ntype counts")
    for name in TYPE_CLASS_NAMES:
        print(f"{name:10s}: {type_counts[name]}")

    month_counts = Counter(sample["month"] for sample in samples)
    print("\nmonth counts")
    for month in MONTH_CLASS_NAMES:
        print(f"{month:>2s}: {month_counts[month]}")


if __name__ == "__main__":
    main(parse_args())

