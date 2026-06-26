#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwatu.analysis import (  # noqa: E402
    class_accuracy_rows,
    collect_predictions,
    collate_with_meta,
    confusion_matrix,
    save_analysis_outputs,
    wrong_cases,
)
from hwatu.config import BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS, class_names_for, final_checkpoint_for  # noqa: E402
from hwatu.data import make_datasets  # noqa: E402
from hwatu.model import DEVICE, load_inference_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Hwatu checkpoints on the validation split")
    parser.add_argument("--target", choices=["month", "type"], required=True)
    parser.add_argument("--root", default=None, help="project/data root; defaults to HWATU_ROOT or repo root")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    return parser.parse_args()


def print_accuracy(results):
    correct = sum(item["correct"] for item in results)
    total = len(results)
    accuracy = correct / total if total else 0.0
    print(f"accuracy: {accuracy:.4f} ({correct}/{total})")


def print_class_accuracy(rows):
    print("\nclass accuracy")
    for row in rows:
        print(f"{row['class_name']:12s} acc={float(row['accuracy']):.4f} count={row['total']}")


def print_confusion_matrix(matrix, class_names):
    print("\nconfusion matrix")
    print("true\\pred".ljust(12) + " ".join(name.rjust(10) for name in class_names))
    for true_name, row in zip(class_names, matrix):
        print(true_name.ljust(12) + " ".join(str(value).rjust(10) for value in row))


def main(args):
    class_names = class_names_for(args.target)
    checkpoint = args.checkpoint or final_checkpoint_for(args.target)
    output_dir = args.output_dir / args.target

    print(f"device: {DEVICE}")
    print(f"target: {args.target}")
    print(f"checkpoint: {checkpoint}")
    print(f"output dir: {output_dir}")

    _, val_dataset = make_datasets(target=args.target, root=args.root, image_size=IMAGE_SIZE, return_meta=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_with_meta,
    )

    model = load_inference_model(checkpoint, class_names)
    results = collect_predictions(model, val_loader, class_names, DEVICE)
    class_rows = class_accuracy_rows(results, class_names)
    matrix = confusion_matrix(results, class_names)
    wrong_rows = wrong_cases(results)

    print_accuracy(results)
    print_class_accuracy(class_rows)
    print_confusion_matrix(matrix, class_names)
    print(f"\nwrong cases: {len(wrong_rows)}")
    save_analysis_outputs(results, class_rows, matrix, wrong_rows, class_names, args.target, output_dir)
    print(f"\nsaved analysis outputs: {output_dir}")


if __name__ == "__main__":
    main(parse_args())

