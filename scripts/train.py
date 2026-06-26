#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hwatu.config import (  # noqa: E402
    BALANCED_SAMPLER,
    BATCH_SIZE,
    DROPOUT_RATE,
    IMAGE_SIZE,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    WEIGHT_DECAY,
    class_names_for,
    final_checkpoint_for,
)
from hwatu.data import make_loaders  # noqa: E402
from hwatu.model import (  # noqa: E402
    DEVICE,
    build_resnet18,
    load_model_weights,
    load_training_checkpoint,
    save_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Hwatu ResNet18 classifier")
    parser.add_argument("--target", choices=["month", "type"], required=True)
    parser.add_argument("--root", default=None, help="project/data root; defaults to HWATU_ROOT or repo root")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def run_epoch(model, loader, loss_fn, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = loss_fn(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (predictions == labels).sum().item()
            total_count += images.size(0)

    return total_loss / total_count, total_correct / total_count


def save_history_csv(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "train_loss": f"{row['train_loss']:.4f}",
                    "train_acc": f"{row['train_acc']:.4f}",
                    "val_loss": f"{row['val_loss']:.4f}",
                    "val_acc": f"{row['val_acc']:.4f}",
                }
            )


def save_training_plot(history, path):
    if not history:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, val_loss, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="train acc")
    plt.plot(epochs, val_acc, label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.ylim(max(0.0, min(train_acc + val_acc) - 0.01), 1.001)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def print_dataset_summary(train_loader, val_loader):
    print(f"train samples: {len(train_loader.dataset)}")
    print(f"val samples: {len(val_loader.dataset)}")
    print(f"train counts: {train_loader.dataset.count_by_class()}")
    print(f"val counts: {val_loader.dataset.count_by_class()}")


def train(args):
    class_names = class_names_for(args.target)
    output = args.output or final_checkpoint_for(args.target)
    run_dir = output.parent / f"{args.target}_training"
    last_checkpoint = run_dir / "last.pt"

    print(f"device: {DEVICE}")
    print(f"target: {args.target}")
    print(f"classes: {class_names}")
    print(f"best checkpoint: {output}")

    train_loader, val_loader = make_loaders(
        target=args.target,
        root=args.root,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        balanced=BALANCED_SAMPLER,
    )
    print_dataset_summary(train_loader, val_loader)

    model = build_resnet18(len(class_names), dropout_rate=DROPOUT_RATE).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    resume_path = args.resume or last_checkpoint
    start_epoch, best_val_loss, resumed = load_training_checkpoint(resume_path, model, optimizer)
    if resumed:
        print(f"resume from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    elif args.source:
        load_model_weights(args.source, model)
        print(f"loaded source weights: {args.source}")
    else:
        print("start training from scratch")

    history = []
    early_stop_count = 0
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
            save_checkpoint(output, model, optimizer, epoch, best_val_loss, class_names, args.target)
            print(f"saved best: {output}")
        else:
            early_stop_count += 1
            print(f"early stop count: {early_stop_count}/{PATIENCE}")

        save_checkpoint(last_checkpoint, model, optimizer, epoch, best_val_loss, class_names, args.target)
        save_history_csv(history, run_dir / "history.csv")
        save_training_plot(history, run_dir / "training_curves.png")

        if early_stop_count >= PATIENCE:
            print("early stopping")
            break


if __name__ == "__main__":
    train(parse_args())

