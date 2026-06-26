from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from hwatu.config import DROPOUT_RATE


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_resnet18(num_classes, dropout_rate=DROPOUT_RATE):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes),
    )
    return model


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, class_names, target):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "class_names": class_names,
            "target_name": target,
        },
        path,
    )


def load_training_checkpoint(path, model, optimizer):
    path = Path(path)
    if not path.exists():
        return 0, float("inf"), False

    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint.get("epoch", -1) + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    return start_epoch, best_val_loss, True


def load_model_weights(path, model):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"source checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])


def load_inference_model(checkpoint_path, class_names, device=DEVICE):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    model = build_resnet18(num_classes=len(class_names)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

