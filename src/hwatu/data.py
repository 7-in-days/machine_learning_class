from collections import Counter
import csv
import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from hwatu.config import (
    BATCH_SIZE,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    MONTH_CLASS_NAMES,
    MONTH_TO_INDEX,
    NUM_WORKERS,
    SEED,
    TYPE_CLASS_NAMES,
    TYPE_TO_INDEX,
    VAL_RATIO,
    data_paths,
)


def load_label_rows(labels_csv):
    labels_csv = Path(labels_csv)
    if not labels_csv.exists():
        raise FileNotFoundError(f"labels.csv not found: {labels_csv}")

    with labels_csv.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = set(reader.fieldnames or [])

    required = {"folder_path", "card_id", "month", "type", "type_id", "description"}
    missing = required - columns
    if missing:
        raise ValueError(f"labels.csv missing columns: {sorted(missing)}")

    return rows


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []

    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_samples(raw_dir, labels_csv):
    samples = []
    for row in load_label_rows(labels_csv):
        folder = Path(raw_dir) / row["folder_path"]
        for image_path in list_images(folder):
            samples.append(
                {
                    "image_path": image_path,
                    "month": str(row["month"]),
                    "type": row["type"],
                    "folder_path": row["folder_path"],
                    "description": row["description"],
                }
            )
    return samples


def split_samples(samples, val_ratio=VAL_RATIO, seed=SEED, stratify_name=None):
    if stratify_name is None:
        shuffled = list(samples)
        random.seed(seed)
        random.shuffle(shuffled)
        val_size = int(len(shuffled) * val_ratio)
        return shuffled[val_size:], shuffled[:val_size]

    grouped = {}
    for sample in samples:
        grouped.setdefault(sample[stratify_name], []).append(sample)

    train_samples = []
    val_samples = []
    random.seed(seed)
    for label_samples in grouped.values():
        shuffled = list(label_samples)
        random.shuffle(shuffled)
        val_size = int(len(shuffled) * val_ratio)
        if val_size == 0 and len(shuffled) >= 2:
            val_size = 1
        val_samples.extend(shuffled[:val_size])
        train_samples.extend(shuffled[val_size:])

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    return train_samples, val_samples


def make_base_transform(image_size=IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def make_light_transform(image_size=IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(25, fill=0),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.06, 0.06),
                scale=(0.9, 1.1),
                fill=0,
            ),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def make_strong_transform(image_size=IMAGE_SIZE):
    # 좌우반전은 화투 그림 방향을 바꿔 라벨을 깨뜨릴 수 있어 사용하지 않는다.
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(180, fill=0),
            transforms.RandomPerspective(distortion_scale=0.45, p=1.0),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.16, 0.16),
                scale=(0.70, 1.25),
                shear=(-14, 14),
                fill=0,
            ),
            transforms.ColorJitter(
                brightness=0.35,
                contrast=0.35,
                saturation=0.2,
                hue=0.03,
            ),
            transforms.RandomAutocontrast(p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.6)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class MixedTrainTransform:
    def __init__(self, image_size=IMAGE_SIZE):
        self.original = make_base_transform(image_size)
        self.light = make_light_transform(image_size)
        self.strong = make_strong_transform(image_size)

    def __call__(self, image):
        value = random.random()
        if value < 0.4:
            return self.original(image)
        if value < 0.8:
            return self.light(image)
        return self.strong(image)


class HwatuDataset(Dataset):
    def __init__(self, samples, target, transform=None, return_meta=False):
        self.samples = samples
        self.target = target
        self.transform = transform
        self.return_meta = return_meta

        if target == "month":
            self.class_names = MONTH_CLASS_NAMES
            self.label_to_index = MONTH_TO_INDEX
        elif target == "type":
            self.class_names = TYPE_CLASS_NAMES
            self.label_to_index = TYPE_TO_INDEX
        else:
            raise ValueError("target must be 'month' or 'type'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = self.label_to_index[sample[self.target]]
        if self.return_meta:
            return image, label, sample
        return image, label

    def count_by_class(self):
        counts = Counter(sample[self.target] for sample in self.samples)
        return {name: counts.get(name, 0) for name in self.class_names}


def make_weighted_sampler(samples, target):
    labels = [sample[target] for sample in samples]
    counts = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )


def make_datasets(
    target="type",
    root=None,
    image_size=IMAGE_SIZE,
    val_ratio=VAL_RATIO,
    seed=SEED,
    return_meta=False,
):
    paths = data_paths(root)
    samples = build_samples(paths["raw_dir"], paths["labels_csv"])
    if not samples:
        raise ValueError(f"No image samples found under {paths['raw_dir']}")

    train_samples, val_samples = split_samples(
        samples,
        val_ratio=val_ratio,
        seed=seed,
        stratify_name=target,
    )
    return (
        HwatuDataset(train_samples, target, MixedTrainTransform(image_size), return_meta),
        HwatuDataset(val_samples, target, make_base_transform(image_size), return_meta),
    )


def make_loaders(
    target="type",
    root=None,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    balanced=True,
    val_ratio=VAL_RATIO,
    seed=SEED,
):
    train_dataset, val_dataset = make_datasets(
        target=target,
        root=root,
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
    )

    sampler = make_weighted_sampler(train_dataset.samples, target) if balanced else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader

