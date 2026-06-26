import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 2
VAL_RATIO = 0.2
SEED = 42

LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.0
MAX_EPOCHS = 8
PATIENCE = 3
DROPOUT_RATE = 0.3
BALANCED_SAMPLER = True

MONTH_CLASS_NAMES = [
    "1", "2", "3", "4", "5", "6",
    "7", "8", "9", "10", "11", "12",
    "no_card",
]
TYPE_CLASS_NAMES = ["gwang", "yeolggeut", "tti", "pi", "no_card"]

MONTH_TO_INDEX = {name: index for index, name in enumerate(MONTH_CLASS_NAMES)}
TYPE_TO_INDEX = {name: index for index, name in enumerate(TYPE_CLASS_NAMES)}

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
MONTH_CHECKPOINT = CHECKPOINTS_DIR / "month_final.pt"
TYPE_CHECKPOINT = CHECKPOINTS_DIR / "type_final.pt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def resolve_root(root=None):
    if root is not None:
        return Path(root).expanduser().resolve()

    env_root = os.environ.get("HWATU_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    return PROJECT_ROOT


def data_paths(root=None):
    hwatu_root = resolve_root(root)
    data_dir = hwatu_root / "data"
    return {
        "root": hwatu_root,
        "data_dir": data_dir,
        "raw_dir": data_dir / "raw",
        "labels_csv": data_dir / "labels.csv",
        "captures_csv": data_dir / "captures.csv",
    }


def class_names_for(target):
    if target == "month":
        return MONTH_CLASS_NAMES
    if target == "type":
        return TYPE_CLASS_NAMES
    raise ValueError("target must be 'month' or 'type'")


def final_checkpoint_for(target):
    if target == "month":
        return MONTH_CHECKPOINT
    if target == "type":
        return TYPE_CHECKPOINT
    raise ValueError("target must be 'month' or 'type'")

