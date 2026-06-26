from collections import defaultdict
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import torch

try:
    import seaborn as sns
except ImportError:
    sns = None


def collate_with_meta(batch):
    images, labels, metas = [], [], []
    for image, label, meta in batch:
        images.append(image)
        labels.append(label)
        metas.append(meta)
    return torch.stack(images, dim=0), torch.tensor(labels), metas


def collect_predictions(model, loader, class_names, device):
    results = []
    with torch.no_grad():
        for images, labels, metas in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            scores, predictions = probabilities.max(dim=1)

            for meta, label, prediction, score in zip(
                metas, labels.cpu(), predictions.cpu(), scores.cpu()
            ):
                true_index = label.item()
                pred_index = prediction.item()
                results.append(
                    {
                        "path": str(meta["image_path"]),
                        "true_name": class_names[true_index],
                        "pred_name": class_names[pred_index],
                        "score": score.item(),
                        "correct": true_index == pred_index,
                    }
                )
    return results


def class_accuracy_rows(results, class_names):
    total_by_class = defaultdict(int)
    correct_by_class = defaultdict(int)
    for item in results:
        total_by_class[item["true_name"]] += 1
        if item["correct"]:
            correct_by_class[item["true_name"]] += 1

    rows = []
    for name in class_names:
        total = total_by_class[name]
        correct = correct_by_class[name]
        rows.append(
            {
                "class_name": name,
                "correct": correct,
                "total": total,
                "accuracy": f"{correct / total if total else 0.0:.4f}",
            }
        )
    return rows


def confusion_matrix(results, class_names):
    table = {
        true_name: {pred_name: 0 for pred_name in class_names}
        for true_name in class_names
    }
    for item in results:
        table[item["true_name"]][item["pred_name"]] += 1

    return [
        [table[true_name][pred_name] for pred_name in class_names]
        for true_name in class_names
    ]


def wrong_cases(results):
    return sorted(
        [item for item in results if not item["correct"]],
        key=lambda item: item["score"],
        reverse=True,
    )


def save_rows_csv(rows, path, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix_plot(matrix, class_names, target, path):
    plt.figure(figsize=(8, 6))
    if sns is not None:
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
    else:
        plt.imshow(matrix, cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)
        for y_index, row in enumerate(matrix):
            for x_index, value in enumerate(row):
                plt.text(x_index, y_index, str(value), ha="center", va="center")

    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title(f"{target} confusion matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_class_accuracy_plot(class_rows, target, path):
    names = [row["class_name"] for row in class_rows]
    accuracies = [float(row["accuracy"]) for row in class_rows]

    plt.figure(figsize=(9, 4.8))
    bars = plt.bar(names, accuracies)
    plt.ylim(max(0.0, min(accuracies) - 0.01 if accuracies else 0.0), 1.001)
    plt.xlabel("class")
    plt.ylabel("accuracy")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    plt.title(f"{target} class accuracy")
    for bar, accuracy in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy,
            f"{accuracy:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_confidence_histogram(results, target, path):
    correct_scores = [item["score"] for item in results if item["correct"]]
    wrong_scores = [item["score"] for item in results if not item["correct"]]

    plt.figure(figsize=(8, 4))
    plt.hist(correct_scores, bins=20, alpha=0.7, label="correct")
    plt.hist(wrong_scores, bins=20, alpha=0.7, label="wrong")
    plt.xlabel("confidence")
    plt.ylabel("count")
    plt.title(f"{target} confidence histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_analysis_outputs(results, class_rows, matrix, wrong_rows, class_names, target, output_dir):
    output_dir = Path(output_dir)
    suffix = target
    save_rows_csv(
        results,
        output_dir / f"predictions_{suffix}.csv",
        ["path", "true_name", "pred_name", "score", "correct"],
    )
    save_rows_csv(
        class_rows,
        output_dir / f"class_accuracy_{suffix}.csv",
        ["class_name", "correct", "total", "accuracy"],
    )
    save_rows_csv(
        wrong_rows,
        output_dir / f"wrong_cases_{suffix}.csv",
        ["path", "true_name", "pred_name", "score", "correct"],
    )
    save_confusion_matrix_plot(matrix, class_names, target, output_dir / f"confusion_matrix_{suffix}.png")
    save_class_accuracy_plot(class_rows, target, output_dir / f"class_accuracy_{suffix}.png")
    save_confidence_histogram(results, target, output_dir / f"confidence_histogram_{suffix}.png")

