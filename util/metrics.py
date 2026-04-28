import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from sklearn.metrics import confusion_matrix
except Exception:
    confusion_matrix = None

import matplotlib.pyplot as plt


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)


def save_confusion_matrix(y_true, y_pred, result_dir, inv_class_map, eval_exclude_label_ids=None, split_name="val", epoch: Optional[int] = None):
    
    if confusion_matrix is None:
        return

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if len(y_true) == 0 or len(y_pred) == 0:
        return

    eval_exclude_label_ids = set(eval_exclude_label_ids or [])

    if isinstance(inv_class_map, dict) and len(inv_class_map) > 0:
        label_ids = [int(i) for i in sorted(inv_class_map.keys()) if int(i) not in eval_exclude_label_ids]
    else:
        label_ids = sorted({int(x) for x in np.concatenate([y_true.reshape(-1), y_pred.reshape(-1)], axis=0)} - eval_exclude_label_ids)

    if len(label_ids) == 0:
        return

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    class_names = [str(inv_class_map.get(i, i)) for i in label_ids]

    save_dir = Path(result_dir) / "confusion_matrix" / str(split_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    stem = "confusion_matrix" if epoch is None else f"confusion_matrix_epoch_{epoch + 1}"
    np.save(save_dir / f"{stem}.npy", cm)

    for normalize, suffix in [(False, ""), (True, "_norm")]:
        cm_plot = cm.astype(np.float64)

        if normalize:
            row_sum = cm_plot.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            cm_plot = cm_plot / row_sum

        plt.figure(figsize=(10, 8))
        plt.imshow(cm_plot, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
        plt.colorbar()

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, class_names)

        thresh = cm_plot.max() / 2.0 if cm_plot.size > 0 else 0.0

        for i in range(cm_plot.shape[0]):
            for j in range(cm_plot.shape[1]):
                text = f"{cm_plot[i, j]:.2f}" if normalize else str(int(cm[i, j]))
                plt.text(j, i, text, horizontalalignment="center", color="white" if cm_plot[i, j] > thresh else "black", fontsize=8)

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(save_dir / f"{stem}{suffix}.png", dpi=200)
        plt.close()


def save_eval_summary(
    result_dir,
    split_name,
    epoch,
    metrics: dict,
    eval_exclude_classes=None,
):
    save_dir = Path(result_dir) / "eval_summary" / str(split_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    stem = "infer" if split_name == "infer" else f"epoch_{epoch + 1}"

    payload = {
        "split": split_name,
        "epoch": None if split_name == "infer" or epoch is None else int(epoch + 1),
        **metrics,
        "eval_exclude_classes": sorted(list(eval_exclude_classes or [])),
    }

    with open(save_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_instance_csv(result_dir, split_name, epoch, instance_rows):
    save_dir = Path(result_dir) / "eval_summary" / str(split_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    stem = "infer" if split_name == "infer" else f"epoch_{epoch + 1}"

    fieldnames = [
        "file",
        "target_idx",
        "group_idx",
        "gt_label",
        "gt_name",
        "pred_label",
        "pred_name",
        "correct",
        "eval_role",
    ]

    with open(save_dir / f"{stem}_instances.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in instance_rows:
            writer.writerow(row)