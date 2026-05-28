import torch
import logging
import numpy as np
import random
from datetime import datetime, timedelta, timezone
import torch.distributed as dist
from tqdm import tqdm 
import sys

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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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

def _percentile_log_float(
    data: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.5,
    log_gain: float = 9.0,
) -> np.ndarray:
    """
    One pass:
        percentile clipping + normalize + log1p enhancement
    Return float32 image in [0, 1].
    """
    x = np.asarray(data, dtype=np.float32)

    if x.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram/image, got shape={x.shape}")

    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)

    p_low = float(p_low)
    p_high = float(p_high)

    if not (0.0 <= p_low < p_high <= 100.0):
        raise ValueError(
            f"Invalid percentile range: p_low={p_low}, p_high={p_high}. "
            "Expected 0 <= p_low < p_high <= 100."
        )

    valid = x[finite]
    lo = float(np.percentile(valid, p_low))
    hi = float(np.percentile(valid, p_high))

    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)

    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)

    if float(log_gain) > 0:
        x = np.log1p(float(log_gain) * x) / np.log1p(float(log_gain))

    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x = np.clip(x, 0.0, 1.0)

    return x.astype(np.float32)


def spectrogram_to_yolo_float(
    data: np.ndarray,
    db_min: float = None,
    db_max: float = None,
    eps: float = 1e-12,
    p_low: float = 1.0,
    p_high: float = 99.5,
    log_gain: float = 9.0,
) -> np.ndarray:
    """
    Temporary test logic:
        Apply percentile + log twice.

    This is used to test whether YOLO training images were effectively
    processed twice, while current detector input was only processed once.

    db_min / db_max / eps are kept only for compatibility with existing calls.
    They are not used in this temporary mode.
    """
    # First pass: raw mat -> [0, 1]
    x1 = _percentile_log_float(
        data=data,
        p_low=p_low,
        p_high=p_high,
        log_gain=log_gain,
    )

    # Simulate saved old PNG quantization.
    # This makes the second pass closer to:
    #     old PNG -> percentile + log
    x1_u8 = np.clip(x1 * 255.0, 0, 255).astype(np.uint8)

    # Second pass: uint8 image -> [0, 1]
    x2 = _percentile_log_float(
        data=x1_u8,
        p_low=p_low,
        p_high=p_high,
        log_gain=log_gain,
    )

    return x2.astype(np.float32)


def spectrogram_to_yolo_uint8(
    data: np.ndarray,
    db_min: float = None,
    db_max: float = None,
    eps: float = 1e-12,
    p_low: float = 1.0,
    p_high: float = 99.5,
    log_gain: float = 9.0,
) -> np.ndarray:
    """
    Convert spectrogram to uint8 image using double percentile + log.
    """
    x = spectrogram_to_yolo_float(
        data=data,
        db_min=db_min,
        db_max=db_max,
        eps=eps,
        p_low=p_low,
        p_high=p_high,
        log_gain=log_gain,
    )
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)
    
# def spectrogram_to_yolo_float(
#     data: np.ndarray,
#     db_min: float = None,
#     db_max: float = None,
#     eps: float = 1e-12,
#     p_low: float = 1.0,
#     p_high: float = 99.5,
#     log_gain: float = 9.0,
# ) -> np.ndarray:
#     """
#     Convert spectrogram to normalized float image in [0, 1]
#     using percentile clipping + log enhancement.

#     Current temporary logic:
#         lo = percentile(data, p_low)
#         hi = percentile(data, p_high)
#         data = clip(data, lo, hi)
#         data = (data - lo) / (hi - lo)
#         data = log1p(log_gain * data) / log1p(log_gain)

#     Note:
#         db_min / db_max are kept only for compatibility with existing calls.
#         They are not used in this temporary percentile+log mode.
#     """
#     x = np.asarray(data, dtype=np.float32)

#     if x.ndim != 2:
#         raise ValueError(f"Expected 2D spectrogram, got shape={x.shape}")

#     finite = np.isfinite(x)
#     if not finite.any():
#         return np.zeros_like(x, dtype=np.float32)

#     p_low = float(p_low)
#     p_high = float(p_high)

#     if not (0.0 <= p_low < p_high <= 100.0):
#         raise ValueError(
#             f"Invalid percentile range: p_low={p_low}, p_high={p_high}. "
#             "Expected 0 <= p_low < p_high <= 100."
#         )

#     valid = x[finite]
#     lo = float(np.percentile(valid, p_low))
#     hi = float(np.percentile(valid, p_high))

#     if hi <= lo:
#         return np.zeros_like(x, dtype=np.float32)

#     x = np.clip(x, lo, hi)
#     x = (x - lo) / (hi - lo)

#     if float(log_gain) > 0:
#         x = np.log1p(float(log_gain) * x) / np.log1p(float(log_gain))

#     x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
#     x = np.clip(x, 0.0, 1.0)

#     return x.astype(np.float32)


# def spectrogram_to_yolo_uint8(
#     data: np.ndarray,
#     db_min: float = None,
#     db_max: float = None,
#     eps: float = 1e-12,
#     p_low: float = 1.0,
#     p_high: float = 99.5,
#     log_gain: float = 9.0,
# ) -> np.ndarray:
#     """
#     Convert spectrogram to uint8 image using percentile clipping + log enhancement.

#     db_min / db_max are accepted for compatibility with current detector.py / vis.py calls,
#     but are not used in this temporary mode.
#     """
#     x = spectrogram_to_yolo_float(
#         data=data,
#         db_min=db_min,
#         db_max=db_max,
#         eps=eps,
#         p_low=p_low,
#         p_high=p_high,
#         log_gain=log_gain,
#     )
#     return np.clip(x * 255.0, 0, 255).astype(np.uint8)
    
def _path_is_set(p):
    if p is None:
        return False
    if isinstance(p, (list, tuple)):
        return len([x for x in p if str(x).strip()]) > 0
    return bool(str(p).strip())

def _make_pbar(iterable, desc: str, leave: bool = False):
    """
    Create tqdm progress bar.

    disable=True when stdout is not a real terminal, e.g. nohup/log file,
    otherwise tqdm progress will be written repeatedly into log files.
    """
    return tqdm(
        iterable,
        desc=desc,
        leave=leave,
        dynamic_ncols=True,
        mininterval=0.5,
        file=sys.stdout,
        disable=not sys.stdout.isatty(),
    )

def _reduce_scalar(val, device, dtype=torch.float32, op=dist.ReduceOp.SUM):
    if isinstance(val, torch.Tensor):
        t = val.detach()
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        if device is not None and t.device != torch.device(device):
            t = t.to(device)
        t = t.clone()   
    else:
        t = torch.scalar_tensor(float(val), dtype=dtype, device=device)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=op)
        if op == dist.ReduceOp.SUM:
            ws = dist.get_world_size()
            if ws > 1:
                t /= ws

    return t.item()

def _set_epoch_for_loaders(epoch, *loaders):
    for ld in loaders:
        if ld is not None and hasattr(ld, "sampler") and hasattr(ld.sampler, "set_epoch"):
            ld.sampler.set_epoch(epoch)
            
class ColorFormatter(logging.Formatter):
    # ANSI 颜色表
    COLORS = {
        logging.DEBUG:    "\033[36m",        # 青色
        logging.INFO:     "\033[32m",        # 绿色
        logging.WARNING:  "\033[33m",        # 黄色
        logging.ERROR:    "\033[31m",        # 红色
        logging.CRITICAL: "\033[1;41m",      # 白字红底
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)
        
class ExpLogger(logging.Logger):
    def init_exp(self, config):
        super().info('---------------------Experiment Settings-------------------------')
        super().info('Nothing here')
        super().info('-----------------------------------------------------------------')
        
logging.setLoggerClass(ExpLogger)

def create_logger(filename: str) -> logging.Logger:
    
    def custom_time(*args):
        utc_plus_8 = datetime.now(tz=timezone.utc) + timedelta(hours=8)
        return utc_plus_8.timetuple()
        
    logger = logging.getLogger(filename)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter.converter = custom_time
    ch.setFormatter(formatter)
    
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

class EarlyStopping:
    def __init__(self, logger, patience=10, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.logger = logger

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.logger.info(
                f'--Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'--EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.logger.info(
                f'--Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            self.counter = 0