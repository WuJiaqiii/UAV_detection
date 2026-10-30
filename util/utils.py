import torch
import logging
import os
import numpy as np
import random
from datetime import datetime, timedelta, timezone
from tqdm import tqdm 
import sys

import csv
import json
from pathlib import Path
from typing import Optional


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
        info = super().info

        info("")
        info("===================== Experiment Settings =====================")

        # --------------------------------------------------
        # Run
        # --------------------------------------------------
        info("[Run]")
        info(f"  run_mode                 : {config.run_mode}")
        info(f"  trainer_type             : {getattr(config, 'trainer_type', 'normal')}")
        info(f"  device                   : {config.device}")

        # --------------------------------------------------
        # Dataset
        # --------------------------------------------------
        info("[Dataset]")
        info(f"  train_dataset_path       : {config.train_dataset_path}")
        info(f"  val_dataset_path         : {config.val_dataset_path}")
        info(f"  input_type               : {config.input_type}")

        # --------------------------------------------------
        # Class Split
        # --------------------------------------------------
        info("[Classes]")
        info(f"  incremental_split_mode   : {getattr(config, 'incremental_split_mode', 'normal')}")
        info(f"  incremental_new_classes  : {getattr(config, 'incremental_new_classes', [])}")
        info(f"  exclude_classes          : {getattr(config, 'exclude_classes', [])}")
        info(f"  eval_exclude_classes     : {getattr(config, 'eval_exclude_classes', [])}")

        if hasattr(config, "data_include_classes"):
            info(f"  train_include_classes    : {config.data_include_classes}")

        if hasattr(config, "data_exclude_classes"):
            info(f"  train_exclude_classes    : {config.data_exclude_classes}")

        if hasattr(config, "classes"):
            info(f"  classes                  : {config.classes}")

        if hasattr(config, "num_classes"):
            info(f"  num_classes              : {config.num_classes}")

        # --------------------------------------------------
        # YOLO
        # --------------------------------------------------
        info("[YOLO]")
        info(f"  yolo_weights             : {config.yolo_weights}")
        info(f"  conf_thres               : {config.yolo_conf_thres}")
        info(f"  iou_thres                : {config.yolo_iou_thres}")
        info(f"  max_det                  : {config.yolo_max_det}")

        info(f"  frontend_mode            : {getattr(config, 'frontend_mode', 'global')}")
        info(f"  detector_tile_width      : {getattr(config, 'detector_tile_width', 768)}")
        info(f"  detector_tile_overlap    : {getattr(config, 'detector_tile_overlap', 192)}")

        # --------------------------------------------------
        # Preprocessor
        # --------------------------------------------------
        info("[Preprocessor]")
        info(f"  min_group_len            : {config.min_group_len}")
        info(f"  min_group_time_span      : {config.min_group_time_span_ratio}")

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------
        info("[Classifier]")
        info(f"  backbone                 : {config.backbone}")
        info(f"  cnn_dropout              : {config.cnn_dropout}")

        # --------------------------------------------------
        # LwF
        # --------------------------------------------------
        if getattr(config, "trainer_type", "normal") == "lwf":

            info("[LwF]")
            info(f"  teacher_checkpoint       : {config.lwf_teacher_checkpoint}")
            info(f"  old_num_classes          : {config.lwf_old_num_classes}")
            info(f"  temperature             : {config.lwf_temperature}")
            info(f"  lambda_kd               : {config.lwf_lambda_kd}")
            info(f"  freeze_backbone         : {config.lwf_freeze_backbone}")

        # --------------------------------------------------
        # Cache
        # --------------------------------------------------
        info("[Cache]")
        info(f"  bbox_cache_mode          : {config.bbox_cache_mode}")
        info(f"  bbox_cache_path          : {config.bbox_cache_path}")

        # --------------------------------------------------
        # Training
        # --------------------------------------------------
        info("[Training]")
        info(f"  batch_size               : {config.batch_size}")
        info(f"  epochs                   : {config.epochs}")
        info(f"  lr                       : {config.lr}")
        info(f"  weight_decay             : {config.weight_decay}")

        if hasattr(config, "cosine_annealing_T0"):
            info(f"  cosine_T0               : {config.cosine_annealing_T0}")

        if hasattr(config, "cosine_annealing_mult"):
            info(f"  cosine_Tmult            : {config.cosine_annealing_mult}")

        info("================================================================")
        info("")
        
logging.setLoggerClass(ExpLogger)

def create_logger(filename: str) -> logging.Logger:
    
    def custom_time(*args):
        from util.run_context import TIMEZONE
        return datetime.fromtimestamp(args[0], TIMEZONE).timetuple()
        
    logger = logging.getLogger(filename)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    project_root = Path(__file__).resolve().parents[1]
    log_path = Path(filename).expanduser().resolve()
    try:
        display_path = Path(os.path.relpath(log_path, project_root)).as_posix()
    except ValueError:
        # Windows paths on another drive cannot be expressed relatively.
        display_path = log_path.as_posix()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(log_path)s - %(levelname)s - %(message)s',
        defaults={"log_path": display_path},
    )
    formatter.converter = custom_time
    ch.setFormatter(formatter)
    
    fh = logging.FileHandler(filename, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter.converter = custom_time
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
