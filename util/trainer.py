import os
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

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


class Trainer:
    def __init__(self, config, dataloaders, logger, detector, preprocessor, classifier, extra_val_loader=None):
        self.config = config
        self.logger = logger
        self.detector = detector
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.extra_val_loader = extra_val_loader

        if isinstance(dataloaders, (tuple, list)) and len(dataloaders) == 2:
            self.train_loader, self.val_loader = dataloaders
        else:
            self.train_loader, self.val_loader = None, None

        self.device = next(self.classifier.parameters()).device
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = AdamW(
            self.classifier.parameters(),
            lr=float(getattr(config, "lr", 1e-4)),
            weight_decay=float(getattr(config, "weight_decay", 1e-2)),
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=int(getattr(config, "cosine_annealing_T0", 50)),
            T_mult=int(getattr(config, "cosine_annealing_mult", 2)),
        )

        self.epochs = int(getattr(config, "epochs", 50))
        self.save_interval = int(getattr(config, "save_interval", 5))
        self.early_stop_patience = int(getattr(config, "early_stop_patience", 20))

        self.result_dir = Path(getattr(config, "result_dir", "./results"))
        self.model_dir = Path(getattr(config, "model_dir", "./models"))
        self.log_dir = Path(getattr(config, "log_dir", "./log"))
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(self.log_dir))

        self.mask_img_size = int(getattr(config, "mask_img_size", 224))
        self.cnn_input_mode = str(getattr(config, "cnn_input_mode", "mask")).lower()
        self.box_draw_thickness = int(getattr(config, "box_draw_thickness", 2))
        self.box_draw_value = int(getattr(config, "box_draw_value", 255))

        self.save_detect_vis_once = bool(getattr(config, "save_detect_vis_once", False))
        self.detect_vis_num_samples = int(getattr(config, "detect_vis_num_samples", 8))
        self._detect_vis_saved = False

        self.match_freq_thresh = float(getattr(config, "match_freq_thresh", 10.0))
        self.match_bandwidth_weight = float(getattr(config, "match_bandwidth_weight", 0.2))
        self.skip_unmatched = bool(getattr(config, "skip_unmatched", True))

        self.run_mode = str(getattr(config, "run_mode", "train")).lower()
        self.train_signal_mode = str(getattr(config, "train_signal_mode", "single")).lower()

        self.best_val_acc = -1.0
        self.no_improve_epochs = 0

        self.inv_class_map = self._build_inv_class_map()
        self.eval_exclude_classes = set(getattr(config, "eval_exclude_classes", []) or [])
        self.eval_exclude_label_ids = {
            int(v) for k, v in getattr(self.config, "classes", {}).items()
            if str(k) in self.eval_exclude_classes
        }

    def _build_inv_class_map(self) -> Dict[int, str]:
        classes = getattr(self.config, "classes", {})
        if isinstance(classes, dict) and len(classes) > 0:
            return {int(v): str(k) for k, v in classes.items()}
        return {}

    def _label_name(self, idx: int) -> str:
        return self.inv_class_map.get(int(idx), str(idx))

    def _should_skip_eval_label(self, label_idx: int) -> bool:
        if not self.eval_exclude_label_ids:
            return False
        return int(label_idx) in self.eval_exclude_label_ids

    def _spec_to_uint8(self, spec: np.ndarray) -> np.ndarray:
        spec = np.asarray(spec)
        if spec.ndim != 2:
            raise ValueError(f"Expected 2D spectrogram, got shape={spec.shape}")
        if spec.dtype == np.uint8:
            return spec.copy()
        x = spec.astype(np.float32)
        x_min, x_max = float(x.min()), float(x.max())
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        else:
            x = np.zeros_like(x, dtype=np.float32)
        return (x * 255.0).astype(np.uint8)

    def _spec_to_uint8_vis_log(self, spec: np.ndarray, p_low: float = 1.0, p_high: float = 99.5, log_gain: float = 9.0) -> np.ndarray:
        x = np.asarray(spec, dtype=np.float32)
        finite_mask = np.isfinite(x)
        if not finite_mask.any():
            return np.zeros_like(x, dtype=np.uint8)
        valid = x[finite_mask]
        lo = float(np.percentile(valid, p_low))
        hi = float(np.percentile(valid, p_high))
        if hi <= lo:
            return np.zeros_like(x, dtype=np.uint8)
        x = np.clip(x, lo, hi)
        x = (x - lo) / (hi - lo)
        x = np.log1p(log_gain * x) / np.log1p(log_gain)
        return np.clip(x * 255.0, 0, 255).astype(np.uint8)

    def _boxes_to_white_mask(self, image_shape: Tuple[int, int], boxes, fill_value: int = 255) -> np.ndarray:
        h, w = int(image_shape[0]), int(image_shape[1])
        mask = np.zeros((h, w), dtype=np.uint8)
        if boxes is None:
            return mask
        arr = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
        for b in arr:
            x1, y1, x2, y2 = [int(v) for v in b]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            mask[y1:y2, x1:x2] = int(fill_value)
        return mask

    def _make_input_tensor(self, spec, final_boxes):
        mode = self.cnn_input_mode
        raw = self._spec_to_uint8(spec)
        if mode == "mask":
            img = self._boxes_to_white_mask(spec.shape, final_boxes, fill_value=255)
        else:
            img = raw.copy()
            if mode == "raw_in_boxes":
                mask = self._boxes_to_white_mask(spec.shape, final_boxes, fill_value=255)
                out = np.zeros_like(img, dtype=np.uint8)
                keep = mask > 0
                out[keep] = img[keep]
                img = out
            elif mode == "raw_with_boxes":
                arr = np.asarray(final_boxes, dtype=np.int32).reshape(-1, 4) if final_boxes is not None else np.zeros((0, 4), dtype=np.int32)
                h, w = img.shape[:2]
                for box in arr:
                    x1, y1, x2, y2 = [int(v) for v in box]
                    x1 = max(0, min(x1, w - 1))
                    x2 = max(0, min(x2, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(0, min(y2, h - 1))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=int(self.box_draw_value), thickness=int(self.box_draw_thickness))
            elif mode != "raw":
                raise ValueError(f"Unsupported cnn_input_mode={mode}")
        x = img.astype(np.float32) / 255.0
        x = cv2.resize(x, (self.mask_img_size, self.mask_img_size), interpolation=cv2.INTER_NEAREST if mode == "mask" else cv2.INTER_LINEAR)
        x = np.expand_dims(x, axis=0)
        return torch.from_numpy(x).float(), img

    def _save_groups_image(self, spec_u8, groups, save_path):
        from PIL import Image, ImageDraw
        img = Image.fromarray(spec_u8).convert("RGB")
        draw = ImageDraw.Draw(img)
        colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,128,0),(128,255,0)]
        for gi, g in enumerate(groups):
            boxes = np.asarray(g.get("boxes", []), dtype=np.int32).reshape(-1, 4)
            if len(boxes) == 0:
                continue
            color = colors[gi % len(colors)]
            for b in boxes:
                x1, y1, x2, y2 = [int(v) for v in b]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            x1, y1, x2, y2 = [int(v) for v in boxes[0]]
            label = f"G{gi}"
            if "score" in g:
                label += f" s={float(g['score']):.2f}"
            if "center_freq" in g:
                label += f" f={float(g['center_freq']):.1f}"
            if "group_type" in g:
                label += f" {g['group_type']}"
            draw.text((x1, max(0, y1 - 12)), label, fill=color)
        img.save(save_path)

    def _save_detect_result_images(self, spec, yolo_boxes, groups, matched_boxes, fp, sample_idx=0, split_name="val"):
        save_dir = self.result_dir / "detect_result" / str(split_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        base = Path(fp).stem if fp else f"sample_{sample_idx}"
        spec_u8 = self._spec_to_uint8_vis_log(spec, p_low=1.0, p_high=99.5, log_gain=9.0)

        from PIL import Image, ImageDraw
        img = Image.fromarray(spec_u8).convert("RGB")
        draw = ImageDraw.Draw(img)
        for b in np.asarray(yolo_boxes, dtype=np.int32).reshape(-1, 4):
            x1, y1, x2, y2 = [int(v) for v in b]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        img.save(save_dir / f"{base}_yolo.png")

        self._save_groups_image(spec_u8, groups, save_dir / f"{base}_groups.png")

        img = Image.fromarray(spec_u8).convert("RGB")
        draw = ImageDraw.Draw(img)
        for b in np.asarray(matched_boxes, dtype=np.int32).reshape(-1, 4):
            x1, y1, x2, y2 = [int(v) for v in b]
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        img.save(save_dir / f"{base}_matched.png")

    def _extract_groups(self, yolo_boxes, spec):
        if hasattr(self.preprocessor, "select_signal_groups"):
            return self.preprocessor.select_signal_groups(yolo_boxes, spectrogram=spec)
        if hasattr(self.preprocessor, "select_main_boxes"):
            main_boxes = self.preprocessor.select_main_boxes(yolo_boxes, spectrogram=spec)
            main_boxes = np.asarray(main_boxes, dtype=np.int32).reshape(-1, 4)
            if len(main_boxes) == 0:
                return []
            return [{"boxes": main_boxes, "score": 1.0, "center_freq": 0.0, "bandwidth": 0.0, "group_type": "single_fallback"}]
        raise AttributeError("Preprocessor has neither select_signal_groups nor select_main_boxes")

    def _select_main_group(self, groups):
        if groups is None or len(groups) == 0:
            return None
        groups_sorted = sorted(groups, key=lambda g: float(g.get("score", 0.0)), reverse=True)
        return groups_sorted[0]

    def _pixel_y_to_freq_mhz(self, y, img_h):
        if img_h <= 1:
            return 0.0
        rel = float(y) / float(img_h - 1) - 0.5
        sr = float(getattr(self.config, "sampling_rate", 122.88e6))
        return rel * sr / 1e6

    def _group_center_freq_mhz(self, group, spec_shape):
        if isinstance(group, dict) and "center_freq" in group:
            return float(group["center_freq"])
        boxes = np.asarray(group.get("boxes", []), dtype=np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return 0.0
        y_center = float(np.mean((boxes[:, 1] + boxes[:, 3]) / 2.0))
        return self._pixel_y_to_freq_mhz(y_center, spec_shape[0])

    def _group_bandwidth_mhz(self, group):
        if isinstance(group, dict) and "bandwidth" in group:
            return float(group["bandwidth"])
        return 0.0

    def _match_groups_to_targets(self, groups, targets, spec_shape):
        if groups is None:
            groups = []
        if targets is None:
            targets = []
        n_g = len(groups)
        n_t = len(targets)
        if n_g == 0 or n_t == 0:
            return [], list(range(n_t)), list(range(n_g))

        group_freqs = [self._group_center_freq_mhz(g, spec_shape) for g in groups]
        group_bws = [self._group_bandwidth_mhz(g) for g in groups]
        target_freqs = [float(t.get("center_freq", 0.0)) for t in targets]
        target_bws = [float(t.get("bandwidth", 0.0)) for t in targets]

        cost = np.zeros((n_t, n_g), dtype=np.float32)
        for ti in range(n_t):
            for gi in range(n_g):
                freq_diff = abs(target_freqs[ti] - group_freqs[gi])
                bw_diff = abs(target_bws[ti] - group_bws[gi])
                cost[ti, gi] = float(freq_diff + self.match_bandwidth_weight * bw_diff)

        pairs = []
        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(cost)
            pairs = list(zip(rows.tolist(), cols.tolist()))
        else:
            used_g = set()
            for ti in range(n_t):
                best_g, best_c = None, None
                for gi in range(n_g):
                    if gi in used_g:
                        continue
                    c = cost[ti, gi]
                    if best_c is None or c < best_c:
                        best_c = c
                        best_g = gi
                if best_g is not None:
                    used_g.add(best_g)
                    pairs.append((ti, best_g))

        matched = []
        used_t, used_g = set(), set()
        for ti, gi in pairs:
            freq_diff = abs(target_freqs[ti] - group_freqs[gi])
            if self.skip_unmatched and freq_diff > self.match_freq_thresh:
                continue
            g = groups[gi]
            matched.append({
                "boxes": np.asarray(g.get("boxes", []), dtype=np.int32).reshape(-1, 4),
                "group_idx": int(gi),
                "target_idx": int(ti),
                "label": int(targets[ti]["label"]),
                "target": targets[ti],
                "group": g,
            })
            used_t.add(ti)
            used_g.add(gi)

        unmatched_targets = [i for i in range(n_t) if i not in used_t]
        unmatched_groups = [i for i in range(n_g) if i not in used_g]
        return matched, unmatched_targets, unmatched_groups

    def _build_single_instances(self, inputs, targets_list, sample_fps=None, save_detect_result=False, max_save_images=8, split_name="val"):
        inputs_np = inputs.detach().cpu().numpy() if torch.is_tensor(inputs) else np.asarray(inputs)
        batch_images, batch_labels, batch_metas, image_infos = [], [], [], []
        saved_count = 0

        for i in range(len(inputs_np)):
            spec = np.asarray(inputs_np[i])
            fp = sample_fps[i] if sample_fps is not None else f"sample_{i}"
            targets = targets_list[i] if targets_list is not None else []
            if len(targets) != 1:
                self.logger.warning(f"[SingleSignalSkip] expected 1 target, got {len(targets)}: {fp}")
                continue

            yolo_boxes = self.detector.detect(spec)
            yolo_boxes = np.asarray(yolo_boxes, dtype=np.int32).reshape(-1, 4) if len(yolo_boxes) > 0 else np.zeros((0, 4), dtype=np.int32)
            groups = self._extract_groups(yolo_boxes, spec)
            main_group = self._select_main_group(groups)
            if main_group is None:
                continue

            final_boxes = np.asarray(main_group.get("boxes", []), dtype=np.int32).reshape(-1, 4)
            if len(final_boxes) == 0:
                continue

            x, _ = self._make_input_tensor(spec, final_boxes)
            label = int(targets[0]["label"])
            batch_images.append(x)
            batch_labels.append(label)
            batch_metas.append({"fp": fp, "target_idx": 0, "group_idx": 0, "label": label, "target_name": self._label_name(label)})
            image_infos.append({"fp": fp, "num_targets": 1, "num_matched": 1, "unmatched_targets": [], "unmatched_groups": []})

            if save_detect_result and saved_count < max_save_images:
                self._save_detect_result_images(spec, yolo_boxes, groups, final_boxes.tolist(), fp, i, split_name)
                saved_count += 1

        if len(batch_images) == 0:
            return None, None, None, 0, 0, image_infos
        images = torch.stack(batch_images, dim=0)
        labels = torch.as_tensor(batch_labels, dtype=torch.long)
        return images, labels, batch_metas, len(batch_labels), len(batch_labels), image_infos

    def _build_multi_matched_instances(self, inputs, targets_list, sample_fps=None, save_detect_result=False, max_save_images=8, split_name="val"):
        inputs_np = inputs.detach().cpu().numpy() if torch.is_tensor(inputs) else np.asarray(inputs)
        batch_images, batch_labels, batch_metas, image_infos = [], [], [], []
        matched_total = 0
        target_total = 0
        saved_count = 0

        for i in range(len(inputs_np)):
            spec = np.asarray(inputs_np[i])
            fp = sample_fps[i] if sample_fps is not None else f"sample_{i}"
            targets = targets_list[i] if targets_list is not None else []

            yolo_boxes = self.detector.detect(spec)
            yolo_boxes = np.asarray(yolo_boxes, dtype=np.int32).reshape(-1, 4) if len(yolo_boxes) > 0 else np.zeros((0, 4), dtype=np.int32)

            groups = self._extract_groups(yolo_boxes, spec)
            matched, unmatched_targets, unmatched_groups = self._match_groups_to_targets(groups, targets, spec.shape)

            matched_total += len(matched)
            target_total += len(targets)

            image_infos.append({
                "fp": fp,
                "num_targets": len(targets),
                "num_matched": len(matched),
                "unmatched_targets": unmatched_targets,
                "unmatched_groups": unmatched_groups,
            })

            if save_detect_result and saved_count < max_save_images:
                matched_boxes = []
                for m in matched:
                    matched_boxes.extend(np.asarray(m["boxes"], dtype=np.int32).reshape(-1, 4).tolist())
                self._save_detect_result_images(spec, yolo_boxes, groups, matched_boxes, fp, i, split_name)
                saved_count += 1

            for m in matched:
                final_boxes = np.asarray(m["boxes"], dtype=np.int32).reshape(-1, 4)
                x, _ = self._make_input_tensor(spec, final_boxes)
                batch_images.append(x)
                batch_labels.append(int(m["label"]))
                batch_metas.append({
                    "fp": fp,
                    "target_idx": int(m["target_idx"]),
                    "group_idx": int(m["group_idx"]),
                    "label": int(m["label"]),
                    "target_name": self._label_name(int(m["label"])),
                })

        if len(batch_images) == 0:
            return None, None, None, matched_total, target_total, image_infos
        images = torch.stack(batch_images, dim=0)
        labels = torch.as_tensor(batch_labels, dtype=torch.long)
        return images, labels, batch_metas, matched_total, target_total, image_infos

    def _compute_image_exact_acc(self, metas, labels, preds, image_infos):
        if metas is None or len(metas) == 0:
            return 0, 0
        labels_np = labels.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()

        per_fp = {}
        for meta, y_true, y_pred in zip(metas, labels_np.tolist(), preds_np.tolist()):
            fp = meta["fp"]
            per_fp.setdefault(fp, [])
            per_fp[fp].append(int(y_true == y_pred))

        info_map = {info["fp"]: info for info in image_infos}
        image_total, image_correct = 0, 0
        for fp, corr_list in per_fp.items():
            info = info_map.get(fp, {})
            num_targets = int(info.get("num_targets", len(corr_list)))
            num_matched = int(info.get("num_matched", len(corr_list)))
            image_total += 1
            if num_targets == num_matched and all(corr_list):
                image_correct += 1
        return image_correct, image_total

    def _save_confusion_matrix(self, y_true, y_pred, split_name="val", epoch: Optional[int] = None):
        if confusion_matrix is None or len(y_true) == 0:
            return

        # Only keep non-excluded label ids on both axes.
        if len(self.inv_class_map) > 0:
            label_ids = [i for i in sorted(self.inv_class_map.keys()) if i not in self.eval_exclude_label_ids]
        else:
            label_ids = [i for i in sorted(set(int(x) for x in y_true) | set(int(x) for x in y_pred)) if i not in self.eval_exclude_label_ids]
        if len(label_ids) == 0:
            return

        cm = confusion_matrix(y_true, y_pred, labels=label_ids)
        class_names = [self._label_name(i) for i in label_ids]

        save_dir = self.result_dir / "confusion_matrix" / str(split_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = "confusion_matrix" if epoch is None else f"confusion_matrix_epoch_{epoch+1}"
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

    @torch.no_grad()
    def evaluate_loader(self, loader, epoch=0, split_name="val", mode="single", save_confusion=True, save_summary=True):
        self.classifier.eval()
        loss_meter = AverageMeter()
        correct = 0
        total = 0
        matched_total = 0
        target_total = 0
        image_correct_total = 0
        image_total = 0
        all_preds, all_targets, instance_rows = [], [], []

        desc = f"Evaluating {split_name}" if epoch is None else f"{split_name.capitalize()} Epoch {epoch + 1}"
        pbar = tqdm(loader, desc=desc, leave=True)

        for batch_idx, batch in enumerate(pbar):
            inputs, targets_list, snrs, fps = batch
            need_save_detect_result = self.save_detect_vis_once and (not self._detect_vis_saved) and batch_idx == 0

            if mode == "single":
                images, labels, metas, batch_matched, batch_targets, image_infos = self._build_single_instances(
                    inputs, targets_list, sample_fps=fps, save_detect_result=need_save_detect_result,
                    max_save_images=self.detect_vis_num_samples, split_name=split_name
                )
            elif mode == "multi":
                images, labels, metas, batch_matched, batch_targets, image_infos = self._build_multi_matched_instances(
                    inputs, targets_list, sample_fps=fps, save_detect_result=need_save_detect_result or (split_name == "infer"),
                    max_save_images=self.detect_vis_num_samples if split_name != "infer" else max(self.detect_vis_num_samples, len(fps)),
                    split_name=split_name
                )
            else:
                raise ValueError(f"Unsupported evaluate mode={mode}")

            matched_total += batch_matched
            target_total += batch_targets

            if need_save_detect_result:
                self._detect_vis_saved = True

            if images is None or labels is None or labels.numel() == 0:
                continue

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.classifier(images)
            preds = logits.argmax(dim=1)

            labels_cpu = labels.detach().cpu().numpy()
            preds_cpu = preds.detach().cpu().numpy()
            keep_mask = np.array([not self._should_skip_eval_label(int(y)) for y in labels_cpu], dtype=bool)
            if not keep_mask.any():
                continue

            keep_mask_t = torch.as_tensor(keep_mask, device=labels.device, dtype=torch.bool)
            logits_kept = logits[keep_mask_t]
            labels_kept = labels[keep_mask_t]
            preds_kept = preds[keep_mask_t]

            loss = self.criterion(logits_kept, labels_kept)
            loss_meter.update(loss.item(), labels_kept.size(0))

            preds_kept_cpu = preds_kept.detach().cpu().numpy()
            labels_kept_cpu = labels_kept.detach().cpu().numpy()
            correct += int((preds_kept_cpu == labels_kept_cpu).sum())
            total += int(labels_kept_cpu.shape[0])

            kept_metas = [m for m, k in zip(metas, keep_mask.tolist()) if k]
            batch_image_correct, batch_image_total = self._compute_image_exact_acc(
                metas=kept_metas, labels=labels_kept, preds=preds_kept, image_infos=image_infos
            )
            image_correct_total += batch_image_correct
            image_total += batch_image_total

            all_preds.append(preds_kept_cpu)
            all_targets.append(labels_kept_cpu)

            for meta, gt, pd in zip(kept_metas, labels_kept_cpu.tolist(), preds_kept_cpu.tolist()):
                instance_rows.append({
                    "file": meta["fp"],
                    "target_idx": meta["target_idx"],
                    "group_idx": meta["group_idx"],
                    "gt_label": gt,
                    "gt_name": self._label_name(gt),
                    "pred_label": pd,
                    "pred_name": self._label_name(pd),
                    "correct": int(gt == pd),
                })

            acc = 100.0 * correct / max(total, 1)
            match_recall = float(matched_total) / max(float(target_total), 1.0)
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc:.2f}", match=f"{match_recall:.3f}")

        if len(all_preds) > 0:
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
        else:
            all_preds = np.array([], dtype=np.int64)
            all_targets = np.array([], dtype=np.int64)

        loss_avg = loss_meter.avg if total > 0 else 0.0
        acc = 100.0 * correct / max(total, 1)
        match_recall = float(matched_total) / max(float(target_total), 1.0)
        image_exact_acc = 100.0 * image_correct_total / max(image_total, 1)

        if save_confusion:
            self._save_confusion_matrix(all_targets, all_preds, split_name=split_name, epoch=(None if split_name == "infer" else epoch))

        if save_summary:
            save_dir = self.result_dir / "eval_summary" / str(split_name)
            save_dir.mkdir(parents=True, exist_ok=True)
            stem = "infer" if split_name == "infer" else f"epoch_{epoch+1}"

            with open(save_dir / f"{stem}.json", "w", encoding="utf-8") as f:
                json.dump({
                    "split": split_name,
                    "epoch": None if split_name == "infer" else int(epoch + 1),
                    "mode": mode,
                    "loss": float(loss_avg),
                    "acc": float(acc),
                    "match_recall": float(match_recall),
                    "image_exact_acc": float(image_exact_acc),
                    "num_instances": int(total),
                    "num_targets": int(target_total),
                    "num_matched": int(matched_total),
                    "eval_exclude_classes": sorted(list(self.eval_exclude_classes)),
                }, f, ensure_ascii=False, indent=2)

            with open(save_dir / f"{stem}_instances.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["file", "target_idx", "group_idx", "gt_label", "gt_name", "pred_label", "pred_name", "correct"])
                writer.writeheader()
                for row in instance_rows:
                    writer.writerow(row)

        return loss_avg, acc, match_recall, image_exact_acc

    @torch.no_grad()
    def validate(self, epoch):
        return self.evaluate_loader(self.val_loader, epoch=epoch, split_name="val", mode="single", save_confusion=True, save_summary=True)

    @torch.no_grad()
    def validate_extra(self, epoch):
        if self.extra_val_loader is None:
            return None, None, None, None
        return self.evaluate_loader(self.extra_val_loader, epoch=epoch, split_name="extra_val", mode="single", save_confusion=True, save_summary=True)

    @torch.no_grad()
    def infer(self, infer_loader):
        self._detect_vis_saved = False
        infer_loss, infer_acc, infer_match_recall, infer_image_exact_acc = self.evaluate_loader(
            infer_loader, epoch=None, split_name="infer", mode="multi", save_confusion=True, save_summary=True
        )
        self.logger.info(f"[Infer] loss={infer_loss:.4f}, acc={infer_acc:.2f}, match={infer_match_recall:.3f}, img_exact={infer_image_exact_acc:.2f}")
        return infer_loss, infer_acc, infer_match_recall, infer_image_exact_acc

    def train_one_epoch(self, epoch):
        self.classifier.train()
        loss_meter = AverageMeter()
        correct = 0
        total = 0
        matched_total = 0
        target_total = 0
        image_correct_total = 0
        image_total = 0

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch + 1}", leave=True)
        for batch in pbar:
            inputs, targets_list, snrs, fps = batch
            images, labels, metas, batch_matched, batch_targets, image_infos = self._build_single_instances(
                inputs, targets_list, sample_fps=fps, save_detect_result=False, max_save_images=0, split_name="train"
            )

            matched_total += batch_matched
            target_total += batch_targets

            if images is None or labels is None or labels.numel() == 0:
                continue

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.classifier(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            preds = logits.argmax(dim=1)
            preds_cpu = preds.detach().cpu().numpy()
            labels_cpu = labels.detach().cpu().numpy()

            correct += int((preds_cpu == labels_cpu).sum())
            total += int(labels_cpu.shape[0])

            batch_image_correct, batch_image_total = self._compute_image_exact_acc(
                metas=metas, labels=labels, preds=preds, image_infos=image_infos
            )
            image_correct_total += batch_image_correct
            image_total += batch_image_total

            loss_meter.update(loss.item(), labels.size(0))
            acc = 100.0 * correct / max(total, 1)
            match_recall = float(matched_total) / max(float(target_total), 1.0)
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc:.2f}", match=f"{match_recall:.3f}")

        self.scheduler.step(epoch + 1)

        train_loss = loss_meter.avg if total > 0 else 0.0
        train_acc = 100.0 * correct / max(total, 1)
        train_match_recall = float(matched_total) / max(float(target_total), 1.0)
        train_image_exact_acc = 100.0 * image_correct_total / max(image_total, 1)

        self.writer.add_scalar("train/loss", train_loss, epoch)
        self.writer.add_scalar("train/acc", train_acc, epoch)
        self.writer.add_scalar("train/match_recall", train_match_recall, epoch)
        self.writer.add_scalar("train/image_exact_acc", train_image_exact_acc, epoch)
        return train_loss, train_acc, train_match_recall, train_image_exact_acc

    def _save_checkpoint(self, epoch, is_best=False):
        save_obj = {
            "epoch": int(epoch),
            "model": self.classifier.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config,
            "best_val_acc": float(self.best_val_acc),
        }
        ckpt_path = self.model_dir / f"epoch_{epoch+1}.pth"
        torch.save(save_obj, ckpt_path)
        if is_best:
            torch.save(save_obj, self.model_dir / "best.pth")

    def train(self):
        if self.train_loader is None or self.val_loader is None:
            raise RuntimeError("train_loader / val_loader is None. Use infer() for inference mode.")

        for epoch in range(self.epochs):
            self.logger.info(f"===== Epoch {epoch + 1}/{self.epochs} =====")

            train_loss, train_acc, train_match_recall, train_image_exact_acc = self.train_one_epoch(epoch)
            val_loss, val_acc, val_match_recall, val_image_exact_acc = self.validate(epoch)

            self.writer.add_scalar("val/loss", val_loss, epoch)
            self.writer.add_scalar("val/acc", val_acc, epoch)
            self.writer.add_scalar("val/match_recall", val_match_recall, epoch)
            self.writer.add_scalar("val/image_exact_acc", val_image_exact_acc, epoch)

            if self.extra_val_loader is not None:
                extra_loss, extra_acc, extra_match, extra_img_acc = self.validate_extra(epoch)
                if extra_loss is not None:
                    self.writer.add_scalar("extra_val/loss", extra_loss, epoch)
                    self.writer.add_scalar("extra_val/acc", extra_acc, epoch)
                    self.writer.add_scalar("extra_val/match_recall", extra_match, epoch)
                    self.writer.add_scalar("extra_val/image_exact_acc", extra_img_acc, epoch)
                    self.logger.info(f"[Extra Val] loss={extra_loss:.4f}, acc={extra_acc:.2f}, match={extra_match:.3f}, img_exact={extra_img_acc:.2f}")

            self.logger.info(f"[Train] loss={train_loss:.4f}, acc={train_acc:.2f}, match={train_match_recall:.3f}, img_exact={train_image_exact_acc:.2f}")
            self.logger.info(f"[Val]   loss={val_loss:.4f}, acc={val_acc:.2f}, match={val_match_recall:.3f}, img_exact={val_image_exact_acc:.2f}")

            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.no_improve_epochs = 0
            else:
                self.no_improve_epochs += 1

            if ((epoch + 1) % self.save_interval == 0) or is_best:
                self._save_checkpoint(epoch, is_best=is_best)

            if self.no_improve_epochs >= self.early_stop_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        self.writer.close()
