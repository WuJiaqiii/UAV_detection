import os

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

from util.utils import EarlyStopping, AverageMeter
from util.checkpoint import save_checkpoint
from util.utils import _reduce_scalar, _set_epoch_for_loaders
from util.boxmask import boxes_to_white_mask, mask_to_tensor
from util.bboxcache import BBoxCache

import cv2

class Trainer:
    def __init__(self, config, data_loaders, logger, detector, preprocessor, classifier):
        self.config = config
        self.logger = logger
        self.detector = detector
        self.preprocessor = preprocessor
        self.classifier = classifier

        self.train_loader, self.val_loader = data_loaders
        self.device = torch.device(config.device)

        self.classifier.to(self.device)

        if torch.distributed.is_initialized():
            self.classifier = torch.nn.parallel.DistributedDataParallel(
                self.classifier,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=self.device.index if self.device.type == "cuda" else None,
                find_unused_parameters=False,
            )
        elif getattr(config, "use_data_parallel", False) and torch.cuda.device_count() > 1:
            self.classifier = torch.nn.DataParallel(self.classifier)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            [p for p in self.classifier.parameters() if p.requires_grad],
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=int(config.cosine_annealing_T0),
            T_mult=int(config.cosine_annealing_mult),
        )
        self.scaler = GradScaler(enabled=bool(config.use_amp_autocast))

        self.mask_img_size = int(getattr(config, "mask_img_size", 224))
        self.mask_source = str(getattr(config, "mask_source", "final")).lower()
        self.mask_fill_value = int(getattr(config, "mask_fill_value", 255))
        self.save_detect_vis_once = bool(getattr(config, "save_detect_vis_once", True))
        self.detect_vis_num_samples = int(getattr(config, "detect_vis_num_samples", 1))
        self._detect_vis_saved = False

        self.cnn_input_mode = str(getattr(config, "cnn_input_mode", "mask")).lower()
        self.box_draw_thickness = int(getattr(config, "box_draw_thickness", 2))
        self.box_draw_value = int(getattr(config, "box_draw_value", 255))

        self.match_freq_thresh = float(getattr(config, "match_freq_thresh", 10.0))
        self.skip_unmatched = bool(getattr(config, "skip_unmatched", True))

        cache_dir = getattr(config, "bbox_cache_path", None)
        if not cache_dir:
            cache_dir = os.path.join(config.result_dir, "bbox_cache")
        self.bbox_cache = BBoxCache(
            base_dir=cache_dir,
            dataset_root=getattr(config, "dataset_path", None),
            mode=str(getattr(config, "bbox_cache_mode", "readwrite")),
            mem_max=0,
            logger=logger,
        )

        self.early_stopping = EarlyStopping(
            self.logger,
            patience=int(config.early_stop_patience),
            delta=0.0,
        )

        self.writer = None
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            self.writer = SummaryWriter(log_dir=config.log_dir)

    # ------------------------- image helpers -------------------------
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

    def _spec_to_uint8_vis_log(
        self,
        spec: np.ndarray,
        p_low: float = 1.0,
        p_high: float = 99.5,
        log_gain: float = 9.0,
    ) -> np.ndarray:
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

    def _make_input_tensor(self, spec, yolo_boxes, final_boxes):
        mode = self.cnn_input_mode
        boxes = final_boxes if self.mask_source == "final" else yolo_boxes

        if mode == "mask":
            img = boxes_to_white_mask(
                image_shape=spec.shape,
                boxes=boxes,
                fill_value=self.mask_fill_value,
                mode="fill",
            )

        else:
            img = self._spec_to_uint8(spec)

            if mode == "raw_in_boxes":
                if boxes is None or len(boxes) == 0:
                    img = np.zeros_like(img, dtype=np.uint8)
                else:
                    mask = boxes_to_white_mask(
                        image_shape=spec.shape,
                        boxes=boxes,
                        fill_value=255,
                        mode="fill",
                    )
                    out = np.zeros_like(img, dtype=np.uint8)
                    keep = mask > 0
                    out[keep] = img[keep]
                    img = out

            elif mode == "raw_with_boxes":
                if final_boxes is not None and len(final_boxes) > 0:
                    arr = np.asarray(final_boxes, dtype=np.int32).reshape(-1, 4)
                    H, W = img.shape[:2]
                    for box in arr:
                        x1, y1, x2, y2 = [int(v) for v in box]
                        x1 = max(0, min(x1, W - 1))
                        x2 = max(0, min(x2, W - 1))
                        y1 = max(0, min(y1, H - 1))
                        y2 = max(0, min(y2, H - 1))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        cv2.rectangle(
                            img,
                            (x1, y1),
                            (x2, y2),
                            color=int(self.box_draw_value),
                            thickness=int(self.box_draw_thickness),
                        )

            elif mode != "raw":
                raise ValueError(f"Unsupported cnn_input_mode={mode}")

        x = img.astype(np.float32) / 255.0
        x = cv2.resize(
            x,
            (self.mask_img_size, self.mask_img_size),
            interpolation=cv2.INTER_NEAREST if mode == "mask" else cv2.INTER_LINEAR,
        )
        x = np.expand_dims(x, axis=0)
        return torch.from_numpy(x).float()

    # ------------------------- detection helpers -------------------------
    def _sanitize_boxes(self, boxes, H, W):
        if boxes is None:
            return []
        arr = np.asarray(boxes, dtype=np.int32).reshape(-1, 4) if len(boxes) > 0 else np.zeros((0, 4), dtype=np.int32)
        out = []
        img_area = float(H * W)
        min_area = int(getattr(self.config, "pre_min_area", 0))
        max_bbox_area_ratio = float(getattr(self.config, "max_bbox_area_ratio", 0.0))
        for b in arr:
            x1, y1, x2, y2 = [int(v) for v in b]
            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H))
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            if min_area > 0 and area < min_area:
                continue
            if max_bbox_area_ratio > 0 and area > max_bbox_area_ratio * img_area:
                continue
            out.append([x1, y1, x2, y2])
        return out

    def _get_boxes_for_sample(self, spec, fp=None):
        H, W = spec.shape[:2]
        if fp is not None:
            got = self.bbox_cache.get(fp)
            if got is not None:
                return got.cpu().numpy().tolist()
        try:
            boxes = self.detector.detect(spec)
            boxes = self._sanitize_boxes(boxes, H, W)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[DetectorError] {os.path.basename(fp) if fp else 'unknown'}: {e}")
            boxes = []
        if fp is not None:
            self.bbox_cache.put(fp, torch.as_tensor(boxes, dtype=torch.int32))
        return boxes

    # ------------------------- multi-signal preprocess + matching -------------------------
    def _extract_groups(self, yolo_boxes, spec):
        if hasattr(self.preprocessor, "select_signal_groups"):
            groups = self.preprocessor.select_signal_groups(yolo_boxes, spectrogram=spec)
        else:
            # fallback to single-signal preprocess
            if self.mask_source == "raw":
                boxes = yolo_boxes
            else:
                boxes = self.preprocessor.select_main_boxes(yolo_boxes, spectrogram=spec)
                boxes = boxes.tolist() if hasattr(boxes, "tolist") else boxes
            groups = [{"boxes": boxes}] if boxes is not None and len(boxes) > 0 else []
        groups = groups if groups is not None else []
        return groups

    def _pixel_y_to_freq_mhz(self, y_center: float, H: int) -> float:
        # same mapping style as old preprocess.process(), convert to MHz for matching
        sr = float(getattr(self.config, "sampling_rate", getattr(self.preprocessor, "sampling_rate", 122.88e6)))
        freq_res = (sr / 2.0) / max(float(H - 1), 1.0)
        freq_hz = (float(y_center) - H / 2.0) * freq_res
        return freq_hz / 1e6

    def _group_center_freq_mhz(self, group: dict, spec_shape) -> float:
        # Try to use precomputed stats first
        for key in ["group_freq_center", "freq_center", "center_freq", "mean_freq"]:
            if key in group:
                return float(group[key])
        boxes = np.asarray(group.get("boxes", []), dtype=np.float32).reshape(-1, 4)
        if boxes.size == 0:
            return float("nan")
        cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
        return float(np.mean([self._pixel_y_to_freq_mhz(v, spec_shape[0]) for v in cy]))

    def _match_groups_to_targets(self, groups, targets, spec_shape):
        """
        targets: list[dict] with keys: label, center_freq, class_name, bandwidth(optional)
        returns list of dicts: {group_idx, target_idx, label, center_freq, group_freq_center, boxes}
        """
        if groups is None:
            groups = []
        if targets is None:
            targets = []
        if len(groups) == 0 or len(targets) == 0:
            return [], list(range(len(targets))), list(range(len(groups)))

        group_freqs = np.array([self._group_center_freq_mhz(g, spec_shape) for g in groups], dtype=np.float32)
        target_freqs = np.array([float(t["center_freq"]) for t in targets], dtype=np.float32)

        valid_group_mask = np.isfinite(group_freqs)
        if not valid_group_mask.any():
            return [], list(range(len(targets))), list(range(len(groups)))

        cost = np.abs(target_freqs[:, None] - group_freqs[None, :])

        matched = []
        used_t = set()
        used_g = set()

        if linear_sum_assignment is not None:
            row_ind, col_ind = linear_sum_assignment(cost)
            for ti, gi in zip(row_ind.tolist(), col_ind.tolist()):
                if not np.isfinite(cost[ti, gi]):
                    continue
                if cost[ti, gi] > self.match_freq_thresh:
                    continue
                matched.append({
                    "target_idx": ti,
                    "group_idx": gi,
                    "label": int(targets[ti]["label"]),
                    "class_name": str(targets[ti].get("class_name", "")),
                    "target_center_freq": float(targets[ti]["center_freq"]),
                    "group_center_freq": float(group_freqs[gi]),
                    "freq_error": float(cost[ti, gi]),
                    "boxes": groups[gi].get("boxes", []),
                    "group": groups[gi],
                })
                used_t.add(ti)
                used_g.add(gi)
        else:
            # greedy fallback
            pairs = []
            for ti in range(len(targets)):
                for gi in range(len(groups)):
                    if np.isfinite(cost[ti, gi]):
                        pairs.append((float(cost[ti, gi]), ti, gi))
            pairs.sort(key=lambda x: x[0])
            for c, ti, gi in pairs:
                if c > self.match_freq_thresh:
                    continue
                if ti in used_t or gi in used_g:
                    continue
                matched.append({
                    "target_idx": ti,
                    "group_idx": gi,
                    "label": int(targets[ti]["label"]),
                    "class_name": str(targets[ti].get("class_name", "")),
                    "target_center_freq": float(targets[ti]["center_freq"]),
                    "group_center_freq": float(group_freqs[gi]),
                    "freq_error": float(c),
                    "boxes": groups[gi].get("boxes", []),
                    "group": groups[gi],
                })
                used_t.add(ti)
                used_g.add(gi)

        unmatched_targets = [i for i in range(len(targets)) if i not in used_t]
        unmatched_groups = [i for i in range(len(groups)) if i not in used_g]
        return matched, unmatched_targets, unmatched_groups

    def _save_detect_result_images(self, spec, yolo_boxes, matched_boxes, fp, sample_idx=0):
        save_dir = os.path.join(self.config.result_dir, "detect_result")
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(fp))[0] if fp else f"sample_{sample_idx}"
        spec_u8 = self._spec_to_uint8_vis_log(spec, p_low=1.0, p_high=99.5, log_gain=9.0)
        from PIL import Image, ImageDraw
        for name, boxes in [("yolo", yolo_boxes), ("matched", matched_boxes)]:
            img = Image.fromarray(spec_u8).convert("RGB")
            draw = ImageDraw.Draw(img)
            color = (255, 0, 0) if name == "yolo" else (0, 255, 0)
            for b in boxes:
                x1, y1, x2, y2 = [int(v) for v in b]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            img.save(os.path.join(save_dir, f"{base}_{name}.png"))

    def _build_matched_instances(self, inputs_bhw, targets_list, sample_fps=None, save_detect_result=False, max_save_images=1):
        if isinstance(inputs_bhw, torch.Tensor):
            batch = inputs_bhw.detach().cpu()
        else:
            batch = torch.as_tensor(inputs_bhw)

        if batch.ndim == 4 and batch.shape[1] == 1:
            batch = batch[:, 0]
        if batch.ndim != 3:
            raise ValueError(f"Expected inputs shape [B,H,W] or [B,1,H,W], got {tuple(batch.shape)}")

        image_tensors = []
        labels = []
        metas = []
        saved_count = 0
        match_total = 0
        target_total = 0

        B = batch.shape[0]
        for i in range(B):
            spec = batch[i].numpy()
            fp = None if sample_fps is None else sample_fps[i]
            targets = targets_list[i] if targets_list is not None else []
            target_total += len(targets)

            yolo_boxes = self._get_boxes_for_sample(spec, fp=fp)
            groups = self._extract_groups(yolo_boxes, spec)
            matched, unmatched_targets, unmatched_groups = self._match_groups_to_targets(groups, targets, spec.shape)
            match_total += len(matched)

            if save_detect_result and saved_count < max_save_images:
                matched_boxes = []
                for m in matched:
                    matched_boxes.extend(np.asarray(m["boxes"], dtype=np.int32).reshape(-1, 4).tolist())
                self._save_detect_result_images(spec, yolo_boxes, matched_boxes, fp, sample_idx=i)
                saved_count += 1

            for m in matched:
                final_boxes = m["boxes"]
                if final_boxes is None or len(final_boxes) == 0:
                    continue
                x = self._build_input_image(spec, yolo_boxes, final_boxes)
                image_tensors.append(x)
                labels.append(int(m["label"]))
                metas.append({
                    "fp": fp,
                    "target_idx": m["target_idx"],
                    "group_idx": m["group_idx"],
                    "freq_error": m["freq_error"],
                    "class_name": m["class_name"],
                })

        if len(image_tensors) == 0:
            return None, None, metas, match_total, target_total

        images = torch.stack(image_tensors, dim=0)
        labels = torch.as_tensor(labels, dtype=torch.long)
        return images, labels, metas, match_total, target_total

    # ------------------------- loss / loops -------------------------
    def _forward_loss(self, images, labels):
        with autocast(enabled=bool(self.config.use_amp_autocast), device_type=self.device.type):
            logits = self.classifier(images)
            loss = self.criterion(logits, labels)
        return logits, loss

    def train_one_epoch(self, epoch):
        self.classifier.train()
        _set_epoch_for_loaders(self.train_loader, epoch)

        loss_meter = AverageMeter()
        correct = 0
        total = 0
        matched_total = 0
        target_total = 0
        skipped_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}", leave=True)
        for batch in pbar:
            inputs, targets_list, snrs, fps = batch

            images, labels, metas, batch_matched, batch_targets = self._build_matched_instances(
                inputs,
                targets_list,
                sample_fps=fps,
                save_detect_result=False,
            )
            matched_total += batch_matched
            target_total += batch_targets

            if images is None or labels is None or labels.numel() == 0:
                skipped_batches += 1
                pbar.set_postfix(skipped=skipped_batches, match=f"{matched_total}/{max(target_total,1)}")
                continue

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits, loss = self._forward_loss(images, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

            loss_meter.update(loss.item(), labels.size(0))
            match_recall = float(matched_total) / max(float(target_total), 1.0)
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{100.0 * correct / max(total,1):.2f}%", match=f"{match_recall:.3f}")

        self.scheduler.step(epoch + 1)

        train_loss = loss_meter.avg if total > 0 else 0.0
        train_acc = 100.0 * correct / max(total, 1)
        train_match_recall = float(matched_total) / max(float(target_total), 1.0)

        if dist.is_initialized():
            train_loss = _reduce_scalar(torch.tensor(train_loss, device=self.device), op="mean").item()
            train_acc = _reduce_scalar(torch.tensor(train_acc, device=self.device), op="mean").item()
            train_match_recall = _reduce_scalar(torch.tensor(train_match_recall, device=self.device), op="mean").item()

        return train_loss, train_acc, train_match_recall

    @torch.no_grad()
    def validate(self, epoch):
        self.classifier.eval()

        loss_meter = AverageMeter()
        correct = 0
        total = 0
        matched_total = 0
        target_total = 0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.val_loader, desc=f"Validating Epoch {epoch + 1}", leave=True)
        for batch_idx, batch in enumerate(pbar):
            inputs, targets_list, snrs, fps = batch

            need_save_detect_result = self.save_detect_vis_once and (not self._detect_vis_saved) and batch_idx == 0

            images, labels, metas, batch_matched, batch_targets = self._build_matched_instances(
                inputs,
                targets_list,
                sample_fps=fps,
                save_detect_result=need_save_detect_result,
                max_save_images=self.detect_vis_num_samples,
            )
            matched_total += batch_matched
            target_total += batch_targets

            if need_save_detect_result:
                self._detect_vis_saved = True

            if images is None or labels is None or labels.numel() == 0:
                pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{100.0 * correct / max(total,1):.2f}%", match=f"{float(matched_total)/max(float(target_total),1.0):.3f}")
                continue

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits, loss = self._forward_loss(images, labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

            loss_meter.update(loss.item(), labels.size(0))
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(labels.detach().cpu().numpy())

            match_recall = float(matched_total) / max(float(target_total), 1.0)
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{100.0 * correct / max(total,1):.2f}%", match=f"{match_recall:.3f}")

        val_loss = loss_meter.avg if total > 0 else 0.0
        val_acc = 100.0 * correct / max(total, 1)
        val_match_recall = float(matched_total) / max(float(target_total), 1.0)

        if len(all_preds) > 0:
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
        else:
            all_preds = np.array([], dtype=np.int64)
            all_targets = np.array([], dtype=np.int64)

        if dist.is_initialized():
            val_loss = _reduce_scalar(torch.tensor(val_loss, device=self.device), op="mean").item()
            val_acc = _reduce_scalar(torch.tensor(val_acc, device=self.device), op="mean").item()
            val_match_recall = _reduce_scalar(torch.tensor(val_match_recall, device=self.device), op="mean").item()

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            self._save_confusion_matrix(all_targets, all_preds, epoch)

        return val_loss, val_acc, val_match_recall

    def _save_confusion_matrix(self, y_true, y_pred, epoch):
        if len(y_true) == 0:
            return
        labels = list(range(self.config.num_classes))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        if isinstance(self.config.classes, dict):
            idx_to_name = {v: k for k, v in self.config.classes.items()}
            names = [idx_to_name.get(i, str(i)) for i in labels]
        else:
            names = [str(i) for i in labels]
        save_dir = os.path.join(self.config.result_dir, "confusion_matrix")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"epoch_{epoch + 1}.png")
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - Epoch {epoch + 1}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    def train(self):
        best_acc = -1.0
        for epoch in range(int(self.config.epochs)):
            train_loss, train_acc, train_match_recall = self.train_one_epoch(epoch)
            val_loss, val_acc, val_match_recall = self.validate(epoch)

            if (not dist.is_initialized()) or dist.get_rank() == 0:
                self.logger.info(
                    f"[Epoch {epoch + 1}/{self.config.epochs}] "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}, train_match={train_match_recall:.3f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}, val_match={val_match_recall:.3f}"
                )

                if self.writer is not None:
                    self.writer.add_scalar("train/loss", train_loss, epoch + 1)
                    self.writer.add_scalar("train/acc", train_acc, epoch + 1)
                    self.writer.add_scalar("train/match_recall", train_match_recall, epoch + 1)
                    self.writer.add_scalar("val/loss", val_loss, epoch + 1)
                    self.writer.add_scalar("val/acc", val_acc, epoch + 1)
                    self.writer.add_scalar("val/match_recall", val_match_recall, epoch + 1)

                is_best = val_acc > best_acc
                if is_best:
                    best_acc = val_acc

                save_checkpoint(
                    models={"model": self.classifier.module if hasattr(self.classifier, "module") else self.classifier},
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    path=os.path.join(self.config.model_dir, f"epoch_{epoch + 1}.pth"),
                    cfg=self.config
                )

                self.early_stopping(val_loss, self.classifier)
                stop = self.early_stopping.early_stop
            else:
                stop = False

            if dist.is_initialized():
                stop_tensor = torch.tensor(int(stop), device=self.device)
                dist.broadcast(stop_tensor, src=0)
                stop = bool(stop_tensor.item())

            if stop:
                if (not dist.is_initialized()) or dist.get_rank() == 0:
                    self.logger.info("Early stopping triggered.")
                break

        if self.writer is not None:
            self.writer.close()
