import os
import json
from collections import OrderedDict

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

from util.utils import EarlyStopping, AverageMeter
from util.checkpoint import save_checkpoint
from util.utils import _reduce_scalar, _set_epoch_for_loaders
from util.boxmask import boxes_to_white_mask, mask_to_tensor

class BBoxCache:
    def __init__(
        self,
        base_dir: str,
        dataset_root: str | None = None,
        mode: str = "readwrite",
        mem_max: int = 0,
        logger=None,
    ):
        self.mode = str(mode).lower()
        self.mem_max = int(mem_max) if mem_max else 0
        self.logger = logger
        self.dataset_root = dataset_root
        self.cache_dir = base_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._mem = OrderedDict()

    @staticmethod
    def _safe_relpath(fp: str, root: str | None) -> str:
        try:
            if root:
                return os.path.relpath(fp, root)
        except Exception:
            pass
        return fp

    def _key_and_meta(self, fp: str):
        rel = self._safe_relpath(fp, self.dataset_root)
        base = os.path.splitext(os.path.basename(fp))[0]
        key = base
        try:
            st = os.stat(fp)
            meta = {
                "rel": rel,
                "size": int(st.st_size),
                "mtime": int(st.st_mtime),
            }
        except Exception:
            meta = {
                "rel": rel,
                "size": None,
                "mtime": None,
            }
        return key, meta

    def _path_for_key(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.pt")

    def get(self, fp: str):
        if self.mode in ("off", "write", "refresh"):
            return None

        key, _ = self._key_and_meta(fp)
        if key in self._mem:
            boxes = self._mem.pop(key)
            self._mem[key] = boxes
            return boxes

        path = self._path_for_key(key)
        if not os.path.isfile(path):
            return None

        try:
            obj = torch.load(path, map_location="cpu")
            boxes = obj.get("boxes", torch.zeros((0, 4), dtype=torch.int32))
            if not isinstance(boxes, torch.Tensor):
                boxes = torch.as_tensor(boxes, dtype=torch.int32)
            boxes = boxes.to(dtype=torch.int32, device="cpu").reshape(-1, 4)

            if self.mem_max > 0:
                self._mem[key] = boxes
                while len(self._mem) > self.mem_max:
                    self._mem.popitem(last=False)

            return boxes
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[BBoxCache] failed reading {fp}: {e}")
            return None

    def put(self, fp: str, boxes: torch.Tensor):
        if self.mode in ("off", "read"):
            return

        key, meta = self._key_and_meta(fp)
        path = self._path_for_key(key)
        obj = {
            "source": meta,
            "boxes": boxes.to(dtype=torch.int32, device="cpu"),
        }
        try:
            torch.save(obj, path)
            if self.mem_max > 0:
                self._mem[key] = obj["boxes"]
                while len(self._mem) > self.mem_max:
                    self._mem.popitem(last=False)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[BBoxCache] failed writing {fp}: {e}")


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

    def _build_mask_from_boxes(self, spec, boxes):
        mask = boxes_to_white_mask(
            image_shape=spec.shape,
            boxes=boxes,
            fill_value=self.mask_fill_value,
            mode="fill",
        )
        x = mask_to_tensor(mask, out_size=self.mask_img_size)   # (1,H,W)
        return torch.from_numpy(x).float()

    def _save_detect_result_images(self, spec, yolo_boxes, final_boxes, fp, sample_idx=0):
        save_dir = os.path.join(self.config.result_dir, "detect_result")
        os.makedirs(save_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(fp))[0] if fp else f"sample_{sample_idx}"

        spec_u8 = np.asarray(spec)
        if spec_u8.dtype != np.uint8:
            s = spec_u8.astype(np.float32)
            s_min, s_max = float(s.min()), float(s.max())
            if s_max > s_min:
                s = (s - s_min) / (s_max - s_min)
            else:
                s = np.zeros_like(s, dtype=np.float32)
            spec_u8 = (s * 255.0).astype(np.uint8)

        from PIL import Image, ImageDraw

        for name, boxes in [("yolo", yolo_boxes), ("final", final_boxes)]:
            img = Image.fromarray(spec_u8).convert("RGB")
            draw = ImageDraw.Draw(img)
            color = (255, 0, 0) if name == "yolo" else (0, 255, 0)
            for b in boxes:
                x1, y1, x2, y2 = [int(v) for v in b]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            img.save(os.path.join(save_dir, f"{base}_{name}.png"))

    def _batch_to_mask_images(self, inputs_bhw, sample_fps=None, save_detect_result=False, max_save_images=1):
        if isinstance(inputs_bhw, torch.Tensor):
            batch = inputs_bhw.detach().cpu()
        else:
            batch = torch.as_tensor(inputs_bhw)

        if batch.ndim == 4 and batch.shape[1] == 1:
            batch = batch[:, 0]
        if batch.ndim != 3:
            raise ValueError(f"Expected inputs shape [B,H,W] or [B,1,H,W], got {tuple(batch.shape)}")

        B = batch.shape[0]
        out = []
        saved_count = 0

        for i in range(B):
            spec = batch[i].numpy()
            fp = None if sample_fps is None else sample_fps[i]

            yolo_boxes = self._get_boxes_for_sample(spec, fp=fp)

            if self.mask_source == "raw":
                boxes = yolo_boxes
                final_boxes = yolo_boxes
            elif self.mask_source == "final":
                final_boxes = self.preprocessor.select_main_boxes(yolo_boxes, spectrogram=spec)
                boxes = final_boxes.tolist() if hasattr(final_boxes, "tolist") else final_boxes
            else:
                raise ValueError(f"Unsupported mask_source={self.mask_source}, use 'raw' or 'final'.")

            x = self._build_mask_from_boxes(spec, boxes)
            out.append(x)

            if save_detect_result and saved_count < max_save_images:
                try:
                    final_boxes_vis = self.preprocessor.select_main_boxes(yolo_boxes, spectrogram=spec)
                    final_boxes_vis = final_boxes_vis.tolist() if hasattr(final_boxes_vis, "tolist") else final_boxes_vis
                except Exception:
                    final_boxes_vis = []
                self._save_detect_result_images(
                    spec=spec,
                    yolo_boxes=yolo_boxes,
                    final_boxes=final_boxes_vis,
                    fp=fp,
                    sample_idx=i,
                )
                saved_count += 1

        return torch.stack(out, dim=0)  # [B,1,H,W]

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

        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}", leave=True)
        for batch in pbar:
            inputs, labels, freq, bw, snr, fps = batch
            labels = labels.to(self.device, non_blocking=True)

            images = self._batch_to_mask_images(inputs, sample_fps=fps, save_detect_result=False)
            images = images.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits, loss = self._forward_loss(images, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

            loss_meter.update(loss.item(), labels.size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{100.0 * correct / max(total,1):.2f}%")

        self.scheduler.step(epoch + 1)

        train_loss = loss_meter.avg
        train_acc = 100.0 * correct / max(total, 1)

        if dist.is_initialized():
            train_loss = _reduce_scalar(torch.tensor(train_loss, device=self.device), op="mean").item()
            train_acc = _reduce_scalar(torch.tensor(train_acc, device=self.device), op="mean").item()

        return train_loss, train_acc

    @torch.no_grad()
    def validate(self, epoch):
        self.classifier.eval()

        loss_meter = AverageMeter()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        pbar = tqdm(self.val_loader, desc=f"Validating Epoch {epoch + 1}", leave=True)
        for batch_idx, batch in enumerate(pbar):
            inputs, labels, freq, bw, snr, fps = batch
            labels = labels.to(self.device, non_blocking=True)

            need_save_detect_result = (
                self.save_detect_vis_once and (not self._detect_vis_saved) and batch_idx == 0
            )

            images = self._batch_to_mask_images(
                inputs,
                sample_fps=fps,
                save_detect_result=need_save_detect_result,
                max_save_images=self.detect_vis_num_samples,
            )
            images = images.to(self.device, non_blocking=True)

            if need_save_detect_result:
                self._detect_vis_saved = True

            logits, loss = self._forward_loss(images, labels)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

            loss_meter.update(loss.item(), labels.size(0))
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(labels.detach().cpu().numpy())

            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{100.0 * correct / max(total,1):.2f}%")

        val_loss = loss_meter.avg
        val_acc = 100.0 * correct / max(total, 1)

        if len(all_preds) > 0:
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
        else:
            all_preds = np.array([], dtype=np.int64)
            all_targets = np.array([], dtype=np.int64)

        if dist.is_initialized():
            val_loss = _reduce_scalar(torch.tensor(val_loss, device=self.device), op="mean").item()
            val_acc = _reduce_scalar(torch.tensor(val_acc, device=self.device), op="mean").item()

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            self._save_confusion_matrix(all_targets, all_preds, epoch)

        return val_loss, val_acc

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
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=names, yticklabels=names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - Epoch {epoch + 1}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    def train(self):
        best_acc = -1.0

        for epoch in range(int(self.config.epochs)):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            if (not dist.is_initialized()) or dist.get_rank() == 0:
                self.logger.info(
                    f"[Epoch {epoch + 1}/{self.config.epochs}] "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}"
                )

                if self.writer is not None:
                    self.writer.add_scalar("train/loss", train_loss, epoch + 1)
                    self.writer.add_scalar("train/acc", train_acc, epoch + 1)
                    self.writer.add_scalar("val/loss", val_loss, epoch + 1)
                    self.writer.add_scalar("val/acc", val_acc, epoch + 1)

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