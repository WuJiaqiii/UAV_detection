import os
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
import cv2

from util.utils import EarlyStopping, AverageMeter
from util.checkpoint import save_checkpoint
from util.utils import _reduce_scalar, _set_epoch_for_loaders
from util.boxmask import boxes_to_white_mask, mask_to_tensor


def spec_to_uint8_vis(spec, p_low=1.0, p_high=99.5, gamma=0.6):
    """
    仅用于可视化保存，不建议直接用于训练输入。
    spec: 2D numpy array
    """
    x = np.asarray(spec, dtype=np.float32)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)

    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)

    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    x = np.power(x, gamma)  # gamma < 1 会提亮暗部
    return (x * 255.0).clip(0, 255).astype(np.uint8)


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
    """
    单信号分支 Trainer
    新增功能：
      - 支持一条额外验证路线 extra_val_loader
      - 每个 epoch 结束后额外评估一次，并单独记录 extra_val loss/acc
    """
    def __init__(self, config, data_loaders, logger, detector, preprocessor, classifier, extra_val_loader=None):
        self.config = config
        self.logger = logger
        self.detector = detector
        self.preprocessor = preprocessor
        self.classifier = classifier

        self.train_loader, self.val_loader = data_loaders
        self.extra_val_loader = extra_val_loader

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

        # 额外验证集可视化仅首次保存一次
        self.extra_save_detect_vis_once = bool(getattr(config, "extra_save_detect_vis_once", False))
        self.extra_detect_vis_num_samples = int(getattr(config, "extra_detect_vis_num_samples", 1))
        self._extra_detect_vis_saved = False

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
        self.cnn_input_mode = str(getattr(config, "cnn_input_mode", "mask")).lower()
        self.box_draw_thickness = int(getattr(config, "box_draw_thickness", 2))
        self.box_draw_value = int(getattr(config, "box_draw_value", 255))

        self.writer = None
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            self.writer = SummaryWriter(log_dir=config.log_dir)

    # ------------------------- image helpers -------------------------
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

    def _spec_to_uint8(self, spec: np.ndarray) -> np.ndarray:
        spec = np.asarray(spec)
        if spec.ndim != 2:
            raise ValueError(f"Expected 2D spectrogram, got shape={spec.shape}")
        if spec.dtype == np.uint8:
            return spec.copy()

        s = spec.astype(np.float32)
        s_min, s_max = float(s.min()), float(s.max())
        if s_max > s_min:
            s = (s - s_min) / (s_max - s_min)
        else:
            s = np.zeros_like(s, dtype=np.float32)
        return (s * 255.0).astype(np.uint8)

    def _gray_image_to_tensor(self, img_u8: np.ndarray) -> torch.Tensor:
        if img_u8.ndim != 2:
            raise ValueError(f"Expected grayscale image, got shape={img_u8.shape}")

        x = img_u8.astype(np.float32) / 255.0
        x = cv2.resize(
            x,
            (self.mask_img_size, self.mask_img_size),
            interpolation=cv2.INTER_NEAREST if self.cnn_input_mode == "mask" else cv2.INTER_LINEAR,
        )
        x = np.expand_dims(x, axis=0)  # [1, H, W]
        return torch.from_numpy(x).float()

    def _draw_final_boxes_on_raw_gray(self, spec: np.ndarray, final_boxes=None) -> np.ndarray:
        img = self._spec_to_uint8(spec)
        H, W = img.shape[:2]

        if final_boxes is None or len(final_boxes) == 0:
            return img

        arr = np.asarray(final_boxes, dtype=np.int32).reshape(-1, 4)
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
        return img

    def _build_input_image(self, spec, yolo_boxes, final_boxes):
        mode = str(self.cnn_input_mode).lower()

        if mode == "mask":
            boxes = final_boxes if self.mask_source == "final" else yolo_boxes
            mask = boxes_to_white_mask(
                image_shape=spec.shape,
                boxes=boxes,
                fill_value=self.mask_fill_value,
                mode="fill",
            )
            return self._gray_image_to_tensor(mask)

        elif mode == "raw":
            raw_img = self._spec_to_uint8(spec)
            return self._gray_image_to_tensor(raw_img)

        elif mode == "raw_with_boxes":
            img = self._draw_final_boxes_on_raw_gray(spec, final_boxes=final_boxes)
            return self._gray_image_to_tensor(img)

        elif mode == "raw_in_boxes":
            raw_img = self._spec_to_uint8(spec)
            boxes = final_boxes if self.mask_source == "final" else yolo_boxes
            mask = boxes_to_white_mask(
                image_shape=spec.shape,
                boxes=boxes,
                fill_value=255,
                mode="fill",
            )
            out = np.zeros_like(raw_img, dtype=np.uint8)
            keep = mask > 0
            out[keep] = raw_img[keep]
            return self._gray_image_to_tensor(out)

        else:
            raise ValueError(f"Unsupported cnn_input_mode={mode}")

    def _save_detect_result_images(self, spec, yolo_boxes, final_boxes, fp, sample_idx=0, split_name="val"):
        save_dir = os.path.join(self.config.result_dir, "detect_result", split_name)
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(fp))[0] if fp else f"sample_{sample_idx}"

        spec_u8 = spec_to_uint8_vis(spec, p_low=1.0, p_high=99.5, gamma=0.6)

        from PIL import Image, ImageDraw
        for name, boxes in [("yolo", yolo_boxes), ("final", final_boxes)]:
            img = Image.fromarray(spec_u8).convert("RGB")
            draw = ImageDraw.Draw(img)
            color = (255, 0, 0) if name == "yolo" else (0, 255, 0)
            for b in boxes:
                x1, y1, x2, y2 = [int(v) for v in b]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            img.save(os.path.join(save_dir, f"{base}_{name}.png"))

    def _batch_to_input_images(self, inputs_bhw, sample_fps=None, save_detect_result=False, max_save_images=1, split_name="val"):
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
                final_boxes = yolo_boxes
            else:
                final_boxes = self.preprocessor.select_main_boxes(yolo_boxes, spectrogram=spec)
                if hasattr(final_boxes, "tolist"):
                    final_boxes = final_boxes.tolist()

            x = self._build_input_image(spec, yolo_boxes, final_boxes)
            out.append(x)

            if save_detect_result and saved_count < max_save_images:
                self._save_detect_result_images(
                    spec=spec,
                    yolo_boxes=yolo_boxes,
                    final_boxes=final_boxes,
                    fp=fp,
                    sample_idx=i,
                    split_name=split_name,
                )
                saved_count += 1

        return torch.stack(out, dim=0)  # [B,1,H,W]

    # ------------------------- loops -------------------------
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

            images = self._batch_to_input_images(inputs, sample_fps=fps, save_detect_result=False, split_name="train")
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
    def evaluate_loader(self, loader, epoch, split_name="val", save_vis_once=False, vis_num_samples=1):
        self.classifier.eval()

        loss_meter = AverageMeter()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        if split_name == "extra_val":
            already_saved = self._extra_detect_vis_saved
        else:
            already_saved = self._detect_vis_saved

        pbar = tqdm(loader, desc=f"Evaluating {split_name} Epoch {epoch + 1}", leave=True)
        for batch_idx, batch in enumerate(pbar):
            inputs, labels, freq, bw, snr, fps = batch
            labels = labels.to(self.device, non_blocking=True)

            need_save_detect_result = save_vis_once and (not already_saved) and batch_idx == 0

            images = self._batch_to_input_images(
                inputs,
                sample_fps=fps,
                save_detect_result=need_save_detect_result,
                max_save_images=vis_num_samples,
                split_name=split_name,
            )
            images = images.to(self.device, non_blocking=True)

            if need_save_detect_result:
                already_saved = True
                if split_name == "extra_val":
                    self._extra_detect_vis_saved = True
                else:
                    self._detect_vis_saved = True

            logits, loss = self._forward_loss(images, labels)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

            loss_meter.update(loss.item(), labels.size(0))
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(labels.detach().cpu().numpy())

            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{100.0 * correct / max(total,1):.2f}%")

        eval_loss = loss_meter.avg
        eval_acc = 100.0 * correct / max(total, 1)

        if len(all_preds) > 0:
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
        else:
            all_preds = np.array([], dtype=np.int64)
            all_targets = np.array([], dtype=np.int64)

        if dist.is_initialized():
            eval_loss = _reduce_scalar(torch.tensor(eval_loss, device=self.device), op="mean").item()
            eval_acc = _reduce_scalar(torch.tensor(eval_acc, device=self.device), op="mean").item()

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            self._save_confusion_matrix(all_targets, all_preds, epoch, split_name=split_name)

        return eval_loss, eval_acc

    @torch.no_grad()
    def validate(self, epoch):
        return self.evaluate_loader(
            self.val_loader,
            epoch=epoch,
            split_name="val",
            save_vis_once=self.save_detect_vis_once,
            vis_num_samples=self.detect_vis_num_samples,
        )

    @torch.no_grad()
    def validate_extra(self, epoch):
        if self.extra_val_loader is None:
            return None, None
        return self.evaluate_loader(
            self.extra_val_loader,
            epoch=epoch,
            split_name="extra_val",
            save_vis_once=self.extra_save_detect_vis_once,
            vis_num_samples=self.extra_detect_vis_num_samples,
        )

    def _save_confusion_matrix(self, y_true, y_pred, epoch, split_name="val"):
        if len(y_true) == 0:
            return

        labels = list(range(self.config.num_classes))
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        if isinstance(self.config.classes, dict):
            idx_to_name = {v: k for k, v in self.config.classes.items()}
            names = [idx_to_name.get(i, str(i)) for i in labels]
        else:
            names = [str(i) for i in labels]

        save_dir = os.path.join(self.config.result_dir, "confusion_matrix", split_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"epoch_{epoch + 1}.png")

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - {split_name} - Epoch {epoch + 1}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    def train(self):
        best_acc = -1.0

        for epoch in range(int(self.config.epochs)):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            extra_val_loss, extra_val_acc = None, None
            if self.extra_val_loader is not None:
                extra_val_loss, extra_val_acc = self.validate_extra(epoch)

            if (not dist.is_initialized()) or dist.get_rank() == 0:
                msg = (
                    f"[Epoch {epoch + 1}/{self.config.epochs}] "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}"
                )
                if extra_val_acc is not None:
                    msg += f", extra_val_loss={extra_val_loss:.4f}, extra_val_acc={extra_val_acc:.2f}"
                self.logger.info(msg)

                if self.writer is not None:
                    self.writer.add_scalar("train/loss", train_loss, epoch + 1)
                    self.writer.add_scalar("train/acc", train_acc, epoch + 1)
                    self.writer.add_scalar("val/loss", val_loss, epoch + 1)
                    self.writer.add_scalar("val/acc", val_acc, epoch + 1)
                    if extra_val_acc is not None:
                        self.writer.add_scalar("extra_val/loss", extra_val_loss, epoch + 1)
                        self.writer.add_scalar("extra_val/acc", extra_val_acc, epoch + 1)

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
