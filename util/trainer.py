import os
import json
import hashlib
from collections import OrderedDict
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.distributed as dist
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from util.utils import EarlyStopping, AverageMeter
from util.checkpoint import load_checkpoint, save_checkpoint
from util.utils import _reduce_scalar, _set_epoch_for_loaders

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class BBoxCache:
    """
    Simplified bbox cache for YOLO detections.

    Cache format (.pt per sample):
      {
        "source": {"rel":..., "size":..., "mtime":...},
        "boxes": IntTensor [N,4],   # xyxy in original image coordinates
      }

    File naming:
      <mat_basename>.pt
      e.g. FPV1-...-Figure-1.mat -> FPV1-...-Figure-1.pt
    """

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

        self._mem = OrderedDict()   # key -> boxes
        self.hits = 0
        self.misses = 0
        self.disk_writes = 0

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
        """
        Return:
            boxes: CPU IntTensor [N,4]
            or None if cache miss
        """
        if self.mode in ("off", "write", "refresh"):
            return None

        key, _ = self._key_and_meta(fp)

        # memory cache
        if key in self._mem:
            boxes = self._mem.pop(key)
            self._mem[key] = boxes
            self.hits += 1
            return boxes

        path = self._path_for_key(key)
        if not os.path.isfile(path):
            self.misses += 1
            return None

        try:
            obj = torch.load(path, map_location="cpu")
            if not isinstance(obj, dict):
                self.misses += 1
                return None

            boxes = obj.get("boxes", None)
            if boxes is None:
                boxes = torch.zeros((0, 4), dtype=torch.int32)
            elif not isinstance(boxes, torch.Tensor):
                boxes = torch.as_tensor(boxes, dtype=torch.int32)

            boxes = boxes.to(dtype=torch.int32, device="cpu").reshape(-1, 4)

            if self.mem_max > 0:
                self._mem[key] = boxes
                while len(self._mem) > self.mem_max:
                    self._mem.popitem(last=False)

            self.hits += 1
            return boxes

        except Exception as e:
            self.misses += 1
            if self.logger:
                self.logger.warning(f"[BBoxCache] failed to read cache for {os.path.basename(fp)}: {e}")
            return None

    def put(self, fp: str, boxes: torch.Tensor):
        """
        Persist boxes to disk.

        Args:
            boxes: Tensor [N,4] on any device / dtype
        """
        if self.mode in ("off", "read"):
            return False

        key, meta = self._key_and_meta(fp)
        path = self._path_for_key(key)

        # write/readwrite 模式下，如果已存在就不重写
        if self.mode in ("write", "readwrite") and os.path.isfile(path):
            return False

        try:
            if boxes is None:
                boxes_t = torch.zeros((0, 4), dtype=torch.int32)
            else:
                boxes_t = torch.as_tensor(boxes, dtype=torch.int32, device="cpu").reshape(-1, 4)

            obj = {
                "source": meta,
                "boxes": boxes_t,
            }

            tmp = path + f".tmp.{os.getpid()}"
            torch.save(obj, tmp)
            os.replace(tmp, path)

            self.disk_writes += 1

            if self.mem_max > 0:
                self._mem[key] = boxes_t
                while len(self._mem) > self.mem_max:
                    self._mem.popitem(last=False)

            return True

        except Exception as e:
            if self.logger:
                self.logger.warning(f"[BBoxCache] failed to write cache for {os.path.basename(fp)}: {e}")
            return False

    @classmethod
    def from_config(cls, config, logger=None):
        mode = str(getattr(config, "bbox_cache_mode", "off")).lower()
        if mode == "off":
            return None

        dataset_root = getattr(config, "dataset_path", None)
        
        if config.bbox_cache_path is not None and os.path.exists(config.bbox_cache_path):
            base_dir = config.bbox_cache_path
        else:
            base_dir = config.cache_dir

        return cls(
            base_dir=base_dir,
            dataset_root=dataset_root,
            mode=mode,
            mem_max=0,
            logger=logger,
        )

class Trainer:
    def __init__(self, config, data_loaders, logger, detector, preprocessor, classifier):
        
        self.config = config
        self.logger = logger
        self.device = self.config.device
        self.train_loader, self.val_loader = data_loaders
            
        self.classifier = classifier.to(self.device)
        self.detector = detector
        self.preprocessor = preprocessor
        
        if dist.is_initialized():
            if dist.get_rank() == 0:
                self.logger.info(f"Using DistributedDataParallel on {dist.get_world_size()} processes")
            self.classifier = nn.parallel.DistributedDataParallel(
                module=self.classifier,
                device_ids=[self.config.local_rank],
                broadcast_buffers=False,
                find_unused_parameters=False,
                output_device=self.config.local_rank)
        elif torch.cuda.device_count() > 1 and self.config.use_data_parallel:
            self.logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
            self.classifier = nn.DataParallel(self.classifier)
            
        self.criterion = torch.nn.CrossEntropyLoss()
            
        self.optimizer = torch.optim.AdamW(list(self.classifier.parameters()), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=self.config.cosine_annealing_T0, T_mult=self.config.cosine_annealing_mult)
        self.early_stopping = EarlyStopping(logger=self.logger, patience=self.config.early_stop_patience, delta=0)
        
        self.scaler = GradScaler(enabled=(self.config.use_amp_autocast and self.device.type == "cuda"))
        self.writer = SummaryWriter(log_dir=config.result_dir)
        
        self.bbox_cache = BBoxCache.from_config(self.config, logger=self.logger)
        self._bbox_stats = {"hit": 0, "miss": 0, "write": 0}
        
    def train_one_epoch(self, epoch):
        """
        - inputs: (B, 512, 750) 原始矩阵 torch.Tensor
        - labels: (B,) 类别标签 torch.Tensor 
        """

        loss_record = AverageMeter()
        self.classifier.train()

        for batch_idx, (inputs, labels, freq, bw, snr, fps) in enumerate(tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}", leave=True)):
            
            labels = labels.to(self.device, non_blocking=True)

            # 构造 token 序列（检测+预处理）
            tokens, key_padding_mask = self._batch_to_tokens(inputs, sample_fps=fps)
            tokens = tokens.to(self.config.device, non_blocking=True)
            key_padding_mask = key_padding_mask.to(self.config.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.config.use_amp_autocast, device_type=self.device.type):
                try:
                    logits = self.classifier(tokens, key_padding_mask=key_padding_mask)
                except TypeError:
                    try:
                        logits = self.classifier(tokens, key_padding_mask)
                    except TypeError:
                        logits = self.classifier(tokens)

                loss = self.criterion(logits, labels)

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            loss_record.update(loss.item(), labels.size(0))

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            self.logger.info(f"Train Epoch: {epoch + 1}, Avg Loss: {loss_record.avg:.4f}")
            self.writer.add_scalar("Loss/Train", loss_record.avg, epoch)

        return loss_record.avg

    @torch.no_grad()
    def validate(self, epoch):

        loss_record = AverageMeter()
        self.classifier.eval()

        total_correct, total_count = 0, 0
        all_preds, all_targets = [], []

        for batch_idx, (inputs, labels, freq, bw, snr, fps) in enumerate(tqdm(self.val_loader, desc=f"Validating Epoch {epoch + 1}", leave=True)):

            labels = labels.to(self.device, non_blocking=True)

            tokens, key_padding_mask = self._batch_to_tokens(inputs, sample_fps=fps, save_detect_result=True)
            tokens = tokens.to(self.config.device, non_blocking=True)
            key_padding_mask = key_padding_mask.to(self.config.device, non_blocking=True)

            with autocast(enabled=self.config.use_amp_autocast, device_type=self.device.type):
                try:
                    logits = self.classifier(tokens, key_padding_mask=key_padding_mask)
                except TypeError:
                    try:
                        logits = self.classifier(tokens, key_padding_mask)
                    except TypeError:
                        logits = self.classifier(tokens)

                loss = self.criterion(logits, labels)

            loss_record.update(loss.item(), labels.size(0))

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_count += labels.numel()

            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(labels.detach().cpu().numpy())
        
        local_loss_sum = loss_record.sum if hasattr(loss_record, "sum") else (loss_record.avg * max(1, loss_record.count))
        local_loss_count = loss_record.count if hasattr(loss_record, "count") else total_count

        if dist.is_initialized():
            stat_tensor = torch.tensor(
                [total_correct, total_count, float(local_loss_sum), float(local_loss_count)],
                device=self.device,
                dtype=torch.float64
            )
            dist.all_reduce(stat_tensor, op=dist.ReduceOp.SUM)

            total_correct = int(stat_tensor[0].item())
            total_count = int(stat_tensor[1].item())
            global_loss_sum = float(stat_tensor[2].item())
            global_loss_count = float(stat_tensor[3].item())
        else:
            global_loss_sum = float(local_loss_sum)
            global_loss_count = float(local_loss_count)

        acc = total_correct / max(1, total_count)
        global_loss_avg = global_loss_sum / max(1.0, global_loss_count)

        y_pred_local = np.concatenate(all_preds) if len(all_preds) else np.array([], dtype=np.int64)
        y_true_local = np.concatenate(all_targets) if len(all_targets) else np.array([], dtype=np.int64)

        if dist.is_initialized():
            gathered_preds = [None for _ in range(dist.get_world_size())]
            gathered_targets = [None for _ in range(dist.get_world_size())]

            dist.all_gather_object(gathered_preds, y_pred_local)
            dist.all_gather_object(gathered_targets, y_true_local)

            if dist.get_rank() == 0:
                y_pred = np.concatenate(
                    [x for x in gathered_preds if x is not None and len(x) > 0]
                ) if len(gathered_preds) else np.array([], dtype=np.int64)

                y_true = np.concatenate(
                    [x for x in gathered_targets if x is not None and len(x) > 0]
                ) if len(gathered_targets) else np.array([], dtype=np.int64)
            else:
                y_pred = np.array([], dtype=np.int64)
                y_true = np.array([], dtype=np.int64)
        else:
            y_pred = y_pred_local
            y_true = y_true_local

        # 只在 rank0 保存日志和混淆矩阵
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            self.logger.info(f"Validate Epoch: {epoch + 1}, Loss: {global_loss_avg:.4f}, Acc: {acc:.4f}")
            self.writer.add_scalar("Loss/Validation", global_loss_avg, epoch)
            self.writer.add_scalar("Acc/Validation", acc, epoch)

            if y_true.size > 0:
                if hasattr(self.config, "classes") and isinstance(self.config.classes, dict):
                    idx_to_name = {v: k for k, v in self.config.classes.items()}
                    class_ids = sorted(idx_to_name.keys())
                    class_names = [idx_to_name[i] for i in class_ids]
                else:
                    n_cls = int(max(np.max(y_true), np.max(y_pred)) + 1)
                    class_ids = list(range(n_cls))
                    class_names = [str(i) for i in class_ids]

                cm = confusion_matrix(y_true, y_pred, labels=class_ids)

                fig_w = max(8, 0.9 * len(class_names) + 4)
                fig_h = max(6, 0.8 * len(class_names) + 3)

                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                ax.set_title(f"Confusion Matrix - Epoch {epoch + 1}")
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")

                ax.set_xticks(np.arange(len(class_names)))
                ax.set_yticks(np.arange(len(class_names)))
                ax.set_xticklabels(class_names, rotation=45, ha="right")
                ax.set_yticklabels(class_names)

                thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(
                            j, i, str(cm[i, j]),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black",
                            fontsize=9
                        )

                fig.tight_layout()
                
                save_path = os.path.join(self.config.result_dir, 'confusion_matrix')
                os.makedirs(save_path, exist_ok=True)

                cm_png_path = os.path.join(save_path, f"confusion_matrix_epoch_{epoch+1}.png")
                fig.savefig(cm_png_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

                self.logger.info(f"Confusion matrix saved at {cm_png_path}")

        return global_loss_avg, acc
        
    def train(self):
        
        best_path = None
        best_val_loss = np.inf
        for epoch in range(self.config.epochs):
            
            _set_epoch_for_loaders(epoch, self.train_loader, self.val_loader)
            
            train_loss = self.train_one_epoch(epoch)
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                self.logger.info('*' * 25 + f' Train Epoch [{epoch + 1}/{self.config.epochs}]' + '*' * 25)
                self.logger.info(f"Train Loss: {train_loss:.4f}")
            
            val_loss, val_acc = self.validate(epoch)
            global_val_main = _reduce_scalar(torch.tensor(val_loss, device=self.device), self.device)
            self.scheduler.step()
                
            early_stop_flag = torch.zeros(1, device=self.device)
            
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                self.logger.info(f"--Current learning rate: {self.scheduler.get_last_lr()}")
            
                if (epoch + 1) % self.config.save_interval == 0:  
                    checkpoint_path = os.path.join(self.config.model_dir, f'epoch_{epoch + 1}.pth')
                    save_checkpoint({"model": self.classifier}, optimizer=None, scheduler=None, epoch=epoch, path=checkpoint_path, cfg=None, logger=self.logger)
                    self.logger.info(f"--Checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")
                    
                if global_val_main < best_val_loss:
                    best_val_loss = global_val_main 
                    best_path = os.path.join(self.config.model_dir, f'best.pth')
                    save_checkpoint({"model": self.classifier}, optimizer=None, scheduler=None, epoch=epoch, path=best_path, cfg=None, logger=self.logger)
                    self.logger.info(f"--Best model saved at epoch {epoch + 1} with validation loss: {global_val_main:.4f}")
                    
                self.early_stopping(global_val_main, self)
                if self.early_stopping.early_stop:
                    early_stop_flag.fill_(1)
                    
            if dist.is_initialized():
                dist.broadcast(early_stop_flag, src=0)

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            checkpoint_path = os.path.join(self.config.model_dir, f'last.pth')
            save_checkpoint({"model": self.classifier}, optimizer=None, scheduler=None, epoch=epoch, path=checkpoint_path, cfg=None, logger=self.logger)
            
        if dist.is_initialized():
            dist.barrier()
        return best_path if (not dist.is_initialized() or dist.get_rank()==0) else None

    def _batch_to_tokens(self, inputs_bhw: torch.Tensor, sample_fps=None, save_detect_result=False):
        """
        inputs_bhw: (B, H, W) torch.Tensor
        sample_fps: Optional[list[str]]

        return:
        tokens: (B, Smax, F) float32 on CPU
        key_padding_mask: (B, Smax) bool on CPU; True=padding
        """
        B, H, W = inputs_bhw.shape

        feats_list = []
        lengths = []

        with torch.inference_mode():
            for i in range(B):
                fp = None
                if sample_fps is not None:
                    try:
                        fp = sample_fps[i]
                    except Exception:
                        fp = None

                # spec = inputs_bhw[i].detach().cpu().numpy().astype(np.float32)
                # boxes = self._get_boxes_for_sample(spec, fp=fp)

                # if save_detect_result:
                #     try:
                #         final_boxes = self.preprocessor.select_main_boxes(boxes)
                #         if hasattr(final_boxes, "tolist"):
                #             final_boxes = final_boxes.tolist()
                #     except Exception:
                #         final_boxes = []

                #     self._save_detect_result_images(
                #         spec=spec,
                #         yolo_boxes=boxes,
                #         final_boxes=final_boxes,
                #         fp=fp,
                #         sample_idx=i,
                #     )
                
                # feats = self.preprocessor.process(boxes, spec)
                
                spec = inputs_bhw[i].detach().cpu().numpy().astype(np.float32)
                boxes = self._get_boxes_for_sample(spec, fp=fp)

                if save_detect_result:
                    try:
                        final_boxes = self.preprocessor.select_main_boxes(boxes)
                        if hasattr(final_boxes, "tolist"):
                            final_boxes = final_boxes.tolist()
                    except Exception:
                        final_boxes = []

                    self._save_detect_result_images(
                        spec=spec,
                        yolo_boxes=boxes,
                        final_boxes=final_boxes,
                        fp=fp,
                        sample_idx=i,
                    )
                
                feats = self.preprocessor.process(boxes, spec)


                if feats is None or feats.size == 0:
                    feats = np.zeros((1, self.config.feature_dim), dtype=np.float32)

                if feats.shape[0] > self.config.max_tokens:
                    feats = feats[: self.config.max_tokens]

                feats_t = torch.from_numpy(feats).to(dtype=torch.float32).contiguous()

                if feats_t.ndim != 2:
                    feats_t = feats_t.reshape(1, -1)

                feats_list.append(feats_t)
                lengths.append(int(feats_t.shape[0]))

        Smax = int(max(lengths))
        F = int(feats_list[0].shape[1])

        tokens = torch.zeros((B, Smax, F), dtype=torch.float32)
        key_padding_mask = torch.ones((B, Smax), dtype=torch.bool)  # True=padding

        for i, ft in enumerate(feats_list):
            si = int(ft.shape[0])
            tokens[i, :si] = ft
            key_padding_mask[i, :si] = False

        return tokens, key_padding_mask
    
    # @torch.no_grad()
    # def precompute_boxes(self, loader=None):
    #     """
    #     Precompute & cache YOLO bboxes only.
    #     """
    #     if self.bbox_cache is None:
    #         self.logger.info("BBox cache disabled; skip precompute_boxes().")
    #         return

    #     if self.bbox_cache.mode == "read":
    #         self.logger.info("BBox cache is read-only; skip precompute_boxes().")
    #         return

    #     dl = loader if loader is not None else self.train_loader
    #     if dl is None:
    #         raise ValueError("train_loader is None; pass loader to precompute_boxes(loader=...).")

    #     self._bbox_stats = {"hit": 0, "miss": 0, "write": 0}

    #     total = len(dl.dataset) if hasattr(dl, "dataset") else None
    #     pbar = tqdm(total=total, desc="Precomputing YOLO bboxes", unit="img")

    #     for batch in dl:
    #         if not isinstance(batch, (list, tuple)):
    #             raise ValueError("Expected batch to be tuple/list")

    #         if len(batch) == 6:
    #             inputs, labels, freqs, bws, snrs, fps = batch
    #         elif len(batch) == 5:
    #             inputs, labels, freqs, bws, snrs = batch
    #             fps = None
    #         else:
    #             raise ValueError(f"Unexpected batch size: {len(batch)}")

    #         B, H, W = inputs.shape

    #         for i in range(B):
    #             fp = fps[i] if fps is not None else None

    #             # 没有文件路径时无法持久化缓存
    #             if fp is None:
    #                 pbar.update(1)
    #                 continue

    #             # 先查缓存
    #             got = self.bbox_cache.get(fp)
    #             if got is not None:
    #                 self._bbox_stats["hit"] += 1
    #                 pbar.update(1)
    #                 continue

    #             self._bbox_stats["miss"] += 1

    #             # 取单张谱图（保持在 CPU）
    #             spec = inputs[i].detach().cpu().numpy().astype(np.float32)

    #             # YOLO 检测
    #             try:
    #                 boxes = self.detector.detect(spec)   # List[[x1,y1,x2,y2], ...]
    #             except Exception as e:
    #                 if self.logger:
    #                     self.logger.warning(f"[YOLO detect] failed during precompute for {os.path.basename(fp)}: {e}")
    #                 boxes = []

    #             # 合法性过滤
    #             boxes = self._sanitize_boxes(boxes, H, W)

    #             boxes_t = (
    #                 torch.as_tensor(boxes, dtype=torch.int32).reshape(-1, 4)
    #                 if len(boxes) > 0 else
    #                 torch.zeros((0, 4), dtype=torch.int32)
    #             )

    #             wrote = self.bbox_cache.put(fp, boxes_t)
    #             if wrote:
    #                 self._bbox_stats["write"] += 1

    #             pbar.update(1)

    #     pbar.close()

    #     st = self._bbox_stats
    #     self.logger.info(
    #         f"YOLO bbox precompute done. hit={st['hit']} miss={st['miss']} write={st['write']} | mode={self.bbox_cache.mode}"
    #     )

    def _sanitize_boxes(self, boxes, H, W):
        """
        boxes: List[[x1,y1,x2,y2], ...]

        Return:
            sanitized List[[x1,y1,x2,y2], ...]
        """
        out = []

        min_area = int(getattr(self.config, "pre_min_area", 0) or 0)
        max_bbox_area_ratio = float(getattr(self.config, "max_bbox_area_ratio", 0.0) or 0.0)
        img_area = float(H * W)

        for box in boxes:
            if box is None or len(box) != 4:
                continue

            x1, y1, x2, y2 = map(int, box)

            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W - 1))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)

            # if min_area > 0 and area < min_area:
            #     continue

            # if max_bbox_area_ratio > 0 and area > max_bbox_area_ratio * img_area:
            #     continue

            out.append([x1, y1, x2, y2])

        return out
    
    def _get_boxes_for_sample(self, spec: np.ndarray, fp=None):
        """
        Return:
            boxes: List[[x1,y1,x2,y2], ...]
        """
        H, W = spec.shape

        # 1) 尝试读缓存
        if fp is not None and self.bbox_cache is not None:
            got = self.bbox_cache.get(fp)
            if got is not None:
                self._bbox_stats["hit"] += 1
                return got.cpu().numpy().tolist()

        # 2) cache miss -> YOLO detect
        self._bbox_stats["miss"] += 1

        try:
            boxes = self.detector.detect(spec)
        except Exception as e:
            if self.logger:
                name = os.path.basename(fp) if fp is not None else "<memory_sample>"
                self.logger.warning(f"[YOLO detect] failed on {name}: {e}")
            boxes = []

        boxes = self._sanitize_boxes(boxes, H, W)

        # 3) 写缓存
        if fp is not None and self.bbox_cache is not None:
            boxes_t = (
                torch.as_tensor(boxes, dtype=torch.int32).reshape(-1, 4)
                if len(boxes) > 0 else
                torch.zeros((0, 4), dtype=torch.int32)
            )
            wrote = self.bbox_cache.put(fp, boxes_t)
            if wrote:
                self._bbox_stats["write"] += 1

        return boxes
    
    def _draw_boxes_on_rgb(self, rgb_u8, boxes, color=(255, 0, 0), width=2):
        """
        rgb_u8: np.ndarray, shape (H, W, 3), dtype uint8
        boxes: List[[x1,y1,x2,y2], ...] or np.ndarray [N,4]
        """
        if boxes is None:
            boxes = []

        if hasattr(boxes, "tolist"):
            boxes = boxes.tolist()

        img = Image.fromarray(rgb_u8.copy())
        draw = ImageDraw.Draw(img)

        for box in boxes:
            if box is None or len(box) != 4:
                continue
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        return np.array(img)


    def _save_detect_result_images(self, spec, yolo_boxes, final_boxes, fp=None, sample_idx=None):
        """
        Save two images:
        - YOLO raw detection boxes
        - final selected boxes

        Saved to:
        <result_dir>/detect_result/<basename>_yolo.png
        <result_dir>/detect_result/<basename>_final.png
        """
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        save_dir = os.path.join(self.config.result_dir, "detect_result")
        os.makedirs(save_dir, exist_ok=True)

        # base filename
        if fp is not None:
            base = os.path.splitext(os.path.basename(fp))[0]
        else:
            base = f"sample_{sample_idx if sample_idx is not None else 0}"

        gray_u8 = self._to_uint8_gray(spec)
        rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)

        yolo_img = self._draw_boxes_on_rgb(rgb, yolo_boxes, color=(255, 0, 0), width=2)
        final_img = self._draw_boxes_on_rgb(rgb, final_boxes, color=(0, 255, 0), width=2)

        yolo_path = os.path.join(save_dir, f"{base}_yolo.png")
        final_path = os.path.join(save_dir, f"{base}_final.png")

        Image.fromarray(yolo_img).save(yolo_path)
        Image.fromarray(final_img).save(final_path)
        
    def _to_uint8_gray(self, spec: np.ndarray) -> np.ndarray:
        """
        spec: (H, W), float/uint16/uint8
        return: uint8 grayscale image in [0, 255]
        """
        if spec.ndim != 2:
            raise ValueError(f"Expected 2D grayscale spectrogram, got shape={spec.shape}")

        if spec.dtype == np.uint8:
            return spec

        x = spec.astype(np.float32, copy=False)

        # 用分位数做稳健拉伸，避免极端值影响显示
        lo = np.percentile(x, 1.0)
        hi = np.percentile(x, 99.0)

        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
            lo = float(np.min(x))
            hi = float(np.max(x))
            if hi <= lo:
                return np.zeros_like(x, dtype=np.uint8)

        x = np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)
        return (x * 255.0).astype(np.uint8)