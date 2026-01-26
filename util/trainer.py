import os
import json
import hashlib
from collections import OrderedDict
import numpy as np

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
import util.detection as detlib

import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import confusion_matrix


class TokenCache:
    """Cache for per-sample token features produced by (SAM2 detection + preprocessing).

    - In-memory: small LRU to avoid repeated disk reads within an epoch.
    - On-disk: stores each sample's variable-length token tensor as a .pt file.
    - Safe for multi-process (DDP): writes use atomic rename (tmp -> final).
    """

    def __init__(self, base_dir: str, signature: dict, dataset_root: str | None,
                 mode: str = "readwrite", mem_max: int = 0, logger=None):
        self.mode = str(mode).lower()
        self.mem_max = int(mem_max) if mem_max else 0
        self.logger = logger
        self.dataset_root = dataset_root

        self.sig_json = json.dumps(signature, sort_keys=True, ensure_ascii=False)
        self.sig_hash = hashlib.sha1(self.sig_json.encode("utf-8")).hexdigest()[:12]

        self.cache_dir = os.path.join(base_dir, self.sig_hash)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Write signature for reproducibility (rank0 only)
        try:
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                sig_path = os.path.join(self.cache_dir, "signature.json")
                if not os.path.isfile(sig_path):
                    with open(sig_path, "w", encoding="utf-8") as f:
                        f.write(self.sig_json)
        except Exception:
            pass

        self._mem = OrderedDict()  # key -> torch.Tensor (CPU)
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
        key = hashlib.sha1(rel.encode("utf-8")).hexdigest()
        try:
            st = os.stat(fp)
            meta = {"rel": rel, "size": int(st.st_size), "mtime": int(st.st_mtime)}
        except Exception:
            meta = {"rel": rel, "size": None, "mtime": None}
        return key, meta

    def _path_for_key(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.pt")

    def get(self, fp: str):
        """Return cached feats tensor on CPU, or None."""
        if self.mode in ("off", "write"):
            return None
        if self.mode == "refresh":
            return None

        key, meta = self._key_and_meta(fp)

        # memory LRU
        if key in self._mem:
            t = self._mem.pop(key)
            self._mem[key] = t
            self.hits += 1
            return t

        path = self._path_for_key(key)
        if not os.path.isfile(path):
            return None

        try:
            obj = torch.load(path, map_location="cpu")
            if not isinstance(obj, dict) or obj.get("sig_hash") != self.sig_hash:
                return None
            src = obj.get("source", {})
            # if we can stat, verify size/mtime to avoid stale cache
            if meta["size"] is not None and src.get("size") is not None and int(src.get("size")) != int(meta["size"]):
                return None
            if meta["mtime"] is not None and src.get("mtime") is not None and int(src.get("mtime")) != int(meta["mtime"]):
                return None

            t = obj.get("feats", None)
            if not isinstance(t, torch.Tensor):
                return None
            t = t.to(dtype=torch.float32, copy=False).contiguous()

            if self.mem_max > 0:
                self._mem[key] = t
                while len(self._mem) > self.mem_max:
                    self._mem.popitem(last=False)

            self.hits += 1
            return t
        except Exception:
            return None

    def put(self, fp: str, feats_cpu: torch.Tensor):
        """Persist feats tensor (CPU) to disk cache (best-effort)."""
        if self.mode in ("off", "read"):
            return
        key, meta = self._key_and_meta(fp)
        path = self._path_for_key(key)

        # in refresh mode we always overwrite; otherwise avoid rewriting if exists
        if self.mode != "refresh" and os.path.isfile(path):
            if self.mem_max > 0 and key not in self._mem:
                self._mem[key] = feats_cpu
                while len(self._mem) > self.mem_max:
                    self._mem.popitem(last=False)
            return

        obj = {
            "sig_hash": self.sig_hash,
            "feats": feats_cpu.to(dtype=torch.float32, copy=False).contiguous(),
            "source": meta,
        }

        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)  # atomic on POSIX
            self.disk_writes += 1
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass

        if self.mem_max > 0:
            self._mem[key] = obj["feats"]
            while len(self._mem) > self.mem_max:
                self._mem.popitem(last=False)

    @classmethod
    def from_config(cls, config, logger=None):
        mode = str(getattr(config, "token_cache_mode", "off")).lower()
        if mode == "off":
            return None

        base_dir = getattr(config, "token_cache_dir", None)
        if not base_dir:
            base_dir = os.path.join(getattr(config, "dataset_path", "."), ".token_cache")

        mem_max = int(getattr(config, "token_cache_mem", 0) or 0)

        # Anything that changes tokenization should be part of the signature.
        signature = {
            "version": 1,
            "max_tokens": int(getattr(config, "max_tokens", 0) or 0),
            "feature_dim": int(getattr(config, "feature_dim", 0) or 0),
            "mask_filter": {
                "min_rectangularity": float(getattr(config, "min_rectangularity", 0.0) or 0.0),
                "max_bbox_area_ratio": float(getattr(config, "max_bbox_area_ratio", 0.0) or 0.0),
            },
            "sam2": {
                "model_cfg": str(getattr(config, "model_cfg", "")),
                "checkpoint": os.path.basename(str(getattr(config, "sam2_checkpoint", ""))),
                "points_per_side": int(getattr(config, "sam2_points_per_side", 0) or 0),
                "points_per_batch": int(getattr(config, "sam2_points_per_batch", 0) or 0),
                "crop_n_layers": int(getattr(config, "sam2_crop_n_layers", 0) or 0),
                "pred_iou_thresh": float(getattr(config, "sam2_pred_iou_thresh", 0.0) or 0.0),
                "stability_score_thresh": float(getattr(config, "sam2_stability_score_thresh", 0.0) or 0.0),
                "box_nms_thresh": float(getattr(config, "sam2_box_nms_thresh", 0.0) or 0.0),
                "crop_nms_thresh": float(getattr(config, "sam2_crop_nms_thresh", 0.0) or 0.0),
                "min_mask_region_area": int(getattr(config, "sam2_min_mask_region_area", 0) or 0),
            },
            "preprocess": {
                "sampling_rate": float(getattr(config, "sampling_rate", 0.0) or 0.0),
                "n_fft": int(getattr(config, "n_fft", 0) or 0),
                "hop_length": int(getattr(config, "hop_length", 0) or 0),
                "min_area": int(getattr(config, "pre_min_area", 0) or 0),
                "min_ratio": float(getattr(config, "pre_min_ratio", 0.0) or 0.0),
                "freq_eps": float(getattr(config, "pre_freq_eps", 0.0) or 0.0),
                "freq_min_samples": int(getattr(config, "pre_freq_min_samples", 0) or 0),
                "nms_iou_thresh": float(getattr(config, "pre_nms_iou_thresh", 0.0) or 0.0),
            },
        }

        return cls(
            base_dir=str(base_dir),
            signature=signature,
            dataset_root=getattr(config, "dataset_path", None),
            mode=mode,
            mem_max=mem_max,
            logger=logger,
        )


class Trainer:
    def __init__(self, config, data_loaders, logger, model, mask_generator, preprocessor):
        
        self.config = config
        self.logger = logger
        self.device = self.config.device
        self.train_loader, self.val_loader = data_loaders
            
        self.model = model.to(self.device)
        self.mask_generator = mask_generator
        self.preprocessor = preprocessor
        
        if dist.is_initialized():
            if dist.get_rank() == 0:
                self.logger.info(f"Using DistributedDataParallel on {dist.get_world_size()} processes")
            self.model = nn.parallel.DistributedDataParallel(
                module=model,
                device_ids=[self.config.local_rank],
                broadcast_buffers=False,
                find_unused_parameters=False,
                output_device=self.config.local_rank)
        elif torch.cuda.device_count() > 1 and self.config.use_data_parallel:
            self.logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
            
        self.criterion = torch.nn.CrossEntropyLoss()
            
        self.optimizer = torch.optim.AdamW(list(self.model.parameters()), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=self.config.cosine_annealing_T0, T_mult=self.config.cosine_annealing_mult)
        self.early_stopping = EarlyStopping(logger=self.logger, patience=self.config.early_stop_patience, delta=0)
        
        self.scaler = GradScaler(enabled=(self.config.use_amp_autocast and self.device.type == "cuda"))
        self.writer = SummaryWriter(log_dir=config.result_dir)
        # Token cache for (SAM2 detection + preprocessing) outputs
        self.token_cache = TokenCache.from_config(self.config, logger=self.logger)
        self._cache_stats = {'hit': 0, 'miss': 0, 'write': 0}

        
    def train_one_epoch(self, epoch):
        """
        - inputs: (B, 512, 750) 原始矩阵 torch.Tensor
        - labels: (B,) 类别标签 torch.Tensor 
        """

        loss_record = AverageMeter()
        self.model.train()

        for batch_idx, (inputs, labels, freq, bw, snr, fps) in enumerate(tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}", leave=True)):
            
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # 构造 token 序列（检测+预处理）
            tokens, key_padding_mask = self._batch_to_tokens(inputs, sample_fps=fps)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.config.use_amp_autocast, device_type=self.device.type):
                try:
                    logits = self.model(tokens, key_padding_mask=key_padding_mask)
                except TypeError:
                    try:
                        logits = self.model(tokens, key_padding_mask)
                    except TypeError:
                        logits = self.model(tokens)

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
        self.model.eval()

        total_correct, total_count = 0, 0
        all_preds, all_targets = [], []

        for batch_idx, (inputs, labels, freq, bw, snr, fps) in enumerate(tqdm(self.val_loader, desc=f"Validating Epoch {epoch + 1}", leave=True)):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            tokens, key_padding_mask = self._batch_to_tokens(inputs, sample_fps=fps)

            with autocast(enabled=self.config.use_amp_autocast, device_type=self.device.type):
                try:
                    logits = self.model(tokens, key_padding_mask=key_padding_mask)
                except TypeError:
                    try:
                        logits = self.model(tokens, key_padding_mask)
                    except TypeError:
                        logits = self.model(tokens)

                loss = self.criterion(logits, labels)

            loss_record.update(loss.item(), labels.size(0))

            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_count += labels.numel()

            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(labels.detach().cpu().numpy())

        # DDP 下做全局 acc 汇总
        if dist.is_initialized():
            c = torch.tensor([total_correct, total_count], device=self.device, dtype=torch.long)
            dist.all_reduce(c, op=dist.ReduceOp.SUM)
            total_correct = int(c[0].item())
            total_count = int(c[1].item())

        acc = (total_correct / max(1, total_count))

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            self.logger.info(f"Validate Epoch: {epoch + 1}, Loss: {loss_record.avg:.4f}, Acc: {acc:.4f}")
            self.writer.add_scalar("Loss/Validation", loss_record.avg, epoch)
            self.writer.add_scalar("Acc/Validation", acc, epoch)

            y_pred = np.concatenate(all_preds) if len(all_preds) else np.array([], dtype=np.int64)
            y_true = np.concatenate(all_targets) if len(all_targets) else np.array([], dtype=np.int64)
            if y_true.size > 0:
                cm = confusion_matrix(y_true, y_pred)
                cm_path = os.path.join(self.config.result_dir, f"confusion_matrix_epoch_{epoch+1}.npy")
                np.save(cm_path, cm)

        return loss_record.avg, acc

    def precompute_tokens(self, loader=None):
        """Precompute and persist token cache for a loader (train by default)."""
        if self.token_cache is None:
            self.logger.warning("[TokenCache] token_cache_mode=off; nothing to precompute.")
            return

        self.model.eval()
        dl = loader if loader is not None else self.train_loader
        self._cache_stats = {'hit': 0, 'miss': 0, 'write': 0}

        with torch.inference_mode():
            for batch in tqdm(dl, desc="Precomputing tokens", leave=True):
                if len(batch) == 5:
                    inputs, labels, freq, bw, snr = batch
                    fps = None
                else:
                    inputs, labels, freq, bw, snr, fps = batch

                inputs = inputs.to(self.device, non_blocking=True)
                _ = self._batch_to_tokens(inputs, sample_fps=fps)

        # DDP: aggregate stats
        if dist.is_initialized():
            t = torch.tensor([self._cache_stats['hit'], self._cache_stats['miss'], self._cache_stats['write']],
                             device=self.device, dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            if dist.get_rank() == 0:
                self.logger.info(f"[TokenCache] precompute done: hit={int(t[0])} miss={int(t[1])} write={int(t[2])}")
        else:
            self.logger.info(f"[TokenCache] precompute done: hit={self._cache_stats['hit']} miss={self._cache_stats['miss']} write={self._cache_stats['write']}")



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
                    save_checkpoint({"model": self.model}, optimizer=None, scheduler=None, epoch=epoch, path=checkpoint_path, cfg=None, logger=self.logger)
                    self.logger.info(f"--Checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")
                    
                if global_val_main < best_val_loss:
                    best_val_loss = global_val_main 
                    best_path = os.path.join(self.config.model_dir, f'best.pth')
                    save_checkpoint({"model": self.model}, optimizer=None, scheduler=None, epoch=epoch, path=best_path, cfg=None, logger=self.logger)
                    self.logger.info(f"--Best model saved at epoch {epoch + 1} with validation loss: {global_val_main:.4f}")
                    
                self.early_stopping(global_val_main, self)
                if self.early_stopping.early_stop:
                    early_stop_flag.fill_(1)
                    
            if dist.is_initialized():
                dist.broadcast(early_stop_flag, src=0)

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            checkpoint_path = os.path.join(self.config.model_dir, f'last.pth')
            save_checkpoint({"model": self.model}, optimizer=None, scheduler=None, epoch=epoch, path=checkpoint_path, cfg=None, logger=self.logger)
            
        if dist.is_initialized():
            dist.barrier()
        return best_path if (not dist.is_initialized() or dist.get_rank()==0) else None

    def _batch_to_tokens(self, inputs_bhw: torch.Tensor, sample_fps=None):
        """
        inputs_bhw: (B,H,W) torch.Tensor (on GPU/CPU)
        sample_fps: Optional[list[str]]; if provided, enables persistent token caching.
        return:
          tokens: (B, Smax, F) float32 on self.device
          key_padding_mask: (B, Smax) bool on self.device; True=padding
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

                feats_t = None
                if fp is not None and self.token_cache is not None:
                    feats_t = self.token_cache.get(fp)
                    if feats_t is not None:
                        self._cache_stats['hit'] += 1

                if feats_t is None:
                    # Cache miss -> run SAM2 + preprocessing
                    self._cache_stats['miss'] += 1

                    spec = inputs_bhw[i].detach().to("cpu").numpy()  # (H,W)
                    gray_u8 = detlib.to_uint8_grayscale(spec)
                    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)

                    masks = self.mask_generator.generate(rgb)

                    masks = detlib.filter_masks_rect_and_area(
                        masks, H, W,
                        min_rectangularity=self.config.min_rectangularity,
                        max_bbox_area_ratio=self.config.max_bbox_area_ratio,
                    )
                    boxes = [m["_bbox_xyxy"] for m in masks if "_bbox_xyxy" in m]

                    feats = self.preprocessor.process(boxes, spec.astype(np.float32))

                    # 没检测到就塞一个全 0 token，避免序列长度为 0
                    if feats is None or feats.size == 0:
                        feats = np.zeros((1, self.config.feature_dim), dtype=np.float32)

                    # 限制最大 token 数
                    if feats.shape[0] > self.config.max_tokens:
                        feats = feats[: self.config.max_tokens]

                    feats_t = torch.from_numpy(feats).to(dtype=torch.float32).contiguous()  # CPU

                    # Persist
                    if fp is not None and self.token_cache is not None:
                        before = self.token_cache.disk_writes
                        self.token_cache.put(fp, feats_t)
                        if self.token_cache.disk_writes > before:
                            self._cache_stats['write'] += 1

                # Ensure correct dtype/shape
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

        return tokens.to(self.device, non_blocking=True), key_padding_mask.to(self.device, non_blocking=True)
