import os
import json
import hashlib
from collections import OrderedDict
import numpy as np
from PIL import Image

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
            self.model = nn.parallel.DistributedDataParallel(
                module=classifier,
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
            tokens = tokens.to(self.config.device)
            key_padding_mask = key_padding_mask.to(self.config.device)

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
            tokens = tokens.to(self.config.device)
            key_padding_mask = key_padding_mask.to(self.config.device)

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

                # 使用 self.config.classes 来替换数字标签为真实名称
                labels = list(self.config.classes.keys())  # 获取所有标签名称

                # Create a heatmap for the confusion matrix
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
                plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')

                # Rotate the axis labels to avoid overlap
                plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels by 45 degrees
                plt.yticks(rotation=45, ha='right')  # Rotate y-axis labels by 45 degrees

                # Save the figure as PNG
                cm_path = os.path.join(self.config.result_dir, f"confusion_matrix_epoch_{epoch+1}.png")
                plt.savefig(cm_path, bbox_inches='tight')  # Save the figure as .png
                plt.close()  # Close the plot to avoid display issues

                self.logger.info(f"Confusion matrix saved at {cm_path}")

        return loss_record.avg, acc
        
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

    def _batch_to_tokens(self, inputs_bhw: torch.Tensor):
        """
        inputs_bhw: (B,H,W) torch.Tensor
        return:
          tokens: (B, Smax, F) float32 on self.device
          key_padding_mask: (B, Smax) bool on self.device; True=padding
        """
        B, H, W = inputs_bhw.shape

        feats_list = []
        lengths = []

        with torch.inference_mode():
            for i in range(B):

                spec = inputs_bhw[i].detach().to("cpu").numpy().astype(np.float32)  # (H,W)
                
                # yolo detect
                try:
                    boxes = self.detector.detect(spec)   # List[[x1,y1,x2,y2], ...]
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"[YOLO detect] failed on sample {i}: {e}")
                    boxes = []
                    
                valid_boxes = []
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

                    valid_boxes.append([x1, y1, x2, y2])

                
                feats = self.preprocessor.process(boxes, spec)

                if feats is None or feats.size == 0:
                    feats = np.zeros((1, self.config.feature_dim), dtype=np.float32)

                if feats.shape[0] > self.config.max_tokens:
                    feats = feats[: self.config.max_tokens]

                feats_t = torch.from_numpy(feats).to(dtype=torch.float32).contiguous()  # CPU

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

        return tokens, key_padding_mask
