import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from util.trainer import Trainer
from util.utils import _make_pbar, AverageMeter
from model.resnet import MaskImageClassifier


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if not str(first_key).startswith("module."):
        return state_dict
    return {str(k).replace("module.", "", 1): v for k, v in state_dict.items()}


def load_state_skip_mismatch(model, checkpoint_path, device, logger=None):
    """
    Load checkpoint weights into model, skipping keys with shape mismatch.
    This is needed when old_num_classes != new_num_classes.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" not in ckpt:
        raise RuntimeError(f"No model_state_dict found in checkpoint: {checkpoint_path}")

    state_dict = _strip_module_prefix(ckpt["model_state_dict"])
    model_state = model.state_dict()

    filtered = {}
    skipped = []

    for k, v in state_dict.items():
        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape):
            filtered[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    if logger is not None:
        logger.info(f"[LwF] loaded compatible student weights from: {checkpoint_path}")
        if skipped:
            logger.warning(f"[LwF] skipped mismatch keys: {skipped[:20]}")
        if missing:
            logger.warning(f"[LwF] missing keys: {missing[:20]}")
        if unexpected:
            logger.warning(f"[LwF] unexpected keys: {unexpected[:20]}")

    return ckpt


class LwFTrainer(Trainer):
    """
    Learning without Forgetting trainer.

    Assumptions:
        1. Old class ids are [0, old_num_classes - 1].
        2. New classes are appended after old classes.
        3. Teacher model uses old_num_classes.
        4. Student model uses current config.num_classes.
    """

    def __init__(self, config, dataloaders, logger, detector, preprocessor, classifier, bbox_cache=None):
        # Prevent base Trainer from loading checkpoint directly,
        # because old/new classifier heads may have different shapes.
        original_checkpoint_path = config.checkpoint_path
        was_frozen = getattr(config, "_frozen", False)
        if was_frozen:
            config.unfreeze()
        config.checkpoint_path = None

        super().__init__(
            config=config,
            dataloaders=dataloaders,
            logger=logger,
            detector=detector,
            preprocessor=preprocessor,
            classifier=classifier,
            bbox_cache=bbox_cache,
        )

        config.checkpoint_path = original_checkpoint_path
        if was_frozen:
            config.freeze()
        self.checkpoint_path = original_checkpoint_path

        self.old_num_classes = int(config.lwf_old_num_classes)
        self.temperature = float(config.lwf_temperature)
        self.lambda_kd = float(config.lwf_lambda_kd)

        if self.old_num_classes <= 0:
            raise ValueError("--lwf_old_num_classes must be > 0")

        if self.old_num_classes >= int(config.num_classes):
            raise ValueError(
                f"LwF expects new classes to be appended. "
                f"old_num_classes={self.old_num_classes}, "
                f"num_classes={config.num_classes}"
            )

        teacher_ckpt = str(config.lwf_teacher_checkpoint or original_checkpoint_path or "").strip()
        if not teacher_ckpt:
            raise ValueError("LwF requires --lwf_teacher_checkpoint or --checkpoint_path")

        # Student: load old compatible weights, skip old classifier head mismatch.
        load_state_skip_mismatch(
            model=self.classifier,
            checkpoint_path=teacher_ckpt,
            device=self.device,
            logger=self.logger,
        )

        # Teacher: old model with old_num_classes.
        self.teacher = MaskImageClassifier(
            backbone=config.backbone,
            num_classes=self.old_num_classes,
            in_chans=config.mask_in_chans,
            pretrained=False,
            dropout=config.cnn_dropout,
            freeze_backbone=False,
        ).to(self.device)

        load_state_skip_mismatch(
            model=self.teacher,
            checkpoint_path=teacher_ckpt,
            device=self.device,
            logger=self.logger,
        )

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        if getattr(config, "lwf_freeze_backbone", False):
            self.logger.info("[LwF] freeze student backbone, train classifier head only.")
            self.classifier.freeze_backbone()

        # Rebuild optimizer after optional freezing.
        self.optimizer = AdamW(
            [p for p in self.classifier.parameters() if p.requires_grad],
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
        )

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=int(getattr(config, "cosine_annealing_T0", 50)),
            T_mult=int(getattr(config, "cosine_annealing_mult", 2)),
        )

        self.logger.info(
            f"[LwF] enabled: old_num_classes={self.old_num_classes}, "
            f"new_num_classes={config.num_classes}, "
            f"T={self.temperature}, lambda_kd={self.lambda_kd}"
        )

    def _kd_loss(self, student_logits, teacher_logits):
        """
        KD only on old-class logits.
        """
        T = self.temperature

        student_old = student_logits[:, :self.old_num_classes]
        teacher_old = teacher_logits[:, :self.old_num_classes]

        loss_kd = F.kl_div(
            F.log_softmax(student_old / T, dim=1),
            F.softmax(teacher_old / T, dim=1),
            reduction="batchmean",
        ) * (T * T)

        return loss_kd

    def train_one_epoch(self, epoch):
        self.classifier.train()
        self.teacher.eval()

        loss_meter = AverageMeter()
        ce_meter = AverageMeter()
        kd_meter = AverageMeter()

        correct = 0
        total = 0
        matched_total = 0
        target_total = 0

        pbar = _make_pbar(
            self.train_loader,
            desc=f"LwF Train Epoch {epoch + 1}",
            leave=True,
        )

        for batch in pbar:
            inputs, targets_list, snrs, fps = batch

            images, labels, metas, batch_matched, batch_targets, _ = self._build_single_instances(
                inputs=inputs,
                targets_list=targets_list,
                sample_fps=fps,
                save_detect_result=False,
                max_save_images=0,
                split_name="train",
            )

            matched_total += batch_matched
            target_total += batch_targets

            if images is None or labels is None or labels.numel() == 0:
                continue

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            student_logits = self.classifier(images)

            with torch.no_grad():
                teacher_logits = self.teacher(images)

            loss_ce = self.criterion(student_logits, labels)
            loss_kd = self._kd_loss(student_logits, teacher_logits)
            loss = loss_ce + self.lambda_kd * loss_kd

            loss.backward()
            self.optimizer.step()

            preds = student_logits.argmax(dim=1)

            correct += int((preds == labels).sum().item())
            total += int(labels.numel())

            loss_meter.update(loss.item(), labels.size(0))
            ce_meter.update(loss_ce.item(), labels.size(0))
            kd_meter.update(loss_kd.item(), labels.size(0))

            acc = 100.0 * correct / max(total, 1)
            match_recall = float(matched_total) / max(float(target_total), 1.0)

            pbar.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                ce=f"{ce_meter.avg:.4f}",
                kd=f"{kd_meter.avg:.4f}",
                acc=f"{acc:.2f}",
                match=f"{match_recall:.3f}",
            )

        self.scheduler.step(epoch + 1)

        train_loss = loss_meter.avg if total > 0 else 0.0
        train_acc = 100.0 * correct / max(total, 1)
        train_match_recall = float(matched_total) / max(float(target_total), 1.0)

        self.writer.add_scalar("train/loss", train_loss, epoch)
        self.writer.add_scalar("train/lwf_ce_loss", ce_meter.avg, epoch)
        self.writer.add_scalar("train/lwf_kd_loss", kd_meter.avg, epoch)
        self.writer.add_scalar("train/acc", train_acc, epoch)
        self.writer.add_scalar("train/match_recall", train_match_recall, epoch)

        return train_loss, train_acc, train_match_recall