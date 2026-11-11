import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from util.trainer import Trainer
from util.checkpoint import _strip_module_prefix, _validate_checkpoint_classes
from util.utils import _make_pbar, AverageMeter
from model.resnet import MaskImageClassifier


def load_state_skip_mismatch(model, checkpoint_path, device, logger=None, checkpoint=None):
    """
    Load checkpoint weights into model, skipping keys with shape mismatch.
    This is needed when old_num_classes != new_num_classes.
    """
    from util.checkpoint import load_checkpoint_object
    ckpt = checkpoint if checkpoint is not None else load_checkpoint_object(checkpoint_path, device)

    if "model_state_dict" not in ckpt:
        raise RuntimeError(f"No model_state_dict found in checkpoint: {checkpoint_path}")

    state_dict = _strip_module_prefix(ckpt["model_state_dict"])
    model_state = model.state_dict()
    head_prefix = "net.fc.1." if model.backbone_name.startswith("resnet") else "net.classifier.3."
    incompatible = [key for key, value in model_state.items() if not key.startswith(head_prefix) and
                    (key not in state_dict or tuple(state_dict[key].shape) != tuple(value.shape))]
    unexpected = set(state_dict) - set(model_state)
    if incompatible or unexpected:
        raise ValueError(f"LwF backbone mismatch; missing/incompatible={incompatible}, unexpected={sorted(unexpected)}")

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
        old_count = int(config.lwf_old_num_classes)
        if not 0 < old_count < int(config.num_classes):
            raise ValueError("LwF requires 0 < lwf_old_num_classes < num_classes")
        teacher_ckpt = str(config.lwf_teacher_checkpoint or "").strip()
        if not teacher_ckpt:
            raise ValueError("LwF requires --lwf_teacher_checkpoint")
        from util.checkpoint import load_checkpoint_object
        teacher_checkpoint = load_checkpoint_object(teacher_ckpt)
        if not isinstance(teacher_checkpoint.get("config", {}).get("classes"), dict):
            raise ValueError("LwF teacher checkpoint must contain config.classes to verify old class semantics")
        expected_old = {name: idx for name, idx in config.classes.items() if idx < old_count}
        _validate_checkpoint_classes(teacher_checkpoint, expected_old, logger)
        teacher_model = MaskImageClassifier(backbone=config.backbone, num_classes=old_count,
            in_chans=config.mask_in_chans, pretrained=False, dropout=config.cnn_dropout,
            freeze_backbone=False).to(config.device)
        teacher_model.load_state_dict(_strip_module_prefix(teacher_checkpoint["model_state_dict"]), strict=True)
        load_state_skip_mismatch(classifier, teacher_ckpt, config.device, logger, checkpoint=teacher_checkpoint)
        super().__init__(config=config, dataloaders=dataloaders, logger=logger, detector=detector,
                         preprocessor=preprocessor, classifier=classifier, bbox_cache=bbox_cache)

        self.old_num_classes = old_count
        self.temperature = float(config.lwf_temperature)
        self.lambda_kd = float(config.lwf_lambda_kd)
        self.kd_new_weight = float(getattr(config, "lwf_kd_new_weight", 1.))
        self.feature_kd_weight = float(getattr(config, "lwf_feature_kd_weight", 0.))

        self.teacher = teacher_model

        # Copy old classifier rows from teacher to student.
        if self.classifier.backbone_name.startswith("resnet"):
            s_fc = self.classifier.net.fc[-1]
            t_fc = self.teacher.net.fc[-1]
        else:
            s_fc = self.classifier.net.classifier[-1]
            t_fc = self.teacher.net.classifier[-1]

        with torch.no_grad():
            s_fc.weight[:self.old_num_classes].copy_(
                t_fc.weight[:self.old_num_classes]
            )
            s_fc.bias[:self.old_num_classes].copy_(
                t_fc.bias[:self.old_num_classes]
            )

        self.logger.info(
            f"[LwF] copied classifier rows 0~{self.old_num_classes - 1}"
        )

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        if getattr(config, "lwf_freeze_backbone", False):
            self.logger.info("[LwF] freeze student backbone, train classifier head only.")
            self.classifier.freeze_backbone()

        # Rebuild optimizer after optional freezing.
        head = self.classifier.net.fc if self.classifier.backbone_name.startswith("resnet") else self.classifier.net.classifier
        head_ids = {id(p) for p in head.parameters()}
        backbone = [p for p in self.classifier.parameters() if p.requires_grad and id(p) not in head_ids]
        groups = [{"params": [p for p in head.parameters() if p.requires_grad], "lr": float(config.lr)}]
        if backbone:
            groups.append({"params": backbone, "lr": float(config.lr)*float(getattr(config, "lwf_backbone_lr_multiplier", 1.))})
        self.optimizer = AdamW(
            groups,
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

    def _kd_loss(self, student_logits, teacher_logits, labels=None, metas=None):
        """
        KD only on old-class logits.
        """
        T = self.temperature

        student_old = student_logits[:, :self.old_num_classes]
        teacher_old = teacher_logits[:, :self.old_num_classes].detach()

        losses = F.kl_div(
            F.log_softmax(student_old / T, dim=1),
            F.softmax(teacher_old / T, dim=1),
            reduction="none",
        ).sum(dim=1) * (T * T)
        weights = torch.ones_like(losses)
        if labels is not None:
            weights = torch.where(labels < self.old_num_classes, weights,
                                  weights * getattr(self, "kd_new_weight", 1.))
        if metas is not None:
            weights = weights * weights.new_tensor([m.get("loss_weight", 1.) for m in metas])
        return (losses*weights).sum()/weights.sum().clamp_min(1e-12)

    def _feature_loss(self, student, teacher, labels, metas):
        old = labels < self.old_num_classes
        losses = 1-F.cosine_similarity(student, teacher.detach(), dim=1)
        weights = old.to(losses.dtype)
        if metas is not None:
            weights = weights * weights.new_tensor([m.get("loss_weight", 1.) for m in metas])
        return (losses*weights).sum()/weights.sum().clamp_min(1e-12)

    def train_one_epoch(self, epoch):
        self.classifier.train()
        if getattr(self.config, "lwf_freeze_bn", False):
            for module in self.classifier.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        self.teacher.eval()
        sampler = getattr(self.train_loader, "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        ce_meter = AverageMeter()
        kd_meter = AverageMeter()
        supcon_meter = AverageMeter()
        feature_meter = AverageMeter()
        steps = old_instances = new_instances = 0

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

            if self.feature_kd_weight:
                student_logits, classification_loss, loss_ce, loss_supcon, student_features = self._training_objective(
                    images, labels, metas, return_features=True)
                with torch.no_grad():
                    teacher_logits, teacher_features = self.teacher(images, return_features=True)
                loss_feature = self._feature_loss(student_features, teacher_features, labels, metas)
            else:
                student_logits, classification_loss, loss_ce, loss_supcon = self._training_objective(images, labels, metas)
                with torch.no_grad():
                    teacher_logits = self.teacher(images)
                loss_feature = student_logits.sum()*0.

            loss_kd = self._kd_loss(student_logits, teacher_logits, labels, metas)
            loss = classification_loss + self.lambda_kd * loss_kd + self.feature_kd_weight * loss_feature

            loss.backward()
            self.optimizer.step()
            steps += 1
            old_instances += int((labels < self.old_num_classes).sum())
            new_instances += int((labels >= self.old_num_classes).sum())

            preds = student_logits.argmax(dim=1)

            correct += int((preds == labels).sum().item())
            total += int(labels.numel())

            loss_meter.update(loss.item(), labels.size(0))
            ce_meter.update(loss_ce.item(), labels.size(0))
            kd_meter.update(loss_kd.item(), labels.size(0))
            supcon_meter.update(loss_supcon.item(), labels.size(0))
            feature_meter.update(loss_feature.item(), labels.size(0))

            acc = 100.0 * correct / max(total, 1)
            match_recall = float(matched_total) / max(float(target_total), 1.0)

            pbar.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                ce=f"{ce_meter.avg:.4f}",
                kd=f"{kd_meter.avg:.4f}",
                feature=f"{feature_meter.avg:.4f}",
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
        self.writer.add_scalar("train/lwf_feature_loss", feature_meter.avg, epoch)
        self.writer.add_scalar("train/supcon_loss", supcon_meter.avg, epoch)
        self.writer.add_scalar("train/acc", train_acc, epoch)
        self.writer.add_scalar("train/match_recall", train_match_recall, epoch)
        self.optimizer_steps_total = getattr(self, "optimizer_steps_total", 0) + steps
        audit = {"epoch": epoch+1, "optimizer_steps": steps, "optimizer_steps_total": self.optimizer_steps_total,
                 "old_instances": old_instances, "new_instances": new_instances,
                 "loss": train_loss, "ce_loss": ce_meter.avg, "kd_loss": kd_meter.avg,
                 "feature_loss": feature_meter.avg, "match_recall": train_match_recall}
        target = Path(self.config.result_dir)/"train_metrics"/f"epoch_{epoch+1}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
        self.logger.info(f"[LwF exposure] steps={steps}; old={old_instances}; new={new_instances}")

        return train_loss, train_acc, train_match_recall
