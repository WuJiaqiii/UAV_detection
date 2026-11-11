import numpy as np
import cv2
import re
import json
from time import perf_counter
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

from util.mask_input import build_mask_input
from util.vis import DetectionVisualizer
from util.spectrogram_frontend import FrontendConfig, render_gray
from util.checkpoint import save_training_checkpoint
from util.utils import _make_pbar, AverageMeter, save_confusion_matrix, save_eval_summary, save_instance_csv

class Trainer:
    """
    Main pipeline:

    1) Single-signal training & validation:
        image -> YOLO -> groups -> best group -> CNN input -> classifier loss

    Deployment and labeled evaluation use the shared predict/infer pipeline.
    """

    def __init__(self, config, dataloaders, logger, detector, preprocessor, classifier, bbox_cache=None):

        self.config = config
        self.logger = logger
        self.detector = detector
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.bbox_cache = bbox_cache

        if isinstance(dataloaders, (tuple, list)) and len(dataloaders) == 2:
            self.train_loader, self.val_loader = dataloaders
        else:
            self.train_loader, self.val_loader = None, None

        self.device = config.device
        self.classifier.to(self.device)

        from model.representation import SupervisedContrastiveLoss, validate_representation_options
        self.supcon_weight = validate_representation_options(config)
        self.supcon_criterion = SupervisedContrastiveLoss(float(getattr(config, "supcon_temperature", .2)))

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.classifier.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))

        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer,
            T_0=int(getattr(config, "cosine_annealing_T0", 50)),
            T_mult=int(getattr(config, "cosine_annealing_mult", 2)))

        self.visualizer = DetectionVisualizer(config=self.config, logger=self.logger)

        self.best_val_acc = -1.0
        self.best_selection_score = -1.0
        self.no_improve_epochs = 0
        self.start_epoch = 0

        self.inv_class_map = {int(v): str(k) for k, v in self.config.classes.items()}
        self.eval_exclude_label_ids = {int(v) for k, v in getattr(self.config, "classes", {}).items() if str(k) in self.config.eval_exclude_classes}

        background_id = self.config.classes.get("Background")
        self.background_label_id = int(background_id) if background_id is not None else None

        if self.config.val_detect_vis_ratio < 0:
            raise ValueError("--val_detect_vis_ratio must be >= 0")
        if self.config.val_detect_vis_ratio > 1:
            raise ValueError("--val_detect_vis_ratio must be <= 1")
        if int(getattr(self.config, "val_detect_vis_min_per_class", 1)) < 0:
            raise ValueError("--val_detect_vis_min_per_class must be >= 0")
        if config.checkpoint_path:
            raise ValueError('Training starts fresh in v3; use lwf_teacher_checkpoint for an incremental stage')
        self.logger.info('Training a fresh model; inference uses the unified predict/infer pipeline.')
        self.writer = SummaryWriter(str(self.config.log_dir))


    def _label_name(self, idx: int) -> str:
        return self.inv_class_map.get(int(idx), str(idx))

    def _should_skip_eval_label(self, label_idx: int) -> bool:
        return int(label_idx) in self.eval_exclude_label_ids

    # ------------------------------------------------------------------
    # Group helpers
    # ------------------------------------------------------------------
    def _detect_boxes(self, spec, fp):
        """
        Unified YOLO detection entry with optional bbox cache.

        Returns:
            np.ndarray, shape [N, 4], dtype int32
        """
        # No cache: run YOLO directly.
        if self.bbox_cache is None:
            boxes = self.detector.detect(spec)
            return (
                np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
                if len(boxes) > 0
                else np.zeros((0, 4), dtype=np.int32)
            )

        # Try reading cache.
        cached = self.bbox_cache.get(fp)
        if cached is not None:
            if torch.is_tensor(cached):
                cached = cached.detach().cpu().numpy()
            boxes = np.asarray(cached, dtype=np.int32).reshape(-1, 4)
            return boxes

        # Cache miss: run YOLO.
        boxes = self.detector.detect(spec)
        boxes = (
            np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
            if len(boxes) > 0
            else np.zeros((0, 4), dtype=np.int32)
        )

        # Write cache if mode allows.
        self.bbox_cache.put(
            fp,
            torch.as_tensor(boxes, dtype=torch.int32),
        )

        return boxes

    def _extract_groups(self, yolo_boxes, spec):
        if not hasattr(self.preprocessor, "select_signal_groups"):
            raise AttributeError("preprocessor must provide select_signal_groups()")

        if getattr(self.config,'event_mode','legacy')=='physical':
            from util.event_detection import select_event_groups
            details=getattr(self.detector,'last_detections',[])
            scores=[x['confidence'] for x in details] if len(details)==len(yolo_boxes) else None
            groups,_=select_event_groups(self.preprocessor,yolo_boxes,spec,self.config,scores,
                self.config.iq_sample_rate if self.config.input_type=='iq_dat' else None)
        else:
            groups = self.preprocessor.select_signal_groups(yolo_boxes, spectrogram=spec)
        return groups or []

    @staticmethod
    def _select_main_group(groups):
        if groups is None or len(groups) == 0:
            return None
        return max(groups, key=lambda g: float(g.get("score", 0.0)))

    def _select_training_group(self, groups, target):
        """Use the known frequency label when available to avoid wrong labels."""
        if not groups:
            return None
        if not target.get("has_frequency_label", True):
            return self._select_main_group(groups)
        target_freq = float(target["center_freq"])
        if not np.isfinite(target_freq):
            return self._select_main_group(groups)
        target_bw = float(target.get("bandwidth", 0.0))
        candidates = []
        for group in groups:
            freq_diff = abs(float(group["center_freq"]) - target_freq)
            bw_diff = abs(float(group.get("bandwidth", 0.0)) - target_bw)
            if bool(self.config.skip_unmatched) and freq_diff > float(self.config.match_freq_thresh):
                continue
            if bool(self.config.match_use_bandwidth) and bw_diff > float(self.config.match_bw_thresh):
                continue
            cost = freq_diff
            if bool(self.config.match_use_bandwidth):
                cost += float(self.config.match_bandwidth_weight) * bw_diff
            candidates.append((cost, -float(group.get("score", 0.0)), group))
        return min(candidates, key=lambda item: item[:2])[2] if candidates else None





    def _build_mask_input(self, spec, boxes):
        return build_mask_input(spec, boxes, self.config)

    # ------------------------------------------------------------------
    # Classifier-input mask visualization
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_stem_for_vis(fp):
        """
        Convert a file path/name into a safe stem for visualization files.
        """
        stem = Path(str(fp)).stem
        stem = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", stem)
        stem = re.sub(r"_+", "_", stem).strip("_")
        return stem if stem else "sample"

    def _to_uint8_gray_for_vis(self, img):
        return render_gray(np.squeeze(np.asarray(img)), FrontendConfig.from_config(self.config),
                           getattr(self.config, "detector_spec_domain", "amplitude"))

    def _save_classifier_input_mask_vis(
        self,
        spec,
        mask,
        x_tensor,
        fp,
        sample_idx,
        group_idx,
        split_name="infer",
        epoch=None,
        class_name=None,
    ):
        """
        Save the binary image immediately before the classifier.

        Saved files:
            *_binary_full.png
                Centered binary mask at original spectrogram resolution.

            *_binary_input.png
                Resized binary mask with size mask_img_size x mask_img_size.
                This is exactly the spatial image sent into the classifier.

            *_overlay.png
                Spectrogram with mask region highlighted for PPT explanation.
        """
        root = self.visualizer.output_dir('classifier_input_masks', split_name, epoch, class_name)

        stem = (self.visualizer.sample_prefix(fp, sample_idx)
                if epoch is not None or class_name is not None else self._safe_stem_for_vis(fp))
        prefix = f"{stem}_sample-{int(sample_idx):04d}_group-{int(group_idx):03d}"

        full_mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
        Image.fromarray(full_mask).save(root / f"{prefix}_binary_full.png")

        if torch.is_tensor(x_tensor):
            x_np = x_tensor.detach().cpu().numpy()
        else:
            x_np = np.asarray(x_tensor)

        if x_np.ndim == 3 and x_np.shape[0] == 2:
            Image.fromarray(np.rint(np.clip(x_np[1], 0, 1)*255).astype(np.uint8)).save(
                root / f'{prefix}_amplitude_input.png')
            x_np = x_np[0]
        x_np = np.squeeze(x_np)
        input_mask = (x_np > 0.5).astype(np.uint8) * 255
        Image.fromarray(input_mask).save(root / f"{prefix}_binary_input.png")

        try:
            spec_u8 = self._to_uint8_gray_for_vis(spec)
            overlay = cv2.cvtColor(spec_u8, cv2.COLOR_GRAY2BGR)
            mask_bool = full_mask > 0

            # Highlight the classifier-selected region in red.
            overlay[mask_bool] = (
                0.45 * overlay[mask_bool]
                + 0.55 * np.array([0, 0, 255], dtype=np.float32)
            ).astype(np.uint8)

            Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)).save(root / f"{prefix}_overlay.png")
        except Exception as error:
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.warning(
                    f"[MaskInputVis] failed to save overlay for {fp}, "
                    f"group={group_idx}: {error}"
                )


    # ------------------------------------------------------------------
    # Instance builders
    # ------------------------------------------------------------------
    def _build_single_instances(self, inputs, targets_list, sample_fps=None,
        save_detect_result=False, max_save_images=8, split_name="val", save_fp_set=None,
        epoch=None,
    ):
        """
        Used by single-signal train/val.

        One input image should have exactly one target.
        If no group is generated, it contributes to target_total but not matched_total.
        """
        inputs_np = inputs.detach().cpu().numpy() if torch.is_tensor(inputs) else np.asarray(inputs)

        batch_images = []
        batch_labels = []
        batch_metas = []
        image_infos = []

        matched_total = 0
        target_total = 0
        saved_count = 0

        for i in range(len(inputs_np)):
            spec = np.asarray(inputs_np[i])
            fp = sample_fps[i] if sample_fps is not None else f"sample_{i}"
            targets = targets_list[i] if targets_list is not None else []

            if len(targets) != 1:
                self.logger.warning(
                    f"[SingleSignalSkip] expected 1 target, got {len(targets)}: {fp}"
                )
                continue

            target_total += 1
            label = int(targets[0]["label"])

            yolo_boxes = self._detect_boxes(spec, fp)

            groups = self._extract_groups(yolo_boxes, spec)
            main_group = self._select_training_group(groups, targets[0])
            all_components = split_name == "train" and getattr(self.config, "train_group_policy", "primary") == "all"
            selected_groups = [(index, g) for index, g in enumerate(groups)
                               if len(np.asarray(g.get("boxes", [])).reshape(-1, 4))] if all_components else (
                                   [(0, main_group)] if main_group is not None else [])
            if all_components:
                main_group = self._select_main_group([g for _, g in selected_groups])

            image_info = {
                "fp": fp,
                "spec": spec,
                "yolo_boxes": yolo_boxes,
                "groups": groups,
                "num_targets": 1,
                "num_matched": 0,
                "unmatched_targets": [0],
                "unmatched_groups": list(range(len(groups))),
            }

            final_boxes = np.asarray(
                main_group.get("boxes", []) if main_group is not None else [],
                dtype=np.int32,
            ).reshape(-1, 4)
            should_save = bool(save_detect_result) and saved_count < int(max_save_images)
            if save_fp_set is not None:
                should_save = should_save and (str(fp) in save_fp_set)
            if should_save:
                self.visualizer.save_detect_result(
                    spec=spec, yolo_boxes=yolo_boxes, groups=groups,
                    matched_boxes=final_boxes.tolist(), fp=fp,
                    sample_idx=i, split_name=split_name, epoch=epoch,
                    class_name=self._label_name(label),
                    diagnostics=getattr(getattr(self, "preprocessor", None), "last_diagnostics", None),
                )
                saved_count += 1

            if main_group is None:
                image_infos.append(image_info)
                continue

            if len(final_boxes) == 0:
                image_infos.append(image_info)
                continue

            matched_total += 1
            image_info["num_matched"] = 1
            image_info["unmatched_targets"] = []
            image_infos.append(image_info)

            for group_idx, selected_group in selected_groups:
                component_boxes = np.asarray(selected_group["boxes"], dtype=np.int32).reshape(-1, 4)
                x, mask = self._build_mask_input(spec, component_boxes)
                batch_images.append(x)
                batch_labels.append(label)
                batch_metas.append({
                    "fp": fp, "target_idx": 0, "group_idx": group_idx, "label": label,
                    "target_name": self._label_name(label), "boxes": component_boxes.copy(),
                    "spec_shape": tuple(spec.shape), "loss_weight": 1./len(selected_groups),
                    **({'spectrum': spec} if getattr(self.config, 'classifier_input_mode', 'mask') == 'mask_amplitude' else {}),
                })
                if should_save:
                    self._save_classifier_input_mask_vis(spec=spec, mask=mask, x_tensor=x, fp=fp,
                        sample_idx=i, group_idx=group_idx, split_name=split_name, epoch=epoch,
                        class_name=self._label_name(label))

        if len(batch_images) == 0:
            return None, None, None, matched_total, target_total, image_infos

        images = torch.stack(batch_images, dim=0)
        labels = torch.as_tensor(batch_labels, dtype=torch.long)

        return images, labels, batch_metas, matched_total, target_total, image_infos


    # ------------------------------------------------------------------
    # Inference evaluation
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Train / validation
    # ------------------------------------------------------------------
    def _training_objective(self, images, labels, metas, return_features=False):
        """Known-only CE/SupCon; feature interface has no additional parameters."""
        weights = None
        if getattr(self.config, "train_group_policy", "primary") == "all":
            if metas is None or len(metas) != len(images):
                raise ValueError("All-component training requires per-component record weights")
            weights = torch.tensor([m["loss_weight"] for m in metas], device=images.device, dtype=torch.float32)
            if not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("Component record weights must be finite and positive")
        def cross_entropy(logits, targets, sample_weights):
            if sample_weights is None:
                return self.criterion(logits, targets)
            losses = nn.functional.cross_entropy(logits, targets, reduction="none")
            return (losses*sample_weights).sum()/sample_weights.sum()
        if not self.supcon_weight:
            if return_features:
                logits, features = self.classifier(images, return_features=True)
            else:
                logits = self.classifier(images)
            ce = cross_entropy(logits, labels, weights)
            result = (logits, ce, ce, ce.detach().new_zeros(()))
            return (*result, features) if return_features else result
        from model.representation import contrastive_mask_batch
        alternative = contrastive_mask_batch(metas, self.config).to(images.device, non_blocking=True)
        count = len(images)
        logits, features = self.classifier(torch.cat([images, alternative]), return_features=True)
        ce = cross_entropy(logits, labels.repeat(2), weights.repeat(2) if weights is not None else None)
        views = torch.stack([features[:count], features[count:]], dim=1)
        supcon = self.supcon_criterion(views, labels, sample_weights=weights)
        result = (logits[:count], ce + self.supcon_weight * supcon, ce, supcon)
        return (*result, features[:count]) if return_features else result

    def train_one_epoch(self, epoch):
        started = perf_counter()
        optimizer_steps = 0
        self.classifier.train()

        loss_meter = AverageMeter()
        ce_meter, supcon_meter = AverageMeter(), AverageMeter()
        correct = total = matched_total = target_total = 0

        # pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch + 1}", leave=True)
        pbar = _make_pbar(self.train_loader, desc=f"Train Epoch {epoch + 1}", leave=True)

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

            logits, loss, loss_ce, loss_supcon = self._training_objective(images, labels, metas)

            loss.backward()
            self.optimizer.step()
            optimizer_steps += 1

            preds = logits.argmax(dim=1)

            correct += int((preds == labels).sum().item())
            total += int(labels.numel())

            loss_meter.update(loss.item(), labels.size(0))
            ce_meter.update(loss_ce.item(), labels.size(0))
            supcon_meter.update(loss_supcon.item(), labels.size(0))

            acc = 100.0 * correct / max(total, 1)
            match_recall = float(matched_total) / max(float(target_total), 1.0)

            pbar.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                acc=f"{acc:.2f}",
                match=f"{match_recall:.3f}",
            )

        self.scheduler.step(epoch + 1)

        if total == 0:
            raise RuntimeError("No training instances survived YOLO detection and group filtering")

        train_loss = loss_meter.avg if total > 0 else 0.0
        train_acc = 100.0 * correct / max(total, 1)
        train_match_recall = float(matched_total) / max(float(target_total), 1.0)

        self.writer.add_scalar("train/loss", train_loss, epoch)
        self.writer.add_scalar("train/ce_loss", ce_meter.avg, epoch)
        self.writer.add_scalar("train/supcon_loss", supcon_meter.avg, epoch)
        self.writer.add_scalar("train/acc", train_acc, epoch)
        self.writer.add_scalar("train/match_recall", train_match_recall, epoch)
        self.last_train_metrics = dict(epoch=epoch+1,optimizer_steps=optimizer_steps,
            valid_instances=total,source_targets=target_total,matched_records=matched_total,
            loss=train_loss,ce_loss=ce_meter.avg,supcon_loss=supcon_meter.avg,
            classifier_input_mode=getattr(self.config,'classifier_input_mode','mask'),
            supcon_weight=self.supcon_weight,wall_seconds=perf_counter()-started)
        if getattr(self.config,'result_dir',None):
            directory=Path(self.config.result_dir)/'train_metrics';directory.mkdir(parents=True,exist_ok=True)
            (directory/f'epoch_{epoch+1}.json').write_text(
                json.dumps(self.last_train_metrics,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

        return train_loss, train_acc, train_match_recall

    @torch.no_grad()
    def evaluate_single_loader(self, loader, epoch=0):
        self.classifier.eval()

        loss_meter = AverageMeter()
        correct = 0
        total = 0
        matched_total = 0
        target_total = 0

        all_preds = []
        all_targets = []
        instance_rows = []
        from collections import Counter
        class_totals, class_correct = Counter(), Counter()

        val_vis_fp_set = None

        if (self.config.save_val_detect_vis and self.config.val_detect_vis_ratio > 0):

            val_vis_fp_set = self.visualizer.sample_validation_fps(
                loader,
                self.config.val_detect_vis_ratio,
                min_per_class=getattr(self.config, "val_detect_vis_min_per_class", 1),
                epoch=epoch,
            )

            self.logger.info(
                f"[Val Vis] epoch={epoch + 1}, "
                f"ratio={self.config.val_detect_vis_ratio:.4f}, "
                f"selected={len(val_vis_fp_set)} samples"
            )

        # pbar = tqdm(loader, desc=f"Val Epoch {epoch + 1}", leave=True)
        pbar = _make_pbar(loader, desc=f"Val Epoch {epoch + 1}", leave=True)

        for batch in pbar:
            inputs, targets_list, snrs, fps = batch

            for targets in targets_list:
                if len(targets) == 1 and not self._should_skip_eval_label(int(targets[0]["label"])):
                    class_totals[int(targets[0]["label"])] += 1

            need_save_detect_result = (val_vis_fp_set is not None and len(val_vis_fp_set) > 0)

            images, labels, metas, batch_matched, batch_targets, _ = self._build_single_instances(
                inputs=inputs,
                targets_list=targets_list,
                sample_fps=fps,
                save_detect_result=need_save_detect_result,
                max_save_images=len(val_vis_fp_set) if val_vis_fp_set is not None else 0,
                split_name='val',
                save_fp_set=val_vis_fp_set,
                epoch=epoch,
            )

            matched_total += batch_matched
            target_total += batch_targets

            if images is None or labels is None or labels.numel() == 0:
                continue

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.classifier(images)
            preds = logits.argmax(dim=1)

            labels_cpu = labels.detach().cpu().numpy()
            keep_mask = np.array(
                [not self._should_skip_eval_label(int(y)) for y in labels_cpu],
                dtype=bool,
            )

            if not keep_mask.any():
                continue

            keep_mask_t = torch.as_tensor(
                keep_mask,
                device=labels.device,
                dtype=torch.bool,
            )

            logits_kept = logits[keep_mask_t]
            labels_kept = labels[keep_mask_t]
            preds_kept = preds[keep_mask_t]

            loss = self.criterion(logits_kept, labels_kept)
            loss_meter.update(loss.item(), labels_kept.size(0))

            preds_cpu = preds_kept.detach().cpu().numpy()
            labels_cpu = labels_kept.detach().cpu().numpy()

            correct += int((preds_cpu == labels_cpu).sum())
            total += int(labels_cpu.shape[0])
            class_correct.update(int(y) for y, prediction in zip(labels_cpu, preds_cpu) if y == prediction)

            all_preds.append(preds_cpu)
            all_targets.append(labels_cpu)

            kept_metas = [m for m, k in zip(metas, keep_mask.tolist()) if k]

            for meta, gt, pd in zip(kept_metas, labels_cpu.tolist(), preds_cpu.tolist()):
                instance_rows.append({
                    "file": meta["fp"],
                    "target_idx": meta["target_idx"],
                    "group_idx": meta["group_idx"],
                    "gt_label": int(gt),
                    "gt_name": self._label_name(int(gt)),
                    "pred_label": int(pd),
                    "pred_name": self._label_name(int(pd)),
                    "correct": int(gt == pd),
                    "eval_role": "single_signal",
                })

            acc = 100.0 * correct / max(total, 1)
            match_recall = float(matched_total) / max(float(target_total), 1.0)

            pbar.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                acc=f"{acc:.2f}",
                match=f"{match_recall:.3f}",
            )

        if total == 0:
            raise RuntimeError("No validation instances survived YOLO detection and group filtering")

        if len(all_preds) > 0:
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
        else:
            all_preds = np.array([], dtype=np.int64)
            all_targets = np.array([], dtype=np.int64)

        loss_avg = loss_meter.avg if total > 0 else 0.0
        acc = 100.0 * correct / max(total, 1)
        match_recall = float(matched_total) / max(float(target_total), 1.0)

        save_confusion_matrix(
            y_true=all_targets,
            y_pred=all_preds,
            result_dir=self.config.result_dir,
            inv_class_map=self.inv_class_map,
            eval_exclude_label_ids=self.eval_exclude_label_ids,
            split_name='val',
            epoch=epoch,
        )

        self.last_validation_metrics = {}
        if getattr(self.config, "checkpoint_selection", "acc") == "old_new_hmean":
            from util.incremental_metrics import incremental_validation_metrics
            self.last_validation_metrics = incremental_validation_metrics(self.config.classes,
                int(self.config.lwf_old_num_classes), class_totals, class_correct)

        save_eval_summary(
            result_dir=self.config.result_dir,
            split_name='val',
            epoch=epoch,
            metrics={
                "mode": "single",
                "loss": float(loss_avg),
                "acc": float(acc),
                "match_recall": float(match_recall),
                "num_instances": int(total),
                "num_targets": int(target_total),
                "num_matched": int(matched_total),
                **self.last_validation_metrics,
            },
            eval_exclude_classes=self.config.eval_exclude_classes,
        )

        save_instance_csv(
            result_dir=self.config.result_dir,
            split_name='val',
            epoch=epoch,
            instance_rows=instance_rows,
        )

        return loss_avg, acc, match_recall

    # ------------------------------------------------------------------
    # Training lifecycle
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch):
        return self.evaluate_single_loader(loader=self.val_loader, epoch=epoch)


    def train(self):
        if self.train_loader is None or self.val_loader is None:
            raise RuntimeError("train_loader / val_loader is None.")
        if self.start_epoch >= self.config.epochs:
            raise ValueError(f"epochs={self.config.epochs} must be greater than checkpoint start_epoch={self.start_epoch}")

        for epoch in range(self.start_epoch, self.config.epochs):
            self.logger.info(f"===== Epoch {epoch + 1}/{self.config.epochs} =====")

            train_loss, train_acc, train_match_recall = self.train_one_epoch(epoch)
            val_loss, val_acc, val_match_recall = self.validate(epoch)

            self.writer.add_scalar("val/loss", val_loss, epoch)
            self.writer.add_scalar("val/acc", val_acc, epoch)
            self.writer.add_scalar("val/match_recall", val_match_recall, epoch)

            self.logger.info(f"[Train] loss={train_loss:.4f}, acc={train_acc:.2f}, match={train_match_recall:.3f}")
            self.logger.info(f"[Val]   loss={val_loss:.4f}, acc={val_acc:.2f}, match={val_match_recall:.3f}" )

            metric = getattr(self.config, "checkpoint_selection", "acc")
            metrics = getattr(self, "last_validation_metrics", {})
            score = val_acc if metric == "acc" else metrics[metric]
            self.writer.add_scalar("val/selection_score", score, epoch)
            self.logger.info(f"[Selection] {metric}={score:.4f}; old={metrics.get('old_macro_recall')}, new={metrics.get('new_macro_recall')}")
            is_best = score > self.best_selection_score
            if is_best:
                self.best_selection_score = score
                self.best_val_acc = val_acc
                self.no_improve_epochs = 0
            else:
                self.no_improve_epochs += 1

            if ((epoch + 1) % self.config.save_interval == 0) or is_best:
                save_training_checkpoint(
                    model=self.classifier,
                    epoch=epoch,
                    best_val_acc=self.best_val_acc,
                    model_dir=self.config.model_dir,
                    config=self.config,
                    logger=self.logger,
                    is_best=is_best,
                    selection_metric=metric, selection_score=score, validation_metrics=metrics,
                )

            if (self.no_improve_epochs >= self.config.early_stop_patience and
                    epoch + 1 >= getattr(self.config, "early_stop_min_epochs", 0)):
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        self.writer.close()
