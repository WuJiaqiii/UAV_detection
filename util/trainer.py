import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

from util.input_builder import CNNInputBuilder
from util.vis import DetectionVisualizer
from util.metrics import AverageMeter, save_confusion_matrix, save_eval_summary, save_instance_csv
from util.checkpoint import save_training_checkpoint, load_training_checkpoint
from util.utils import _make_pbar

class Trainer:
    """
    Main pipeline:

    1) Single-signal training & validation:
        image -> YOLO -> groups -> best group -> CNN input -> classifier loss

    2) Universal inference:
        image -> YOLO -> groups -> each group classification
        Background prediction is treated as ignored output.
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

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.classifier.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))

        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer,
            T_0=int(getattr(config, "cosine_annealing_T0", 50)),
            T_mult=int(getattr(config, "cosine_annealing_mult", 2)))

        self.writer = SummaryWriter(str(self.config.log_dir))

        self.input_builder = CNNInputBuilder(
            mode=str(config.cnn_input_mode).lower(),
            out_size=int(config.mask_img_size),
            box_draw_thickness=int(config.box_draw_thickness),
            box_draw_value=int(config.box_draw_value),
            mask_fill_value=int(config.mask_fill_value)
        )

        self.visualizer = DetectionVisualizer(result_dir=self.config.result_dir, logger=self.logger)

        self.best_val_acc = -1.0
        self.no_improve_epochs = 0

        self.inv_class_map = {int(v): str(k) for k, v in self.config.classes.items()}
        self.eval_exclude_label_ids = {int(v) for k, v in getattr(self.config, "classes", {}).items() if str(k) in self.config.eval_exclude_classes}

        self.background_label_id = int(self.config.classes["Background"])

        if self.config.val_detect_vis_ratio < 0:
            raise ValueError("--val_detect_vis_ratio must be >= 0")
        if self.config.val_detect_vis_ratio > 1:
            raise ValueError("--val_detect_vis_ratio must be <= 1")
        if self.config.infer_detect_vis_ratio < 0:
            raise ValueError("--infer_detect_vis_ratio must be >= 0")
        if self.config.infer_detect_vis_ratio > 1:
            raise ValueError("--infer_detect_vis_ratio must be <= 1")

        self.checkpoint_path = config.checkpoint_path

        if self.checkpoint_path:
            ckpt = load_training_checkpoint(
                model=self.classifier,
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                logger=self.logger,
                load_optimizer=(self.config.run_mode == "train"),
            )
            if isinstance(ckpt, dict) and "best_val_acc" in ckpt:
                try:
                    self.best_val_acc = float(ckpt["best_val_acc"])
                except Exception:
                    pass
        else:
            self.logger.info("No checkpoint path provided, training from scratch.")


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

        groups = self.preprocessor.select_signal_groups(yolo_boxes, spectrogram=spec)
        return groups or []

    @staticmethod
    def _select_main_group(groups):
        if groups is None or len(groups) == 0:
            return None
        return max(groups, key=lambda g: float(g.get("score", 0.0)))

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

    @staticmethod
    def _group_bandwidth_mhz(group):
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
                cost[ti, gi] = float(freq_diff + self.config.match_bandwidth_weight * bw_diff)

        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(cost)
            pairs = list(zip(rows.tolist(), cols.tolist()))
        else:
            pairs = []
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
            if self.config.skip_unmatched and freq_diff > self.config.match_freq_thresh:
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

    # ------------------------------------------------------------------
    # Instance builders
    # ------------------------------------------------------------------
    def _build_single_instances(self, inputs, targets_list, sample_fps=None,
        save_detect_result=False, max_save_images=8, split_name="val", save_fp_set=None,
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
            main_group = self._select_main_group(groups)

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

            if main_group is None:
                image_infos.append(image_info)
                continue

            final_boxes = np.asarray(main_group.get("boxes", []), dtype=np.int32).reshape(-1, 4)
            if len(final_boxes) == 0:
                image_infos.append(image_info)
                continue

            matched_total += 1
            image_info["num_matched"] = 1
            image_info["unmatched_targets"] = []
            image_infos.append(image_info)

            x, _ = self.input_builder.build(spec, final_boxes)

            batch_images.append(x)
            batch_labels.append(label)
            batch_metas.append({
                "fp": fp,
                "target_idx": 0,
                "group_idx": 0,
                "label": label,
                "target_name": self._label_name(label),
            })

            should_save = bool(save_detect_result) and saved_count < int(max_save_images)
            if save_fp_set is not None:
                should_save = should_save and (str(fp) in save_fp_set)

            if should_save:
                self.visualizer.save_detect_result(
                    spec=spec,
                    yolo_boxes=yolo_boxes,
                    groups=groups,
                    matched_boxes=final_boxes.tolist(),
                    fp=fp,
                    sample_idx=i,
                    split_name=split_name,
                )
                saved_count += 1

        if len(batch_images) == 0:
            return None, None, None, matched_total, target_total, image_infos

        images = torch.stack(batch_images, dim=0)
        labels = torch.as_tensor(batch_labels, dtype=torch.long)

        return images, labels, batch_metas, matched_total, target_total, image_infos

    def _build_multi_group_instances(self, inputs, targets_list, sample_fps=None,
        save_detect_result=False, max_save_images=8, split_name="infer", save_fp_set=None):
        """
        Used by universal/multi-signal inference.

        It builds one CNN instance for every final group.
        GT matching is not used here; it is only used later for evaluation.
        """
        inputs_np = inputs.detach().cpu().numpy() if torch.is_tensor(inputs) else np.asarray(inputs)

        batch_images = []
        batch_metas = []
        image_infos = []

        saved_count = 0

        for i in range(len(inputs_np)):
            spec = np.asarray(inputs_np[i])
            fp = sample_fps[i] if sample_fps is not None else f"sample_{i}"
            targets = targets_list[i] if targets_list is not None else []

            yolo_boxes = self._detect_boxes(spec, fp)

            groups = self._extract_groups(yolo_boxes, spec)

            image_infos.append({
                "fp": fp,
                "spec": spec,
                "yolo_boxes": yolo_boxes,
                "groups": groups,
                "targets": targets,
            })

            should_save = bool(save_detect_result) and saved_count < int(max_save_images)
            if save_fp_set is not None:
                should_save = should_save and (str(fp) in save_fp_set)

            if should_save:
                all_group_boxes = []
                for g in groups:
                    all_group_boxes.extend(
                        np.asarray(g.get("boxes", []), dtype=np.int32).reshape(-1, 4).tolist()
                    )

                self.visualizer.save_detect_result(
                    spec=spec,
                    yolo_boxes=yolo_boxes,
                    groups=groups,
                    matched_boxes=all_group_boxes,
                    fp=fp,
                    sample_idx=i,
                    split_name=split_name,
                )
                saved_count += 1

            for gi, g in enumerate(groups):
                final_boxes = np.asarray(g.get("boxes", []), dtype=np.int32).reshape(-1, 4)
                if len(final_boxes) == 0:
                    continue

                x, _ = self.input_builder.build(spec, final_boxes)

                batch_images.append(x)
                batch_metas.append({
                    "fp": fp,
                    "group_idx": int(gi),
                    "boxes": final_boxes.copy(),
                })

        if len(batch_images) == 0:
            return None, None, image_infos

        images = torch.stack(batch_images, dim=0)

        return images, batch_metas, image_infos

    # ------------------------------------------------------------------
    # Inference evaluation
    # ------------------------------------------------------------------
    def _evaluate_infer_predictions(self, image_infos, metas, preds):
        """
        End-to-end multi-signal inference evaluation.

        Rules:
        1) matched group:
           GT = matched signal label, Pred = classifier prediction.

        2) unmatched group:
           GT = Background, Pred = classifier prediction.

        3) unmatched target:
           GT = true signal label, Pred = Background.
           This represents missed detection / missed grouping / no final output.
        """
        if image_infos is None:
            image_infos = []
        if metas is None:
            metas = []

        if preds is None:
            preds_np = np.asarray([], dtype=np.int64)
        else:
            preds_np = preds.detach().cpu().numpy() if torch.is_tensor(preds) else np.asarray(preds)

        pred_map_by_fp = {}
        for meta, pred in zip(metas, preds_np.tolist()):
            fp = meta["fp"]
            pred_map_by_fp.setdefault(fp, {})
            pred_map_by_fp[fp][int(meta["group_idx"])] = int(pred)

        correct = 0
        total = 0
        matched_total = 0
        target_total = 0
        image_correct_total = 0
        image_total = 0

        all_targets = []
        all_preds = []
        instance_rows = []

        bg_id = self.background_label_id

        for info in image_infos:
            fp = info.get("fp", "")
            spec = info.get("spec", None)
            groups = info.get("groups", []) or []
            targets = info.get("targets", []) or []

            if spec is None:
                image_total += 1
                continue

            matched, unmatched_targets, unmatched_groups = self._match_groups_to_targets(
                groups,
                targets,
                spec.shape,
            )

            matched_total += len(matched)
            target_total += len(targets)
            image_total += 1

            per_image_all_correct = (len(unmatched_targets) == 0)

            # 1) matched groups -> signal labels
            for m in matched:
                gi = int(m["group_idx"])

                if fp not in pred_map_by_fp or gi not in pred_map_by_fp[fp]:
                    per_image_all_correct = False
                    continue

                gt = int(m["label"])
                pd = int(pred_map_by_fp[fp][gi])

                if self._should_skip_eval_label(gt):
                    continue

                is_corr = int(pd == gt)

                total += 1
                correct += is_corr
                all_targets.append(gt)
                all_preds.append(pd)

                instance_rows.append({
                    "file": fp,
                    "target_idx": int(m["target_idx"]),
                    "group_idx": gi,
                    "gt_label": gt,
                    "gt_name": self._label_name(gt),
                    "pred_label": pd,
                    "pred_name": self._label_name(pd),
                    "correct": is_corr,
                    "eval_role": "matched_signal",
                })

                if is_corr == 0:
                    per_image_all_correct = False

            # 2) unmatched groups -> Background GT
            if bg_id is not None:
                for gi in unmatched_groups:
                    gi = int(gi)

                    if fp not in pred_map_by_fp or gi not in pred_map_by_fp[fp]:
                        per_image_all_correct = False
                        continue

                    gt = int(bg_id)
                    pd = int(pred_map_by_fp[fp][gi])

                    if self._should_skip_eval_label(gt):
                        continue

                    is_corr = int(pd == gt)

                    total += 1
                    correct += is_corr
                    all_targets.append(gt)
                    all_preds.append(pd)

                    instance_rows.append({
                        "file": fp,
                        "target_idx": -1,
                        "group_idx": gi,
                        "gt_label": gt,
                        "gt_name": self._label_name(gt),
                        "pred_label": pd,
                        "pred_name": self._label_name(pd),
                        "correct": is_corr,
                        "eval_role": "unmatched_group_as_background",
                    })

                    if is_corr == 0:
                        per_image_all_correct = False
            else:
                if len(unmatched_groups) > 0:
                    per_image_all_correct = False

            # 3) unmatched targets -> predicted as Background
            if bg_id is not None:
                for ti in unmatched_targets:
                    ti = int(ti)
                    gt = int(targets[ti]["label"])
                    pd = int(bg_id)

                    if self._should_skip_eval_label(gt):
                        continue

                    is_corr = int(pd == gt)

                    total += 1
                    correct += is_corr
                    all_targets.append(gt)
                    all_preds.append(pd)

                    instance_rows.append({
                        "file": fp,
                        "target_idx": ti,
                        "group_idx": -1,
                        "gt_label": gt,
                        "gt_name": self._label_name(gt),
                        "pred_label": pd,
                        "pred_name": self._label_name(pd),
                        "correct": is_corr,
                        "eval_role": "unmatched_target_as_background",
                    })

                    per_image_all_correct = False
            else:
                if len(unmatched_targets) > 0:
                    per_image_all_correct = False

            if per_image_all_correct:
                image_correct_total += 1

        return {
            "correct": correct,
            "total": total,
            "matched_total": matched_total,
            "target_total": target_total,
            "image_correct_total": image_correct_total,
            "image_total": image_total,
            "all_targets": np.asarray(all_targets, dtype=np.int64),
            "all_preds": np.asarray(all_preds, dtype=np.int64),
            "instance_rows": instance_rows,
        }

    def _save_infer_classified_results(self, metas, preds, image_infos, split_name="infer"):
        if not self.config.save_infer_classified_vis:
            return

        if metas is None or len(metas) == 0:
            return

        preds_np = preds.detach().cpu().numpy() if torch.is_tensor(preds) else np.asarray(preds)

        pred_map_by_fp = {}
        for meta, pred in zip(metas, preds_np.tolist()):
            fp = meta["fp"]
            pred_map_by_fp.setdefault(fp, {})
            pred_map_by_fp[fp][int(meta["group_idx"])] = {
                "pred_label": int(pred),
                "pred_name": self._label_name(int(pred)),
            }

        for info in image_infos:
            fp = info.get("fp", "")
            if fp not in pred_map_by_fp:
                continue

            spec = info.get("spec", None)
            groups = info.get("groups", None)

            if spec is None or groups is None:
                continue

            self.visualizer.save_classified_groups(
                spec=spec,
                groups=groups,
                group_pred_map=pred_map_by_fp[fp],
                fp=fp,
                split_name=split_name,
            )

    # ------------------------------------------------------------------
    # Train / validation
    # ------------------------------------------------------------------
    def train_one_epoch(self, epoch):
        self.classifier.train()

        loss_meter = AverageMeter()
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

            logits = self.classifier(images)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            preds = logits.argmax(dim=1)

            correct += int((preds == labels).sum().item())
            total += int(labels.numel())

            loss_meter.update(loss.item(), labels.size(0))

            acc = 100.0 * correct / max(total, 1)
            match_recall = float(matched_total) / max(float(target_total), 1.0)

            pbar.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                acc=f"{acc:.2f}",
                match=f"{match_recall:.3f}",
            )

        self.scheduler.step(epoch + 1)

        train_loss = loss_meter.avg if total > 0 else 0.0
        train_acc = 100.0 * correct / max(total, 1)
        train_match_recall = float(matched_total) / max(float(target_total), 1.0)

        self.writer.add_scalar("train/loss", train_loss, epoch)
        self.writer.add_scalar("train/acc", train_acc, epoch)
        self.writer.add_scalar("train/match_recall", train_match_recall, epoch)

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

        val_vis_fp_set = None

        if (self.config.save_val_detect_vis and self.config.val_detect_vis_ratio > 0):
            
            if self.config.clear_val_detect_vis_each_epoch:
                self.visualizer.clear_split_dir("val")

            val_vis_fp_set = self.visualizer.sample_fps_by_ratio(
                loader,
                self.config.val_detect_vis_ratio,
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

            need_save_detect_result = (val_vis_fp_set is not None and len(val_vis_fp_set) > 0)

            images, labels, metas, batch_matched, batch_targets, _ = self._build_single_instances(
                inputs=inputs,
                targets_list=targets_list,
                sample_fps=fps,
                save_detect_result=need_save_detect_result,
                max_save_images=len(val_vis_fp_set) if val_vis_fp_set is not None else 0,
                split_name='val',
                save_fp_set=val_vis_fp_set,
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
    # Universal / multi-signal inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def infer_multi_loader(self, loader):
        
        self.classifier.eval()

        correct = 0
        total = 0
        matched_total = 0
        target_total = 0
        image_correct_total = 0
        image_total = 0

        all_preds = []
        all_targets = []
        instance_rows = []

        infer_vis_fp_set = None
        if self.config.save_infer_detect_vis and self.config.infer_detect_vis_ratio > 0:
            infer_vis_fp_set = self.visualizer.sample_fps_by_ratio(
                loader,
                self.config.infer_detect_vis_ratio,
            )
            self.logger.info(
                f"[Infer Vis] ratio={self.config.infer_detect_vis_ratio:.4f}, "
                f"selected={len(infer_vis_fp_set)} samples"
            )

        # pbar = tqdm(loader, desc="Infer", leave=True)
        pbar = _make_pbar(loader, desc=f"Infer", leave=True)

        for batch in pbar:
            inputs, targets_list, snrs, fps = batch

            need_save_detect_result = (
                infer_vis_fp_set is not None
                and len(infer_vis_fp_set) > 0
            )

            images, metas, image_infos = self._build_multi_group_instances(
                inputs=inputs,
                targets_list=targets_list,
                sample_fps=fps,
                save_detect_result=need_save_detect_result,
                max_save_images=len(infer_vis_fp_set) if infer_vis_fp_set is not None else 0,
                split_name="infer",
                save_fp_set=infer_vis_fp_set,
            )

            if images is None:
                batch_eval = self._evaluate_infer_predictions(
                    image_infos=image_infos,
                    metas=[],
                    preds=None,
                )
            else:
                images = images.to(self.device, non_blocking=True)
                logits = self.classifier(images)
                preds = logits.argmax(dim=1)

                self._save_infer_classified_results(
                    metas=metas,
                    preds=preds,
                    image_infos=image_infos,
                    split_name="infer",
                )

                batch_eval = self._evaluate_infer_predictions(
                    image_infos=image_infos,
                    metas=metas,
                    preds=preds,
                )

            correct += int(batch_eval["correct"])
            total += int(batch_eval["total"])
            matched_total += int(batch_eval["matched_total"])
            target_total += int(batch_eval["target_total"])
            image_correct_total += int(batch_eval["image_correct_total"])
            image_total += int(batch_eval["image_total"])

            if len(batch_eval["all_preds"]) > 0:
                all_preds.append(batch_eval["all_preds"])
                all_targets.append(batch_eval["all_targets"])

            instance_rows.extend(batch_eval["instance_rows"])

            acc = 100.0 * correct / max(total, 1)
            match_recall = float(matched_total) / max(float(target_total), 1.0)
            image_exact_acc = 100.0 * image_correct_total / max(image_total, 1)

            pbar.set_postfix(
                acc=f"{acc:.2f}",
                match=f"{match_recall:.3f}",
                img_exact=f"{image_exact_acc:.2f}",
            )

        if len(all_preds) > 0:
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
        else:
            all_preds = np.array([], dtype=np.int64)
            all_targets = np.array([], dtype=np.int64)

        acc = 100.0 * correct / max(total, 1)
        match_recall = float(matched_total) / max(float(target_total), 1.0)
        image_exact_acc = 100.0 * image_correct_total / max(image_total, 1)

        save_confusion_matrix(
            y_true=all_targets,
            y_pred=all_preds,
            result_dir=self.config.result_dir,
            inv_class_map=self.inv_class_map,
            eval_exclude_label_ids=self.eval_exclude_label_ids,
            split_name="infer",
            epoch=None,
        )
    
        save_eval_summary(
            result_dir=self.config.result_dir,
            split_name="infer",
            epoch=None,
            metrics={
                "mode": "multi",
                "loss": 0.0,
                "acc": float(acc),
                "match_recall": float(match_recall),
                "image_exact_acc": float(image_exact_acc),
                "num_instances": int(total),
                "num_targets": int(target_total),
                "num_matched": int(matched_total),
                "num_images": int(image_total),
                "num_image_exact": int(image_correct_total),
            },
            eval_exclude_classes=self.config.eval_exclude_classes,
        )

        save_instance_csv(
            result_dir=self.config.result_dir,
            split_name="infer",
            epoch=None,
            instance_rows=instance_rows,
        )

        return 0.0, acc, match_recall, image_exact_acc

    # ------------------------------------------------------------------
    # Public APIs
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch):
        return self.evaluate_single_loader(loader=self.val_loader, epoch=epoch)

    @torch.no_grad()
    def infer(self, infer_loader):
        infer_loss, infer_acc, infer_match_recall, infer_image_exact_acc = self.infer_multi_loader(loader=infer_loader)
        self.logger.info(f"[Infer] loss={infer_loss:.4f}, acc={infer_acc:.2f}, match={infer_match_recall:.3f}, img_exact={infer_image_exact_acc:.2f}")
        return infer_loss, infer_acc, infer_match_recall, infer_image_exact_acc

    def train(self):
        if self.train_loader is None or self.val_loader is None:
            raise RuntimeError("train_loader / val_loader is None.")

        for epoch in range(self.config.epochs):
            self.logger.info(f"===== Epoch {epoch + 1}/{self.config.epochs} =====")

            train_loss, train_acc, train_match_recall = self.train_one_epoch(epoch)
            val_loss, val_acc, val_match_recall = self.validate(epoch)

            self.writer.add_scalar("val/loss", val_loss, epoch)
            self.writer.add_scalar("val/acc", val_acc, epoch)
            self.writer.add_scalar("val/match_recall", val_match_recall, epoch)

            self.logger.info(f"[Train] loss={train_loss:.4f}, acc={train_acc:.2f}, match={train_match_recall:.3f}")
            self.logger.info(f"[Val]   loss={val_loss:.4f}, acc={val_acc:.2f}, match={val_match_recall:.3f}" )

            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.no_improve_epochs = 0
            else:
                self.no_improve_epochs += 1

            if ((epoch + 1) % self.config.save_interval == 0) or is_best:
                save_training_checkpoint(
                    model=self.classifier,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    best_val_acc=self.best_val_acc,
                    model_dir=self.config.model_dir,
                    config=self.config,
                    logger=self.logger,
                    is_best=is_best,
                )

            if self.no_improve_epochs >= self.config.early_stop_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        self.writer.close()