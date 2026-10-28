from pathlib import Path
import hashlib
import json
import re

import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin

from util.spectrogram_frontend import FrontendConfig, render_gray

class DetectionVisualizer:
    def __init__(self, config, logger=None):
        
        self.config = config
        self.logger = logger
        
        self.result_dir = Path(self.config.result_dir)
        self.frontend = FrontendConfig.from_config(config)
        self.input_domain = getattr(config, "detector_spec_domain", "amplitude")

    def _spec_to_u8(self, spec):
        # Whole-image visualization only. Exact per-tile model rasters are
        # produced by the shared native frontend; never enhance PNGs twice.
        return render_gray(spec, self.frontend, self.input_domain)

    def _log_warning(self, msg):
        if self.logger is not None:
            self.logger.warning(msg)
        else:
            print(msg)


    def collect_fps_from_dataset(self, dataset):
        if dataset is None:
            return []

        if hasattr(dataset, "samples"):
            fps = []
            for s in dataset.samples:
                if isinstance(s, dict) and "fp" in s:
                    fps.append(str(s["fp"]))
            return fps

        if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
            base_fps = self.collect_fps_from_dataset(dataset.dataset)
            fps = []
            for idx in list(dataset.indices):
                idx = int(idx)
                if 0 <= idx < len(base_fps):
                    fps.append(str(base_fps[idx]))
            return fps

        return []

    def sample_fps_by_ratio(self, loader, ratio: float):
        if loader is None:
            return set()

        ratio = float(ratio)
        if ratio <= 0:
            return set()

        fps = self.collect_fps_from_dataset(getattr(loader, "dataset", None))
        if len(fps) == 0:
            return set()

        num_samples = int(np.ceil(len(fps) * ratio))
        num_samples = max(1, min(num_samples, len(fps)))

        chosen = np.random.choice(
            np.asarray(fps, dtype=object),
            size=num_samples,
            replace=False,
        )

        return set(str(x) for x in chosen.tolist())

    @staticmethod
    def _safe_component(value, limit=96):
        value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(value))
        value = value[:limit].strip(' .') or 'unknown'
        # Prefixing also avoids Windows reserved names such as CON and NUL.
        return 'class_' + value

    def output_dir(self, kind, split_name, epoch=None, class_name=None):
        path = self.result_dir / kind / str(split_name)
        if epoch is not None:
            path /= f"epoch_{int(epoch) + 1:04d}"
        if class_name is not None:
            path /= self._safe_component(class_name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def sample_prefix(fp, sample_idx=0):
        source = str(Path(fp).resolve()) if fp else f"sample_{sample_idx}"
        stem = Path(fp).stem if fp else source
        stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', stem)[:80].strip(' .')
        digest = hashlib.sha256(source.encode('utf-8')).hexdigest()[:12]
        return f"sample_{stem or 'unknown'}__{digest}"

    def collect_sample_records(self, dataset):
        """Read metadata only, preserving nested Subset indices without loading IQ."""
        if dataset is None:
            return []
        if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
            records = self.collect_sample_records(dataset.dataset)
            return [records[int(i)] for i in dataset.indices]
        return list(getattr(dataset, 'samples', []))

    def sample_validation_fps(self, loader, ratio, min_per_class=1, epoch=0):
        """Sample each class independently from the actual validation subset."""
        if loader is None or ratio <= 0:
            return set()
        inverse = {int(v): str(k) for k, v in self.config.classes.items()}
        by_class = {}
        for sample in self.collect_sample_records(getattr(loader, 'dataset', None)):
            targets = sample.get('targets')
            if targets is not None:
                if len(targets) != 1:
                    continue
                target = targets[0]
            else:
                target = sample
            name = target.get('class_name') or inverse.get(int(target['label']), 'unknown')
            by_class.setdefault(str(name), {})[str(sample['fp'])] = sample

        # Local RNG keeps visualization sampling independent of training randomness.
        rng = np.random.default_rng(42 + int(epoch))
        selected = []
        counts = {}
        for name, samples in sorted(by_class.items()):
            fps = sorted(samples)
            count = min(len(fps), max(int(min_per_class), int(np.ceil(len(fps) * ratio))))
            chosen = rng.choice(fps, size=count, replace=False).tolist()
            counts[name] = count
            selected.extend({
                'file': fp, 'absolute_path': str(Path(fp).resolve()),
                'filename': Path(fp).name, 'class_name': name,
                'image_prefix': self.sample_prefix(fp),
                'directory': self._safe_component(name),
            } for fp in chosen)

        path = self.output_dir('detect_result', 'val', epoch) / 'selected_samples.json'
        path.write_text(json.dumps({
            'epoch': int(epoch) + 1, 'ratio_per_class': float(ratio),
            'min_per_class': int(min_per_class), 'counts': counts, 'samples': selected,
        }, ensure_ascii=False, indent=2), encoding='utf-8')
        if self.logger is not None:
            self.logger.info(f"[Val Vis] epoch={epoch + 1}, selected per class={counts}")
        return {sample['file'] for sample in selected}

    @staticmethod
    def _draw_boxes(draw, boxes, color, width=2):
        arr = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
        for b in arr:
            x1, y1, x2, y2 = [int(v) for v in b]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    def _save_groups_image(self, spec_u8, groups, save_path, caption=None, pnginfo=None):
        img = Image.fromarray(spec_u8).convert("RGB")
        draw = ImageDraw.Draw(img)

        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (128, 255, 0),
        ]

        for gi, g in enumerate(groups or []):
            boxes = np.asarray(g.get("boxes", []), dtype=np.int32).reshape(-1, 4)
            if len(boxes) == 0:
                continue

            color = colors[gi % len(colors)]
            self._draw_boxes(draw, boxes, color=color, width=2)

            x1, y1, _, _ = [int(v) for v in boxes[0]]
            label = f"G{gi}"

            if "score" in g:
                label += f" s={float(g['score']):.2f}"
            if "center_freq" in g:
                label += f" f={float(g['center_freq']):.1f}"
            if "group_type" in g:
                label += f" {g['group_type']}"

            draw.text((x1, max(0, y1 - 12)), label, fill=color)

        self._save_image(img, save_path, caption, pnginfo)

    @staticmethod
    def _save_image(img, path, caption=None, pnginfo=None):
        if caption:
            header = Image.new('RGB', (img.width, 38), 'black')
            draw = ImageDraw.Draw(header)
            # Full Unicode names remain available in filenames, JSON and PNG metadata.
            for i, line in enumerate(caption):
                draw.text((4, 2 + i * 16), line.encode('ascii', 'replace').decode(), fill='white')
            canvas = Image.new('RGB', (img.width, img.height + header.height))
            canvas.paste(header, (0, 0))
            canvas.paste(img, (0, header.height))
            img = canvas
        img.save(path, pnginfo=pnginfo)

    def save_detect_result(
        self,
        spec,
        yolo_boxes,
        groups,
        matched_boxes,
        fp,
        sample_idx=0,
        split_name="val",
        epoch=None,
        class_name=None,
        diagnostics=None,
    ):
        save_dir = self.output_dir('detect_result', split_name, epoch, class_name)

        base = (self.sample_prefix(fp, sample_idx) if epoch is not None or class_name is not None
                else Path(fp).stem if fp else f"sample_{sample_idx}")
        if diagnostics is not None:
            (save_dir / f"{base}_grouping.json").write_text(
                json.dumps({"source_file": str(fp), "ground_truth": class_name,
                            **diagnostics}, ensure_ascii=False, indent=2), encoding="utf-8",
            )
        spec_u8 = self._spec_to_u8(spec)
        caption = None
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text('source_file', str(fp))
        if class_name is not None:
            metadata.add_text('ground_truth', str(class_name))
            status = 'matched' if len(matched_boxes) else 'no matched group'
            caption = [f"GT: {class_name} | {status}", f"File: {Path(fp).name}"]

        img = Image.fromarray(spec_u8).convert("RGB")
        draw = ImageDraw.Draw(img)
        self._draw_boxes(draw, yolo_boxes, color=(255, 0, 0), width=2)
        self._save_image(img, save_dir / f"{base}_yolo.png", caption, metadata)

        self._save_groups_image(spec_u8, groups, save_dir / f"{base}_groups.png", caption, metadata)

        img = Image.fromarray(spec_u8).convert("RGB")
        draw = ImageDraw.Draw(img)
        self._draw_boxes(draw, matched_boxes, color=(0, 255, 0), width=2)
        self._save_image(img, save_dir / f"{base}_matched.png", caption, metadata)

    def save_classified_groups(
        self,
        spec,
        groups,
        group_pred_map,
        fp,
        sample_idx=0,
        split_name="infer",
    ):
        save_dir = self.result_dir / "detect_result" / str(split_name)
        save_dir.mkdir(parents=True, exist_ok=True)

        base = Path(fp).stem if fp else f"sample_{sample_idx}"
        spec_u8 = self._spec_to_u8(spec)

        img = Image.fromarray(spec_u8).convert("RGB")
        draw = ImageDraw.Draw(img)

        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (128, 255, 0),
        ]

        for gi, g in enumerate(groups or []):
            if gi not in group_pred_map:
                continue

            boxes = np.asarray(g.get("boxes", []), dtype=np.int32).reshape(-1, 4)
            if len(boxes) == 0:
                continue

            color = colors[gi % len(colors)]
            self._draw_boxes(draw, boxes, color=color, width=2)

            pred_name = str(
                group_pred_map[gi].get(
                    "pred_name",
                    group_pred_map[gi].get("pred_label", "unknown"),
                )
            )

            extra = []
            if "score" in g:
                extra.append(f"s={float(g['score']):.2f}")
            if "center_freq" in g:
                extra.append(f"f={float(g['center_freq']):.1f}")

            label = f"G{gi}: {pred_name}"
            if extra:
                label += " | " + " ".join(extra)

            x1, y1, _, _ = [int(v) for v in boxes[0]]
            draw.text((x1, max(0, y1 - 12)), label, fill=color)

        img.save(save_dir / f"{base}_classified_groups.png")
