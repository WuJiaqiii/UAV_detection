from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from util.utils import spectrogram_to_yolo_uint8

class DetectionVisualizer:
    def __init__(self, config, logger=None):
        
        self.config = config
        self.logger = logger
        
        self.result_dir = Path(self.config.result_dir)
        self.yolo_db_min = float(self.config.yolo_db_min)
        self.yolo_db_max = float(self.config.yolo_db_max)
        self.yolo_db_eps = float(self.config.yolo_db_eps)
        
    def _spec_to_u8(self, spec):
        return spectrogram_to_yolo_uint8(
            data=spec,
            db_min=self.yolo_db_min,
            db_max=self.yolo_db_max,
            eps=self.yolo_db_eps,
        )

    def _log_warning(self, msg):
        if self.logger is not None:
            self.logger.warning(msg)
        else:
            print(msg)

    def clear_split_dir(self, split_name: str):
        save_dir = self.result_dir / "detect_result" / str(split_name)
        save_dir.mkdir(parents=True, exist_ok=True)

        for p in save_dir.glob("*.png"):
            try:
                p.unlink()
            except Exception as e:
                self._log_warning(f"[Vis] failed to remove {p}: {e}")

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
    def _draw_boxes(draw, boxes, color, width=2):
        arr = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
        for b in arr:
            x1, y1, x2, y2 = [int(v) for v in b]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    def _save_groups_image(self, spec_u8, groups, save_path):
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

        img.save(save_path)

    def save_detect_result(
        self,
        spec,
        yolo_boxes,
        groups,
        matched_boxes,
        fp,
        sample_idx=0,
        split_name="val",
    ):
        save_dir = self.result_dir / "detect_result" / str(split_name)
        save_dir.mkdir(parents=True, exist_ok=True)

        base = Path(fp).stem if fp else f"sample_{sample_idx}"
        spec_u8 = self._spec_to_u8(spec)

        img = Image.fromarray(spec_u8).convert("RGB")
        draw = ImageDraw.Draw(img)
        self._draw_boxes(draw, yolo_boxes, color=(255, 0, 0), width=2)
        img.save(save_dir / f"{base}_yolo.png")

        self._save_groups_image(spec_u8, groups, save_dir / f"{base}_groups.png")

        img = Image.fromarray(spec_u8).convert("RGB")
        draw = ImageDraw.Draw(img)
        self._draw_boxes(draw, matched_boxes, color=(0, 255, 0), width=2)
        img.save(save_dir / f"{base}_matched.png")

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