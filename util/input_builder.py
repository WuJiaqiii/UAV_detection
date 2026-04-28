import cv2
import numpy as np
import torch


class CNNInputBuilder:
    """
    Build CNN input tensor from spectrogram + selected boxes.

    Supported modes:
      - mask
      - raw
      - raw_with_boxes
      - raw_in_boxes
    """

    def __init__(self, mode: str = "mask",
        out_size: int = 224,
        box_draw_thickness: int = 2,
        box_draw_value: int = 255,
        mask_fill_value: int = 255,
    ):
        self.mode = str(mode).lower()
        self.out_size = int(out_size)
        self.box_draw_thickness = int(box_draw_thickness)
        self.box_draw_value = int(box_draw_value)
        self.mask_fill_value = int(mask_fill_value)

        if self.mode not in {"mask", "raw", "raw_with_boxes", "raw_in_boxes"}:
            raise ValueError(
                f"Unsupported cnn_input_mode={mode}. "
                f"Expected one of: mask, raw, raw_with_boxes, raw_in_boxes."
            )

    @staticmethod
    def spec_to_uint8(spec: np.ndarray) -> np.ndarray:
        spec = np.asarray(spec)
        if spec.ndim != 2:
            raise ValueError(f"Expected 2D spectrogram, got shape={spec.shape}")

        if spec.dtype == np.uint8:
            return spec.copy()

        x = spec.astype(np.float32)
        finite = np.isfinite(x)
        if not finite.any():
            return np.zeros_like(x, dtype=np.uint8)

        valid = x[finite]
        x_min, x_max = float(valid.min()), float(valid.max())
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        else:
            x = np.zeros_like(x, dtype=np.float32)

        return np.clip(x * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def boxes_to_white_mask(image_shape, boxes, fill_value=255) -> np.ndarray:
        h, w = int(image_shape[0]), int(image_shape[1])
        mask = np.zeros((h, w), dtype=np.uint8)

        if boxes is None:
            return mask

        arr = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
        for b in arr:
            x1, y1, x2, y2 = [int(v) for v in b]

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            mask[y1:y2, x1:x2] = int(fill_value)

        return mask

    def boxes_to_centered_white_mask(self, image_shape, boxes, fill_value=255) -> np.ndarray:
        h, w = int(image_shape[0]), int(image_shape[1])
        mask = np.zeros((h, w), dtype=np.uint8)

        if boxes is None:
            return mask

        arr = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
        if len(arr) == 0:
            return mask

        valid = []
        for b in arr:
            x1, y1, x2, y2 = [int(v) for v in b]

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            valid.append([x1, y1, x2, y2])

        if len(valid) == 0:
            return mask

        arr = np.asarray(valid, dtype=np.int32)

        gy1 = int(arr[:, 1].min())
        gy2 = int(arr[:, 3].max())

        group_cy = 0.5 * (gy1 + gy2)
        canvas_cy = 0.5 * (h - 1)

        dx = 0
        dy = int(round(canvas_cy - group_cy))

        shifted = arr.copy()
        shifted[:, [0, 2]] += dx
        shifted[:, [1, 3]] += dy

        for b in shifted:
            x1, y1, x2, y2 = [int(v) for v in b]

            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                continue

            mask[y1:y2, x1:x2] = int(fill_value)

        return mask

    def build(self, spec, boxes):
        """
        Return:
            x: torch.FloatTensor, shape [1, out_size, out_size]
            img: uint8 image before resizing, shape [H, W]
        """
        spec = np.asarray(spec)
        raw = self.spec_to_uint8(spec)

        if self.mode == "mask":
            img = self.boxes_to_centered_white_mask(
                    spec.shape,
                    boxes,
                    fill_value=self.mask_fill_value,
                )
            
        elif self.mode == "raw":
            img = raw.copy()

        elif self.mode == "raw_in_boxes":
            mask = self.boxes_to_white_mask(
                spec.shape,
                boxes,
                fill_value=self.mask_fill_value,
            )
            img = np.zeros_like(raw, dtype=np.uint8)
            keep = mask > 0
            img[keep] = raw[keep]

        elif self.mode == "raw_with_boxes":
            img = raw.copy()
            arr = np.asarray(boxes, dtype=np.int32).reshape(-1, 4) if boxes is not None else np.zeros((0, 4), dtype=np.int32)
            h, w = img.shape[:2]

            for b in arr:
                x1, y1, x2, y2 = [int(v) for v in b]

                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    color=int(self.box_draw_value),
                    thickness=int(self.box_draw_thickness),
                )

        else:
            raise ValueError(f"Unsupported cnn_input_mode={self.mode}")

        interpolation = cv2.INTER_NEAREST if self.mode == "mask" else cv2.INTER_LINEAR
        x = img.astype(np.float32) / 255.0
        x = cv2.resize(x, (self.out_size, self.out_size), interpolation=interpolation)
        x = np.expand_dims(x, axis=0)

        return torch.from_numpy(x).float(), img