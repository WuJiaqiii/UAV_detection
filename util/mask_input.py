"""Shared classifier mask construction for training and standalone inference."""
import cv2
import numpy as np
import torch


def boxes_to_centered_mask(image_shape, boxes, config):
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

    dy = int(round(canvas_cy - group_cy))

    shifted = arr.copy()
    shifted[:, [1, 3]] += dy

    fill_value = int(config.mask_fill_value)

    for b in shifted:
        x1, y1, x2, y2 = [int(v) for v in b]

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            continue

        mask[y1:y2, x1:x2] = fill_value

    return mask


def build_mask_input(spec, boxes, config):
    mask = boxes_to_centered_mask(spec.shape, boxes, config)

    x = mask.astype(np.float32) / 255.0
    x = cv2.resize(
        x,
        (int(config.mask_img_size), int(config.mask_img_size)),
        interpolation=cv2.INTER_NEAREST,
    )
    x = np.expand_dims(x, axis=0)

    return torch.from_numpy(x).float(), mask
