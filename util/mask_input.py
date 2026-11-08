"""Shared classifier mask construction for training and standalone inference."""
import cv2
import numpy as np
import torch


def validate_classifier_input(config):
    mode = getattr(config, 'classifier_input_mode', 'mask')
    if mode not in ('mask', 'mask_amplitude'):
        raise ValueError('classifier_input_mode must be mask or mask_amplitude')
    expected = 1 if mode == 'mask' else 2
    if getattr(config, 'mask_in_chans', 1) != expected:
        raise ValueError(f'{mode} requires mask_in_chans={expected}')
    return mode


def _frequency_shift(image_shape, boxes):
    h, w = map(int, image_shape)
    arr = np.asarray([] if boxes is None else boxes, dtype=np.int32).reshape(-1, 4).copy()
    if not len(arr):
        return 0
    arr[:, 0] = np.clip(arr[:, 0], 0, w-1)
    arr[:, 2] = np.clip(arr[:, 2], 0, w)
    arr[:, 1] = np.clip(arr[:, 1], 0, h-1)
    arr[:, 3] = np.clip(arr[:, 3], 0, h)
    arr = arr[(arr[:, 2] > arr[:, 0]) & (arr[:, 3] > arr[:, 1])]
    if not len(arr):
        return 0
    return int(round(.5 * (h - 1) - .5 * (arr[:, 1].min() + arr[:, 3].max())))


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
    """Shared train/infer input; amplitude and mask use the same translation.

    mask_amplitude needs canonical scientific amplitude, never rendered PNGs.
    The mask-only path retains its original numerical contract.
    """
    mode = validate_classifier_input(config)
    mask = boxes_to_centered_mask(spec.shape, boxes, config)

    x = mask.astype(np.float32) / 255.0
    x = cv2.resize(
        x,
        (int(config.mask_img_size), int(config.mask_img_size)),
        interpolation=cv2.INTER_NEAREST,
    )
    x = np.expand_dims(x, axis=0)
    if mode == 'mask_amplitude':
        from util.spectrogram_frontend import FrontendConfig, amplitude_to_db, background_standardize
        frontend = FrontendConfig.from_config(config)
        z = background_standardize(amplitude_to_db(spec), frontend)
        amplitude = np.clip((z-frontend.z_min)/(frontend.z_max-frontend.z_min), 0, 1)
        shifted = np.zeros_like(amplitude)
        dy = _frequency_shift(spec.shape, boxes)
        h = len(amplitude)
        start, stop = max(0, -dy), min(h, h-dy)
        if stop > start:
            shifted[start+dy:stop+dy] = amplitude[start:stop]
        shifted *= mask > 0
        # Nearest interpolation preserves the identical support of both channels.
        amp = cv2.resize(shifted, (int(config.mask_img_size), int(config.mask_img_size)),
                         interpolation=cv2.INTER_NEAREST)
        x = np.concatenate([x, amp[None]], axis=0)

    return torch.from_numpy(np.ascontiguousarray(x)).float(), mask
