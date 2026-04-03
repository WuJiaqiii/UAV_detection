import numpy as np
import cv2


def boxes_to_white_mask(
    image_shape,
    boxes,
    fill_value=255,
    mode="fill",
    thickness=1,
):
    """
    image_shape: (H, W)
    boxes: list[[x1,y1,x2,y2]] or ndarray [N,4]
    return: uint8 mask, white boxes on black background
    """
    H, W = image_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    if boxes is None:
        return mask

    if isinstance(boxes, np.ndarray):
        arr = boxes.reshape(-1, 4) if boxes.size > 0 else np.zeros((0, 4), dtype=np.int32)
    else:
        if len(boxes) == 0:
            arr = np.zeros((0, 4), dtype=np.int32)
        else:
            arr = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)

    for box in arr:
        x1, y1, x2, y2 = [int(v) for v in box]

        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H))

        if x2 <= x1 or y2 <= y1:
            continue

        if mode == "fill":
            mask[y1:y2, x1:x2] = fill_value
        elif mode == "outline":
            cv2.rectangle(mask, (x1, y1), (x2 - 1, y2 - 1), color=fill_value, thickness=thickness)
        else:
            raise ValueError(f"Unsupported mode={mode}, expected 'fill' or 'outline'.")

    return mask


def mask_to_tensor(mask_u8, out_size=None):
    """
    mask_u8: (H, W), uint8
    return: float32 ndarray (1, H, W) in [0,1]
    """
    if not isinstance(mask_u8, np.ndarray):
        mask_u8 = np.asarray(mask_u8)

    if mask_u8.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={mask_u8.shape}")

    x = mask_u8.astype(np.float32) / 255.0

    if out_size is not None:
        if isinstance(out_size, int):
            out_h = out_w = out_size
        else:
            out_h, out_w = out_size
        x = cv2.resize(x, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    x = np.expand_dims(x, axis=0)  # (1, H, W)
    return x


def render_boxes_to_tensor(
    image_shape,
    boxes,
    out_size=None,
    fill_value=255,
    mode="fill",
    thickness=1,
):
    mask = boxes_to_white_mask(
        image_shape=image_shape,
        boxes=boxes,
        fill_value=fill_value,
        mode=mode,
        thickness=thickness,
    )
    return mask_to_tensor(mask, out_size=out_size)