import os
import glob
import argparse
from contextlib import nullcontext

import numpy as np
from PIL import Image, ImageDraw

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

def load_summed_submatrices(mat_path: str, key: str = "summed_submatrices") -> np.ndarray:
    """
    Load `key` from a .mat file.
    Supports MATLAB v7 (scipy.io.loadmat) and v7.3 (HDF5 via h5py).
    """
    try:
        from scipy.io import loadmat
        d = loadmat(mat_path)
        if key not in d:
            raise KeyError(f"Key '{key}' not found in {mat_path}. Keys: {list(d.keys())[:20]} ...")
        arr = np.asarray(d[key])
        return arr
    except NotImplementedError:
        pass
    except Exception:
        pass
    import h5py
    with h5py.File(mat_path, "r") as f:
        if key not in f:
            raise KeyError(f"Key '{key}' not found in HDF5 mat: {mat_path}. Keys: {list(f.keys())}")
        arr = np.array(f[key])
    return arr


def ensure_hw(x: np.ndarray, target_hw=(512, 750)) -> np.ndarray:
    """
    Ensure x is 2D and (H,W) = target_hw if possible.
    Some v7.3 mats may load transposed.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={x.shape}")

    Ht, Wt = target_hw
    if x.shape == (Ht, Wt):
        return x
    if x.shape == (Wt, Ht):
        return x.T

    # allow other shapes but warn
    return x


def to_uint8_grayscale(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.5) -> np.ndarray:
    """
    uint16/float -> uint8 grayscale (percentile normalization).
    """
    x = np.asarray(x).astype(np.float32)
    lo, hi = np.percentile(x, [p_low, p_high])
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi <= lo):
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi <= lo:
            hi = lo + 1.0

    x01 = (x - lo) / (hi - lo)
    x01 = np.clip(x01, 0.0, 1.0)
    return (x01 * 255.0).astype(np.uint8)


# Geometry helpers (rectangularity + bbox area)
def _bbox_from_segmentation(seg: np.ndarray):
    """seg: HxW bool"""
    ys, xs = np.where(seg)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def _parse_bbox_any(bbox):
    """
    Accept SAM2-style bbox=[x,y,w,h] or xyxy=[x0,y0,x1,y1].
    Return xyxy ints.
    """
    if bbox is None:
        return None
    b = list(bbox)
    if len(b) != 4:
        return None

    x0, y0, a, c = b
    # Heuristic: if a,c look like w,h (positive and small-ish) -> xywh
    # Otherwise treat as xyxy.
    if a >= 0 and c >= 0 and (a + x0) >= x0 and (c + y0) >= y0:
        # could still be xyxy, but we decide by comparing
        # if a > x0 and c > y0 then it's likely xyxy; yet w/h can also be > x0/y0.
        # We'll use a simple rule: if a <= 1e4 and c <= 1e4 and (a + c) < 2e4 it's ambiguous.
        # Better rule: assume SAM2 default is xywh.
        x, y, w, h = map(float, b)
        return int(x), int(y), int(x + w), int(y + h)

    return int(x0), int(y0), int(a), int(c)


def filter_masks_rect_and_area(
    masks: list,
    H: int,
    W: int,
    min_rectangularity: float,
    max_bbox_area_ratio: float,
):
    """
    Filter by:
      1) bbox_area <= max_bbox_area_ratio * (H*W)
      2) rectangularity = mask_area / bbox_area >= min_rectangularity
    Mutates mask dict by adding:
      _bbox_xyxy, _rectangularity, _bbox_area_ratio
    """
    img_area = float(H * W)
    max_bbox_area = max_bbox_area_ratio * img_area

    kept = []
    for m in masks:
        seg = m.get("segmentation", None)
        if seg is None:
            continue
        seg = np.asarray(seg).astype(bool)

        mask_area = float(m.get("area", seg.sum()))

        xyxy = None
        if "bbox" in m and m["bbox"] is not None:
            xyxy = _parse_bbox_any(m["bbox"])
        if xyxy is None:
            xyxy = _bbox_from_segmentation(seg)
        if xyxy is None:
            continue

        x0, y0, x1, y1 = xyxy
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)
        bbox_area = float(bw * bh)

        bbox_area_ratio = bbox_area / img_area
        rectangularity = mask_area / bbox_area

        if (bbox_area <= max_bbox_area) and (rectangularity >= min_rectangularity):
            m["_bbox_xyxy"] = (x0, y0, x1, y1)
            m["_rectangularity"] = float(rectangularity)
            m["_bbox_area_ratio"] = float(bbox_area_ratio)
            kept.append(m)

    return kept

def overlay_masks_on_rgb(
    rgb_u8: np.ndarray,
    masks: list,
    alpha: float = 0.6,
    random_color: bool = True,
    draw_borders: bool = True,
    draw_bboxes: bool = True,
):
    """
    Produce an overlay image (uint8 RGB).
    - If cv2 available and draw_borders=True: draw contours like your original show_mask.
    - bbox uses green rectangle like your original show_box.
    """
    img = rgb_u8.astype(np.float32).copy()
    H, W = img.shape[:2]

    rng = np.random.default_rng(3)
    base_color = np.array([30, 144, 255], dtype=np.float32)  # same as your show_mask default (RGB)

    has_cv2 = False
    cv2 = None
    if draw_borders or draw_bboxes:
        try:
            import cv2 as _cv2
            cv2 = _cv2
            has_cv2 = True
        except Exception:
            has_cv2 = False

    # mask fill overlay
    for m in masks:
        seg = np.asarray(m["segmentation"]).astype(bool)
        if seg.shape != (H, W):
            continue

        color = base_color
        if random_color:
            color = rng.integers(0, 256, size=(3,), dtype=np.uint8).astype(np.float32)

        img[seg] = (1.0 - alpha) * img[seg] + alpha * color

        # borders (contours)
        if draw_borders and has_cv2:
            mask_u8 = seg.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
            # draw white-ish border
            cv2.drawContours(img, contours, -1, (255, 255, 255), thickness=2)

    out = np.clip(img, 0, 255).astype(np.uint8)

    # bbox drawing
    if draw_bboxes:
        if has_cv2:
            for m in masks:
                if "_bbox_xyxy" not in m:
                    continue
                x0, y0, x1, y1 = m["_bbox_xyxy"]
                cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
        else:
            # fallback to PIL if cv2 not installed
            pil = Image.fromarray(out)
            draw = ImageDraw.Draw(pil)
            for m in masks:
                if "_bbox_xyxy" not in m:
                    continue
                x0, y0, x1, y1 = m["_bbox_xyxy"]
                draw.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=2)
            out = np.array(pil)

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .mat files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")

    parser.add_argument("--mat_key", type=str, default="summed_submatrices", help="Key inside .mat")

    # Default paths per your message
    parser.add_argument("--sam2_checkpoint", type=str, default="sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--model_cfg", type=str, default="sam2/configs/sam2.1/sam2.1_hiera_l.yaml")

    # Filter constraints
    parser.add_argument("--min_rectangularity", type=float, default=0.75, help="mask_area / bbox_area threshold")
    parser.add_argument("--max_bbox_area_ratio", type=float, default=0.20, help="bbox_area / image_area <= this")

    # Visualization / save options
    parser.add_argument("--save_gray", action="store_true", help="Also save grayscale input pngs")
    parser.add_argument("--alpha", type=float, default=0.6, help="Overlay alpha for masks")
    parser.add_argument("--random_color", action="store_true", help="Use random colors for each mask")
    parser.add_argument("--no_borders", action="store_true", help="Disable contour borders")
    parser.add_argument("--no_bboxes", action="store_true", help="Disable bbox drawing")
    parser.add_argument("--max_images", type=int, default=-1, help="Process at most N mats (-1 all)")

    # Optional SAM2 generator knobs (keep default if not set)
    parser.add_argument("--points_per_side", type=int, default=64, help="SAM2 points_per_side (0 means default)")
    parser.add_argument("--pred_iou_thresh", type=float, default=0.55, help="SAM2 pred_iou_thresh (-1 means default)")
    parser.add_argument("--stability_score_thresh", type=float, default=-1.0, help="SAM2 stability_score_thresh (-1 means default)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_pred = os.path.join(args.output_dir, "pred_overlays")
    os.makedirs(out_pred, exist_ok=True)
    out_gray = os.path.join(args.output_dir, "gray_inputs")
    if args.save_gray:
        os.makedirs(out_gray, exist_ok=True)

    # device selection (same idea as your original script)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[INFO] using device: {device}")

    if device.type == "cuda":
        # allow tf32 on Ampere+
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "[WARN] MPS support is preliminary; outputs may differ vs CUDA."
        )

    # Build SAM2 + automatic mask generator
    from sam2.sam2.build_sam import build_sam2
    try:
        from sam2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except Exception as e:
        raise RuntimeError(
            "Import sam2.automatic_mask_generator failed. "
            "Please ensure your SAM2 installation includes SAM2AutomaticMaskGenerator.\n"
            f"Error: {e}"
        )

    sam2_model = build_sam2(args.model_cfg, args.sam2_checkpoint, device=device)

    gen_kwargs = {}
    if args.points_per_side and args.points_per_side > 0:
        gen_kwargs["points_per_side"] = args.points_per_side
    if args.pred_iou_thresh is not None and args.pred_iou_thresh >= 0:
        gen_kwargs["pred_iou_thresh"] = args.pred_iou_thresh
    if args.stability_score_thresh is not None and args.stability_score_thresh >= 0:
        gen_kwargs["stability_score_thresh"] = args.stability_score_thresh

    # mask_generator = SAM2AutomaticMaskGenerator(sam2_model, **gen_kwargs)
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=32,
        points_per_batch=128,
        crop_n_layers=0,
        # crop_n_layers=1,
        # crop_n_points_downscale_factor=2,
        # crop_overlap_ratio=0.25,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.85,
        box_nms_thresh=0.6,
        crop_nms_thresh=0.6,
        min_mask_region_area=120,
    )

    # gather files
    mat_files = sorted(glob.glob(os.path.join(args.input_dir, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in: {args.input_dir}")
    if args.max_images is not None and args.max_images > 0:
        mat_files = mat_files[: args.max_images]

    print(f"[INFO] Found {len(mat_files)} mat files.")
    print(f"[INFO] Filter: rectangularity >= {args.min_rectangularity}, bbox_area_ratio <= {args.max_bbox_area_ratio}")

    # autocast for cuda
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    with torch.inference_mode():
        with amp_ctx:
            for i, mat_path in enumerate(mat_files, 1):
                name = os.path.splitext(os.path.basename(mat_path))[0]
                try:
                    x = load_summed_submatrices(mat_path, key=args.mat_key)
                    x = ensure_hw(x, target_hw=(512, 750))

                    gray_u8 = to_uint8_grayscale(x)

                    if args.save_gray:
                        Image.fromarray(gray_u8, mode="L").save(os.path.join(out_gray, f"{name}.png"))

                    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)  # HxWx3 uint8
                    H, W = rgb.shape[:2]

                    masks = mask_generator.generate(rgb)
                    print("raw masks:", len(masks))

                    # filter by rectangularity + bbox area ratio
                    masks = filter_masks_rect_and_area(
                        masks,
                        H, W,
                        min_rectangularity=args.min_rectangularity,
                        max_bbox_area_ratio=args.max_bbox_area_ratio,
                    )
                    print("kept masks:", len(masks))

                    overlay = overlay_masks_on_rgb(
                        rgb_u8=rgb,
                        masks=masks,
                        alpha=args.alpha,
                        random_color=args.random_color,
                        draw_borders=(not args.no_borders),
                        draw_bboxes=(not args.no_bboxes),
                    )

                    out_path = os.path.join(out_pred, f"{name}_overlay.png")
                    Image.fromarray(overlay).save(out_path)

                    print(f"[{i:4d}/{len(mat_files):4d}] OK: {name} | kept_masks={len(masks)} -> {out_path}")

                except Exception as e:
                    print(f"[{i:4d}/{len(mat_files):4d}] FAIL: {name} | {e}")


if __name__ == "__main__":
    main()
