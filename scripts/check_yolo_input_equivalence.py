#!/usr/bin/env python3
"""Rebuild exported pixels from their source and optionally capture YOLO input.

Without --yolo_weights: pixel/PNG/CHW and coordinate checks.
With --yolo_weights: also capture the real AutoShape backend input, compare it
with the PNG path and compare detections. Requires the local YOLO submodule.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from scipy.io import loadmat
from util.spectrogram_frontend import (
    FrontendConfig, FRONTEND_VERSION, iq_to_db, render_tiles, rgb_to_tensor_array,
)


def restore(root, row, source):
    path = root / row["source"]
    if source["input_type"] == "iq_dat":
        from data.iq_dat_loader import read_iq_dat
        seg = row["segment"]
        iq = read_iq_dat(path, seg["start_sample"], seg["num_samples"],
                         byte_order=seg["byte_order"])
        return iq_to_db(iq, source["iq_n_fft"], source["iq_hop_length"])
    if source["input_type"] == "mat":
        return loadmat(path, variable_names=[source["mat_key"]])[source["mat_key"]]
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Cannot read " + str(path))
    return image


def check(dataset, source_root, limit=20, weights=None, device="cpu"):
    manifest = json.loads((dataset / "frontend_config.json").read_text())
    if manifest.get("version") != FRONTEND_VERSION:
        raise ValueError("Frontend version mismatch")
    cfg = FrontendConfig(**manifest["frontend"])
    source = manifest["source"]
    rows = [json.loads(line) for line in (dataset / "metadata.jsonl").read_text().splitlines()]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No exported images to check")
    model = None
    if weights:
        import yolov5.hubconf as hubconf
        model = hubconf.custom(path=str(weights), autoshape=True, device=device, _verbose=False)
        model.float().eval()
        model.amp = False
    training_loaders = {}
    cache_key, matrix = None, None
    for row in rows:
        if row["frontend"] != cfg.fingerprint():
            raise AssertionError("Mixed frontend versions in metadata")
        key = (row["source"], json.dumps(row["segment"], sort_keys=True))
        if key != cache_key:
            matrix = restore(source_root, row, source)
            cache_key = key
        candidates = render_tiles(matrix, cfg, row["domain"])
        tile = next(t for t in candidates if t.x0 == row["transform"]["x0"])
        if tile.metadata() != row["transform"] or list(matrix.shape) != row["matrix_shape"]:
            raise AssertionError("Coordinate/shape metadata mismatch")
        path = dataset / row["image"]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        np.testing.assert_array_equal(rgb, tile.image)
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == row["pixel_sha256"]
        expected = rgb_to_tensor_array(tile.image)
        np.testing.assert_array_equal(expected, rgb_to_tensor_array(rgb))
        box = np.array([[tile.x0, 0, tile.x0+tile.width, tile.height]], np.float32)
        np.testing.assert_allclose(tile.to_source(tile.from_source(box)), box, atol=1e-4)
        if model is not None:
            from yolov5.utils.dataloaders import LoadImagesAndLabels
            split = row["split"]
            if split not in training_loaders:
                training_loaders[split] = LoadImagesAndLabels(
                    str(dataset / "images" / split), img_size=cfg.image_size,
                    batch_size=1, augment=False, rect=False, stride=32,
                )
            loader = training_loaders[split]
            indices = [i for i, name in enumerate(loader.im_files)
                       if Path(name).resolve() == path.resolve()]
            if len(indices) != 1:
                raise AssertionError("Exported image missing from actual YOLO training loader")
            training_image = loader[indices[0]][0].float().numpy() / 255.0
            np.testing.assert_array_equal(training_image, expected)
            captured = []
            handle = model.model.register_forward_pre_hook(
                lambda module, args: captured.append(args[0].detach().cpu().float().numpy().copy()))
            try:
                import torch
                with torch.inference_mode():
                    a = model(tile.image, size=cfg.image_size).xyxy[0].cpu().numpy()
                    b = model(str(path), size=cfg.image_size).xyxy[0].cpu().numpy()
            finally:
                handle.remove()
            if len(captured) != 2:
                raise AssertionError("Unexpected YOLO wrapper: expected one backend call per input")
            np.testing.assert_array_equal(captured[0], expected[None])
            np.testing.assert_array_equal(captured[1], expected[None])
            np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-5)
    return {"checked": len(rows), "pixel_equal": True, "tensor_conversion_equal": True,
            "actual_yolo_tensor_and_predictions": "passed" if model is not None else "not_run_no_weights",
            "actual_training_loader": "passed_augment_false" if model is not None else "not_run_no_weights"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--source_root", type=Path, required=True)
    p.add_argument("--limit", type=int, default=20, help="0 checks every image")
    p.add_argument("--yolo_weights", type=Path)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.limit < 0:
        p.error("--limit must be nonnegative")
    print(json.dumps(check(args.dataset, args.source_root, args.limit,
                           args.yolo_weights, args.device), indent=2))


if __name__ == "__main__":
    main()
