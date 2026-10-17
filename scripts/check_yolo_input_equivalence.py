#!/usr/bin/env python3
"""Verify native PNGs, actual dataset adapters, and optionally real YOLO tensors."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from util.spectrogram_frontend import (
    FrontendConfig, FRONTEND_VERSION, SPEC_SHAPE, read_mat_spectrum,
    render_rgb, rgb_to_tensor_array, prepare_yolo_image, spectrogram_contract,
)
from data.iq_dat_loader import read_iq_dat, iq_to_spectrogram


def restore(root, row, source):
    """Use exactly the adapters used in framework datasets, not an alternate FFT."""
    path = root / row["source"]
    if source["input_type"] == "iq_dat":
        seg = row["segment"]
        iq = read_iq_dat(path, seg["start_sample"], seg["num_samples"],
                         byte_order=seg["byte_order"])
        return iq_to_spectrogram(iq, sample_rate=source["iq_sample_rate"])
    if source["input_type"] == "mat":
        return read_mat_spectrum(path, source["mat_key"], source["mat_spec_domain"])
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Cannot read " + str(path))
    return image


def check(dataset, source_root, limit=20, weights=None, device="cpu"):
    manifest = json.loads((dataset / "frontend_config.json").read_text())
    if manifest.get("version") != FRONTEND_VERSION or manifest.get("spectrogram") != spectrogram_contract():
        raise ValueError("Regenerate images with the v2 native-amplitude frontend")
    cfg = FrontendConfig(**manifest["frontend"])
    source = manifest["source"]
    rows = [json.loads(line) for line in (dataset / "metadata.jsonl").read_text().splitlines()]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No exported images to check")
    model, loaders = None, {}
    if weights:
        from util.detector import load_yolo_model
        from util.yolo_dataset_contract import install_yolo_dataset_contract
        model = load_yolo_model(weights, device)
        model.float().eval()
        model.amp = False
        yolo_module = install_yolo_dataset_contract()
    key, matrix = None, None
    for row in rows:
        if row["frontend"] != cfg.fingerprint():
            raise AssertionError("Mixed frontend versions")
        current = (row["source"], json.dumps(row["segment"], sort_keys=True))
        if current != key:
            matrix = restore(source_root, row, source)
            key = current
        rgb = render_rgb(matrix, cfg, row["domain"])
        path = dataset / row["image"]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        loaded = np.ascontiguousarray(bgr[:, :, ::-1])
        if loaded.shape[:2] != SPEC_SHAPE:
            raise AssertionError("Annotation PNG must be exactly 512x750")
        np.testing.assert_array_equal(loaded, rgb)
        assert hashlib.sha256(loaded.tobytes()).hexdigest() == row["pixel_sha256"]
        if list(matrix.shape) != row["matrix_shape"] or list(loaded.shape[:2]) != row["png_shape"]:
            raise AssertionError("Matrix/PNG shape metadata mismatch")
        runtime = prepare_yolo_image(rgb, cfg)
        from_png = prepare_yolo_image(loaded, cfg)
        np.testing.assert_array_equal(runtime.image, from_png.image)
        expected = rgb_to_tensor_array(runtime.image)
        box = np.array([[0, 0, 750, 512]], np.float32)
        np.testing.assert_allclose(runtime.to_source(runtime.from_source(box)), box, atol=1e-4)
        if model is not None:
            split = row["split"]
            if split not in loaders:
                loaders[split] = yolo_module.LoadImagesAndLabels(
                    str(dataset/"images"/split), img_size=cfg.image_size,
                    batch_size=1, augment=False, rect=False, stride=32,
                )
            loader = loaders[split]
            indices = [i for i, name in enumerate(loader.im_files)
                       if Path(name).resolve() == path.resolve()]
            if len(indices) != 1:
                raise AssertionError("Missing PNG in actual YOLO training loader")
            training_image = loader[indices[0]][0].float().numpy()/255.0
            np.testing.assert_array_equal(training_image, expected)
            captured = []
            handle = model.model.register_forward_pre_hook(
                lambda module, args: captured.append(args[0].detach().cpu().float().numpy().copy()))
            try:
                import torch
                with torch.inference_mode():
                    a = model(runtime.image, size=cfg.image_size).xyxy[0].cpu().numpy()
                    b = model(from_png.image, size=cfg.image_size).xyxy[0].cpu().numpy()
            finally:
                handle.remove()
            if len(captured) != 2:
                raise AssertionError("Unexpected AutoShape backend call count")
            np.testing.assert_array_equal(captured[0], expected[None])
            np.testing.assert_array_equal(captured[1], expected[None])
            np.testing.assert_allclose(a, b, atol=1e-4, rtol=1e-5)
    return {"checked": len(rows), "native_png_shape": [512, 750],
            "native_pixels_equal": True, "model_geometry_equal": True,
            "actual_yolo_tensor_and_predictions": "passed" if weights else "not_run_no_weights",
            "actual_training_loader": "passed_shared_contract" if weights else "not_run_no_weights"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--source_root", type=Path, required=True)
    p.add_argument("--limit", type=int, default=20, help="0 checks all images")
    p.add_argument("--yolo_weights", type=Path)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.limit < 0:
        p.error("--limit must be nonnegative")
    print(json.dumps(check(args.dataset, args.source_root, args.limit,
                           args.yolo_weights, args.device), indent=2))


if __name__ == "__main__":
    main()
