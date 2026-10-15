#!/usr/bin/env python3
"""Train YOLO using the native PNG/online geometry contract, without random augmentation."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from util.spectrogram_frontend import FRONTEND_VERSION, FrontendConfig, spectrogram_contract


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True,
                   help="Finalized export directory containing data.yaml and frontend_config.json")
    p.add_argument("--weights", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", default="yolov5/runs/train")
    p.add_argument("--name", default="uav_native_v2")
    args = p.parse_args()
    manifest = json.loads((args.dataset/"frontend_config.json").read_text())
    if manifest.get("version") != FRONTEND_VERSION or manifest.get("spectrogram") != spectrogram_contract():
        raise ValueError("Use a v2 512x750 dataset")
    data = args.dataset/"data.yaml"
    if not data.is_file():
        raise ValueError("Annotate images and run exporter --finalize before training")
    cfg = FrontendConfig(**manifest["frontend"])
    repo = ROOT/"yolov5"
    if not (repo/"train.py").is_file():
        raise FileNotFoundError("Initialize the actual YOLO submodule")
    sys.path.insert(0, str(repo))
    from util.yolo_dataset_contract import install_yolo_dataset_contract
    # Install before importing train/val, which bind create_dataloader locally.
    install_yolo_dataset_contract()
    import train as yolo_train
    # This baseline intentionally disables augmentation and rectangular batching.
    # Merely setting hsv/mosaic gains to zero does not disable all loader transforms.
    if not Path(args.weights).is_file():
        raise FileNotFoundError("Provide an existing initial YOLO weight file")
    yolo_train.run(data=str(data.resolve()), weights=args.weights, imgsz=cfg.image_size,
                   epochs=args.epochs, batch_size=args.batch_size, device=args.device,
                   workers=args.workers, project=args.project, name=args.name,
                   rect=False, multi_scale=False, cache=False)


if __name__ == "__main__":
    main()
