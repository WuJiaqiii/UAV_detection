#!/usr/bin/env python3
"""Export deterministic, lossless detector inputs from IQ, MAT or grayscale PNG.

No labels are invented. Run --finalize after manual annotation to validate every
label file and write data.yaml. Split original recordings BEFORE slicing.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from scipy.io import loadmat
from util.spectrogram_frontend import (
    FRONTEND_VERSION, FrontendConfig, add_frontend_arguments,
    apply_frontend_manifest, iq_to_db, render_tiles, read_mat_matrix,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", type=Path)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--input_type", choices=["iq_dat", "mat", "png"], default="iq_dat")
    p.add_argument("--iq_sample_rate", type=int, default=100_000_000)
    p.add_argument("--iq_segment_ms", type=float, default=100.0)
    p.add_argument("--iq_n_fft", type=int, default=512)
    p.add_argument("--iq_hop_length", type=int, default=256)
    p.add_argument("--iq_byte_order", choices=["<", ">"], default="<")
    p.add_argument("--segments_per_file", type=int, default=3)
    p.add_argument("--mat_key", default="summed_submatrices")
    p.add_argument("--split_mode", choices=["group", "none"], default="group",
                   help="group: split independent source groups; none: export unassigned annotation/preview images")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--groups_json", type=Path,
                   help="Optional JSON: relative source path -> acquisition/session ID")
    p.add_argument("--finalize", action="store_true")
    p.add_argument("--yolo_imgsz_h", type=int, default=640)
    p.add_argument("--yolo_imgsz_w", type=int, default=640)
    add_frontend_arguments(p)
    return p


def source_group(path, root, groups):
    rel = path.relative_to(root).as_posix()
    if groups is not None:
        if rel not in groups:
            raise ValueError("Missing acquisition group for " + rel)
        return str(groups[rel])
    # Group Figure slices and SNR variants together where filenames carry them.
    stem = re.sub(r"-Figure-\d+$", "", path.stem.split("-SNR-")[0])
    return str(path.parent.relative_to(root) / stem)


def source_matrices(path, args):
    if args.input_type == "iq_dat":
        # Same byte reader as RawIQDataset; no alternate IQ conversion.
        from data.iq_dat_loader import read_iq_dat
        size = path.stat().st_size
        if size % 4:
            raise ValueError("Incomplete int16 IQ pair: " + str(path))
        count = round(args.iq_sample_rate * args.iq_segment_ms / 1000)
        available = size // 4
        if count < args.iq_n_fft or available < count:
            raise ValueError("Recording shorter than configured segment: " + str(path))
        starts = ([(available-count)//2] if args.segments_per_file == 1 else
                  sorted(set(np.rint(np.linspace(
                      0, available-count, args.segments_per_file)).astype(np.int64).tolist())))
        for start in starts:
            iq = read_iq_dat(path, start, count, byte_order=args.iq_byte_order)
            yield iq_to_db(iq, args.iq_n_fft, args.iq_hop_length), "db", {
                "start_sample": start, "num_samples": count,
                "byte_order": args.iq_byte_order,
            }
    elif args.input_type == "mat":
        yield read_mat_matrix(path, args.mat_key, args.mat_spec_domain), args.mat_spec_domain, {}
    else:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Cannot read " + str(path))
        yield image, "image", {}


def finalize(output):
    meta = output / "metadata.jsonl"
    if not meta.is_file():
        raise FileNotFoundError(meta)
    counts = {"train": 0, "val": 0}
    for line in meta.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("split") not in counts:
            raise ValueError("Unassigned preview exports cannot be finalized as a training dataset. "
                             "Export with --split_mode group after collecting at least two independent groups.")
        image = output / row["image"]
        label = output / row["label"]
        if not image.is_file() or not label.is_file():
            raise ValueError("Every image needs an explicitly reviewed label file: " + str(label))
        for text in label.read_text(encoding="utf-8").splitlines():
            fields = text.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError("Expected YOLO class xc yc w h: " + str(label))
            cls, xc, yc, w, h = map(float, fields)
            if cls != 0 or not np.isfinite([xc, yc, w, h]).all():
                raise ValueError("Only class 0 (signal_frame) and finite labels are supported")
            if not (w > 0 and h > 0 and xc-w/2 >= -1e-6 and yc-h/2 >= -1e-6
                    and xc+w/2 <= 1+1e-6 and yc+h/2 <= 1+1e-6):
                raise ValueError("Label outside image: " + str(label))
        counts[row["split"]] += 1
    if not all(counts.values()):
        raise ValueError("Need nonempty train and val splits")
    (output / "data.yaml").write_text(
        "path: " + json.dumps(str(output.resolve())) + "\n"
        "train: images/train\nval: images/val\nnc: 1\nnames: ['signal_frame']\n",
        encoding="utf-8",
    )
    print(json.dumps({"reviewed_images": counts, "data_yaml": str(output / "data.yaml")}))


def export(args):
    apply_frontend_manifest(args)
    cfg = FrontendConfig.from_config(args)
    root = args.input_dir.resolve()
    output = args.output_dir.resolve()
    if output == root or root in output.parents:
        raise ValueError("Output must be outside the source tree")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory; existing annotations are never overwritten")
    if args.segments_per_file < 1:
        raise ValueError("Need segments_per_file >= 1")
    if args.split_mode == "group" and not 0 < args.val_ratio < 1:
        raise ValueError("Group splitting requires 0 < val_ratio < 1")
    if args.iq_sample_rate <= 0 or args.iq_segment_ms <= 0:
        raise ValueError("Invalid IQ sample rate or segment duration")
    suffix = {"iq_dat": ".dat", "mat": ".mat", "png": ".png"}[args.input_type]
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == suffix)
    if not paths:
        raise FileNotFoundError("No source files under " + str(root))
    groups = json.loads(args.groups_json.read_text()) if args.groups_json else None
    group_ids = {p: source_group(p, root, groups) for p in paths}
    unique = sorted(set(group_ids.values()))
    print(f"Found {len(paths)} source file(s) in {len(unique)} independent group(s).")
    if args.split_mode == "group":
        if len(unique) < 2:
            raise ValueError(
                f"Found {len(paths)} source file(s), but only {len(unique)} independent group(s). "
                "Figure slices and SNR variants of the same source are grouped together. "
                "Use --split_mode none to export preview/annotation images without train/val splitting. "
                "For training, provide at least two genuinely independent recording groups."
            )
        shuffled = np.random.default_rng(args.seed).permutation(unique)
        n_val = min(len(unique)-1, max(1, round(len(unique)*args.val_ratio)))
        val_groups = set(shuffled[:n_val].tolist())
        splits = ("train", "val")
    else:
        val_groups = set()
        splits = ("unassigned",)
    for split in splits:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
    source = {name: getattr(args, name) for name in (
        "input_type", "iq_sample_rate", "iq_segment_ms", "iq_n_fft",
        "iq_hop_length", "mat_key", "mat_spec_domain", "frequency_axis",
    )}
    source["iq_byte_order"] = args.iq_byte_order
    source["segments_per_file"] = args.segments_per_file
    source["frequency_axis"] = "fft" if args.input_type == "iq_dat" else args.frequency_axis
    manifest = {"version": FRONTEND_VERSION, "frontend": asdict(cfg),
                "source": source, "seed": args.seed, "split_mode": args.split_mode}
    (output / "frontend_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output / "metadata.jsonl").open("w", encoding="utf-8") as metadata:
        for path in paths:
            rel = path.relative_to(root).as_posix()
            split = ("unassigned" if args.split_mode == "none" else
                     ("val" if group_ids[path] in val_groups else "train"))
            for matrix, domain, segment in source_matrices(path, args):
                for tile in render_tiles(matrix, cfg, domain):
                    identity = json.dumps([rel, segment, tile.x0, cfg.fingerprint()], sort_keys=True)
                    name = hashlib.sha256(identity.encode()).hexdigest()[:24] + ".png"
                    image_rel = (Path("images") / split / name).as_posix()
                    label_rel = (Path("labels") / split / Path(name).with_suffix(".txt")).as_posix()
                    if not cv2.imwrite(str(output / image_rel), tile.image[:, :, ::-1]):
                        raise IOError("PNG write failed: " + image_rel)
                    row = {"image": image_rel, "label": label_rel, "split": split,
                           "source": rel, "group": group_ids[path], "segment": segment,
                           "domain": domain, "matrix_shape": list(matrix.shape),
                           "transform": tile.metadata(), "frontend": cfg.fingerprint(),
                           "pixel_sha256": hashlib.sha256(tile.image.tobytes()).hexdigest()}
                    metadata.write(json.dumps(row) + "\n")
    if args.split_mode == "none":
        print("Preview/annotation images exported under images/unassigned; no training split was created.")
    else:
        print("Images exported. Annotate every image, then run this command with --finalize.")
    print("No data.yaml or implicit background labels were created.")


def main():
    args = parser().parse_args()
    if args.finalize:
        finalize(args.output_dir)
    else:
        if args.input_dir is None:
            raise ValueError("--input_dir is required for export")
        export(args)


if __name__ == "__main__":
    main()
