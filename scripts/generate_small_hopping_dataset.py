#!/usr/bin/env python3
"""Generate labeled thin/small hopping bursts on the shared native amplitude grid.

This is spectrogram-domain augmentation, not an IQ waveform/channel simulator.
Every emitted pulse is labeled; absent pulses and negative scenes have no boxes.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import gaussian_filter1d

from util.spectrogram_frontend import (
    FRONTEND_VERSION, SPEC_SHAPE, FrontendConfig, add_frontend_arguments,
    apply_frontend_manifest, canonical_amplitude, prepare_yolo_image,
    read_mat_spectrum, render_rgb, spectrogram_contract,
)

GENERATOR_VERSION = "small-hopping-v1"
H, W = SPEC_SHAPE


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--num_images", type=int, default=2000)
    p.add_argument("--variants_per_scene", type=int, default=1)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--negative_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--thin_probability", type=float, default=0.55)
    p.add_argument("--small_probability", type=float, default=0.35)
    p.add_argument("--thin_width", type=int, nargs=2, default=[20, 70], metavar=("MIN", "MAX"))
    p.add_argument("--thin_height", type=int, nargs=2, default=[2, 4], metavar=("MIN", "MAX"))
    p.add_argument("--small_width", type=int, nargs=2, default=[3, 12], metavar=("MIN", "MAX"))
    p.add_argument("--small_height", type=int, nargs=2, default=[2, 8], metavar=("MIN", "MAX"))
    p.add_argument("--regular_width", type=int, nargs=2, default=[12, 35], metavar=("MIN", "MAX"))
    p.add_argument("--regular_height", type=int, nargs=2, default=[5, 16], metavar=("MIN", "MAX"))
    p.add_argument("--signals_per_scene", type=int, nargs=2, default=[1, 2])
    p.add_argument("--frames_per_signal", type=int, nargs=2, default=[4, 16])
    p.add_argument("--burst_length", type=int, nargs=2, default=[1, 4])
    p.add_argument("--pattern", choices=["mixed", "staircase", "random", "pingpong", "bursts"], default="mixed")
    p.add_argument("--dropout_probability", type=float, default=0.1,
                   help="Simulated absent transmissions: remove BOTH signal and label, never leave unlabeled pulses.")
    p.add_argument("--contrast_db", type=float, nargs=2, default=[6., 26.],
                   help="Added amplitude relative to local median, NOT calibrated IQ SNR.")
    p.add_argument("--weak_probability", type=float, default=0.25)
    p.add_argument("--weak_contrast_db", type=float, nargs=2, default=[2., 8.])
    p.add_argument("--edge_probability", type=float, default=0.15)
    p.add_argument("--background_train_dir", type=Path,
                   help="Optional confirmed signal-free MAT sources, used only by training scenes.")
    p.add_argument("--background_val_dir", type=Path,
                   help="Separate confirmed signal-free MAT sources for validation; both directories required.")
    p.add_argument("--mat_key", default="summed_submatrices")
    p.add_argument("--background_jitter", type=float, default=0.03,
                   help="Log-amplitude noise sigma added to real MAT backgrounds per variant; 0 preserves them.")
    p.add_argument("--imgsz", type=int, default=None,
                   help="Network size in manifest; default 640 or the supplied frontend manifest's size.")
    p.add_argument("--save_amplitudes", action="store_true",
                   help="Save float32 .npy sources for exact rerender verification (about 1.5 MB/image).")
    p.add_argument("--preview_count", type=int, default=8)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--verify_only", action="store_true")
    p.add_argument("--rewrite_data_yaml", action="store_true",
                   help="With --verify_only, update absolute dataset path after moving the dataset.")
    add_frontend_arguments(p)
    return p


def _range_valid(name, values, minimum=1, maximum=None):
    if not np.isfinite(values).all() or values[0] < minimum or values[1] < values[0]:
        raise ValueError(f"Invalid {name}: {values}")
    if maximum is not None and values[1] > maximum:
        raise ValueError(f"{name} must not exceed {maximum}")


def validate_args(args):
    if args.seed < 0 or args.variants_per_scene < 1 or args.num_images % args.variants_per_scene:
        raise ValueError("Use a nonnegative seed and num_images divisible by positive variants_per_scene")
    if args.num_images // args.variants_per_scene < 2:
        raise ValueError("At least two independent scenes are required")
    for name in ("val_ratio", "negative_ratio", "thin_probability", "small_probability",
                 "weak_probability", "edge_probability", "dropout_probability"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    if not 0 < args.val_ratio < 1 or args.negative_ratio >= 1 or args.dropout_probability >= 1:
        raise ValueError("Need 0 < val_ratio < 1, negative_ratio < 1 and dropout_probability < 1")
    if args.thin_probability + args.small_probability > 1:
        raise ValueError("thin_probability + small_probability must be <= 1")
    for kind in ("thin", "small", "regular"):
        _range_valid(kind + "_width", getattr(args, kind + "_width"), maximum=W)
        _range_valid(kind + "_height", getattr(args, kind + "_height"), maximum=H)
    for name in ("signals_per_scene", "frames_per_signal", "burst_length"):
        _range_valid(name, getattr(args, name), maximum=100)
    for name in ("contrast_db", "weak_contrast_db"):
        _range_valid(name, getattr(args, name), minimum=-20, maximum=60)
    if args.preview_count < 0:
        raise ValueError("preview_count must be nonnegative")
    if not np.isfinite(args.background_jitter) or not 0 <= args.background_jitter <= 0.5:
        raise ValueError("background_jitter must be in [0,0.5]")
    if bool(args.background_train_dir) != bool(args.background_val_dir):
        raise ValueError("Provide both background_train_dir and background_val_dir")
    if args.mat_spec_domain == "image" and args.background_train_dir:
        raise ValueError("Background MATs must be amplitude/power/db, not rendered image values")


def _int(rng, values):
    return int(rng.integers(values[0], values[1] + 1))


def _background_sources(args, output):
    pools = {"train": [], "val": []}
    seen = {}
    for split in pools:
        directory = getattr(args, f"background_{split}_dir")
        if directory is None:
            continue
        directory = directory.resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        if output == directory or directory in output.parents or output in directory.parents:
            raise ValueError("Background source and output directories must be separate")
        paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() == ".mat")
        if not paths:
            raise FileNotFoundError("No MAT backgrounds in " + str(directory))
        for path in paths:
            # Validate every background before writing; reject exact cross-split duplicates.
            matrix = read_mat_spectrum(path, args.mat_key, args.mat_spec_domain)
            identity = hashlib.sha256(matrix.tobytes()).hexdigest()
            if identity in seen and seen[identity] != split:
                raise ValueError("Same background matrix appears in train and val")
            seen[identity] = split
            pools[split].append(path)
    return pools


def synthetic_background(rng):
    # Amplitude-domain receiver-like noise; no calibrated RF SNR is implied.
    frequency = (np.arange(H) - H // 2) * 0.24
    cutoff = rng.uniform(44, 51)
    envelope = 0.008 + 0.992 / (1 + np.exp((np.abs(frequency) - cutoff) / rng.uniform(1.0, 2.5)))
    row_gain = np.exp(gaussian_filter1d(rng.normal(0, 0.7, H), 10))
    time_gain = np.exp(gaussian_filter1d(rng.normal(0, 0.5, W), 20))
    noise = rng.rayleigh(1., SPEC_SHAPE).astype(np.float32)
    return noise * (envelope * row_gain)[:, None] * time_gain[None, :]


def scene_boxes(rng, args):
    boxes, kinds, tracks = [], [], []
    probabilities = [args.thin_probability, args.small_probability,
                     max(0., 1 - args.thin_probability - args.small_probability)]
    for track_id in range(_int(rng, args.signals_per_scene)):
        kind = str(rng.choice(["thin", "small", "regular"], p=probabilities))
        width = _int(rng, getattr(args, kind + "_width"))
        height = _int(rng, getattr(args, kind + "_height"))
        requested = _int(rng, args.frames_per_signal)
        pattern = (str(rng.choice(["staircase", "random", "pingpong", "bursts"]))
                   if args.pattern == "mixed" else args.pattern)
        channels = int(rng.integers(3, 9))
        levels = np.linspace(max(height / 2, rng.uniform(55, 150)),
                             min(H - height / 2, rng.uniform(360, 455)), channels)
        if rng.random() < .5:
            levels = levels[::-1]
        period = max(width * 1.3, W / requested * rng.uniform(.85, 1.15))
        start = rng.uniform(0, max(1., period * .45))
        allow_edges = rng.random() < args.edge_probability
        if allow_edges:
            start = -width * rng.uniform(.1, .7)
        hold = _int(rng, args.burst_length)
        candidates = []
        for i in range(requested):
            if pattern == "random":
                channel = int(rng.integers(channels))
            elif pattern == "pingpong":
                phase = i % (2 * channels - 2)
                channel = min(phase, 2 * channels - 2 - phase)
            else:
                channel = (i // hold if pattern == "bursts" else i) % channels
            # Keep requested native size ranges; jitter within those bounds.
            width_range, height_range = getattr(args, kind + "_width"), getattr(args, kind + "_height")
            bw = int(np.clip(round(width * rng.uniform(.9, 1.1)), *width_range))
            bh = int(np.clip(round(height * rng.uniform(.9, 1.1)), *height_range))
            x1 = int(round(start + i * period + rng.uniform(-.06, .06) * period))
            y1 = int(round(levels[channel] - bh / 2 + rng.uniform(-.3, .3)))
            if i == 0 and allow_edges:
                y1 = -max(0, bh // 3) if rng.random() < .5 else H - max(1, bh // 2)
            if not allow_edges and (x1 < 0 or x1 + bw > W or y1 < 0 or y1 + bh > H):
                continue
            x2, y2 = min(W, x1 + bw), min(H, y1 + bh)
            x1, y1 = max(0, x1), max(0, y1)
            if x2 <= x1 or y2 <= y1:
                continue
            candidate = [x1, y1, x2, y2]
            if any(min(x2, b[2]) > max(x1, b[0]) and min(y2, b[3]) > max(y1, b[1])
                   for b in boxes):
                continue
            candidates.append(candidate)
        if not candidates and not boxes:
            candidates = [[(W-width)//2, (H-height)//2, (W-width)//2+width, (H-height)//2+height]]
        emitted = [box for box in candidates if rng.random() >= args.dropout_probability]
        if not emitted and candidates:
            emitted = candidates[:1]
        for box in emitted:
            boxes.append(box)
            kinds.append(kind)
            tracks.append({"track": track_id, "pattern": pattern})
    if not boxes:
        raise RuntimeError("Scene placement produced no positive frames; reduce size/density constraints")
    return boxes, kinds, tracks


def inject_frames(background, boxes, rng, args):
    matrix = np.asarray(background, dtype=np.float32).copy()
    global_floor = max(float(np.median(matrix)), 1e-6)
    strengths = []
    for x1, y1, x2, y2 in boxes:
        local = matrix[y1:y2, x1:x2]
        weak = rng.random() < args.weak_probability
        bounds = args.weak_contrast_db if weak else args.contrast_db
        contrast = float(rng.uniform(*bounds))
        floor = max(float(np.median(local)), global_floor * .25)
        yy = np.sin(np.pi * (np.arange(y2-y1) + .5) / (y2-y1)) ** .3
        xx = np.sin(np.pi * (np.arange(x2-x1) + .5) / (x2-x1)) ** .3
        texture = rng.uniform(.8, 1.2, local.shape).astype(np.float32)
        local += floor * 10 ** (contrast / 20) * yy[:, None] * xx[None, :] * texture
        strengths.append({"relative_added_amplitude_db": contrast, "weak": bool(weak)})
    return canonical_amplitude(matrix), strengths


def labels_for_boxes(boxes):
    return "".join(f"0 {(x1+x2)/2/W:.9f} {(y1+y2)/2/H:.9f} {(x2-x1)/W:.9f} {(y2-y1)/H:.9f}\n"
                   for x1, y1, x2, y2 in boxes)


def write_data_yaml(output):
    (output / "data.yaml").write_text(
        "path: " + json.dumps(output.resolve().as_posix()) + "\n"
        "train: images/train\nval: images/val\nnc: 1\nnames: ['signal_frame']\n",
        encoding="utf-8",
    )


def _preview(rgb, boxes, title):
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2-1, y2-1], outline=(255, 80, 40), width=1)
    image = ImageOps.expand(image, border=(0, 22, 0, 0), fill="black")
    ImageDraw.Draw(image).text((5, 4), title, fill="white")
    return image


def generate(args):
    validate_args(args)
    args.yolo_imgsz_h = args.yolo_imgsz_w = 640
    apply_frontend_manifest(args)
    if args.imgsz is not None:
        args.yolo_imgsz_h = args.yolo_imgsz_w = args.imgsz
    cfg = FrontendConfig.from_config(args)
    output = args.output_dir.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use an empty output directory; existing data is never overwritten")
    backgrounds = _background_sources(args, output)
    scenes = args.num_images // args.variants_per_scene
    split_rng = np.random.default_rng(args.seed)
    shuffled = split_rng.permutation(scenes)
    n_val = min(scenes - 1, max(1, round(scenes * args.val_ratio)))
    val_ids = set(shuffled[:n_val].tolist())
    negatives = set()
    for ids in (shuffled[:n_val], shuffled[n_val:]):
        # Keep at least one positive independent scene in both splits.
        n_negative = min(len(ids)-1, round(len(ids) * args.negative_ratio))
        negatives.update(split_rng.permutation(ids)[:n_negative].tolist())
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    if args.preview_count:
        (output / "previews").mkdir()
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest = {
        "version": FRONTEND_VERSION, "spectrogram": spectrogram_contract(), "frontend": asdict(cfg),
        "source": {"input_type": "synthetic_amplitude", "generator": GENERATOR_VERSION,
                   "simulation_domain": "spectrogram_amplitude", "not_iq_simulation": True},
        "split_mode": "synthetic_scene_group", "generator_settings": settings,
    }
    (output / "frontend_config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    previews = []
    with (output / "metadata.jsonl").open("w", encoding="utf-8") as metadata:
        for scene_id in range(scenes):
            split = "val" if scene_id in val_ids else "train"
            scene_rng = np.random.default_rng(np.random.SeedSequence([args.seed, scene_id, 0]))
            negative = scene_id in negatives
            boxes, kinds, tracks = ([], [], []) if negative else scene_boxes(scene_rng, args)
            background_path = (backgrounds[split][int(scene_rng.integers(len(backgrounds[split])))]
                               if backgrounds[split] else None)
            background = (read_mat_spectrum(background_path, args.mat_key, args.mat_spec_domain)
                          if background_path else None)
            for variant in range(args.variants_per_scene):
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, scene_id, variant, 1]))
                raw_background = synthetic_background(rng) if background is None else background
                if background is not None and args.background_jitter:
                    raw_background = background * rng.lognormal(0., args.background_jitter, SPEC_SHAPE)
                matrix, strengths = inject_frames(raw_background, boxes, rng, args)
                rgb = render_rgb(matrix, cfg, "amplitude")
                name = f"scene_{scene_id:06d}_v{variant:02d}"
                image_rel = f"images/{split}/{name}.png"
                label_rel = f"labels/{split}/{name}.txt"
                Image.fromarray(rgb).save(output / image_rel)
                labels = labels_for_boxes(boxes)
                (output / label_rel).write_text(labels, encoding="utf-8")
                prepared = prepare_yolo_image(rgb, cfg)
                model_boxes = prepared.from_source(np.asarray(boxes, dtype=np.float32).reshape(-1, 4))
                tiny = int(np.count_nonzero(np.minimum(model_boxes[:, 2]-model_boxes[:, 0],
                                                       model_boxes[:, 3]-model_boxes[:, 1]) < 2))
                row = {
                    "image": image_rel, "label": label_rel, "split": split,
                    "group": f"synthetic_{args.seed}_{scene_id:06d}", "variant": variant,
                    "background_source": str(background_path) if background_path else None,
                    "synthetic": True, "domain": "amplitude", "png_shape": list(SPEC_SHAPE),
                    "frontend": cfg.fingerprint(), "pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                    "boxes_xyxy": boxes, "size_kinds": kinds, "tracks": tracks,
                    "strengths": strengths, "negative": negative, "model_min_side_below_2px": tiny,
                }
                if args.save_amplitudes:
                    amplitude_rel = f"amplitudes/{split}/{name}.npy"
                    (output / amplitude_rel).parent.mkdir(parents=True, exist_ok=True)
                    np.save(output / amplitude_rel, matrix, allow_pickle=False)
                    row["amplitude"] = amplitude_rel
                metadata.write(json.dumps(row, ensure_ascii=False) + "\n")
                if len(previews) < args.preview_count and variant == 0:
                    preview = _preview(rgb, boxes, f"{split}/{name} | {len(boxes)} frames | {','.join(sorted(set(kinds))) or 'negative'}")
                    preview.save(output / "previews" / f"{name}.png")
                    previews.append(preview.resize((375, 267)))
                written = scene_id * args.variants_per_scene + variant + 1
                if not args.quiet and (written % 100 == 0 or written == args.num_images):
                    print(f"Generated {written}/{args.num_images} images", file=sys.stderr, flush=True)
    if previews:
        grid = Image.new("RGB", (750, 267 * ((len(previews)+1)//2)), "black")
        for i, preview in enumerate(previews):
            grid.paste(preview, ((i % 2) * 375, (i // 2) * 267))
        grid.save(output / "previews" / "contact_sheet.jpg", quality=92)
    summary = verify_dataset(output)
    # Publish the training entry only after every generated sample validates.
    write_data_yaml(output)
    (output / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def verify_dataset(output):
    output = Path(output)
    manifest = json.loads((output / "frontend_config.json").read_text(encoding="utf-8"))
    if manifest.get("version") != FRONTEND_VERSION or manifest.get("spectrogram") != spectrogram_contract():
        raise ValueError("Incompatible frontend contract")
    if manifest.get("source", {}).get("generator") != GENERATOR_VERSION:
        raise ValueError("This verifier expects a small-hopping generator dataset")
    cfg = FrontendConfig(**manifest["frontend"])
    counts = {split: {"images": 0, "negative_images": 0, "boxes": 0} for split in ("train", "val")}
    group_splits, images, cross_hashes = {}, set(), {}
    kinds = {"thin": 0, "small": 0, "regular": 0}
    tiny, amplitudes = 0, 0
    sizes = []
    for line in (output / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        split = row["split"]
        if split not in counts or row["image"] in images:
            raise ValueError("Invalid split or duplicate image entry")
        images.add(row["image"])
        if row["group"] in group_splits and group_splits[row["group"]] != split:
            raise ValueError("A scene group occurs in both train and val")
        group_splits[row["group"]] = split
        for key in ("image", "label", "amplitude"):
            if key in row and not (output / row[key]).resolve().is_relative_to(output.resolve()):
                raise ValueError("Dataset path must remain inside output directory")
        with Image.open(output / row["image"]) as image:
            rgb = np.array(image)
        if rgb.shape != (H, W, 3) or rgb.dtype != np.uint8 or not np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) or not np.array_equal(rgb[:, :, 1], rgb[:, :, 2]):
            raise ValueError("Expected native 512x750 equal-channel RGB PNG")
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        if digest != row["pixel_sha256"] or row["frontend"] != cfg.fingerprint():
            raise ValueError("PNG hash or frontend fingerprint mismatch")
        if digest in cross_hashes and cross_hashes[digest] != split:
            raise ValueError("Identical image pixels occur in train and val")
        cross_hashes[digest] = split
        boxes = row["boxes_xyxy"]
        if any(len(row[key]) != len(boxes) for key in ("size_kinds", "tracks", "strengths")):
            raise ValueError("Per-frame metadata does not match bounding box count")
        for box in boxes:
            if len(box) != 4 or any(int(v) != v for v in box) or not (0 <= box[0] < box[2] <= W and 0 <= box[1] < box[3] <= H):
                raise ValueError("Invalid native bounding box")
            sizes.append([box[2]-box[0], box[3]-box[1]])
        actual = (output / row["label"]).read_text(encoding="utf-8")
        if actual != labels_for_boxes(boxes):
            raise ValueError("YOLO labels do not match emitted pulse boxes")
        if bool(boxes) == bool(row["negative"]):
            raise ValueError("Negative/positive image label mismatch")
        if "amplitude" in row:
            matrix = np.load(output / row["amplitude"], allow_pickle=False)
            if not np.array_equal(render_rgb(matrix, cfg, "amplitude"), rgb):
                raise ValueError("Saved amplitude does not reconstruct PNG exactly")
            amplitudes += 1
        counts[split]["images"] += 1
        counts[split]["negative_images"] += int(row["negative"])
        counts[split]["boxes"] += len(boxes)
        for kind in row["size_kinds"]:
            kinds[kind] += 1
        tiny += row["model_min_side_below_2px"]
    if not all(value["images"] for value in counts.values()):
        raise ValueError("Need nonempty train and val splits")
    return {"verified": True, "generator": GENERATOR_VERSION, "splits": counts,
            "size_kinds": kinds, "independent_scenes": len(group_splits),
            "network_image_size": cfg.image_size, "model_min_side_below_2px": tiny,
            "native_width_height_min": np.min(sizes, axis=0).tolist() if sizes else [],
            "native_width_height_max": np.max(sizes, axis=0).tolist() if sizes else [],
            "amplitude_reconstructions": amplitudes,
            "negative_ratio_note": "Scene-level rounded counts; each split retains a positive scene.",
            "not_calibrated_snr_or_iq_simulation": True}


def main():
    args = parser().parse_args()
    if args.rewrite_data_yaml and not args.verify_only:
        raise ValueError("--rewrite_data_yaml requires --verify_only")
    result = verify_dataset(args.output_dir) if args.verify_only else generate(args)
    if args.rewrite_data_yaml:
        write_data_yaml(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
