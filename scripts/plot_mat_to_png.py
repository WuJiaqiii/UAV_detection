import os
import argparse
from pathlib import Path

import numpy as np
import cv2
from scipy.io import loadmat
from tqdm import tqdm


def spectrogram_to_yolo_uint8(
    data: np.ndarray,
    db_min: float = -80.0,
    db_max: float = 0.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Convert raw power/energy spectrogram to YOLO-style uint8 image.

    Logic:
        data_norm = data / max(data)
        data_db = 10 * log10(data_norm + eps)
        data_db = clip(data_db, db_min, db_max)
        out = (data_db - db_min) / (db_max - db_min)
        uint8 = out * 255
    """
    x = np.asarray(data, dtype=np.float32)

    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={x.shape}")

    if db_max <= db_min:
        raise ValueError(f"Invalid db range: db_min={db_min}, db_max={db_max}")

    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.uint8)

    x = np.maximum(x, 0.0)
    ref = float(np.max(x[finite]))

    if (not np.isfinite(ref)) or ref <= 0:
        return np.zeros_like(x, dtype=np.uint8)

    x_norm = x / (ref + float(eps))
    x_db = 10.0 * np.log10(x_norm + float(eps))

    x_db = np.nan_to_num(
        x_db,
        nan=float(db_min),
        posinf=float(db_max),
        neginf=float(db_min),
    )

    x_db = np.clip(x_db, float(db_min), float(db_max))

    out = (x_db - float(db_min)) / (float(db_max) - float(db_min))
    out = np.clip(out, 0.0, 1.0)

    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def collect_mat_files(input_dir: Path, recursive: bool = False):
    if recursive:
        return sorted(input_dir.rglob("*.mat"))
    return sorted(input_dir.glob("*.mat"))


def convert_one_mat(
    mat_path: Path,
    input_dir: Path,
    output_dir: Path,
    mat_key: str,
    db_min: float,
    db_max: float,
    eps: float,
    save_rgb: bool,
    preserve_tree: bool,
):
    mat_data = loadmat(str(mat_path), variable_names=[mat_key])

    if mat_key not in mat_data:
        raise KeyError(f"Key '{mat_key}' not found in {mat_path.name}")

    arr = np.asarray(mat_data[mat_key])

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={arr.shape}")

    img_u8 = spectrogram_to_yolo_uint8(
        arr,
        db_min=db_min,
        db_max=db_max,
        eps=eps,
    )

    if preserve_tree:
        rel = mat_path.relative_to(input_dir)
        output_path = output_dir / rel.with_suffix(".png")
    else:
        output_path = output_dir / f"{mat_path.stem}.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if save_rgb:
        img_to_save = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    else:
        img_to_save = img_u8

    ok = cv2.imwrite(str(output_path), img_to_save)

    if not ok:
        raise RuntimeError(f"Failed to save image: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert .mat spectrogram files to YOLO PNG images using fixed dB clipping."
    )

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--mat_key",
        type=str,
        default="summed_submatrices",
        help="mat 文件中的矩阵 key",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="是否递归读取子目录中的 mat 文件",
    )

    parser.add_argument(
        "--preserve_tree",
        action="store_true",
        help="是否保留输入目录的子目录结构",
    )

    parser.add_argument(
        "--db_min",
        type=float,
        default=-80.0,
        help="固定 dB 裁剪下界",
    )

    parser.add_argument(
        "--db_max",
        type=float,
        default=0.0,
        help="固定 dB 裁剪上界。若先按每张图最大值归一化，0 dB 表示当前图最大值",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="避免 log10(0) 的极小值",
    )

    parser.add_argument(
        "--save_rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否保存三通道 PNG，推荐开启以贴近 YOLO 输入",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir not found or not a directory: {input_dir}")

    if args.db_max <= args.db_min:
        raise ValueError(f"Invalid db range: db_min={args.db_min}, db_max={args.db_max}")

    output_dir.mkdir(parents=True, exist_ok=True)

    mat_files = collect_mat_files(input_dir, recursive=args.recursive)

    if len(mat_files) == 0:
        print(f"[WARN] No .mat files found in: {input_dir}")
        return

    print(f"[INFO] Found {len(mat_files)} mat files")
    print(f"[INFO] Input dir: {input_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] mat_key: {args.mat_key}")
    print(f"[INFO] db_range: [{args.db_min}, {args.db_max}]")
    print(f"[INFO] eps: {args.eps}")
    print(f"[INFO] save_rgb: {args.save_rgb}")
    print(f"[INFO] preserve_tree: {args.preserve_tree}")

    failed = []

    for mat_path in tqdm(mat_files, desc="Converting"):
        try:
            convert_one_mat(
                mat_path=mat_path,
                input_dir=input_dir,
                output_dir=output_dir,
                mat_key=args.mat_key,
                db_min=args.db_min,
                db_max=args.db_max,
                eps=args.eps,
                save_rgb=args.save_rgb,
                preserve_tree=args.preserve_tree,
            )
        except Exception as e:
            failed.append((str(mat_path), str(e)))
            print(f"\n[FAILED] {mat_path.name}: {e}")

    print(f"\n[DONE] Converted {len(mat_files) - len(failed)} / {len(mat_files)} files")

    if failed:
        failed_log = output_dir / "failed_files.txt"
        with open(failed_log, "w", encoding="utf-8") as f:
            for fp, err in failed:
                f.write(f"{fp}\t{err}\n")
        print(f"[WARN] Failed files saved to: {failed_log}")


if __name__ == "__main__":
    main()