#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import cv2
from scipy.io import loadmat
from tqdm import tqdm


def mat_to_uint8_vis(
    mat_array: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.5,
    log_gain: float = 9.0,
) -> np.ndarray:
    """
    将 2D mat 矩阵转换为 uint8 灰度图。

    当前逻辑：
        1. percentile 裁剪，避免极端值影响显示；
        2. 归一化到 [0, 1]；
        3. log1p 映射，增强弱信号；
        4. 转换到 uint8 [0, 255]。
    """
    x = np.asarray(mat_array, dtype=np.float32)

    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={x.shape}")

    finite_mask = np.isfinite(x)
    if not finite_mask.any():
        return np.zeros_like(x, dtype=np.uint8)

    valid = x[finite_mask]

    lo = float(np.percentile(valid, p_low))
    hi = float(np.percentile(valid, p_high))

    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)

    # 1) 分位数裁剪
    x = np.clip(x, lo, hi)

    # 2) 归一化到 [0, 1]
    x = (x - lo) / (hi - lo)

    # 3) log 增强
    if log_gain > 0:
        x = np.log1p(float(log_gain) * x) / np.log1p(float(log_gain))

    # 4) 转 uint8
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x = np.clip(x * 255.0, 0.0, 255.0)

    return x.astype(np.uint8)


def collect_mat_files(input_dir: Path, recursive: bool = False):
    if recursive:
        return sorted(input_dir.rglob("*.mat"))
    return sorted(input_dir.glob("*.mat"))


def make_output_path(
    mat_path: Path,
    input_dir: Path,
    output_dir: Path,
    preserve_tree: bool,
) -> Path:
    if preserve_tree:
        rel = mat_path.relative_to(input_dir)
        return output_dir / rel.with_suffix(".png")
    return output_dir / f"{mat_path.stem}.png"


def convert_one_mat(
    mat_path: Path,
    input_dir: Path,
    output_dir: Path,
    mat_key: str,
    p_low: float,
    p_high: float,
    log_gain: float,
    save_rgb: bool,
    preserve_tree: bool,
):
    mat_data = loadmat(str(mat_path), variable_names=[mat_key])

    if mat_key not in mat_data:
        raise KeyError(f"Key '{mat_key}' not found in {mat_path.name}")

    arr = np.asarray(mat_data[mat_key])

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={arr.shape}")

    img_u8 = mat_to_uint8_vis(
        arr,
        p_low=p_low,
        p_high=p_high,
        log_gain=log_gain,
    )

    output_path = make_output_path(
        mat_path=mat_path,
        input_dir=input_dir,
        output_dir=output_dir,
        preserve_tree=preserve_tree,
    )
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
        description="Convert .mat spectrogram files to .png images using percentile clipping + log enhancement."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="输入 mat 文件目录",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出 png 图片目录",
    )

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
        "--p_low",
        type=float,
        default=1.0,
        help="低分位裁剪百分比",
    )

    parser.add_argument(
        "--p_high",
        type=float,
        default=99.5,
        help="高分位裁剪百分比",
    )

    parser.add_argument(
        "--log_gain",
        type=float,
        default=9.0,
        help="log 增强系数；设为 0 可关闭 log 映射",
    )

    parser.add_argument(
        "--save_rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否保存为三通道 PNG",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir not found or not a directory: {input_dir}")

    if not (0.0 <= args.p_low < args.p_high <= 100.0):
        raise ValueError(
            f"Invalid percentile range: p_low={args.p_low}, p_high={args.p_high}. "
            "Expected 0 <= p_low < p_high <= 100."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    mat_files = collect_mat_files(input_dir, recursive=args.recursive)

    if len(mat_files) == 0:
        print(f"[WARN] No .mat files found in: {input_dir}")
        return

    print(f"[INFO] Found {len(mat_files)} mat files")
    print(f"[INFO] Input dir: {input_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] mat_key: {args.mat_key}")
    print(f"[INFO] p_low: {args.p_low}")
    print(f"[INFO] p_high: {args.p_high}")
    print(f"[INFO] log_gain: {args.log_gain}")
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
                p_low=args.p_low,
                p_high=args.p_high,
                log_gain=args.log_gain,
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