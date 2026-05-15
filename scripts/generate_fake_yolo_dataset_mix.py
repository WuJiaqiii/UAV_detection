import os
import cv2
import math
import yaml
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm


class DatasetGenerator:
    def __init__(
        self,
        img_dir,
        label_dir,
        out_dir,
        out_h=512,
        out_w=750,
        min_objects=1,
        max_objects=5,
        train_ratio=0.8,
        rotations=(0,),
        scale_x_range=(0.8, 1.3),
        scale_y_range=(0.8, 1.3),
        noise_prob=0.5,
        noise_sigma_range=(5.0, 20.0),
        background_mode="median",
        background_noise_std=3.0,
        max_try_per_object=50,
        max_retry_per_image=20,
        preserve_class_id=True,
        class_names=None,
        seed=None,
    ):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.out_dir = out_dir

        self.H = int(out_h)
        self.W = int(out_w)

        self.min_objects = int(min_objects)
        self.max_objects = int(max_objects)
        self.train_ratio = float(train_ratio)

        self.rotations = tuple(rotations)
        self.scale_x_range = tuple(scale_x_range)
        self.scale_y_range = tuple(scale_y_range)

        self.noise_prob = float(noise_prob)
        self.noise_sigma_range = tuple(noise_sigma_range)

        self.background_mode = background_mode
        self.background_noise_std = float(background_noise_std)

        self.max_try_per_object = int(max_try_per_object)
        self.max_retry_per_image = int(max_retry_per_image)

        self.preserve_class_id = bool(preserve_class_id)
        self.class_names = class_names if class_names is not None else None

        self.patch_pool = []
        self.background_pool = []
        self.class_ids = set()

        # 原始真实有标签数据，用于混入最终数据集
        self.real_pairs = []

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # =========================
    # 基础工具
    # =========================
    @staticmethod
    def ensure_uint8(img):
        if img is None:
            return None

        if img.dtype == np.uint8:
            return img

        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            if info.max == 0:
                return np.zeros_like(img, dtype=np.uint8)

            scale = 255.0 / float(info.max)
            return np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)

        img = img.astype(np.float32)
        finite_mask = np.isfinite(img)

        if not finite_mask.any():
            return np.zeros_like(img, dtype=np.uint8)

        valid = img[finite_mask]
        vmin, vmax = float(valid.min()), float(valid.max())

        if math.isclose(vmin, vmax):
            return np.zeros_like(img, dtype=np.uint8)

        img = (img - vmin) / (vmax - vmin)
        img = np.clip(img * 255.0, 0, 255)
        return img.astype(np.uint8)

    @staticmethod
    def read_image(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            return None

        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return DatasetGenerator.ensure_uint8(img)

    @staticmethod
    def parse_label_line(line):
        parts = line.strip().split()

        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label line: {line!r}")

        class_id = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:])

        return class_id, xc, yc, w, h

    def normalize_class_id(self, class_id):
        """
        preserve_class_id=True:
            保留原始类别。

        preserve_class_id=False:
            所有类别统一为 0，适合单类别检测训练。
        """
        return int(class_id) if self.preserve_class_id else 0

    def is_valid_yolo_values(self, class_id, xc, yc, w, h):
        if class_id < 0:
            return False

        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            return False

        if w <= 0 or h <= 0:
            return False

        return True

    def yolo_to_xyxy(self, label_line, img_shape):
        class_id, xc, yc, bw, bh = self.parse_label_line(label_line)
        H, W = img_shape[:2]

        x1 = int(round((xc - bw / 2.0) * W))
        y1 = int(round((yc - bh / 2.0) * H))
        x2 = int(round((xc + bw / 2.0) * W))
        y2 = int(round((yc + bh / 2.0) * H))

        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(1, min(W, x2))
        y2 = max(1, min(H, y2))

        if x2 <= x1:
            if x1 < W - 1:
                x2 = x1 + 1
            else:
                x1 = max(0, x2 - 1)

        if y2 <= y1:
            if y1 < H - 1:
                y2 = y1 + 1
            else:
                y1 = max(0, y2 - 1)

        return class_id, x1, y1, x2, y2

    def xyxy_to_yolo_label(self, class_id, x1, y1, x2, y2):
        bw = (x2 - x1) / self.W
        bh = (y2 - y1) / self.H
        xc = (x1 + x2) / 2.0 / self.W
        yc = (y1 + y2) / 2.0 / self.H

        return f"{int(class_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

    # =========================
    # IoU 与防重叠
    # =========================
    @staticmethod
    def compute_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        inter_x1 = max(x1, x1g)
        inter_y1 = max(y1, y1g)
        inter_x2 = min(x2, x2g)
        inter_y2 = min(y2, y2g)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area1 = max(0, x2 - x1) * max(0, y2 - y1)
        area2 = max(0, x2g - x1g) * max(0, y2g - y1g)

        union = area1 + area2 - inter_area

        return inter_area / union if union > 0 else 0.0

    def is_valid_position(self, box, existing_boxes, iou_thresh=0.0):
        for b in existing_boxes:
            if self.compute_iou(box, b) > iou_thresh:
                return False
        return True

    # =========================
    # Step 1: 构建 patch 池、背景池、真实样本列表
    # =========================
    def build_pools(self):
        print("Building patch pool, background pool and real pair list...")

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        img_names = sorted(os.listdir(self.img_dir))

        skipped_no_label = 0
        skipped_bad_image = 0
        skipped_invalid_label = 0
        skipped_empty_patch = 0
        valid_real_images = 0

        for img_name in tqdm(img_names, desc="Scanning source dataset"):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                continue

            img_path = os.path.join(self.img_dir, img_name)
            label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")

            if not os.path.exists(label_path):
                skipped_no_label += 1
                continue

            img = self.read_image(img_path)

            if img is None:
                skipped_bad_image += 1
                continue

            H, W = img.shape[:2]

            with open(label_path, "r", encoding="utf-8") as f:
                raw_lines = [line.strip() for line in f if line.strip()]

            if not raw_lines:
                skipped_invalid_label += 1
                continue

            valid_boxes = []
            valid_label_count_for_this_image = 0

            for line in raw_lines:
                try:
                    class_id, xc, yc, bw, bh = self.parse_label_line(line)
                except Exception:
                    skipped_invalid_label += 1
                    continue

                if not self.is_valid_yolo_values(class_id, xc, yc, bw, bh):
                    skipped_invalid_label += 1
                    continue

                try:
                    class_id, x1, y1, x2, y2 = self.yolo_to_xyxy(line, img.shape)
                except Exception:
                    skipped_invalid_label += 1
                    continue

                patch = img[y1:y2, x1:x2].copy()

                if patch.size == 0 or patch.shape[0] < 1 or patch.shape[1] < 1:
                    skipped_empty_patch += 1
                    continue

                out_class_id = self.normalize_class_id(class_id)

                self.patch_pool.append(
                    {
                        "patch": patch,
                        "class_id": out_class_id,
                    }
                )

                self.class_ids.add(out_class_id)
                valid_boxes.append((x1, y1, x2, y2))
                valid_label_count_for_this_image += 1

            if valid_label_count_for_this_image == 0:
                continue

            # 记录真实有标签样本，后续混入最终数据集
            self.real_pairs.append((img_path, label_path))
            valid_real_images += 1

            # 根据非目标区域生成背景
            bg = self.make_background_from_image(
                img=img,
                valid_boxes=valid_boxes,
                mode=self.background_mode,
                noise_std=self.background_noise_std,
            )

            if (W, H) != (self.W, self.H):
                bg = cv2.resize(bg, (self.W, self.H), interpolation=cv2.INTER_AREA)

            self.background_pool.append(bg)

        print("\n========== Source Scan Summary ==========")
        print(f"patches             : {len(self.patch_pool)}")
        print(f"backgrounds         : {len(self.background_pool)}")
        print(f"valid real images   : {valid_real_images}")
        print(f"classes             : {sorted(self.class_ids) if self.class_ids else [0]}")
        print(f"skipped_no_label    : {skipped_no_label}")
        print(f"skipped_bad_image   : {skipped_bad_image}")
        print(f"skipped_invalid_row : {skipped_invalid_label}")
        print(f"skipped_empty_patch : {skipped_empty_patch}")
        print("=========================================\n")

        if len(self.patch_pool) == 0:
            raise RuntimeError("patch_pool is empty. 请检查图像、标签和坐标转换是否正确。")

        if len(self.background_pool) == 0:
            raise RuntimeError("background_pool is empty. 请检查标签是否存在且是否能成功生成背景。")

        if len(self.real_pairs) == 0:
            print("[WARNING] 没有找到可混入的真实有标签图片。")

    # =========================
    # 背景生成
    # =========================
    def make_background_from_image(self, img, valid_boxes, mode="median", noise_std=3.0):
        """
        根据非目标区域生成背景图。

        mode:
            median: 用非目标区域像素中位数生成背景
            mean  : 用非目标区域像素均值生成背景
            black : 黑色背景
        """
        H, W = img.shape[:2]

        mask = np.zeros((H, W), dtype=np.uint8)

        for x1, y1, x2, y2 in valid_boxes:
            mask[y1:y2, x1:x2] = 255

        bg_mask = mask == 0

        if img.ndim == 2:
            pixels = img[bg_mask]

            if pixels.size == 0:
                base_value = 0.0
            else:
                if mode == "mean":
                    base_value = float(np.mean(pixels))
                elif mode == "black":
                    base_value = 0.0
                else:
                    base_value = float(np.median(pixels))

            bg = np.full((H, W), base_value, dtype=np.float32)

        else:
            pixels = img[bg_mask]

            if pixels.size == 0:
                base_value = np.zeros((img.shape[2],), dtype=np.float32)
            else:
                if mode == "mean":
                    base_value = np.mean(pixels, axis=0).astype(np.float32)
                elif mode == "black":
                    base_value = np.zeros((img.shape[2],), dtype=np.float32)
                else:
                    base_value = np.median(pixels, axis=0).astype(np.float32)

            bg = np.ones_like(img, dtype=np.float32)
            bg = bg * base_value.reshape(1, 1, -1)

        if noise_std is not None and noise_std > 0:
            noise = np.random.normal(0, noise_std, bg.shape).astype(np.float32)
            bg = bg + noise

        bg = np.clip(bg, 0, 255).astype(np.uint8)
        return bg

    # =========================
    # patch 增强
    # =========================
    def augment_patch(self, patch):
        aug = patch.copy()

        angle = random.choice(self.rotations)

        if angle == 90:
            aug = cv2.rotate(aug, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            aug = cv2.rotate(aug, cv2.ROTATE_180)
        elif angle == 270:
            aug = cv2.rotate(aug, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w = aug.shape[:2]

        sx = random.uniform(*self.scale_x_range)
        sy = random.uniform(*self.scale_y_range)

        new_w = max(1, int(round(w * sx)))
        new_h = max(1, int(round(h * sy)))

        aug = cv2.resize(aug, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        return aug

    @staticmethod
    def match_patch_channels_to_bg(patch, bg):
        """
        保证 patch 和 background 通道数一致，避免粘贴时报错。
        """
        if bg.ndim == 2 and patch.ndim == 3:
            return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        if bg.ndim == 3 and patch.ndim == 2:
            return cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)

        return patch

    # =========================
    # paste：随机粘贴且防重叠
    # =========================
    def paste_patch_no_overlap(self, bg, patch, class_id, existing_boxes, max_try=None, iou_thresh=0.0):
        if max_try is None:
            max_try = self.max_try_per_object

        patch = self.match_patch_channels_to_bg(patch, bg)

        ph, pw = patch.shape[:2]

        if ph < 1 or pw < 1:
            return False, None, None

        if ph > self.H or pw > self.W:
            return False, None, None

        for _ in range(max_try):
            x = random.randint(0, self.W - pw)
            y = random.randint(0, self.H - ph)

            new_box = (x, y, x + pw, y + ph)

            if not self.is_valid_position(new_box, existing_boxes, iou_thresh=iou_thresh):
                continue

            bg[y:y + ph, x:x + pw] = patch

            label = self.xyxy_to_yolo_label(class_id, x, y, x + pw, y + ph)

            return True, label, new_box

        return False, None, None

    # =========================
    # 高斯噪声
    # =========================
    def add_gaussian_noise(self, img):
        sigma = random.uniform(*self.noise_sigma_range)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        out = img.astype(np.float32) + noise
        return np.clip(out, 0, 255).astype(np.uint8)

    # =========================
    # 输出目录
    # =========================
    def prepare_output_dirs(self, clean_output=False):
        if clean_output and os.path.isdir(self.out_dir):
            print(f"[INFO] Removing existing output directory: {self.out_dir}")
            shutil.rmtree(self.out_dir)

        for split in ["train", "val"]:
            os.makedirs(os.path.join(self.out_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "labels", split), exist_ok=True)

    # =========================
    # 真实标签清洗与复制
    # =========================
    def clean_label_lines_for_output(self, label_path):
        """
        清洗真实标签文件。

        - 删除格式错误行；
        - 删除坐标越界行；
        - 删除 w<=0 或 h<=0 行；
        - preserve_class_id=False 时，将所有类别改为 0。
        """
        cleaned = []
        bad_count = 0

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            try:
                class_id, xc, yc, w, h = self.parse_label_line(line)
            except Exception:
                bad_count += 1
                continue

            if not self.is_valid_yolo_values(class_id, xc, yc, w, h):
                bad_count += 1
                continue

            out_class_id = self.normalize_class_id(class_id)

            cleaned.append(
                f"{int(out_class_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
            )

            self.class_ids.add(out_class_id)

        return cleaned, bad_count

    def copy_one_real_pair(self, img_path, label_path, split, out_name):
        """
        复制一张真实图片和对应标签到最终 YOLO 数据集中。
        """
        dst_img_path = os.path.join(self.out_dir, "images", split, out_name + ".png")
        dst_label_path = os.path.join(self.out_dir, "labels", split, out_name + ".txt")

        img = self.read_image(img_path)

        if img is None:
            return False, 0

        h, w = img.shape[:2]

        if h != self.H or w != self.W:
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)

        cleaned_lines, bad_count = self.clean_label_lines_for_output(label_path)

        if len(cleaned_lines) == 0:
            return False, bad_count

        cv2.imwrite(dst_img_path, img)

        with open(dst_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(cleaned_lines) + "\n")

        return True, bad_count

    def copy_real_dataset_to_output(self, real_prefix="real"):
        """
        将原始真实有标签数据混入最终 YOLOv5 数据集。
        """
        if len(self.real_pairs) == 0:
            print("[WARNING] real_pairs 为空，没有真实数据被混入。")
            return 0, 0, 0

        pairs = self.real_pairs.copy()
        random.shuffle(pairs)

        train_num = int(round(len(pairs) * self.train_ratio))

        train_pairs = pairs[:train_num]
        val_pairs = pairs[train_num:]

        copied_train = 0
        copied_val = 0
        total_bad_labels = 0

        for idx, (img_path, label_path) in enumerate(tqdm(train_pairs, desc="Copying real train")):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            out_name = f"{real_prefix}_{idx:06d}_{stem}"

            ok, bad_count = self.copy_one_real_pair(
                img_path=img_path,
                label_path=label_path,
                split="train",
                out_name=out_name,
            )

            total_bad_labels += bad_count

            if ok:
                copied_train += 1

        for idx, (img_path, label_path) in enumerate(tqdm(val_pairs, desc="Copying real val")):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            out_name = f"{real_prefix}_{idx:06d}_{stem}"

            ok, bad_count = self.copy_one_real_pair(
                img_path=img_path,
                label_path=label_path,
                split="val",
                out_name=out_name,
            )

            total_bad_labels += bad_count

            if ok:
                copied_val += 1

        print("\n========== Real Data Copy Summary ==========")
        print(f"Real pairs found       : {len(self.real_pairs)}")
        print(f"Copied real train      : {copied_train}")
        print(f"Copied real val        : {copied_val}")
        print(f"Removed bad label rows : {total_bad_labels}")
        print("===========================================\n")

        return copied_train, copied_val, total_bad_labels

    # =========================
    # data.yaml
    # =========================
    def get_class_name_list(self):
        if self.class_names is not None:
            return [str(x) for x in self.class_names]

        if not self.preserve_class_id:
            return ["class_0"]

        if not self.class_ids:
            return ["class_0"]

        max_class_id = max(self.class_ids)

        names = []
        for i in range(max_class_id + 1):
            names.append(f"class_{i}")

        return names

    def write_data_yaml(self):
        yaml_path = os.path.join(self.out_dir, "data.yaml")
        names = self.get_class_name_list()

        if self.class_ids:
            max_class_id = max(self.class_ids)
            if max_class_id >= len(names):
                raise ValueError(
                    f"class_names 数量不足。当前最大类别ID为 {max_class_id}，"
                    f"但 names 只有 {len(names)} 个。"
                    f"如果你是单类别训练，请加 --single_class。"
                )

        data = {
            "path": os.path.abspath(self.out_dir),
            "train": "images/train",
            "val": "images/val",
            "nc": len(names),
            "names": names,
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

        print(f"data.yaml generated: {yaml_path}")

    # =========================
    # 生成混合数据集
    # =========================
    def generate(
        self,
        num_synthetic_images=1000,
        clean_output=False,
        include_real=True,
        real_prefix="real",
        synth_prefix="synth",
    ):
        """
        生成混合 YOLOv5 数据集。

        include_real=True:
            输出目录中包含原始真实有标签数据。

        num_synthetic_images:
            额外生成的仿真图片数量。
        """
        if len(self.patch_pool) == 0 or len(self.background_pool) == 0:
            raise RuntimeError("请先执行 build_pools()，并确保 patch_pool/background_pool 非空。")

        self.prepare_output_dirs(clean_output=clean_output)

        real_train = 0
        real_val = 0
        bad_real_label_rows = 0

        if include_real:
            real_train, real_val, bad_real_label_rows = self.copy_real_dataset_to_output(
                real_prefix=real_prefix,
            )

        train_num = int(round(num_synthetic_images * self.train_ratio))

        synth_train = 0
        synth_val = 0

        for i in tqdm(range(num_synthetic_images), desc="Generating synthetic"):
            split = "train" if i < train_num else "val"

            success = False
            labels = []
            bg = None

            for _ in range(self.max_retry_per_image):
                bg = random.choice(self.background_pool).copy()
                target_num_obj = random.randint(self.min_objects, self.max_objects)

                labels = []
                boxes = []

                for _ in range(target_num_obj):
                    item = random.choice(self.patch_pool)

                    patch = self.augment_patch(item["patch"])
                    class_id = item["class_id"]

                    ok, label, new_box = self.paste_patch_no_overlap(
                        bg=bg,
                        patch=patch,
                        class_id=class_id,
                        existing_boxes=boxes,
                        max_try=self.max_try_per_object,
                        iou_thresh=0.0,
                    )

                    if ok:
                        labels.append(label)
                        boxes.append(new_box)

                if len(labels) > 0:
                    success = True
                    break

            if not success or bg is None:
                raise RuntimeError(
                    f"第 {i} 张仿真图像在 {self.max_retry_per_image} 次尝试后仍未成功放置目标。"
                    f"请检查 patch 尺寸、输出尺寸或 max_try_per_object 设置。"
                )

            if random.random() < self.noise_prob:
                bg = self.add_gaussian_noise(bg)

            img_name = f"{synth_prefix}_{i:06d}.png"
            label_name = f"{synth_prefix}_{i:06d}.txt"

            img_path = os.path.join(self.out_dir, "images", split, img_name)
            label_path = os.path.join(self.out_dir, "labels", split, label_name)

            cv2.imwrite(img_path, bg)

            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(labels) + "\n")

            if split == "train":
                synth_train += 1
            else:
                synth_val += 1

        self.write_data_yaml()

        print("\n========== Mixed Dataset Summary ==========")
        print(f"Output dir             : {self.out_dir}")
        print(f"Real train             : {real_train}")
        print(f"Real val               : {real_val}")
        print(f"Synthetic train        : {synth_train}")
        print(f"Synthetic val          : {synth_val}")
        print(f"Total train            : {real_train + synth_train}")
        print(f"Total val              : {real_val + synth_val}")
        print(f"Bad real label rows    : {bad_real_label_rows}")
        print(f"Classes                : {sorted(self.class_ids) if self.class_ids else [0]}")
        print("==========================================\n")


def parse_rotations(rotation_str):
    """
    将命令行输入的旋转角度字符串转为 tuple。

    例如：
        "0" -> (0,)
        "0,180" -> (0, 180)
        "0,90,180,270" -> (0, 90, 180, 270)
    """
    rotations = []

    for item in rotation_str.split(","):
        item = item.strip()

        if not item:
            continue

        angle = int(item)

        if angle not in [0, 90, 180, 270]:
            raise ValueError("rotations 只能包含 0, 90, 180, 270")

        rotations.append(angle)

    if not rotations:
        rotations = [0]

    return tuple(rotations)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate mixed YOLOv5 dataset with real labeled data and synthetic pasted-patch data."
    )

    parser.add_argument(
        "--img_dir",
        type=str,
        default="/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/dataset/images",
        help="原始图片目录。",
    )

    parser.add_argument(
        "--label_dir",
        type=str,
        default="/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/dataset/labels",
        help="原始 YOLO 标签目录。",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/data/dataset_mix",
        help="输出 YOLOv5 数据集目录。",
    )

    parser.add_argument(
        "--num_images",
        type=int,
        default=1500,
        help="需要额外生成的仿真图片数量，不包含真实图片数量。",
    )

    parser.add_argument(
        "--out_h",
        type=int,
        default=512,
        help="输出图片高度。",
    )

    parser.add_argument(
        "--out_w",
        type=int,
        default=750,
        help="输出图片宽度。",
    )

    parser.add_argument(
        "--min_objects",
        type=int,
        default=5,
        help="每张仿真图最少目标数。",
    )

    parser.add_argument(
        "--max_objects",
        type=int,
        default=15,
        help="每张仿真图最多目标数。",
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集比例。",
    )

    parser.add_argument(
        "--rotations",
        type=str,
        default='0,180',
        help="patch 随机旋转角度，例如 '0'、'0,180'、'0,90,180,270'。频谱图建议默认只用 0。",
    )

    parser.add_argument(
        "--scale_x_min",
        type=float,
        default=0.6,
        help="patch 横向缩放最小倍率。",
    )

    parser.add_argument(
        "--scale_x_max",
        type=float,
        default=1.5,
        help="patch 横向缩放最大倍率。",
    )

    parser.add_argument(
        "--scale_y_min",
        type=float,
        default=0.6,
        help="patch 纵向缩放最小倍率。",
    )

    parser.add_argument(
        "--scale_y_max",
        type=float,
        default=1.5,
        help="patch 纵向缩放最大倍率。",
    )

    parser.add_argument(
        "--noise_prob",
        type=float,
        default=0.8,
        help="仿真图添加高斯噪声的概率。",
    )

    parser.add_argument(
        "--noise_sigma_min",
        type=float,
        default=5.0,
        help="高斯噪声 sigma 最小值。",
    )

    parser.add_argument(
        "--noise_sigma_max",
        type=float,
        default=40.0,
        help="高斯噪声 sigma 最大值。",
    )

    parser.add_argument(
        "--background_mode",
        type=str,
        default="median",
        choices=["median", "mean", "black"],
        help="背景生成方式。",
    )

    parser.add_argument(
        "--background_noise_std",
        type=float,
        default=6.0,
        help="背景轻微噪声标准差。",
    )

    parser.add_argument(
        "--max_try_per_object",
        type=int,
        default=50,
        help="每个目标尝试随机放置的最大次数。",
    )

    parser.add_argument(
        "--max_retry_per_image",
        type=int,
        default=20,
        help="每张仿真图生成失败后的重试次数。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="生成前删除 out_dir 后重新生成。",
    )

    parser.add_argument(
        "--no_real",
        action="store_true",
        help="不混入真实数据，只生成仿真数据。",
    )

    parser.add_argument(
        "--single_class",
        action="store_true",
        help="将真实和仿真标签中的所有类别都改为 0，适合单类别 YOLO 训练。",
    )

    parser.add_argument(
        "--class_names",
        type=str,
        nargs="+",
        default=None,
        help="类别名称。例如单类别可写：--class_names UAV",
    )

    parser.add_argument(
        "--real_prefix",
        type=str,
        default="real",
        help="真实数据输出文件名前缀。",
    )

    parser.add_argument(
        "--synth_prefix",
        type=str,
        default="synth",
        help="仿真数据输出文件名前缀。",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    rotations = parse_rotations(args.rotations)

    generator = DatasetGenerator(
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        out_dir=args.out_dir,
        out_h=args.out_h,
        out_w=args.out_w,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        train_ratio=args.train_ratio,
        rotations=rotations,
        scale_x_range=(args.scale_x_min, args.scale_x_max),
        scale_y_range=(args.scale_y_min, args.scale_y_max),
        noise_prob=args.noise_prob,
        noise_sigma_range=(args.noise_sigma_min, args.noise_sigma_max),
        background_mode=args.background_mode,
        background_noise_std=args.background_noise_std,
        max_try_per_object=args.max_try_per_object,
        max_retry_per_image=args.max_retry_per_image,
        preserve_class_id=not args.single_class,
        class_names=args.class_names,
        seed=args.seed,
    )

    generator.build_pools()

    generator.generate(
        num_synthetic_images=args.num_images,
        clean_output=args.clean,
        include_real=not args.no_real,
        real_prefix=args.real_prefix,
        synth_prefix=args.synth_prefix,
    )


if __name__ == "__main__":
    main()