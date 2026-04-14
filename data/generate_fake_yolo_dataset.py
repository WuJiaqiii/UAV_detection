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
        rotations=(0, 90, 180, 270),
        scale_x_range=(0.8, 1.3),
        scale_y_range=(0.8, 1.3),
        noise_prob=0.5,
        noise_sigma_range=(5.0, 20.0),
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

        self.max_try_per_object = int(max_try_per_object)
        self.max_retry_per_image = int(max_retry_per_image)

        self.preserve_class_id = bool(preserve_class_id)
        self.class_names = class_names if class_names is not None else None

        self.patch_pool = []
        self.background_pool = []
        self.class_ids = set()

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

    def yolo_to_xyxy(self, label_line, img_shape):
        class_id, xc, yc, bw, bh = self.parse_label_line(label_line)
        H, W = img_shape[:2]

        # 将归一化坐标转为像素坐标，并做边界裁剪
        x1 = int(round((xc - bw / 2.0) * W))
        y1 = int(round((yc - bh / 2.0) * H))
        x2 = int(round((xc + bw / 2.0) * W))
        y2 = int(round((yc + bh / 2.0) * H))

        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = max(1, min(W, x2))
        y2 = max(1, min(H, y2))

        # 保证至少 1 像素宽高
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
    # IoU 与重叠判断
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

    def is_valid(self, box, existing_boxes, iou_thresh=0.0):
        for b in existing_boxes:
            if self.compute_iou(box, b) > iou_thresh:
                return False
        return True

    # =========================
    # Step1: 构建素材池
    # =========================
    def build_pools(self):
        print("Building patch pool and background pool...")

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        img_names = sorted(os.listdir(self.img_dir))

        skipped_no_label = 0
        skipped_bad_image = 0
        skipped_invalid_label = 0
        skipped_empty_patch = 0

        for img_name in tqdm(img_names):
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
            for line in raw_lines:
                try:
                    class_id, x1, y1, x2, y2 = self.yolo_to_xyxy(line, img.shape)
                except Exception:
                    skipped_invalid_label += 1
                    continue

                patch = img[y1:y2, x1:x2].copy()
                if patch.size == 0 or patch.shape[0] < 1 or patch.shape[1] < 1:
                    skipped_empty_patch += 1
                    continue

                self.patch_pool.append(
                    {
                        "patch": patch,
                        "class_id": class_id if self.preserve_class_id else 0,
                    }
                )
                self.class_ids.add(class_id if self.preserve_class_id else 0)
                valid_boxes.append((x1, y1, x2, y2))

            if not valid_boxes:
                continue

            # 生成背景图：将目标区域抹除后作为背景池素材
            mask = np.zeros((H, W), dtype=np.uint8)
            for x1, y1, x2, y2 in valid_boxes:
                mask[y1:y2, x1:x2] = 255

            # bg = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
            # if (W, H) != (self.W, self.H):
            #     bg = cv2.resize(bg, (self.W, self.H), interpolation=cv2.INTER_AREA)
            # self.background_pool.append(bg)
            
            bg = self.make_background_from_image(
                img=img,
                valid_boxes=valid_boxes,
                mode="median",      # 可改为 "mean" 或 "black"
                noise_std=3.0
            )

            if (W, H) != (self.W, self.H):
                bg = cv2.resize(bg, (self.W, self.H), interpolation=cv2.INTER_AREA)

            self.background_pool.append(bg)

        print(f"patches: {len(self.patch_pool)}")
        print(f"backgrounds: {len(self.background_pool)}")
        print(f"classes: {sorted(self.class_ids) if self.class_ids else [0]}")
        print(f"skipped_no_label: {skipped_no_label}")
        print(f"skipped_bad_image: {skipped_bad_image}")
        print(f"skipped_invalid_label: {skipped_invalid_label}")
        print(f"skipped_empty_patch: {skipped_empty_patch}")

        if len(self.patch_pool) == 0:
            raise RuntimeError("patch_pool is empty. 请检查图像、标签和坐标转换是否正确。")
        if len(self.background_pool) == 0:
            raise RuntimeError("background_pool is empty. 请检查标签是否存在且是否能成功生成背景。")

    # =========================
    # patch增强
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

    # =========================
    # paste（防重叠）
    # =========================
    def paste_patch_no_overlap(self, bg, patch, class_id, existing_boxes, max_try=None, iou_thresh=0.0):
        if max_try is None:
            max_try = self.max_try_per_object

        ph, pw = patch.shape[:2]
        if ph < 1 or pw < 1:
            return False, None, None
        if ph > self.H or pw > self.W:
            return False, None, None

        for _ in range(max_try):
            x = random.randint(0, self.W - pw)
            y = random.randint(0, self.H - ph)
            new_box = (x, y, x + pw, y + ph)

            if not self.is_valid(new_box, existing_boxes, iou_thresh=iou_thresh):
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
            shutil.rmtree(self.out_dir)

        for split in ["train", "val"]:
            os.makedirs(os.path.join(self.out_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, "labels", split), exist_ok=True)

    def get_class_name_list(self):
        if self.class_names is not None:
            return list(self.class_names)

        if not self.class_ids:
            return ["rectangle"]

        max_class_id = max(self.class_ids)
        names = []
        for i in range(max_class_id + 1):
            names.append(f"class_{i}")
        return names

    def write_data_yaml(self):
        yaml_path = os.path.join(self.out_dir, "data.yaml")
        names = self.get_class_name_list()

        data = {
            "path": self.out_dir,
            "train": "images/train",
            "val": "images/val",
            "nc": len(names),
            "names": names,
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

        print(f"data.yaml generated: {yaml_path}")
        
    def make_background_from_image(self, img, valid_boxes, mode="median", noise_std=3.0):
        """
        根据非目标区域生成背景图
        mode:
            - "median": 用非目标区域像素中位数填充整张图
            - "mean":   用非目标区域像素均值填充整张图
            - "black":  直接纯黑背景
        """
        H, W = img.shape[:2]

        mask = np.zeros((H, W), dtype=np.uint8)
        for x1, y1, x2, y2 in valid_boxes:
            mask[y1:y2, x1:x2] = 255

        bg_mask = (mask == 0)

        # 灰度图
        if img.ndim == 2:
            pixels = img[bg_mask]
            if pixels.size == 0:
                base_value = 0
            else:
                if mode == "mean":
                    base_value = float(np.mean(pixels))
                elif mode == "black":
                    base_value = 0.0
                else:
                    base_value = float(np.median(pixels))

            bg = np.full((H, W), base_value, dtype=np.float32)

        # 彩色图
        else:
            pixels = img[bg_mask]   # shape: [N, C]
            if pixels.size == 0:
                if img.shape[2] == 3:
                    base_value = np.array([0, 0, 0], dtype=np.float32)
                else:
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

        # 加一点轻噪声，避免背景过于僵硬
        if noise_std is not None and noise_std > 0:
            noise = np.random.normal(0, noise_std, bg.shape).astype(np.float32)
            bg = bg + noise

        bg = np.clip(bg, 0, 255).astype(np.uint8)
        return bg

    # =========================
    # 生成数据（YOLO格式）
    # =========================
    def generate(self, num_images=1000, clean_output=False):
        if len(self.patch_pool) == 0 or len(self.background_pool) == 0:
            raise RuntimeError("请先执行 build_pools()，并确保 patch_pool/background_pool 非空。")

        self.prepare_output_dirs(clean_output=clean_output)

        train_num = int(round(num_images * self.train_ratio))
        written = 0

        for i in tqdm(range(num_images), desc="Generating"):
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
                    f"第 {i} 张图像在 {self.max_retry_per_image} 次尝试后仍未成功放置目标。"
                    f"请检查 patch 尺寸、输出尺寸或 max_try_per_object 设置。"
                )

            if random.random() < self.noise_prob:
                bg = self.add_gaussian_noise(bg)

            img_path = os.path.join(self.out_dir, "images", split, f"{i:06d}.png")
            label_path = os.path.join(self.out_dir, "labels", split, f"{i:06d}.txt")

            cv2.imwrite(img_path, bg)
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(labels))

            written += 1

        print(f"Dataset generation completed. written={written}")
        self.write_data_yaml()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic YOLO dataset from labeled patches.")
    parser.add_argument("--img_dir", type=str,
                        default="/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/dataset/images")
    parser.add_argument("--label_dir", type=str,
                        default="/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/label")
    parser.add_argument("--out_dir", type=str,
                        default="/media/kaneki/5490675f-8f6a-4932-bae3-f457edde3ca0/wujiaqi/code/data/dataset")

    parser.add_argument("--num_images", type=int, default=10000)
    parser.add_argument("--out_h", type=int, default=512)
    parser.add_argument("--out_w", type=int, default=750)
    parser.add_argument("--min_objects", type=int, default=1)
    parser.add_argument("--max_objects", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.8)

    parser.add_argument("--scale_x_min", type=float, default=0.8)
    parser.add_argument("--scale_x_max", type=float, default=1.3)
    parser.add_argument("--scale_y_min", type=float, default=0.8)
    parser.add_argument("--scale_y_max", type=float, default=1.3)

    parser.add_argument("--noise_prob", type=float, default=0.5)
    parser.add_argument("--noise_sigma_min", type=float, default=5.0)
    parser.add_argument("--noise_sigma_max", type=float, default=20.0)

    parser.add_argument("--max_try_per_object", type=int, default=50)
    parser.add_argument("--max_retry_per_image", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="删除 out_dir 后重新生成")
    return parser.parse_args()


def main():
    args = parse_args()

    generator = DatasetGenerator(
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        out_dir=args.out_dir,
        out_h=args.out_h,
        out_w=args.out_w,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        train_ratio=args.train_ratio,
        rotations=(0, 90, 180, 270),
        scale_x_range=(args.scale_x_min, args.scale_x_max),
        scale_y_range=(args.scale_y_min, args.scale_y_max),
        noise_prob=args.noise_prob,
        noise_sigma_range=(args.noise_sigma_min, args.noise_sigma_max),
        max_try_per_object=args.max_try_per_object,
        max_retry_per_image=args.max_retry_per_image,
        preserve_class_id=True,
        class_names=None,
        seed=args.seed,
    )

    generator.build_pools()
    generator.generate(num_images=args.num_images, clean_output=args.clean)


if __name__ == "__main__":
    main()
