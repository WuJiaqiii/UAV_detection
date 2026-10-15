import os
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
import torch


def detector_signature_from_config(config) -> str:
    """Fingerprint the detector settings that determine cached coordinates."""
    weights_path = Path(config.yolo_weights).resolve()
    weights_stat = weights_path.stat()
    settings = {
        "weights": str(weights_path),
        "weights_size": weights_stat.st_size,
        "weights_mtime_ns": weights_stat.st_mtime_ns,
        "imgsz": [config.yolo_imgsz_h, config.yolo_imgsz_w],
        "conf": config.yolo_conf_thres,
        "iou": config.yolo_iou_thres,
        "classes": config.yolo_classes,
        "max_det": config.yolo_max_det,
        "normalization": [config.yolo_input_norm, config.yolo_input_p_low,
                          config.yolo_input_p_high],
        "db_compat": [config.yolo_db_min, config.yolo_db_max, config.yolo_db_eps],
    }
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()

class BBoxCache:
    def __init__(
        self,
        base_dir: str,
        dataset_root: str | None = None,
        mode: str = "refresh",
        mem_max: int = 0,
        detector_signature: str = "",
        logger=None,
    ):
        self.mode = str(mode).lower()
        self.mem_max = int(mem_max) if mem_max else 0
        self.logger = logger
        self.dataset_root = dataset_root
        self.detector_signature = str(detector_signature)
        self.cache_dir = base_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._mem = OrderedDict()

    @staticmethod
    def _safe_relpath(fp: str, root: str | None) -> str:
        try:
            if root:
                return os.path.relpath(fp, root)
        except Exception:
            pass
        return fp

    def _key_and_meta(self, fp: str):
        rel = self._safe_relpath(fp, self.dataset_root)
        stem = os.path.splitext(os.path.basename(fp))[0]
        digest = hashlib.sha1((rel + "|" + self.detector_signature).encode("utf-8")).hexdigest()[:12]
        key = f"{stem}_{digest}"

        try:
            st = os.stat(fp)
            meta = {
                "rel": rel,
                "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
                "detector_signature": self.detector_signature,
            }
        except Exception:
            meta = {
                "rel": rel,
                "size": None,
                "mtime_ns": None,
                "detector_signature": self.detector_signature,
            }

        return key, meta

    def _path_for_key(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.pt")

    def get(self, fp: str):
        if self.mode in ("off", "write", "refresh"):
            return None

        key, current_meta = self._key_and_meta(fp)
        if key in self._mem:
            saved_meta, boxes = self._mem.pop(key)
            if saved_meta == current_meta:
                self._mem[key] = (saved_meta, boxes)
                return boxes

        path = self._path_for_key(key)
        if not os.path.isfile(path):
            return None

        try:
            obj = torch.load(path, map_location="cpu")
            if obj.get("source") != current_meta:
                return None
            boxes = obj.get("boxes", torch.zeros((0, 4), dtype=torch.int32))
            if not isinstance(boxes, torch.Tensor):
                boxes = torch.as_tensor(boxes, dtype=torch.int32)
            boxes = boxes.to(dtype=torch.int32, device="cpu").reshape(-1, 4)

            if self.mem_max > 0:
                self._mem[key] = (current_meta, boxes)
                while len(self._mem) > self.mem_max:
                    self._mem.popitem(last=False)

            return boxes
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[BBoxCache] failed reading {fp}: {e}")
            return None

    def put(self, fp: str, boxes: torch.Tensor):
        if self.mode in ("off", "read"):
            return

        key, meta = self._key_and_meta(fp)
        path = self._path_for_key(key)
        obj = {
            "source": meta,
            "boxes": boxes.to(dtype=torch.int32, device="cpu"),
        }
        try:
            tmp_path = path + ".tmp"
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
            if self.mem_max > 0:
                self._mem[key] = (meta, obj["boxes"])
                while len(self._mem) > self.mem_max:
                    self._mem.popitem(last=False)
        except Exception as e:
            if os.path.isfile(path + ".tmp"):
                os.remove(path + ".tmp")
            if self.logger:
                self.logger.warning(f"[BBoxCache] failed writing {fp}: {e}")
