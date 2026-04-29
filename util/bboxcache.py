import os
import hashlib
from collections import OrderedDict
import torch

class BBoxCache:
    def __init__(
        self,
        base_dir: str,
        dataset_root: str | None = None,
        mode: str = "refresh",
        mem_max: int = 0,
        logger=None,
    ):
        self.mode = str(mode).lower()
        self.mem_max = int(mem_max) if mem_max else 0
        self.logger = logger
        self.dataset_root = dataset_root
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
        digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:12]
        key = f"{stem}_{digest}"

        try:
            st = os.stat(fp)
            meta = {
                "rel": rel,
                "size": int(st.st_size),
                "mtime": int(st.st_mtime),
            }
        except Exception:
            meta = {
                "rel": rel,
                "size": None,
                "mtime": None,
            }

        return key, meta

    def _path_for_key(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.pt")

    def get(self, fp: str):
        if self.mode in ("off", "write", "refresh"):
            return None

        key, _ = self._key_and_meta(fp)
        if key in self._mem:
            boxes = self._mem.pop(key)
            self._mem[key] = boxes
            return boxes

        path = self._path_for_key(key)
        if not os.path.isfile(path):
            return None

        try:
            obj = torch.load(path, map_location="cpu")
            boxes = obj.get("boxes", torch.zeros((0, 4), dtype=torch.int32))
            if not isinstance(boxes, torch.Tensor):
                boxes = torch.as_tensor(boxes, dtype=torch.int32)
            boxes = boxes.to(dtype=torch.int32, device="cpu").reshape(-1, 4)

            if self.mem_max > 0:
                self._mem[key] = boxes
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
            torch.save(obj, path)
            if self.mem_max > 0:
                self._mem[key] = obj["boxes"]
                while len(self._mem) > self.mem_max:
                    self._mem.popitem(last=False)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[BBoxCache] failed writing {fp}: {e}")