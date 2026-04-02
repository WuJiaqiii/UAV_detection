import os
import re
import glob
import csv
import numpy as np
import torch
import random
import cv2

from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler, Subset
from sklearn.model_selection import train_test_split
import torch.distributed as dist

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None


class UAVDataset(Dataset):

    _FNAME_RE = re.compile(
        r"""^(?P<protocol>.+?)-\[(?P<bracket>[^\]]+)\](?:-SNR-(?P<snr>[-+]?\d+(?:\.\d+)?))?""",
        re.VERBOSE,
    )# 例：Skylink11-[0,-22.0,1000,20]-SNR-20-SNRSPACE1.046149e+01-Figure-1.mat/.png

    def __init__(self, config, logger, validate_on_init: bool = False):
        self.config = config
        self.logger = logger
        self.input_type = str(getattr(config, "input_type", "png")).lower()

        dataset_dir = getattr(config, "dataset_path", None)
        if not dataset_dir or not os.path.isdir(dataset_dir):
            raise ValueError(f"config.dataset_path must be an existing directory, got: {dataset_dir}")

        if self.input_type not in {"mat", "png"}:
            raise ValueError(f"Unsupported input_type={self.input_type}, expected 'mat' or 'png'")

        self.mod2label = {str(k): int(v) for k, v in getattr(config, "classes", {}).items()}

        pattern = "*.png" if self.input_type == "png" else "*.mat"
        data_files = sorted(glob.glob(os.path.join(dataset_dir, pattern)))
        if not data_files:
            raise FileNotFoundError(f"No {pattern} files found under: {dataset_dir}")

        def _load_whitelist(csv_path: str) -> set[str]:
            keep = set()
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError(f"Invalid CSV (no header): {csv_path}")

                col = "filename" if "filename" in reader.fieldnames else reader.fieldnames[0]

                for row in reader:
                    v = (row.get(col) or "").strip()
                    if not v:
                        continue
                    v = os.path.basename(v).strip('"').strip("'")
                    keep.add(v)
            return keep

        whitelist_csv = getattr(config, "whitelist_csv", None)
        if whitelist_csv:
            if not os.path.isfile(whitelist_csv):
                raise FileNotFoundError(f"--whitelist_csv not found: {whitelist_csv}")

            keep_names = _load_whitelist(whitelist_csv)

            before = len(data_files)
            filtered = []
            for fp in data_files:
                fname = os.path.basename(fp)                  # xxx.png / xxx.mat
                base = os.path.splitext(fname)[0]             # xxx
                png_name = base + ".png"
                mat_name = base + ".mat"

                if (fname in keep_names) or (base in keep_names) or (png_name in keep_names) or (mat_name in keep_names):
                    filtered.append(fp)

            data_files = filtered
            logger.info(
                f"[Whitelist] Applied whitelist_csv={whitelist_csv}: keep {len(data_files)}/{before}, drop {before-len(data_files)}"
            )

            if not data_files:
                raise RuntimeError("Whitelist filtered out all files. Check filename mapping and CSV content.")

        files, protocol_list, freq_list, bw_list, snr_list = [], [], [], [], []
        bad = 0

        def skip(reason: str, fname: str):
            nonlocal bad
            bad += 1
            logger.warning(f"[Skip] {reason}: {fname}")

        for fp in data_files:
            fname = os.path.basename(fp)
            m = self._FNAME_RE.search(fname)
            if not m:
                skip("filename regex not matched", fname)
                continue

            protocol = m.group("protocol").strip()
            if protocol not in config.classes:
                continue

            parts = [p.strip() for p in m.group("bracket").split(",")]
            if len(parts) < 4:
                skip("bracket parse failed (need 4 fields)", fname)
                continue

            try:
                freq = float(parts[1])
                bw = float(parts[3])
            except ValueError:
                skip("freq/bw parse failed", fname)
                continue

            snr = float("nan")
            snr_str = m.group("snr")
            if snr_str is not None:
                try:
                    snr = float(snr_str)
                except ValueError:
                    snr = float("nan")

            if validate_on_init:
                try:
                    _ = self._load_x(fp)
                except Exception as e:
                    skip(f"validate_on_init failed ({e})", fname)
                    continue

            files.append(fp)
            protocol_list.append(self.mod2label[protocol])
            freq_list.append(freq)
            bw_list.append(bw)
            snr_list.append(snr)

        if not files:
            raise RuntimeError(f"All files were skipped. bad={bad}, total={len(data_files)}")

        self.files = files
        self.protocol = torch.tensor(protocol_list, dtype=torch.int64)
        self.freq = torch.tensor(freq_list, dtype=torch.float32)
        self.bw = torch.tensor(bw_list, dtype=torch.float32)
        self.snr = torch.tensor(snr_list, dtype=torch.float32)

        logger.info(f"Indexed {len(self.files)} samples from {dataset_dir} (skipped {bad})")
        logger.info(f"Dataset input_type={self.input_type}, validate_on_init={validate_on_init}")

    def _load_x(self, fp: str) -> torch.Tensor:
        if self.input_type == "png":
            x = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if x is None:
                raise RuntimeError(f"Failed to read png: {fp}")

            # 保持 uint8，不做归一化
            x = np.ascontiguousarray(x)
            t = torch.from_numpy(x)  # uint8, shape (H, W)
            return t

        # mat mode
        if loadmat is None:
            raise ImportError("scipy is required for mat mode. Please install scipy.")

        mat = loadmat(fp, variable_names=[self.config.mat_key])
        if self.config.mat_key not in mat:
            raise KeyError(f"key '{self.config.mat_key}' not found in mat")

        x = np.asarray(mat[self.config.mat_key])
        if x.ndim != 2:
            raise ValueError(f"data dim != 2, got shape {x.shape}")

        x = np.asarray(x, dtype=np.float32)
        x = np.ascontiguousarray(x)
        t = torch.from_numpy(x)
        return t

    def __len__(self):
        return self.protocol.numel()

    def __getitem__(self, idx: int):
        fp = self.files[idx]
        try:
            x = self._load_x(fp)
        except Exception as e:
            raise RuntimeError(f"Failed to load '{os.path.basename(fp)}': {e}") from e

        return x, self.protocol[idx], self.freq[idx], self.bw[idx], self.snr[idx], fp


class RandomSampler(Sampler):
    def __init__(self, data_source, sample_ratio):
        self.data_source = data_source
        self.sample_ratio = float(sample_ratio)
        self.num_samples = len(data_source)

    def __iter__(self):
        k = int(self.num_samples * self.sample_ratio)
        if k <= 0:
            return iter([])
        idx = torch.randperm(self.num_samples).tolist()[:k]
        return iter(idx)

    def __len__(self):
        return int(self.num_samples * self.sample_ratio)


def build_ddp_sampler(dataset, shuffle: bool, sample_ratio: float, seed: int = 42):
    rank = dist.get_rank()
    world = dist.get_world_size()
    num_keep = int(round(len(dataset) * float(sample_ratio)))
    num_keep = max(0, min(num_keep, len(dataset)))

    g = torch.Generator()
    g.manual_seed(int(seed))

    idx = torch.randperm(len(dataset), generator=g)[:num_keep].tolist()
    subset = Subset(dataset, idx)

    sampler = DistributedSampler(
        subset, num_replicas=world, rank=rank, shuffle=shuffle, drop_last=False
    )
    return subset, sampler


def create_dataloader(dataset: Dataset, config, shuffle):
    if dist.is_initialized():
        dataset, sampler = build_ddp_sampler(dataset, shuffle=shuffle, sample_ratio=config.sample_ratio)
    else:
        sampler = RandomSampler(dataset, config.sample_ratio)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None and shuffle),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False
    )


def get_dataloader(dataset, config):
    all_labels = dataset.protocol
    idxs = np.arange(len(dataset))

    if torch.is_tensor(all_labels):
        labels_np = all_labels.detach().cpu().numpy()
    else:
        labels_np = np.asarray(all_labels)

    n_unique = len(np.unique(labels_np)) if labels_np.size > 0 else 0

    if labels_np.size == 0 or n_unique <= 1:
        train_idx, val_idx = train_test_split(
            idxs, test_size=config.val_ratio, shuffle=True, random_state=42
        )
    else:
        train_idx, val_idx = train_test_split(
            idxs, test_size=config.val_ratio, shuffle=True, random_state=42,
            stratify=labels_np
        )

    train_dataset = Subset(dataset, train_idx.tolist())
    val_dataset = Subset(dataset, val_idx.tolist())

    train_loader = create_dataloader(train_dataset, config, shuffle=True)
    val_loader = create_dataloader(val_dataset, config, shuffle=False)

    return train_loader, val_loader