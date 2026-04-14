import os
import re
import glob
import numpy as np
import torch
import cv2

from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler, Subset
from sklearn.model_selection import train_test_split
import torch.distributed as dist

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None


class UAVDataset(Dataset):
    """
    Dataloader for multi-UAV .mat or .png signal data

    Data name example:
        FPV1-[0,-0.1,1000,17]-Ocusync21-[0,65.3,1000,18]-Ocusync41-[0,-5.5,1000,38]-SNR-6-SNRSPACE17-Figure-1.mat
    """

    _SIGNAL_RE = re.compile(r'(?P<protocol>[A-Za-z0-9_]+)-\[(?P<bracket>[^\]]+)\]')
    _SNR_RE = re.compile(r'-SNR-(?P<snr>[-+]?\d+(?:\.\d+)?)')

    def __init__(self, config, logger, validate_on_init: bool = False):
        
        self.config = config
        self.logger = logger
        self.input_type = str(config.input_type).lower()

        if not config.dataset_path or not os.path.isdir(config.dataset_path):
            raise ValueError(f"config.dataset_path must be an existing directory, got: {config.dataset_path}")
        if self.input_type not in {"mat", "png"}:
            raise ValueError(f"Unsupported input_type={self.input_type}, expected 'mat' or 'png'")

        self.mod2label = {str(k): int(v) for k, v in getattr(config, "classes", {}).items()}

        pattern = "*." + self.input_type
        data_files = sorted(glob.glob(os.path.join(config.dataset_path, pattern)))
        if not data_files:
            raise FileNotFoundError(f"No {pattern} files found under: {config.dataset_path}")

        self.samples: List[Dict[str, Any]] = []
        bad = 0

        def skip(reason: str, fname: str):
            nonlocal bad
            bad += 1
            logger.warning(f"[Skip] {reason}: {fname}")

        for fp in data_files:
            fname = os.path.basename(fp)
            base = os.path.splitext(fname)[0]

            snr = float("nan")
            m_snr = self._SNR_RE.search(base)
            if m_snr is not None:
                try:
                    snr = float(m_snr.group("snr"))
                except ValueError:
                    snr = float("nan")

            prefix = base.split("-SNR-")[0]

            matches = list(self._SIGNAL_RE.finditer(prefix))
            if not matches:
                skip("no signal blocks parsed from filename", fname)
                continue

            targets: List[Dict[str, Any]] = []
            parse_failed = False

            for m in matches:
                protocol = m.group("protocol").strip()
                if protocol not in self.mod2label:
                    logger.warning(f"[SkipTarget] protocol not in config.classes: {protocol} ({fname})")
                    continue

                parts = [p.strip() for p in m.group("bracket").split(",")]
                if len(parts) < 4:
                    logger.warning(f"[SkipTarget] bracket parse failed (need >=4 fields): {protocol} ({fname})")
                    continue

                try:
                    center_freq = float(parts[1])
                    upper_freq = float(parts[3])
                    bandwidth = 2.0 * abs(upper_freq - center_freq)
                except ValueError:
                    logger.warning(f"[SkipTarget] center_freq/upper_freq parse failed: {protocol} ({fname})")
                    continue

                targets.append({
                    "label": self.mod2label[protocol],
                    "class_name": protocol,
                    "center_freq": float(center_freq),
                    "bandwidth": float(bandwidth),
                })

            if len(targets) == 0:
                skip("all targets invalid or filtered out", fname)
                continue

            if validate_on_init:
                try:
                    _ = self._load_x(fp)
                except Exception as e:
                    skip(f"validate_on_init failed ({e})", fname)
                    parse_failed = True

            if parse_failed:
                continue

            self.samples.append({
                "fp": fp,
                "targets": targets,
                "snr": snr,
            })

        if not self.samples:
            raise RuntimeError(f"All files were skipped. bad={bad}, total={len(data_files)}")

        logger.info(f"Indexed {len(self.samples)} samples from {config.dataset_path} (skipped {bad})")
        logger.info(f"Dataset input_type={self.input_type}, validate_on_init={validate_on_init}")

    def _load_x(self, fp: str) -> torch.Tensor:
        if self.input_type == "png":
            x = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if x is None:
                raise RuntimeError(f"Failed to read png: {fp}")
            x = np.ascontiguousarray(x)
            return torch.from_numpy(x)  # uint8, (H, W)

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
        return torch.from_numpy(x)  # float32, (H, W)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        fp = sample["fp"]

        try:
            x = self._load_x(fp)
        except Exception as e:
            raise RuntimeError(f"Failed to load '{os.path.basename(fp)}': {e}") from e

        return x, sample["targets"], sample["snr"], fp


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
        subset,
        num_replicas=world,
        rank=rank,
        shuffle=shuffle,
        drop_last=False
    )
    return subset, sampler

def multi_signal_collate_fn(batch: List[Tuple[torch.Tensor, List[Dict[str, Any]], float, str]]):
    xs, targets_list, snrs, fps = zip(*batch)
    xs = torch.stack(xs, dim=0)
    snrs = torch.as_tensor(snrs, dtype=torch.float32)
    return xs, list(targets_list), snrs, list(fps)


def create_dataloader(dataset: Dataset, config, shuffle: bool):
    if dist.is_initialized():
        dataset, sampler = build_ddp_sampler(
            dataset,
            shuffle=shuffle,
            sample_ratio=config.sample_ratio
        )
    else:
        sampler = RandomSampler(dataset, config.sample_ratio)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None and shuffle),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=multi_signal_collate_fn,
    )

def get_dataloader(dataset, config):
    
    idxs = np.arange(len(dataset))

    train_idx, val_idx = train_test_split(
        idxs,
        test_size=config.val_ratio,
        shuffle=True,
        random_state=42,
    )

    train_dataset = Subset(dataset, train_idx.tolist())
    val_dataset = Subset(dataset, val_idx.tolist())

    train_loader = create_dataloader(train_dataset, config, shuffle=True)
    val_loader = create_dataloader(val_dataset, config, shuffle=False)

    return train_loader, val_loader