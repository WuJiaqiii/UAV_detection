import os
import re
import glob
import numpy as np
import torch
import cv2

from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler, Subset
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import torch.distributed as dist

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None


def _normalize_dataset_paths(dataset_path):
    if dataset_path is None:
        return []
    if isinstance(dataset_path, (list, tuple)):
        return [str(p) for p in dataset_path if str(p).strip()]
    s = str(dataset_path).strip()
    return [s] if s else []


def _collect_data_files(dataset_path, input_type: str):
    dataset_paths = _normalize_dataset_paths(dataset_path)
    if len(dataset_paths) == 0:
        raise ValueError("config.dataset_path must not be empty")

    data_files = []
    for path in dataset_paths:
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower().lstrip('.')
            if ext != input_type:
                raise ValueError(f"dataset_path file type {ext} does not match input_type={input_type}: {path}")
            data_files.append(path)
        elif os.path.isdir(path):
            pattern = "*." + input_type
            data_files.extend(sorted(glob.glob(os.path.join(path, pattern))))
        else:
            raise ValueError(f"config.dataset_path element must be an existing file or directory, got: {path}")

    data_files = sorted(set(data_files))
    if not data_files:
        joined = ", ".join(dataset_paths)
        raise FileNotFoundError(f"No *.{input_type} files found under: {joined}")
    return dataset_paths, data_files


def _sample_group_key(fp: str) -> str:
    """
    Group samples by original signal identity, so slices from the same signal
    (e.g. ...-Figure-1.mat, ...-Figure-2.mat) stay in the same split.
    """
    stem = os.path.splitext(os.path.basename(fp))[0]
    stem = re.sub(r"-Figure-\d+$", "", stem)
    return stem


def _grouped_train_val_split(dataset, val_ratio: float, random_state: int = 42):
    idxs = np.arange(len(dataset))
    if len(idxs) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    groups = np.array([_sample_group_key(dataset.samples[i]["fp"]) for i in idxs], dtype=object)
    unique_groups = np.unique(groups)

    # Fallback: if every sample is unique or there is only one group, keep old behavior.
    if len(unique_groups) <= 1:
        return train_test_split(
            idxs,
            test_size=val_ratio,
            shuffle=True,
            random_state=random_state,
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(splitter.split(idxs, groups=groups))
    return idxs[train_idx], idxs[val_idx]


class UAVDataset(Dataset):
    """
    Unified dataloader for single-signal training and multi-signal inference.

    Returns unified sample format:
        x, targets, snr, fp

    where targets is always a list[dict].
    """

    _SIGNAL_RE = re.compile(r'(?P<protocol>[A-Za-z0-9_]+)-\[(?P<bracket>[^\]]+)\]')
    _SNR_RE = re.compile(r'-SNR-(?P<snr>[-+]?\d+(?:\.\d+)?)')

    def __init__(self, config, logger, validate_on_init: bool = False):
        self.config = config
        self.logger = logger
        self.input_type = str(config.input_type).lower()
        self.exclude_set = set(getattr(config, "exclude_classes", []) or [])
        self.run_mode = str(getattr(config, "run_mode", "train")).lower()
        self.train_signal_mode = str(getattr(config, "train_signal_mode", "single")).lower()

        if self.input_type not in {"mat", "png"}:
            raise ValueError(f"Unsupported input_type={self.input_type}, expected 'mat' or 'png'")

        self.mod2label = {str(k): int(v) for k, v in getattr(config, "classes", {}).items()}

        dataset_paths, data_files = _collect_data_files(config.dataset_path, self.input_type)

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

            all_protocols = [m.group("protocol").strip() for m in matches]
            if self.exclude_set and any(p in self.exclude_set for p in all_protocols):
                bad += 1
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

            # Minimal-change single-signal training policy:
            # keep unified dataloader format, but skip samples with multi-target annotations.
            if self.run_mode == "train" and self.train_signal_mode == "single":
                if len(targets) != 1:
                    skip(f"single-signal train requires exactly 1 target, got {len(targets)}", fname)
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

        logger.info(f"Indexed {len(self.samples)} samples from {dataset_paths} (skipped {bad})")
        logger.info(f"Dataset input_type={self.input_type}, validate_on_init={validate_on_init}, run_mode={self.run_mode}, train_signal_mode={self.train_signal_mode}")

    def _load_x(self, fp: str) -> torch.Tensor:
        if self.input_type == "png":
            x = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if x is None:
                raise RuntimeError(f"Failed to read png: {fp}")
            x = np.ascontiguousarray(x)
            return torch.from_numpy(x)

        if loadmat is None:
            raise ImportError("scipy is required for mat mode. Please install scipy.")

        mat_key = getattr(self.config, "mat_key", "summed_submatrices")
        mat = loadmat(fp, variable_names=[mat_key])
        if mat_key not in mat:
            raise KeyError(f"key '{mat_key}' not found in mat")

        x = np.asarray(mat[mat_key])
        if x.ndim != 2:
            raise ValueError(f"data dim != 2, got shape {x.shape}")

        x = np.asarray(x, dtype=np.float32)
        x = np.ascontiguousarray(x)
        return torch.from_numpy(x)

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


def create_infer_dataloader(dataset: Dataset, config):
    sampler = None
    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False,
        )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=multi_signal_collate_fn,
    )


def get_dataloader(dataset, config, mode="train"):
    mode = str(mode).lower()
    if mode == "infer":
        return create_infer_dataloader(dataset, config)

    train_idx, val_idx = _grouped_train_val_split(
        dataset,
        val_ratio=float(config.val_ratio),
        random_state=42,
    )

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError(
            f"Grouped train/val split failed: train={len(train_idx)}, val={len(val_idx)}. "
            f"Please check dataset size / val_ratio / grouping rule."
        )

    if hasattr(dataset, "logger"):
        train_groups = len(set(_sample_group_key(dataset.samples[i]["fp"]) for i in train_idx.tolist()))
        val_groups = len(set(_sample_group_key(dataset.samples[i]["fp"]) for i in val_idx.tolist()))
        dataset.logger.info(
            f"Grouped split by signal source: train_samples={len(train_idx)}, val_samples={len(val_idx)}, "
            f"train_groups={train_groups}, val_groups={val_groups}"
        )

    train_dataset = Subset(dataset, train_idx.tolist())
    val_dataset = Subset(dataset, val_idx.tolist())

    train_loader = create_dataloader(train_dataset, config, shuffle=True)
    val_loader = create_dataloader(val_dataset, config, shuffle=False)
    return train_loader, val_loader
