import os
import re
import glob
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler, Subset
from sklearn.model_selection import train_test_split
import torch.distributed as dist

try:
    from scipy.io import loadmat
except ImportError as e:
    raise ImportError(
        "scipy is required to read .mat files. Please install it with: pip install scipy"
    ) from e

class UAVDataset(Dataset):

    _FNAME_RE = re.compile(
        r"""^(?P<protocol>.+?)-\[(?P<bracket>[^\]]+)\](?:-SNR-(?P<snr>[-+]?\d+(?:\.\d+)?))?""",
        re.VERBOSE,
    )# 例：Skylink11-[0,-22.0,1000,20]-SNR-20-SNRSPACE1.046149e+01-Figure-1.mat

    def __init__(self, config, logger, validate_on_init: bool = False):
        self.config = config
        self.logger = logger

        dataset_dir = getattr(config, "dataset_path", None)
        if not dataset_dir or not os.path.isdir(dataset_dir):
            raise ValueError(f"config.dataset_path must be an existing directory, got: {dataset_dir}")

        self.mod2label = {str(k): int(v) for k, v in getattr(config, "classes", {}).items()}

        mat_files = sorted(glob.glob(os.path.join(dataset_dir, "*.mat")))
        if not mat_files:
            raise FileNotFoundError(f"No .mat files found under: {dataset_dir}")

        files, protocol_list, freq_list, bw_list, snr_list = [], [], [], [], []
        
        bad = 0
        
        def skip(reason: str, fname: str):
            nonlocal bad
            bad += 1
            logger.warning(f"[Skip] {reason}: {fname}")

        for fp in mat_files:
            fname = os.path.basename(fp)
            m = self._FNAME_RE.search(fname)
            if not m:
                skip("filename regex not matched", fname)
                continue

            protocol = m.group("protocol").strip()
            if protocol not in self.mod2label:
                raise KeyError(
                    f"protocol '{protocol}' not found in config.classes mapping. "
                    f"Available keys (sample): {list(self.mod2label.keys())[:10]} ..."
                )

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

            # Optional heavy validation (loads the matrix once at init; turn on only for debugging)
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
            raise RuntimeError(f"All files were skipped. bad={bad}, total={len(mat_files)}")

        self.files = files
        self.protocol = torch.tensor(protocol_list, dtype=torch.int64)
        self.freq = torch.tensor(freq_list, dtype=torch.float32)
        self.bw = torch.tensor(bw_list, dtype=torch.float32)
        self.snr = torch.tensor(snr_list, dtype=torch.float32)

        logger.info(f"Indexed {len(self.files)} samples from {dataset_dir} (skipped {bad})")
        logger.info(f"Lazy-loading mat key: '{self.config.mat_key}', validate_on_init={validate_on_init}")

    def _load_x(self, fp: str) -> torch.Tensor:

        mat = loadmat(fp, variable_names=[self.config.mat_key])
        if self.config.mat_key not in mat:
            raise KeyError(f"key '{self.config.mat_key}' not found in mat")

        x = np.asarray(mat[self.config.mat_key])
        if x.ndim != 2:
            raise ValueError(f"data dim != 2, got shape {x.shape}")

        # Ensure float32 + contiguous for torch.from_numpy
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

        return x, self.protocol[idx], self.freq[idx], self.bw[idx], self.snr[idx]

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
    val_dataset   = Subset(dataset, val_idx.tolist())

    train_loader = create_dataloader(train_dataset, config, shuffle=True)
    val_loader   = create_dataloader(val_dataset,   config, shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":
    from types import SimpleNamespace
    import logging
    config = SimpleNamespace()
    config.dataset_path = '/home/jiaqi/uav/dataset'
    config.val_ratio = 0.2
    config.sample_ratio = 1.0
    config.batch_size = 8
    config.num_workers = 1
    logger = logging.getLogger("ckpt")
    
    dataset = UAVDataset(config, logger)
    for i in range(10):
        data, label, freq, bw = dataset[i]
        print(data.shape, type(data), label, freq, bw)
        
    train_loader, val_loader = get_dataloader(dataset, config)
