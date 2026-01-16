import os
import re
import glob
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler, Subset, TensorDataset
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

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        dataset_dir = config.dataset_path
        if not dataset_dir or not os.path.isdir(dataset_dir):
            raise ValueError(f"config.dataset_path must be an existing directory, got: {dataset_dir}")

        self.mod2label = {str(k): int(v) for k, v in self.config.classes.items()}
        self.mat_key = "summed_submatrices"

        mat_files = sorted(glob.glob(os.path.join(dataset_dir, "*.mat")))
        if not mat_files:
            raise FileNotFoundError(f"No .mat files found under: {dataset_dir}")

        data_list, protocol_list, freq_list, bw_list, snr_list = [], [], [], [], []

        bad_files = 0
        for fp in mat_files:
            fname = os.path.basename(fp)

            m = self._FNAME_RE.search(fname)
            if not m:
                bad_files += 1
                self.logger.warning(f"[Skip] filename regex not matched: {fname}")
                continue

            protocol = m.group("protocol").strip()
            bracket = m.group("bracket")
            snr_str = m.group("snr")

            parts = [p.strip() for p in bracket.split(",")]
            if len(parts) < 4:
                bad_files += 1
                self.logger.warning(f"[Skip] bracket parse failed (need 4 fields): {fname}")
                continue

            try:
                freq = float(parts[1])
                bw = float(parts[3])
            except ValueError:
                bad_files += 1
                self.logger.warning(f"[Skip] freq/bw parse failed: {fname}")
                continue

            snr = None
            if snr_str is not None:
                try:
                    snr = float(snr_str)
                except ValueError:
                    snr = None  

            try:
                mat = loadmat(fp)
                if self.mat_key not in mat:
                    bad_files += 1
                    self.logger.warning(f"[Skip] key '{self.mat_key}' not found in mat: {fname}")
                    continue

                x = mat[self.mat_key]  #  (512, 750)
                x = np.asarray(x)

                if x.ndim != 2:
                    bad_files += 1
                    self.logger.warning(f"[Skip] data dim != 2, got {x.shape}: {fname}")
                    continue

                x = x.astype(np.float32)

            except Exception as e:
                bad_files += 1
                self.logger.warning(f"[Skip] failed to load mat '{fname}': {e}")
                continue

            if protocol not in self.mod2label:
                raise KeyError(
                    f"protocol '{protocol}' not found in config.classes mapping. "
                    f"Available keys (sample): {list(self.mod2label.keys())[:10]} ..."
                )
            proto_out = self.mod2label[protocol]

            data_list.append(x)
            protocol_list.append(proto_out)
            freq_list.append(freq)
            bw_list.append(bw)
            snr_list.append(snr)

        if not data_list:
            raise RuntimeError(
                f"All files were skipped. Check filename format / mat key / directory. "
                f"bad_files={bad_files}, total={len(mat_files)}"
            )

        self.data = torch.from_numpy(np.stack(data_list, axis=0))  #  (N, 512, 750)
        self.protocol = torch.tensor(protocol_list, dtype=torch.int64)
        self.freq = torch.tensor(freq_list, dtype=torch.float32)
        self.bw = torch.tensor(bw_list, dtype=torch.float32)
        self.snr = torch.tensor([float(s) if s is not None else float("nan") for s in snr_list],
                                dtype=torch.float32)

        self.logger.info(f"Loaded {len(self.data)} samples from {dataset_dir} (skipped {bad_files})")
        self.logger.info(f"Data shape: {list(self.data.shape)}")

    def __len__(self):
        return len(self.protocol)

    def __getitem__(self, idx):
        return self.data[idx], self.protocol[idx], self.freq[idx], self.bw[idx], self.snr[idx]


class RandomSampler(Sampler): 
    def __init__(self, data_source, sample_ratio):
        self.data_source = data_source
        self.sample_ratio = sample_ratio
        self.num_samples = len(data_source)

    def __iter__(self):
        k = int(self.num_samples * self.sample_ratio)
        sampled = random.sample(range(self.num_samples), k)
        return iter(sampled)

    def __len__(self):
        return int(self.num_samples * self.sample_ratio)

def build_ddp_sampler(dataset, shuffle: bool, sample_ratio: float):

    rank = dist.get_rank()
    world = dist.get_world_size()
    num_keep = round(len(dataset) * sample_ratio)
    device = torch.device(f"cuda:{rank}") 

    if rank == 0:
        idx_tensor = torch.randperm(len(dataset), device=device)[:num_keep]
    else:
        idx_tensor = torch.empty(num_keep, dtype=torch.int64, device=device)

    dist.broadcast(idx_tensor, src=0)
    subset = Subset(dataset, idx_tensor.cpu().tolist())
    sampler = DistributedSampler(subset, num_replicas=world, rank=rank, shuffle=shuffle, drop_last=False)
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
