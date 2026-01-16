import h5py
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler, Subset
from sklearn.model_selection import train_test_split
import torch.distributed as dist

class SignalDataset(Dataset):
    """
    Signal dataset based on hdf5
      - 'X': shape (N, C, L), IQ signals
      - 'mod': shape (N,), labels(int)
      - 'snr': shape (N,), signals snrs
    返回: views, label, mask_lab, global_idx
    """
    def __init__(self, config, logger, transform=None):

        self.config = config
        self.logger = logger
        self.transform = transform

        self.h5_path = self.config.dataset_path
        self._h5 = None # hdf5文件句柄

        # 只在初始化时读取标签与snr，数据懒加载（节省内存）
        with h5py.File(self.h5_path, 'r') as f:
            self.mods = f['mod'][:]
            self.snrs = f['snr'][:]

        # 根据config设置将标签重新映射，所有未知类映射到最后
        self.mods = np.array([self.config.label_map[x] for x in self.mods], dtype=np.int64)
        # mask_lab: True 表示已知类，False 表示未知类
        self.mask_lab = np.isin(self.mods, self.config.known_classes) 
        
        # 根据config设置挑选固定SNR范围的信号
        snr_values = np.array(self.config.snr_range)
        snr_mask = np.isin(self.snrs, snr_values)
        self.filtered_indices = np.where(snr_mask)[0]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        # 延迟打开 h5 文件（适配多进程 DataLoader）
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')

        # filtered_indices 映射到原始全局索引
        global_idx = int(self.filtered_indices[idx])
        X = self._h5['X'][global_idx]
        signal = torch.from_numpy(X.astype(np.float32))

        label = int(self.mods[global_idx])
        mask_lab = int(self.mask_lab[global_idx])

        views = [signal]
        for _ in range(self.config.num_views - 1):
            if self.transform:
                views.append(self.transform(signal, float(self.snrs[global_idx])))
            else:
                views.append(signal.clone())
                self.logger.warning('Undefined signal tranform, clone from origin signal')

        # 返回 views, label, mask_lab, global_idx（便于上层记录/对齐）
        return views, label, mask_lab, global_idx
    
    def close(self):
        if self._h5 is not None:
            self._h5.close()

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

    def collate_fn(batch):
        # batch 中每个 item = views, label, mask, global_idx
        batch_views = [item[0] for item in batch]
        views = [torch.stack(vs, dim=0) for vs in zip(*batch_views)]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        masks  = torch.tensor([item[2] for item in batch], dtype=torch.bool)
        global_idxs = torch.tensor([item[3] for item in batch], dtype=torch.long)
        return views, labels, masks, global_idxs

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
            collate_fn=collate_fn,
            drop_last=False
        )

def get_dataloader(dataset, config):

    idxs = np.arange(len(dataset))
    labels = dataset.mods[dataset.filtered_indices]
    snrs = dataset.snrs[dataset.filtered_indices]
    keys = [f"{l}_{s}" for l, s in zip(labels, snrs)]
    
    if config.val_ratio is None or config.val_ratio <= 0.0:
        train_idxs = idxs
        val_idxs = idxs
    elif config.val_ratio >= 1.0:
        raise ValueError(f"config.val_ratio must be in [0,1), got {config.val_ratio}")
    else:
        train_idxs, val_idxs = train_test_split(idxs, test_size=config.val_ratio, shuffle=True, random_state=42, stratify=keys)

    train_ds = Subset(dataset, train_idxs.tolist())
    val_ds   = Subset(dataset, val_idxs.tolist())

    train_loader = create_dataloader(train_ds, config, shuffle=True)
    val_loader   = create_dataloader(val_ds,   config, shuffle=False)

    return train_loader, val_loader
