import pickle
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import torch.distributed as dist
import random
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler, Subset

class SignalDataset(Dataset):
    def __init__(self, config, logger, transform=None):
        
        self.config = config
        self.logger = logger
        self.transform = transform

        dataset_dict = pickle.load(open(self.config.dataset_path, 'rb'), encoding='bytes')
        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], dataset_dict.keys())))), [1, 0])

        signals, labels, SNRs = [], [], []
        for mod in mods:
            for snr in snrs:
                signals.append(dataset_dict[(mod, snr)])
                for i in range(dataset_dict[(mod, snr)].shape[0]):
                    labels.append(self.config.label_map[self.config.classes[mod]])
                    SNRs.append(snr)

        self.signals = torch.from_numpy(np.vstack(signals).astype(np.float32))
        self.mods = torch.tensor(labels, dtype=torch.int64)
        self.snrs = torch.tensor(SNRs, dtype=torch.float32)
        
        # select data by snr
        snr_range = torch.tensor(config.snr_range)
        selected_mask = torch.isin(self.snrs, snr_range)
        self.signals, self.mods, self.snrs = self.signals[selected_mask], self.mods[selected_mask], self.snrs[selected_mask]

        self.mask_lab = torch.isin(self.mods, torch.tensor(self.config.known_classes, dtype=torch.int64))
        self.filtered_indices = torch.arange(len(self.mods))

    def __len__(self):
        return len(self.mods)

    # def __getitem__(self, idx):
    #     return self.signals[idx], self.mods[idx], self.snrs[idx]
    
    def __getitem__(self, idx):
        global_idx = int(self.filtered_indices[idx])
        signal = self.signals[global_idx]
        snr = int(self.snrs[global_idx])
        label = int(self.mods[global_idx])
        mask_lab = int(self.mask_lab[global_idx])

        # === views 构造，与 hdf5 逻辑保持一致 ===
        views = [signal]
        for _ in range(self.config.num_views - 1):
            if hasattr(self, "transform") and self.transform is not None:
                views.append(self.transform(signal, float(self.snrs[global_idx])))
            else:
                views.append(signal.clone())
                self.logger.warning('Undefined signal transform, clone from origin signal')

        return views, label, mask_lab, snr, global_idx
    
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
        snrs = torch.tensor([item[3] for item in batch], dtype=torch.long)
        global_idxs = torch.tensor([item[4] for item in batch], dtype=torch.long)
        # return views, labels, masks, snrs, global_idxs
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

    train_idxs, val_idxs = train_test_split(idxs, test_size=config.val_ratio, shuffle=True, random_state=42, stratify=keys)

    train_ds = Subset(dataset, train_idxs.tolist())
    val_ds   = Subset(dataset, val_idxs.tolist())

    train_loader = create_dataloader(train_ds, config, shuffle=True)
    val_loader   = create_dataloader(val_ds,   config, shuffle=False)

    return train_loader, val_loader
