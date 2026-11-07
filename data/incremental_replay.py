"""Recording-level, train-only rehearsal for the IQ incremental pipeline."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
from types import SimpleNamespace

from torch.utils.data import DataLoader, Sampler

from data.data_loader import multi_signal_collate_fn
from data.iq_dat_loader import RawIQDataset
from util.recordings import file_sha256, group_recordings


def select_replay_records(manifest, classes, old_count, per_class, seed, training, validation):
    """Validate partitions, then choose one file per original training recording.

    The manifest must include all new train and validation inputs, plus the old
    training candidates. Hash/identity aliases may never cross split boundaries.
    Calibration and test rows are allowed as audit metadata, never as candidates.
    """
    manifest = Path(manifest).expanduser().resolve()
    document = json.loads(manifest.read_text(encoding="utf-8"))
    source = document.get("records") if isinstance(document, dict) else None
    if not isinstance(source, list) or not source:
        raise ValueError("Replay manifest requires a nonempty records list")
    rows, paths = [], set()
    for original in source:
        row = dict(original)
        path = Path(row["path"]).expanduser()
        path = str((manifest.parent / path).resolve() if not path.is_absolute() else path.resolve())
        if path in paths:
            raise ValueError("Duplicate replay manifest path: " + path)
        paths.add(path)
        if (row.get("split") not in {"train", "val", "calibration", "test"}
                or not isinstance(row.get("class_name"), str)
                or not isinstance(row.get("recording_id"), str) or not row["recording_id"].strip()
                or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("source_sha256", "")))):
            raise ValueError("Invalid replay manifest metadata: " + path)
        row["path"] = path
        rows.append(row)
    groups = group_recordings(rows)
    for group in groups:
        if len({r["split"] for r in group}) != 1:
            raise ValueError("Original recording/content crosses replay partitions: " + group[0]["path"])
    by_path = {r["path"]: r for r in rows}
    for split, samples in (("train", training), ("val", validation)):
        for sample in samples:
            path = str(Path(sample["fp"]).resolve())
            row = by_path.get(path)
            if row is None or row["split"] != split or row["class_name"] != sample["class_name"]:
                raise ValueError(f"{split} input missing from the matching manifest partition: {path}")
    candidates = defaultdict(list)
    old_names = [name for name, idx in classes.items() if idx < old_count]
    for group in groups:
        if group[0]["split"] == "train" and group[0]["class_name"] in old_names:
            candidates[group[0]["class_name"]].append(min(group, key=lambda r: r["path"]))
    selected = []
    for name in old_names:
        available = candidates[name]
        if len(available) < per_class:
            raise ValueError(f"Replay needs {per_class} independent TRAIN recordings for {name}; found {len(available)}")
        available.sort(key=lambda r: hashlib.sha256(f'{seed}:{name}:{r["recording_id"]}'.encode()).hexdigest())
        for row in available[:per_class]:
            if file_sha256(row["path"]) != row["source_sha256"]:
                raise ValueError("Replay recording content changed: " + row["path"])
            selected.append(row)
    return selected


class ReplayBatchSampler(Sampler):
    """Cover every new input once per epoch; cycle class-balanced old records.

    The last batch may be smaller. A fresh local RNG per epoch avoids consuming
    the RNG used for IQ crops. set_epoch makes a resumed schedule reproducible.
    """
    def __init__(self, labels, old_count, batch_size, fraction=.5, seed=42):
        self.new = [i for i, label in enumerate(labels) if label >= old_count]
        self.old = defaultdict(list)
        for i, label in enumerate(labels):
            if label < old_count:
                self.old[int(label)].append(i)
        if not self.new or not self.old or batch_size < 2 or not 0 < fraction < 1:
            raise ValueError("Replay batches require old/new data, batch_size>=2 and 0<fraction<1")
        self.old_size = min(batch_size-1, max(1, int(batch_size * fraction)))
        self.new_size = batch_size-self.old_size
        self.seed, self.epoch = int(seed), 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return math.ceil(len(self.new)/self.new_size)

    def __iter__(self):
        rng = random.Random(self.seed+self.epoch)
        new = self.new.copy()
        rng.shuffle(new)
        classes, pools, positions = [], {}, defaultdict(int)
        for label, indices in self.old.items():
            pools[label] = indices.copy()
            rng.shuffle(pools[label])
        for start in range(0, len(new), self.new_size):
            batch = new[start:start+self.new_size]
            # Preserve the requested ratio in the partial final batch.
            count = max(1, round(len(batch)*self.old_size/self.new_size))
            for _ in range(count):
                if not classes:
                    classes = sorted(self.old)
                    rng.shuffle(classes)
                label = classes.pop()
                if positions[label] == len(pools[label]):
                    rng.shuffle(pools[label])
                    positions[label] = 0
                batch.append(pools[label][positions[label]])
                positions[label] += 1
            rng.shuffle(batch)
            yield batch


def create_replay_loader(config, train_dataset, val_dataset, logger=None):
    selected = select_replay_records(config.lwf_replay_manifest, config.classes,
        int(config.lwf_old_num_classes), int(config.lwf_replay_per_class), int(config.seed),
        train_dataset.samples, val_dataset.samples)
    options = SimpleNamespace(**vars(config))
    options.train_data_include_classes = []
    paths = [s["fp"] for s in train_dataset.samples] + [s["path"] for s in selected]
    dataset = RawIQDataset(options, logger, dataset_path=paths, split_name="train")
    sampler = ReplayBatchSampler([s["label"] for s in dataset.samples], config.lwf_old_num_classes,
                                 config.batch_size, config.lwf_replay_fraction, config.seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=config.num_workers,
                        pin_memory=True, collate_fn=multi_signal_collate_fn)
    audit = {"strategy": "record_balanced_lwf_replay", "manifest": str(Path(config.lwf_replay_manifest).resolve()),
             "manifest_sha256": file_sha256(config.lwf_replay_manifest),
             "per_class_recordings": int(config.lwf_replay_per_class), "records": selected,
             "new_files_per_epoch": len(train_dataset), "batches_per_epoch": len(sampler),
             "old_per_full_batch": sampler.old_size, "new_per_full_batch": sampler.new_size}
    target = Path(config.result_dir)/"replay_selection.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    if logger:
        logger.info(f"[Replay] {len(selected)} old TRAIN recordings; audit={target}")
    return loader
