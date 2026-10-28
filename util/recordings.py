"""Shared source identity and leak-free, class-stratified recording splits."""
from collections import defaultdict
import hashlib
from pathlib import Path
import re

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recording_identity(path, digest, explicit=None):
    path = Path(path)
    if str(path) in (explicit or {}):
        return explicit[str(path)]
    match = re.fullmatch(r"(.+)-Figure-\d+", path.stem, flags=re.IGNORECASE)
    return "figure-source:" + match.group(1) if match else digest


def group_recordings(rows):
    """Union content and recording IDs, including transitive aliases."""
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for key in (("hash", row["source_sha256"]), ("id", row["recording_id"])):
            if key in seen:
                parents[find(i)] = find(seen[key])
            seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    for group in groups.values():
        if len({row["class_name"] for row in group}) != 1:
            raise ValueError("One recording/content has conflicting class labels: " + str([r["path"] for r in group]))
        identity = "recording:" + min(row["source_sha256"] for row in group)
        for row in group:
            row["recording_id"] = identity
    return sorted(groups.values(), key=lambda group: min(row["path"] for row in group))


def train_val_indices(samples, val_ratio, seed=42, recording_ids=None):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be in (0, 1)")
    if any(not isinstance(value, str) or not value.strip() for value in (recording_ids or {}).values()):
        raise ValueError("Recording IDs must be nonempty strings")
    rows = []
    for index, sample in enumerate(samples):
        path = str(Path(sample["fp"]).expanduser().resolve())
        label = sample["label"] if "label" in sample else sample["targets"][0]["label"]
        digest = file_sha256(path)
        rows.append({"index": index, "path": path, "class_name": str(label), "source_sha256": digest,
                     "recording_id": recording_identity(path, digest, recording_ids)})
    by_class = defaultdict(list)
    for group in group_recordings(rows):
        by_class[group[0]["class_name"]].append(group)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for name in sorted(by_class):
        groups = by_class[name]
        if len(groups) < 2:
            raise ValueError(f"Class {name} needs at least two independent recordings for an automatic train/val split")
        order = rng.permutation(len(groups))
        count = min(len(groups) - 1, max(1, int(len(groups) * val_ratio)))
        for target, indices in ((val, order[:count]), (train, order[count:])):
            target.extend(row["index"] for i in indices for row in groups[int(i)])
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(val), dtype=np.int64)


def dataset_train_val_indices(samples, config):
    mapping = None
    if getattr(config, "recording_ids_json", None):
        from util.eval_common import read_path_mapping
        mapping = read_path_mapping(config.recording_ids_json)
    return train_val_indices(samples, float(config.val_ratio), int(getattr(config, "seed", 42)), mapping)
