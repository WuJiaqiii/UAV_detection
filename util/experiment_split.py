"""Reproducible known-only training splits, grouped by original recording."""
from collections import defaultdict
import logging
import math
from types import SimpleNamespace

import numpy as np
from util.recordings import group_recordings as _recording_groups


def split_dataset(args, train_args):
    from util.eval_common import collect_inputs
    from util.multi_signal_eval import load_multi_truth, read_path_mapping
    from util.open_set import file_sha256, recording_identity

    suffix = ".dat" if args.input_type == "iq_dat" else "." + args.input_type
    files = [p for p in collect_inputs(args.inputs, args.recursive) if p.suffix.lower() == suffix]
    if not files:
        raise ValueError(f"No {suffix} files in experiment inputs")
    truth = load_multi_truth(None, files, {}, allow_unknown=True,
                             label_from_parent=args.input_type == "iq_dat",
                             label_from_filename=args.input_type != "iq_dat")
    if any(len(names) != 1 for names in truth.values()):
        raise ValueError("Experiment splitting requires one labeled signal class per file")
    labels = sorted({names[0] for names in truth.values()})
    excluded = args.exclude_classes
    if len(excluded) != len(set(excluded)) or set(excluded) - set(labels):
        raise ValueError("exclude_classes must contain unique discovered class names")
    known, unknown = _class_partition(sorted(set(labels) - set(excluded)), args.known_classes, args.unknown_classes)

    # Reuse the real loaders' metadata validation, without reading spectra or IQ.
    cfg = SimpleNamespace(**vars(train_args))
    cfg.classes = {name: i for i, name in enumerate(labels)}
    active_files = [p for p in files if truth[str(p)][0] not in excluded]
    cfg.dataset_path = list(map(str, active_files))
    logger = logging.getLogger("experiment_split")
    if args.input_type == "iq_dat":
        from data.iq_dat_loader import RawIQDataset
        dataset = RawIQDataset(cfg, logger)
    else:
        from data.data_loader import UAVDataset
        dataset = UAVDataset(cfg, logger)
    if len(dataset.samples) != len(active_files):
        raise ValueError("Some experiment inputs were rejected by the training loader; inspect filenames/labels")

    identities = read_path_mapping(args.recording_ids_json) if args.recording_ids_json else {}
    if any(not isinstance(value, str) or not value.strip() for value in identities.values()):
        raise ValueError("Recording IDs must be nonempty strings")
    rows = []
    for path in files:
        digest = file_sha256(path)
        rows.append({"path": str(path), "class_name": truth[str(path)][0],
                     "source_sha256": digest,
                     "recording_id": recording_identity(path, digest, identities)})
    groups = _recording_groups([row for row in rows if row["class_name"] not in excluded])
    for row in rows:
        if row["class_name"] in excluded:
            row["split"] = "excluded"
    ratios = (args.val_ratio, args.calibration_ratio, args.test_ratio)
    if any(not math.isfinite(r) or r <= 0 for r in ratios) or sum(ratios) >= 1:
        raise ValueError("val/calibration/test ratios must be positive and sum to less than 1")
    rng = np.random.default_rng(args.seed)
    by_class = defaultdict(list)
    for group in groups:
        by_class[group[0]["class_name"]].append(group)
    for name in sorted(by_class):
        items = by_class[name]
        if name in excluded:
            partitions = [("excluded", items)]
        elif name in unknown:
            partitions = [("test_unknown", items)]
        else:
            items = [items[int(i)] for i in rng.permutation(len(items))]
            counts = [max(1, int(len(items) * ratio)) for ratio in ratios]
            if sum(counts) >= len(items):
                raise ValueError(f"Known class {name!r} needs more independent recordings for four nonempty splits; got {len(items)}")
            partitions, start = [], 0
            for split, count in zip(("val", "calibration", "test_known", "train"),
                                    [*counts, len(items) - sum(counts)]):
                partitions.append((split, items[start:start + count]))
                start += count
        for split, assigned in partitions:
            for group in assigned:
                for row in group:
                    row["split"] = split
    summary = {}
    for name in labels:
        summary[name] = {}
        for split in ("train", "val", "calibration", "test_known", "test_unknown", "excluded"):
            selected = [r for r in rows if r["class_name"] == name and r["split"] == split]
            summary[name][split] = {"files": len(selected),
                                    "recordings": len({r["recording_id"] for r in selected})}
    return {"schema_version": 1, "input_type": args.input_type, "seed": args.seed,
            "known_classes": known, "unknown_classes": unknown, "excluded_classes": excluded,
            "classes": {name: i for i, name in enumerate(known)},
            "ratios": dict(zip(("val", "calibration", "test_known"), ratios)),
            "summary": summary, "records": rows}


def _class_partition(labels, known, unknown):
    universe = set(labels)
    for names in (known, unknown):
        if names is not None and (not names or len(names) != len(set(names)) or
                                  any(not str(n).strip() for n in names) or set(names) - universe):
            raise ValueError(f"Class selection must contain unique discovered names; discovered={labels}, selected={names}")
    if known is None and unknown is None:
        raise ValueError("Specify known_classes or unknown_classes; the other defaults to its complement")
    known = list(known) if known is not None else sorted(universe - set(unknown))
    unknown = list(unknown) if unknown is not None else sorted(universe - set(known))
    if not known or not unknown or set(known) & set(unknown) or set(known) | set(unknown) != universe:
        raise ValueError("Known/unknown classes must be nonempty, disjoint and cover the discovered classes")
    return known, unknown


