"""Audited frozen-run inputs and exact binary-mask replay; never reruns detection."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from util.open_set import FEATURE_VERSION, file_sha256, normalize_features, save_npz


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def primary_key(path, track):
    return (-track.get("valid_votes", 0), -track.get("observation_count", 0),
            path, str(track.get("track_id", "")))


@dataclass
class FrozenSplit:
    directory: Path
    metadata: dict
    features: np.ndarray
    records: list
    report: dict
    logits: np.ndarray | None = None
    raw_features: np.ndarray | None = None
    source_recording_ids: dict | None = None

    @property
    def classes(self):
        return self.metadata["classes"]

    @property
    def recording_ids(self):
        return np.asarray([r["recording_id"] for r in self.records])

    def primary_indices(self):
        lookup = {(r["path"], r["track_id"]): i for i, r in enumerate(self.records)}
        identities = {r["path"]: r["recording_id"] for r in self.records}
        identities.update(self.source_recording_ids or {})
        grouped = {}
        for f in self.report["files"]:
            if f["path"] in identities:
                for track in f.get("signals", []):
                    grouped.setdefault(identities[f["path"]], []).append((f["path"], track))
        selected = []
        for _, items in sorted(grouped.items()):
            path, track = min(items, key=lambda pair: primary_key(*pair))
            key = (path, track["track_id"])
            if key in lookup:
                selected.append(lookup[key])
        return np.asarray(selected, dtype=int)


def load_split(directory, archive_name):
    directory = Path(directory)
    archive_path = directory / archive_name
    with np.load(archive_path, allow_pickle=False) as z:
        metadata = json.loads(z["metadata"].item())
        features = normalize_features(z["features"])
        raw = z["raw_features"].copy() if "raw_features" in z else None
        logits = z["logits"].copy() if "logits" in z else None
    if (metadata.get("schema_version") != 1 or metadata.get("feature_version") != FEATURE_VERSION
            or metadata.get("kind") not in {"track_features", "inference_features"}):
        raise ValueError("Expected a current binary-mask track/inference feature archive")
    report = read_json(directory / "results.json")
    tracks = {(f["path"], t["track_id"]): t for f in report["files"] for t in f.get("signals", [])}
    records = []
    seen = set()
    for original in metadata["records"]:
        r = original.copy()
        key = (r["path"], r["track_id"])
        if key in seen or key not in tracks or not r.get("recording_id"):
            raise ValueError(f"Duplicate, missing, or unidentified archived track: {key}")
        seen.add(key)
        t = tracks[key]
        r.update(valid_votes=t.get("valid_votes", 0), observation_count=t.get("observation_count", 0),
                 head_class=t.get("known_candidate_class") or t.get("candidate_class_name") or t.get("class_name"),
                 profile_class_name=t.get("profile_class_name"))
        labels = r.get("true_class_names")
        if "true_class_name" not in r:
            r["true_class_name"] = labels[0] if labels and len(labels) == 1 else None
        records.append(r)
    if not records or len(records) != len(features):
        raise ValueError("Frozen archive records and features must be nonempty and aligned")
    return FrozenSplit(directory, metadata, features, records, report, logits, raw)


def validate_splits(reference, calibration, test):
    """All fitting inputs are known and independent, before reading test truth."""
    splits = (reference, calibration, test)
    if any(s.classes != reference.classes for s in splits):
        raise ValueError("Archive class mappings differ")
    signatures = [s.metadata.get("signature", {}).get("sha256") for s in splits]
    if not signatures[0] or len(set(signatures)) != 1:
        raise ValueError("Feature-producing signatures differ; use one frozen model/frontend")
    if len({s.features.shape[1] for s in splits}) != 1:
        raise ValueError("Feature dimensions differ")
    for s in (reference, calibration):
        if any(r.get("true_class_name") not in reference.classes for r in s.records):
            raise ValueError("Reference/calibration must contain only single-known-class recordings")
        names_by_id = {}
        for r in s.records:
            names_by_id.setdefault(r["recording_id"], set()).add(r["true_class_name"])
        if any(len(v) != 1 for v in names_by_id.values()):
            raise ValueError("A reference/calibration recording has conflicting labels")
    for key in ("recording_id", "source_sha256"):
        sets = [{r[key] for r in s.records if r.get(key)} for s in splits]
        for i in range(3):
            for j in range(i + 1, 3):
                if sets[i] & sets[j]:
                    raise ValueError(f"Reference/calibration/test overlap by {key}")
    return {"signature": signatures[0], "splits": {
        name: {"tracks": len(s.records), "valid_recordings": len(set(s.recording_ids)),
               "archive_directory": str(s.directory)}
        for name, s in zip(("reference", "calibration", "test"), splits)}}


def replay_classifier(splits, checkpoint_path, output, batch_size=32, device="cpu", threads=4,
                      tolerance=5e-3):
    """Recreate the current binary masks from accepted saved groups, with parity checks.

    Only valid for the binary-mask representation. It does not recover amplitude/IQ.
    Tracks use mean raw features/logits, and the original normalized mean-of-unit-
    observations for kNN. Track vote labels stay as originally recorded.
    """
    import torch
    from model.resnet import MaskImageClassifier
    from util.checkpoint import load_checkpoint_object
    from util.eval_common import resolve_config
    from util.mask_input import build_mask_input
    from util.spectrogram_frontend import SPEC_SHAPE

    checkpoint_path = Path(checkpoint_path)
    digest = file_sha256(checkpoint_path)
    expected = splits[0].metadata["signature"]["details"]["checkpoint_sha256"]
    if digest != expected:
        raise ValueError("Replay checkpoint hash differs from the original feature-producing model")
    checkpoint = load_checkpoint_object(checkpoint_path)
    torch.set_num_threads(threads)
    cfg = resolve_config(checkpoint)
    if cfg.classes != splits[0].classes:
        raise ValueError("Replay checkpoint class mapping differs")
    model = MaskImageClassifier(cfg.backbone, cfg.num_classes, cfg.mask_in_chans,
                                pretrained=False, dropout=cfg.cnn_dropout).to(device).eval()
    state = checkpoint["model_state_dict"]
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    head = model.net.fc[-1] if cfg.backbone.startswith("resnet") else model.net.classifier[-1]
    parameters = {"weight": head.weight.detach().cpu().numpy(),
                  "bias": head.bias.detach().cpu().numpy() if head.bias is not None else np.zeros(cfg.num_classes)}
    evidence = []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, split in zip(("reference", "calibration", "test"), splits):
        source_cfg = split.report["effective_config"]
        for key in ("backbone", "mask_img_size", "mask_in_chans", "mask_fill_value"):
            if source_cfg[key] != getattr(cfg, key):
                raise ValueError(f"Saved mask settings differ for {key}")
        wanted = {(r["path"], r["track_id"]): i for i, r in enumerate(split.records)}
        observations, owners, original_probabilities = [], [], []
        with (split.directory / "slices.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                for group in row.get("groups", []):
                    key = (row["path"], group.get("track_id"))
                    if key in wanted and group.get("vote_accepted"):
                        observations.append(group["selected_boxes"])
                        owners.append(wanted[key])
                        original_probabilities.append(group["probabilities"])
        if not observations:
            raise ValueError(f"No accepted masks to replay in {split.directory}")
        raw_parts, logit_parts = [], []
        probability_error, argmax_disagreements = 0., 0
        empty_spectrum = np.empty(SPEC_SHAPE, dtype=np.uint8)
        for start in range(0, len(observations), batch_size):
            batch = torch.stack([build_mask_input(empty_spectrum, b, cfg)[0]
                                 for b in observations[start:start+batch_size]]).to(device)
            with torch.inference_mode():
                logits, raw = model(batch, return_features=True)
            prediction = logits.softmax(1).cpu().numpy()
            expected_prob = np.asarray(original_probabilities[start:start+len(batch)])
            probability_error = max(probability_error, float(np.max(np.abs(prediction - expected_prob))))
            argmax_disagreements += int(np.sum(prediction.argmax(1) != expected_prob.argmax(1)))
            raw_parts.append(raw.cpu().numpy())
            logit_parts.append(logits.cpu().numpy())
        raw_obs, logits_obs = np.concatenate(raw_parts), np.concatenate(logit_parts)
        unit_obs = normalize_features(raw_obs)
        owners = np.asarray(owners)
        if set(owners) != set(range(len(split.records))):
            raise ValueError("Some archived tracks have no accepted replay observations")
        raw = np.asarray([raw_obs[owners == i].mean(0) for i in range(len(split.records))])
        logits = np.asarray([logits_obs[owners == i].mean(0) for i in range(len(split.records))])
        unit = normalize_features([unit_obs[owners == i].mean(0) for i in range(len(split.records))])
        error = float(np.max(np.linalg.norm(unit - split.features, axis=1)))
        if error > tolerance or probability_error > tolerance:
            raise ValueError(f"Replay parity failed: feature L2={error}, probability max={probability_error}")
        linear_error = float(np.max(np.abs(raw @ parameters["weight"].T + parameters["bias"] - logits)))
        split.raw_features, split.logits = raw, logits
        parity = {"split": name, "observations": len(observations), "max_feature_l2_error": error,
                  "max_probability_error": probability_error, "max_linear_head_error": linear_error,
                  "observation_argmax_disagreements": argmax_disagreements,
                  "tolerance": tolerance, "checkpoint_sha256": digest}
        evidence.append(parity)
        save_npz(output / f"{name}.npz", {**split.metadata, "replay": parity,
                 "raw_aggregation": "mean of accepted observation raw features/logits; original voting labels retained"},
                 features=split.features, raw_features=raw, logits=logits)
        print(f"Mask replay {name}: {len(observations)} observations, max feature error {error:.3g}", flush=True)
    save_npz(output / "classifier_head.npz", {"kind": "linear_head", "checkpoint_sha256": digest}, **parameters)
    return parameters, evidence
