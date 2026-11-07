"""Known-only, primary-record local support check for rejected OSR features.

Reference-only leave-one-record-out distances set class scales. Independent
known calibration sets a pooled normalized radius and a class margin. Neither
test features nor cluster sizes fit this gate. This is not an unknown-FAR bound.
"""
from dataclasses import asdict, dataclass
import numpy as np

VERSION = "primary-local-known-recheck-v1"


@dataclass(frozen=True)
class RecheckConfig:
    k: int = 3
    scale_prior_records: float = 5.
    radius_frr: float = .01
    margin_tail: float = .05

    def validate(self):
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise ValueError("recheck k must be a positive integer")
        if (not np.isfinite(self.scale_prior_records) or self.scale_prior_records < 0 or
                not 0 < self.radius_frr < 1 or not 0 <= self.margin_tail < 1):
            raise ValueError("Invalid recheck scale/radius/margin calibration options")
        return self


def _unit(features):
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or x.shape[1] == 0:
        raise ValueError("Invalid recheck feature matrix")
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if (norm <= 1e-12).any():
        raise ValueError("Zero features cannot enter known recheck")
    return x / norm


def _primary(features, records, names):
    from util.recording_features import annotated_primary_indices
    indices = annotated_primary_indices(records)
    if len(features) != len(records):
        raise ValueError("Recheck records/features differ")
    rows = [records[i] for i in indices]
    if any(r.get("true_class_name") not in names for r in rows):
        raise ValueError("Recheck fit requires known-only primary truth")
    return _unit(features[indices]), rows


class KnownRechecker:
    def __init__(self, metadata, arrays):
        self.metadata = metadata
        if metadata.get("version") != VERSION or metadata.get("unit") != "primary":
            raise ValueError("Unsupported known recheck version/unit")
        self.config = RecheckConfig(**metadata["config"]).validate()
        self.names = list(metadata["class_order"])
        self.bank = _unit(arrays["recheck_reference"])
        self.labels = np.asarray(arrays["recheck_labels"])
        self.scales = np.asarray(arrays["recheck_scales"], dtype=float)
        self.radius = float(metadata["radius"])
        self.margin = float(metadata["margin"])
        if (len(self.names) < 2 or len(set(self.names)) != len(self.names) or
                self.labels.shape != (len(self.bank),) or
                self.scales.shape != (len(self.names),) or not np.isfinite(self.scales).all() or
                (self.scales <= 0).any() or not np.isfinite(self.radius) or self.radius < 0 or
                not np.isfinite(self.margin) or not 0 <= self.margin <= 1 or
                set(self.labels.tolist()) != set(self.names)):
            raise ValueError("Invalid known recheck bank/calibration")
        if any(np.sum(self.labels == n) < self.config.k for n in self.names):
            raise ValueError("Recheck bank has insufficient class support")

    def distances(self, features):
        query = _unit(features)
        if query.shape[1] != self.bank.shape[1]:
            raise ValueError("Recheck feature dimension mismatch")
        # Chunk queries to keep large deployment batches bounded in memory.
        blocks = []
        for start in range(0, len(query), 128):
            q = query[start:start + 128]
            blocks.append(np.column_stack([
                np.sort(np.clip(1 - q @ self.bank[self.labels == n].T, 0, 2), axis=1)
                [:, :self.config.k].mean(1) for n in self.names]))
        raw = np.concatenate(blocks) if blocks else np.empty((0, len(self.names)))
        return raw, raw / self.scales

    def decide(self, features, head_names):
        raw, scaled = self.distances(features)
        if len(head_names) != len(raw):
            raise ValueError("Recheck heads/features differ")
        decisions = []
        for d, s, head in zip(raw, scaled, head_names):
            order = np.argsort(s, kind="stable")
            first, second = order[:2]
            margin = float((s[second] - s[first]) / max(float(s[second]), 1e-12))
            name = self.names[first]
            agreement = name == head == self.names[int(np.argmin(d))]
            supported = self.metadata.get('per_class_eligible_calibration', {}).get(name, 1) > 0
            accepted = bool(supported and agreement and s[first] <= self.radius and
                            margin > 1e-12 and margin >= self.margin)
            reason = ("no_agreed_class_calibration" if not supported else
                      "class_disagreement" if not agreement else
                      "outside_local_radius" if s[first] > self.radius else
                      "ambiguous_class_margin" if margin <= 1e-12 or margin < self.margin else
                      "known_local_support")
            decisions.append(dict(accepted=accepted, reason=reason, candidate_class=name,
                head_class=head, raw_distance=float(d[first]), normalized_distance=float(s[first]),
                radius=self.radius, margin=margin, required_margin=self.margin))
        return decisions


def fit_known_rechecker(reference, ref_records, calibration, cal_records, names,
                        config=None, minimum_calibration=5, maximum_reference=2000):
    config = (config or RecheckConfig()).validate()
    names = sorted(names)
    if len(names) < 2:
        raise ValueError("Known recheck needs at least two known classes")
    x, rr = _primary(reference, ref_records, names)
    z, cr = _primary(calibration, cal_records, names)
    for key in ("recording_id", "source_sha256"):
        if {r[key] for r in rr} & {r[key] for r in cr}:
            raise ValueError("Known recheck reference/calibration overlap")
    selected = []
    for name in names:
        indices = sorted([i for i, r in enumerate(rr) if r["true_class_name"] == name],
                         key=lambda i: rr[i]["recording_id"])
        if len(indices) > maximum_reference:
            indices = [indices[i] for i in np.linspace(0, len(indices)-1, maximum_reference, dtype=int)]
        if len(indices) < config.k + 1:
            raise ValueError(f"Recheck {name}: need k+1 independent primary references")
        if sum(r["true_class_name"] == name for r in cr) < minimum_calibration:
            raise ValueError(f"Recheck {name}: insufficient primary calibration recordings")
        selected.extend(indices)
    bank = x[selected]
    labels = np.asarray([rr[i]["true_class_name"] for i in selected])
    # One primary per original recording, so dropping the diagonal drops that
    # complete recording; repeated components cannot count as extra neighbors.
    medians, counts = [], []
    for name in names:
        points = bank[labels == name]
        d = np.clip(1 - points @ points.T, 0, 2)
        np.fill_diagonal(d, np.inf)
        loo = np.sort(d, axis=1)[:, :config.k].mean(1)
        medians.append(float(np.median(loo)))
        counts.append(len(points))
    pooled = max(float(np.median(medians)), 1e-6)
    weights = np.asarray(counts) / (np.asarray(counts) + config.scale_prior_records)
    scales = np.maximum(weights * medians + (1-weights) * pooled, 1e-6)
    metadata = dict(version=VERSION, unit="primary", config=asdict(config),
        class_order=names, radius=0., margin=0., reference_recordings=len(bank),
        calibration_recordings=len(z), scale_fit="reference-only leave-one-record-out",
        threshold_fit="independent known primary calibration only")
    arrays = dict(recheck_reference=bank.astype(np.float32), recheck_labels=labels,
                  recheck_scales=scales)
    gate = KnownRechecker(metadata, arrays)
    raw, scaled = gate.distances(z)
    truth = np.array([names.index(r["true_class_name"]) for r in cr])
    true_scores = scaled[np.arange(len(z)), truth]
    nearest = np.argmin(scaled, axis=1)
    raw_nearest = np.argmin(raw, axis=1)
    heads = [r.get("known_candidate_class") for r in cr]
    # A calibration example that fails the gate's mandatory class agreement
    # cannot determine its radius. Otherwise mislabeled head decisions can
    # inflate a local acceptance radius for examples that can never pass it.
    head_correct = np.array([head == names[t] for head, t in zip(heads, truth)])
    correct = (nearest == truth) & (raw_nearest == truth) & head_correct
    if not correct.any():
        raise ValueError("Known recheck calibration has no agreed correct neighbors")
    metadata["radius"] = float(np.quantile(true_scores[correct], 1-config.radius_frr, method="higher"))
    ordered = np.sort(scaled[correct], axis=1)
    margins = (ordered[:, 1] - ordered[:, 0]) / np.maximum(ordered[:, 1], 1e-12)
    metadata["margin"] = float(np.quantile(margins, config.margin_tail, method="lower"))
    metadata['per_class_eligible_calibration'] = {n: int(sum(correct[i] and cr[i]['true_class_name'] == n
        for i in range(len(cr)))) for n in names}
    gate = KnownRechecker(metadata, arrays)
    decisions = gate.decide(z, heads)
    metadata.update(calibration_neighbor_agreement_accuracy=float(correct.mean()),
        calibration_gate_accept_rate=float(np.mean([d["accepted"] for d in decisions])),
        calibrated_margin_records=int(correct.sum()),
        radius_calibration_scope="conditional on correct head/raw/normalized neighbor agreement",
        warnings=[
            "Known calibration does not bound unknown false acceptance.",
            "Few independent tail records; validate with held-out unknown classes."])
    return metadata, arrays
