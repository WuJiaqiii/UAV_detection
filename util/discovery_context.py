"""Truth-free recording context for online and archived unknown discovery."""
import json
from pathlib import Path


DISCOVERY_FIELDS = (
    "recognition_status", "display_label",
    "valid_votes", "observation_count", "component_types", "discovery_is_primary",
    "discovery_primary_path", "discovery_primary_track_id", "discovery_primary_status",
    "discovery_role", "cluster_assignment_source", "cluster_status", "cluster_membership",
    "cluster_similarity",
)
POSITION_FIELDS = (
    "start_ms", "end_ms", "known_candidate_class", "ood_score", "ood_threshold",
    "frequency_center_offset_mhz", "frequency_low_offset_mhz", "frequency_high_offset_mhz",
    "frequency_bands_offset_mhz", "rf_center_mhz", "frequency_center_absolute_mhz",
    "frequency_low_absolute_mhz", "frequency_high_absolute_mhz", "frequency_bands_absolute_mhz",
)


def _source_primaries(results, identities):
    grouped, lookup = {}, {}
    for row in results:
        identity = identities.get(row["path"], row["path"])
        for track in row.get("signals", []):
            key = (row["path"], track["track_id"])
            if key in lookup:
                raise ValueError("Duplicate source track in discovery context")
            lookup[key] = track
            grouped.setdefault(identity, []).append((row["path"], track))
    primaries = {identity: min(items, key=lambda pair: (
        -pair[1].get("valid_votes", 0), -pair[1].get("observation_count", 0),
        pair[0], str(pair[1]["track_id"]))) for identity, items in grouped.items()}
    return primaries, lookup


def annotate_discovery_candidates(candidates, results, recording_ids=None):
    """Select from ALL tracks, including known/invalid primaries, without truth.

    A secondary unknown track cannot become primary merely because the real
    primary was accepted as known or had no usable feature.
    """
    identities = dict(recording_ids or {})
    for item in candidates:
        identity = item.get("recording_id", identities.get(item["path"], item["path"]))
        if item["path"] in identities and identities[item["path"]] != identity:
            raise ValueError("Conflicting candidate/source recording identities")
        identities[item["path"]] = identity
        item["recording_id"] = identity
    primaries, lookup = _source_primaries(results, identities)
    for item in candidates:
        key = (item["path"], item["track"]["track_id"])
        if key not in lookup:
            raise ValueError("Candidate track is absent from source results")
        path, primary = primaries[item["recording_id"]]
        source = lookup[key]
        item["track"].update({key: source.get(key) for key in
                              ("valid_votes", "observation_count", "component_types")})
        item["track"].update(discovery_is_primary=key == (path, primary["track_id"]),
            discovery_primary_path=path, discovery_primary_track_id=primary["track_id"],
            discovery_primary_status=primary.get("recognition_status", "no_signal"))
    return identities


def attach_recording_predictions(results, recording_ids, input_mode):
    """A single-device decision is explicit and does not overwrite track evidence."""
    primaries, _ = _source_primaries(results, recording_ids)
    for row in results:
        identity = recording_ids.get(row["path"], row["path"])
        row["single_signal_prediction"] = None
        if input_mode != "single":
            continue
        path, track = primaries.get(identity, (row["path"], {}))
        row["single_signal_prediction"] = {
            "recording_id": identity, "primary_path": path, "primary_track_id": track.get("track_id"),
            "recognition_status": track.get("recognition_status", "no_signal"),
            "class_name": track.get("class_name") if track.get("recognition_status") == "known" else None,
            "unknown_cluster_id": track.get("unknown_cluster_id"),
            "display_label": track.get("display_label", "no_signal"),
            "assumption": "one signal/device class per original recording; all component positions remain in signals"}


def candidate_record(item):
    track = item["track"]
    return {"path": item["path"], "recording_id": item.get("recording_id", item["path"]),
            "track_id": track["track_id"], "unknown_cluster_id": track.get("unknown_cluster_id"),
            **{key: track.get(key) for key in POSITION_FIELDS + DISCOVERY_FIELDS}}


def source_recording_ids(report, records, results_path=None):
    """Read identity context from a saved report/metric sidecar, never class truth."""
    identities = dict(report.get("recording_ids", {}))
    if results_path:
        sidecar = Path(results_path).parent / "open_set_metrics.json"
        if sidecar.is_file():
            samples = json.loads(sidecar.read_text(encoding="utf-8")).get("samples", [])
            for sample in samples:
                for path in sample.get("paths", []):
                    identity = sample["recording_id"]
                    if path in identities and identities[path] != identity:
                        raise ValueError("Conflicting report/sidecar recording IDs")
                    identities[path] = identity
    for row in records:
        identity = row["recording_id"]
        if row["path"] in identities and identities[row["path"]] != identity:
            raise ValueError("Candidate archive does not match source recording IDs")
        identities[row["path"]] = identity
    return identities
