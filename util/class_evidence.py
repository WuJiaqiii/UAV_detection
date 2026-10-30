"""Summarize category evidence while preserving independent signal components."""
from collections import defaultdict


def summarize_class_evidence(results, rows_by_file):
    """No truth labels, co-occurrence-based class merges or device IDs are used."""
    for result in results:
        types = defaultdict(set)
        for row in rows_by_file.get(result["path"], []):
            for group in row.get("groups", []):
                types[group["track_id"]].add(group.get("group_type", "unspecified"))
        categories = {}
        unresolved = []
        for track in result.get("signals", []):
            component_types = sorted(types[track["track_id"]]) or ["unspecified"]
            track["component_types"] = component_types
            status = track.get("recognition_status", "known" if track.get("status") == "ok" else "uncertain")
            label = (track.get("class_name") if status == "known" else
                     track.get("unknown_cluster_id") if status == "novel_candidate" else None)
            if not label:
                unresolved.append(track["track_id"])
                continue
            key = (status, label)
            entry = categories.setdefault(key, {"display_label": label, "recognition_status": status,
                "class_name": label if status == "known" else None,
                "identity_status": "category_evidence_only", "components": []})
            component = {name: track.get(name) for name in ("track_id", "start_ms", "end_ms",
                "frequency_center_offset_mhz", "frequency_bands_offset_mhz", "rf_center_mhz",
                "frequency_bands_absolute_mhz", "valid_votes", "observation_count")}
            component["component_types"] = component_types
            entry["components"].append(component)
        result["class_evidence"] = sorted(categories.values(), key=lambda item: (item["recognition_status"], item["display_label"]))
        result["unresolved_component_track_ids"] = unresolved
        result["device_identity_note"] = "Components sharing a category may belong to different physical devices."
