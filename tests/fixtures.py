"""Small deterministic archives and tracks shared by core tests."""
from util.open_set import FEATURE_VERSION, save_npz, normalize_features


def parse_test_args(parser, tokens):
    """Inject internal stage paths for isolated core tests, never public CLI."""
    tokens = list(tokens)
    outputs = {}
    for flag in ("--output_dir", "--test_output_dir"):
        if flag in tokens:
            index = tokens.index(flag)
            from pathlib import Path
            outputs[flag[2:]] = Path(tokens[index + 1])
            del tokens[index:index + 2]
    args = parser.parse_args(tokens)
    for key, value in outputs.items():
        setattr(args, key, value)
    return args


def archive(path, features, labels, split, signature=None, classes=None):
    classes = classes or {"known": 0}
    records = [{"path": f"{split}/{i}.png", "track_id": "T0001", "recording_id": f"{split}_{i}",
                "source_sha256": f"{split}_hash_{i}", "true_class_name": name}
               for i, name in enumerate(labels)]
    metadata = {"schema_version": 1, "kind": "track_features", "feature_version": FEATURE_VERSION,
                "signature": signature or {"sha256": "fixture"}, "classes": classes, "records": records}
    save_npz(path, metadata, features=normalize_features(features))
    return metadata


def track_fixture(embeddings, probabilities=None, min_votes=1):
    from util.multi_signal_eval import FrequencyTracker, group_geometry
    tracker, rows = FrequencyTracker(), []
    for i, embedding in enumerate(embeddings):
        group = {"status": "ok", "group_id": 0, "probabilities": probabilities or [1.], "embedding": embedding,
                 "selected_boxes": [[10, 100, 50, 110]], "observation_id": f"S{i}_G0", "slice_index": i,
                 "confidence": 1., "class_name": "known"}
        group.update(group_geometry(group["selected_boxes"], (512, 750), 122880000, i*50.))
        tracker.update([group])
        rows.append({"slice_index": i, "groups": [group]})
    result = {"path": "test.png", "status": "ok", "signals": tracker.summarize({"known": 0}, {"min_votes": min_votes}),
              "predicted_classes": ["known"], "true_class_names": None}
    return [result], {"test.png": rows}


def known_archive(path, features, split, signature=None, classes=None, recording=None):
    labels = classes or {"known": 0}
    records = [{"path": f"{split}/{i}.png", "recording_id": recording or f"{split}_{i}",
                "source_sha256": f"hash_{split}_{i}", "true_class_name": "known", "track_id": "T0001"}
               for i in range(len(features))]
    save_npz(path, {"kind": "track_features", "schema_version": 1, "feature_version": FEATURE_VERSION,
                   "signature": signature or {"sha256": "fixture"}, "classes": labels, "records": records},
             features=normalize_features(features))
