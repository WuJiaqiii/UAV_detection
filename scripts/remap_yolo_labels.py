#!/usr/bin/env python3
"""Map YOLOv5 detection classes other than 0, 1 and 2 to 0.

Accept a dataset root containing labels/, or the labels/ directory itself.
Only the first token of each changed annotation is replaced; coordinates,
whitespace, line endings and UTF-8 BOMs are preserved. No third-party packages.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid


KEEP_CLASSES = {0, 1, 2}


def rewrite_label(raw, path):
    """Validate one detection label and return bytes, counts and change count."""
    bom = b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b""
    text = raw[len(bom):].decode("utf-8")
    output, before, after = [], Counter(), Counter()
    changed = 0
    for line_number, line in enumerate(text.splitlines(keepends=True), 1):
        fields = line.split()
        if not fields:
            output.append(line)
            continue
        location = f"{path}:{line_number}"
        if len(fields) != 5:
            raise ValueError(f"{location}: expected class xc yc width height (5 columns)")
        if re.fullmatch(r"[0-9]+", fields[0]) is None:
            raise ValueError(f"{location}: class must be a nonnegative integer")
        cls = int(fields[0])
        try:
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"{location}: invalid bounding-box coordinates") from exc
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"{location}: bounding-box coordinates must be finite")
        target = cls if cls in KEEP_CLASSES else 0
        before[cls] += 1
        after[target] += 1
        if target != cls:
            token = re.search(r"\S+", line)
            line = line[:token.start()] + "0" + line[token.end():]
            changed += 1
        output.append(line)
    return bom + "".join(output).encode("utf-8"), before, after, changed


def atomic_write(path, content):
    """Replace a label only after its complete replacement has been written."""
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def remap_dataset(dataset_dir, dry_run=False):
    root = Path(dataset_dir).expanduser().resolve()
    labels = root / "labels"
    if not labels.is_dir():
        if root.name == "labels" and root.is_dir():
            labels = root
        else:
            raise FileNotFoundError(f"No labels/ directory under {root}")
    files = sorted(path for path in labels.rglob("*")
                   if path.is_file() and path.suffix.lower() == ".txt")
    if not files:
        raise FileNotFoundError(f"No .txt label files under {labels}")
    before, after = Counter(), Counter()
    changes, changed_annotations, empty_files = [], 0, 0
    # Complete validation before creating a backup or modifying any labels.
    for path in files:
        raw = path.read_bytes()
        rewritten, old_counts, new_counts, count = rewrite_label(raw, path)
        before.update(old_counts)
        after.update(new_counts)
        empty_files += not bool(old_counts)
        changed_annotations += count
        if count:
            changes.append((path, raw, rewritten))

    # YOLOv5 hashes file paths and sizes; a same-length class edit can otherwise
    # leave its cached labels valid. Include all scanned parents for mixed folders.
    candidates = {labels.with_suffix(".cache")}
    candidates.update(path.parent.with_suffix(".cache") for path in files)
    candidates.update(labels.rglob("*.cache"))
    caches = sorted(path for path in candidates if path.is_file()) if changes else []
    backup = None
    if changes and not dry_run:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = labels.with_name(labels.name + "_backup_" + stamp + "_" + uuid.uuid4().hex[:8])
        backup.mkdir()
        cache_manifest = []
        # Back up every changed original and existing cache before the first edit.
        for path, original, _ in changes:
            if path.read_bytes() != original:
                raise RuntimeError(f"Label changed during scanning; no labels written: {path}")
            destination = backup / "labels" / path.relative_to(labels)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        for index, cache in enumerate(caches):
            destination = backup / "caches" / (str(index) + "-" + cache.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache, destination)
            cache_manifest.append({"source": str(cache),
                                   "backup": destination.relative_to(backup).as_posix()})
        (backup / "manifest.json").write_text(json.dumps({
            "labels_dir": str(labels),
            "changed_labels": [path.relative_to(labels).as_posix() for path, _, _ in changes],
            "caches": cache_manifest,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        # Invalidate cached old classes before editing, including if a later write
        # fails. All originals remain available in the backup directory.
        for cache in caches:
            cache.unlink()
        for path, _, rewritten in changes:
            atomic_write(path, rewritten)

    return {
        "dry_run": dry_run,
        "labels_dir": str(labels),
        "files_scanned": len(files),
        "empty_files": empty_files,
        "annotations": sum(before.values()),
        "files_to_change" if dry_run else "files_changed": len(changes),
        "annotations_to_change" if dry_run else "annotations_changed": changed_annotations,
        "classes_before": dict(sorted(before.items())),
        "classes_after": dict(sorted(after.items())),
        "caches_to_invalidate" if dry_run else "caches_invalidated": len(caches),
        "backup_dir": str(backup) if backup else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True,
                        help="Dataset root containing labels/, or the labels/ directory itself")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate and report planned changes without writing files")
    args = parser.parse_args()
    try:
        result = remap_dataset(args.dataset_dir, args.dry_run)
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
