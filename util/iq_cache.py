"""Lossless cache for fixed validation spectra, never for random training crops."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import warnings

import numpy as np

_warned = set()
_warning_lock = threading.Lock()


def cache_warning(root, error):
    with _warning_lock:
        if str(root) not in _warned:
            _warned.add(str(root))
            warnings.warn(f"Validation spectrum cache unavailable; using fresh conversion: {error}", RuntimeWarning)


@lru_cache(maxsize=16)
def implementation_hash(filename):
    return hashlib.sha256(Path(filename).read_bytes()).hexdigest()


def fixed_spectrum(cache_dir, source, parameters, compute):
    source = Path(source).resolve()
    stat = source.stat()
    identity = {"version": 1, "source": str(source), "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "parameters": parameters}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(cache_dir)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        cache_warning(root, error)
        return compute()
    # Keep managed experiment paths below the traditional Windows path limit.
    target = root/(key[:24]+".npy")
    if target.is_file():
        try:
            spectrum = np.load(target, allow_pickle=False)
            if spectrum.shape == (512, 750) and spectrum.dtype == np.float32 and np.isfinite(spectrum).all():
                return spectrum
        except (OSError, ValueError, EOFError):
            pass
    spectrum = compute()
    if spectrum.shape != (512, 750) or spectrum.dtype != np.float32 or not np.isfinite(spectrum).all():
        raise ValueError("Fixed IQ cache requires a finite float32 native spectrum")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, suffix=".npy.tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, spectrum, allow_pickle=False)
        try:
            os.replace(temporary, target)
        except OSError as error:
            cache_warning(root, error)
    except OSError as error:
        cache_warning(root, error)
    finally:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
    return spectrum
