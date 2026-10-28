"""Read class-folder UAV recordings stored as interleaved int16 I/Q.

The public functions also work independently of the training pipeline::

    spec, time_ms, freq_mhz = read_iq_spectrogram(path, duration_ms=50)
    # plt.pcolormesh(time_ms, freq_mhz, spec, shading="auto")

Frequency coordinates are offsets from the recording's unknown RF center.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from data.data_loader import create_dataloader
from util.spectrogram_frontend import (
    iq_to_amplitude, validate_iq_settings, TARGET_SAMPLE_RATE, N_FFT, FRAMES_PER_COLUMN,
)


DEFAULT_SAMPLE_RATE = 100_000_000
BYTES_PER_COMPLEX_SAMPLE = 4


def _paths(value):
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        value = [value]
    return [Path(p).expanduser() for p in value if str(p).strip()]


def list_iq_files(dataset_path):
    """Find .dat files in a root, class directory, or explicit file list."""
    found = set()
    for path in _paths(dataset_path):
        if path.is_file():
            if path.suffix.lower() != ".dat":
                raise ValueError(f"Expected .dat file: {path}")
            found.add(path.resolve())
        elif path.is_dir():
            found.update(p.resolve() for p in path.glob("*.dat") if p.is_file())
            for child in path.iterdir():
                if child.is_dir():
                    found.update(p.resolve() for p in child.glob("*.dat") if p.is_file())
        else:
            raise FileNotFoundError(path)
    if not found:
        raise FileNotFoundError(f"No .dat files found in {dataset_path}")
    return sorted(found)


def discover_iq_classes(dataset_path, include_background=False):
    """Map folder names to stable IDs without inventing unlabeled classes."""
    names = sorted({p.parent.name for p in list_iq_files(dataset_path)})
    if include_background:
        names = ["Background"] + [name for name in names if name != "Background"]
    return {name: idx for idx, name in enumerate(names)}


def resolve_iq_classes(config):
    """Discover training labels from the same paths the loaders will consume.

    Explicit train/validation paths share one mapping. ``class_names`` may
    reorder that mapping, but cannot silently whitelist or invent folder labels.
    Class exclusion and incremental splits are applied to this mapping later.
    """
    train_paths = _paths(getattr(config, "train_dataset_path", None))
    val_paths = _paths(getattr(config, "val_dataset_path", None))
    if train_paths or val_paths:
        if not train_paths or not val_paths:
            raise ValueError("Provide both --train_dataset_path and --val_dataset_path")
        paths = train_paths + val_paths
    else:
        paths = _paths(getattr(config, "dataset_path", None))
    classes = discover_iq_classes(paths)
    names = getattr(config, "class_names", None)
    if names is not None:
        if not names or len(set(names)) != len(names) or any(not str(n).strip() for n in names):
            raise ValueError("class_names must be nonempty unique names in checkpoint order")
        missing = sorted(set(classes) - set(names))
        extra = sorted(set(names) - set(classes))
        if missing or extra:
            raise ValueError(
                "For iq_dat, class_names may only reorder discovered folder classes; "
                f"missing folder names={missing}, names not found in data={extra}. "
                "Omit class_names for automatic discovery; use exclude_classes to "
                "hold out discovered classes."
            )
        classes = {name: idx for idx, name in enumerate(names)}
    return classes


def read_iq_dat(path, start_sample=0, num_samples: Optional[int] = None,
                normalize=True, byte_order="<"):
    """Read a slice of complex samples without loading the full file.

    ``start_sample`` and ``num_samples`` count complex samples, not int16 values.
    Data layout is I0, Q0, I1, Q1, ... . Normalized output is complex64.
    """
    path = Path(path)
    if byte_order not in ("<", ">"):
        raise ValueError("byte_order must be '<' or '>'")
    size = path.stat().st_size
    if size % BYTES_PER_COMPLEX_SAMPLE:
        raise ValueError(f"File size is not a whole number of int16 IQ pairs: {path}")
    total = size // BYTES_PER_COMPLEX_SAMPLE
    start_sample = int(start_sample)
    if start_sample < 0 or start_sample > total:
        raise ValueError(f"start_sample must be in [0, {total}]")
    if num_samples is None:
        num_samples = total - start_sample
    num_samples = int(num_samples)
    if num_samples < 0 or start_sample + num_samples > total:
        raise ValueError("Requested IQ slice exceeds file length")
    with path.open("rb") as stream:
        stream.seek(start_sample * BYTES_PER_COMPLEX_SAMPLE)
        raw = np.fromfile(stream, dtype=np.dtype(byte_order + "i2"), count=2 * num_samples)
    if raw.size != 2 * num_samples:
        raise IOError(f"Short read from {path}")
    scale = 1.0 / 32768.0 if normalize else 1.0
    iq = (raw[0::2].astype(np.float32) * scale).astype(np.complex64)
    iq.imag = raw[1::2].astype(np.float32) * scale
    return iq


def iq_to_spectrogram(iq, n_fft=512, hop_length=512, db_floor=None,
                      sample_rate=DEFAULT_SAMPLE_RATE):
    """50 ms IQ -> canonical 512x750 relative amplitude, not dB."""
    if db_floor is not None:
        raise ValueError("db_floor is not part of the amplitude profile")
    validate_iq_settings(sample_rate, 50.0, n_fft, hop_length)
    return iq_to_amplitude(iq, sample_rate)


def read_iq_spectrogram(path, start_ms=0.0, duration_ms=50.0,
                        sample_rate=DEFAULT_SAMPLE_RATE, n_fft=512,
                        hop_length=512, db_floor=None, byte_order="<"):
    """Return amplitude, aggregated-column times and target FFT-bin frequencies."""
    count = validate_iq_settings(sample_rate, duration_ms, n_fft, hop_length)
    if start_ms < 0:
        raise ValueError("start_ms must be nonnegative")
    start_sample = round(float(start_ms)*sample_rate/1000)
    iq = read_iq_dat(path, start_sample, count, byte_order=byte_order)
    spec = iq_to_spectrogram(iq, n_fft, hop_length, db_floor, sample_rate)
    step = N_FFT * FRAMES_PER_COLUMN
    time_ms = (start_sample/sample_rate +
               (np.arange(spec.shape[1])*step + (step-1)/2)/TARGET_SAMPLE_RATE)*1000
    freq_mhz = np.fft.fftshift(np.fft.fftfreq(N_FFT, 1/TARGET_SAMPLE_RATE))/1e6
    return spec, time_ms, freq_mhz


class RawIQDataset(Dataset):
    """Build 2D spectrograms on demand from class-folder .dat recordings.

    Training uses a random time slice; validation and inference use the center
    slice. Folder names provide classification labels. RF center frequency,
    signal bandwidth and SNR are absent from this dataset. The target's 0 MHz
    center and full capture bandwidth are nominal placeholders; they must not
    be treated as ground truth for frequency-matching evaluation.
    """

    def __init__(self, config, logger=None, dataset_path=None, split_name="train"):
        self.config = config
        self.logger = logger
        self.split_name = str(split_name).lower()
        self.sample_rate = int(getattr(config, "iq_sample_rate", DEFAULT_SAMPLE_RATE))
        self.n_fft = int(getattr(config, "iq_n_fft", 512))
        self.hop_length = int(getattr(config, "iq_hop_length", 512))
        self.segment_ms = float(getattr(config, "iq_segment_ms", 50.0))
        self.segment_samples = validate_iq_settings(
            self.sample_rate, self.segment_ms, self.n_fft, self.hop_length)
        self.byte_order = str(getattr(config, "iq_byte_order", "<"))
        self.random_crop = self.split_name == "train"
        paths = dataset_path if dataset_path is not None else getattr(config, "dataset_path", None)
        class_map = dict(config.classes)
        if not class_map:
            raise ValueError("config.classes is empty")
        if self.split_name in ("val", "valid", "validation"):
            excluded = set(getattr(config, "val_data_exclude_classes", []))
            included = set(getattr(config, "val_data_include_classes", []))
        else:
            excluded = set(getattr(config, "train_data_exclude_classes", []))
            included = set(getattr(config, "train_data_include_classes", []))
        self.samples = []
        for path in list_iq_files(paths):
            name = path.parent.name
            if name in excluded or (included and name not in included):
                continue
            if name not in class_map:
                raise ValueError(
                    f"IQ folder class {name!r} is missing from config.classes: {path}. "
                    "Discover a shared class mapping before creating the datasets, "
                    "or explicitly exclude this class."
                )
            if path.stat().st_size % BYTES_PER_COMPLEX_SAMPLE:
                raise ValueError(f"Invalid IQ file size: {path}")
            available = path.stat().st_size // BYTES_PER_COMPLEX_SAMPLE
            if available < self.segment_samples:
                raise ValueError(f"File shorter than requested {self.segment_ms:g} ms: {path}")
            self.samples.append({"fp": str(path), "class_name": name,
                                 "label": int(class_map[name]), "num_samples": available})
        if not self.samples:
            raise RuntimeError(f"No usable IQ samples in {paths}")
        if logger is not None:
            logger.info(f"Indexed {len(self.samples)} IQ samples for {self.split_name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        remaining = sample["num_samples"] - self.segment_samples
        start = int(np.random.randint(0, remaining + 1)) if self.random_crop else remaining // 2
        iq = read_iq_dat(sample["fp"], start_sample=start, num_samples=self.segment_samples,
                         byte_order=self.byte_order)
        spec = iq_to_spectrogram(iq, n_fft=self.n_fft, hop_length=self.hop_length,
                                 sample_rate=self.sample_rate)
        target = {"label": sample["label"], "class_name": sample["class_name"],
                  "center_freq": 0.0, "bandwidth": self.sample_rate / 1e6,
                  "has_frequency_label": False}
        return torch.from_numpy(spec), [target], float("nan"), sample["fp"]


def create_iq_train_val_loaders(config, logger=None):
    """Stratified file-level split with random train and fixed validation crops."""
    train_data = RawIQDataset(config, logger, split_name="train")
    val_data = RawIQDataset(config, logger, split_name="val")
    if [s["fp"] for s in train_data.samples] != [s["fp"] for s in val_data.samples]:
        raise ValueError("Automatic split requires identical train/val class filters")
    from util.recordings import dataset_train_val_indices
    train_idx, val_idx = dataset_train_val_indices(train_data.samples, config)
    train_loader = create_dataloader(Subset(train_data, train_idx.tolist()), config,
                                     shuffle=True, sample_ratio=float(config.sample_ratio))
    val_loader = create_dataloader(Subset(val_data, val_idx.tolist()), config,
                                   shuffle=False, sample_ratio=1.0)
    return train_loader, val_loader
