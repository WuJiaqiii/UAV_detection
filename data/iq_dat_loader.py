"""Read class-folder UAV recordings stored as interleaved int16 I/Q.

The public functions also work independently of the training pipeline::

    spec, time_ms, freq_mhz = read_iq_spectrogram(path, duration_ms=10)
    # plt.pcolormesh(time_ms, freq_mhz, spec, shading="auto")

Frequency coordinates are offsets from the recording's unknown RF center.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

from data.data_loader import create_dataloader


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


def iq_to_spectrogram(iq, n_fft=512, hop_length=1024, db_floor=-100.0):
    """Return a centered [frequency, time] relative-power dB image."""
    iq = np.asarray(iq, dtype=np.complex64)
    n_fft, hop_length = int(n_fft), int(hop_length)
    if iq.ndim != 1 or n_fft < 2 or hop_length < 1 or iq.size < n_fft:
        raise ValueError("Need a 1D IQ array with at least n_fft samples")
    frames = np.lib.stride_tricks.sliding_window_view(iq, n_fft)[::hop_length]
    window = np.hanning(n_fft).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft(frames * window, axis=1), axes=1)
    power = np.abs(spectrum) ** 2
    db = 10.0 * np.log10(np.maximum(power, 1e-12))
    db -= db.max()
    return np.ascontiguousarray(np.clip(db.T, float(db_floor), 0.0), dtype=np.float32)


def read_iq_spectrogram(path, start_ms=0.0, duration_ms=10.0,
                        sample_rate=DEFAULT_SAMPLE_RATE, n_fft=512,
                        hop_length=1024, db_floor=-100.0, byte_order="<"):
    """Read a time slice and return (spectrogram, time_ms, freq_offset_mhz).

    The frequency axis is relative to the unknown RF recording center.
    """
    sample_rate = int(sample_rate)
    if sample_rate <= 0 or start_ms < 0 or duration_ms <= 0:
        raise ValueError("sample_rate and duration_ms must be positive; start_ms >= 0")
    start_sample = round(float(start_ms) * sample_rate / 1000.0)
    num_samples = round(float(duration_ms) * sample_rate / 1000.0)
    iq = read_iq_dat(path, start_sample, num_samples, byte_order=byte_order)
    spec = iq_to_spectrogram(iq, n_fft=n_fft, hop_length=hop_length,
                             db_floor=db_floor)
    time_ms = (start_sample + int(n_fft) / 2.0
               + np.arange(spec.shape[1]) * int(hop_length)) / sample_rate * 1000.0
    freq_mhz = np.fft.fftshift(np.fft.fftfreq(int(n_fft), d=1.0 / sample_rate)) / 1e6
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
        self.sample_rate = int(getattr(config, "sampling_rate", DEFAULT_SAMPLE_RATE))
        self.n_fft = int(getattr(config, "iq_n_fft", 512))
        self.hop_length = int(getattr(config, "iq_hop_length", 1024))
        self.segment_ms = float(getattr(config, "iq_segment_ms", 10.0))
        self.segment_samples = round(self.sample_rate * self.segment_ms / 1000.0)
        if self.segment_samples < self.n_fft:
            raise ValueError("iq_segment_ms must contain at least iq_n_fft samples")
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
            if name not in class_map or name in excluded or (included and name not in included):
                continue
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
        iq = read_iq_dat(sample["fp"], start_sample=start, num_samples=self.segment_samples)
        spec = iq_to_spectrogram(iq, n_fft=self.n_fft, hop_length=self.hop_length)
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
    indices = np.arange(len(train_data))
    labels = [s["label"] for s in train_data.samples]
    train_idx, val_idx = train_test_split(indices, test_size=float(config.val_ratio),
                                          random_state=42, stratify=labels)
    train_loader = create_dataloader(Subset(train_data, train_idx.tolist()), config,
                                     shuffle=True, sample_ratio=float(config.sample_ratio))
    val_loader = create_dataloader(Subset(val_data, val_idx.tolist()), config,
                                   shuffle=False, sample_ratio=1.0)
    return train_loader, val_loader
