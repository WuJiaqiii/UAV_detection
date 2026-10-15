"""One deterministic frontend for IQ, MAT spectra and already-rendered PNGs.

All scientific matrices use [frequency, time], low frequency at row zero.
Only render_tiles() creates detector pixels. PNGs are never log-transformed.
"""
from dataclasses import asdict, dataclass
from typing import Iterator
import hashlib
import json
import cv2
import numpy as np
from scipy.ndimage import median_filter

FRONTEND_VERSION = "background-v1"


@dataclass(frozen=True)
class FrontendConfig:
    mode: str = "global"
    z_min: float = -2.0
    z_max: float = 8.0
    min_scale_db: float = 1.0
    frequency_quantile: float = 0.2
    frequency_smooth: int = 9
    image_size: int = 640
    tile_width: int = 768
    tile_overlap: int = 192
    pad_value: int = 114
    nms_iou: float = 0.5

    def __post_init__(self):
        if self.mode not in ("global", "frequency", "none"):
            raise ValueError("frontend mode must be global, frequency or none")
        if not np.isfinite([self.z_min, self.z_max, self.min_scale_db]).all():
            raise ValueError("Non-finite frontend range")
        if self.z_max <= self.z_min or self.min_scale_db <= 0:
            raise ValueError("Need z_max > z_min and min_scale_db > 0")
        if not 0 < self.frequency_quantile < 1:
            raise ValueError("frequency_quantile must be in (0, 1)")
        if self.frequency_smooth < 1 or self.frequency_smooth % 2 != 1:
            raise ValueError("frequency_smooth must be a positive odd integer")
        if self.image_size < 32 or self.image_size % 32:
            raise ValueError("image_size must be a positive multiple of 32")
        if self.tile_width < 0 or self.tile_overlap < 0:
            raise ValueError("Tile width and overlap must be nonnegative")
        if self.tile_width and self.tile_overlap >= self.tile_width:
            raise ValueError("Tile overlap must be smaller than tile width")
        if not 0 <= self.pad_value <= 255 or not 0 <= self.nms_iou <= 1:
            raise ValueError("Invalid pad value or NMS IoU")

    @classmethod
    def from_config(cls, config):
        mapping = {
            "mode": "frontend_mode", "z_min": "frontend_z_min",
            "z_max": "frontend_z_max", "min_scale_db": "frontend_min_scale_db",
            "frequency_quantile": "frontend_frequency_quantile",
            "frequency_smooth": "frontend_frequency_smooth",
            "tile_width": "detector_tile_width", "tile_overlap": "detector_tile_overlap",
            "nms_iou": "detector_tile_nms_iou",
        }
        values = {key: getattr(config, name) for key, name in mapping.items()
                  if hasattr(config, name)}
        values["image_size"] = int(max(getattr(config, "yolo_imgsz_h", 640),
                                       getattr(config, "yolo_imgsz_w", 640)))
        return cls(**values)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(
            {"version": FRONTEND_VERSION, **asdict(self)}, sort_keys=True
        ).encode()).hexdigest()


def add_frontend_arguments(parser):
    g = parser.add_argument_group("Shared spectrogram frontend")
    g.add_argument("--frontend_config", default="", help="JSON manifest exported with the detector dataset")
    g.add_argument("--mat_spec_domain", choices=["power", "amplitude", "db", "image"],
                   default="power", help="Declare MAT values; never infer units from their range.")
    g.add_argument("--frontend_mode", choices=["global", "frequency", "none"], default="global")
    g.add_argument("--frontend_z_min", type=float, default=-2.0)
    g.add_argument("--frontend_z_max", type=float, default=8.0)
    g.add_argument("--frontend_min_scale_db", type=float, default=1.0)
    g.add_argument("--frontend_frequency_quantile", type=float, default=0.2)
    g.add_argument("--frontend_frequency_smooth", type=int, default=9)
    g.add_argument("--detector_tile_width", type=int, default=768,
                   help="Time columns per detector tile; 0 disables tiling.")
    g.add_argument("--detector_tile_overlap", type=int, default=192)
    g.add_argument("--detector_tile_nms_iou", type=float, default=0.5)
    g.add_argument("--frequency_axis", choices=["legacy", "fft"], default="legacy",
                   help="Existing MAT/PNG endpoint mapping, or FFT-bin mapping. IQ uses fft.")


def to_db(data, domain):
    x = np.asarray(data)
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError("Expected nonempty [frequency, time] matrix")
    if domain == "db":
        if np.iscomplexobj(x):
            raise ValueError("dB spectra must be real")
        out = x.astype(np.float64)
    elif domain == "power":
        if np.iscomplexobj(x) or not np.isfinite(x).all() or (x < 0).any():
            raise ValueError("Power must be finite, real and nonnegative; set the correct MAT domain")
        out = 10.0 * np.log10(np.maximum(x.astype(np.float64), 1e-30))
    elif domain == "amplitude":
        out = 20.0 * np.log10(np.maximum(np.abs(x).astype(np.float64), 1e-15))
    else:
        raise ValueError("to_db expects power, amplitude or db; rendered images bypass it")
    if not np.isfinite(out).all():
        raise ValueError("Spectrum contains non-finite values")
    return np.ascontiguousarray(out, dtype=np.float32)


def iq_to_db(iq, n_fft=512, hop_length=256, block_frames=2048):
    """Hann-window periodogram, no peak subtraction, no dB clipping.

    No centering or implicit padding. Compute FFTs in bounded batches.
    Window-energy normalization gives a consistent power convention.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    n_fft, hop_length = int(n_fft), int(hop_length)
    if iq.ndim != 1 or n_fft < 2 or not 1 <= hop_length <= n_fft:
        raise ValueError("Require 1D IQ, n_fft >= 2, and 1 <= hop_length <= n_fft (no blind gaps)")
    if iq.size < n_fft or not np.isfinite(iq).all() or block_frames < 1:
        raise ValueError("IQ must be finite and contain at least one complete window")
    count = 1 + (iq.size - n_fft) // hop_length
    result = np.empty((n_fft, count), dtype=np.float32)
    window = np.hanning(n_fft).astype(np.float32)
    energy = float(np.sum(window ** 2))
    if energy <= 0:
        raise ValueError("Hann window has zero energy; increase n_fft")
    for start in range(0, count, block_frames):
        stop = min(start + block_frames, count)
        left = start * hop_length
        right = (stop - 1) * hop_length + n_fft
        frames = np.lib.stride_tricks.sliding_window_view(iq[left:right], n_fft)[::hop_length]
        fft = np.fft.fftshift(np.fft.fft(frames * window, axis=1), axes=1)
        power = np.abs(fft) ** 2 / energy
        result[:, start:stop] = (10 * np.log10(np.maximum(power, 1e-30))).T
    return result


def background_standardize(db, config):
    """Robust empirical significance, not a calibrated SNR or CFAR statistic."""
    x = to_db(db, "db").astype(np.float64)
    if config.mode == "none":
        return x.astype(np.float32)
    if config.mode == "frequency":
        floor = np.quantile(x, config.frequency_quantile, axis=1, keepdims=True)
        floor = median_filter(floor, size=(config.frequency_smooth, 1), mode="nearest")
        x = x - floor
    center = float(np.median(x))
    scale = max(1.4826 * float(np.median(np.abs(x - center))), config.min_scale_db)
    return ((x - center) / scale).astype(np.float32)


def render_gray(data, config, domain="db"):
    if domain == "image":
        x = np.asarray(data)
        if x.ndim != 2 or x.size == 0 or not np.isfinite(x).all():
            raise ValueError("Rendered images must be finite nonempty 2D grayscale arrays")
        if x.min() < 0 or x.max() > 255:
            raise ValueError("Rendered image range must be [0, 255]")
        return np.ascontiguousarray(np.rint(x).astype(np.uint8))
    z = background_standardize(to_db(data, domain), config)
    scaled = (z - config.z_min) / (config.z_max - config.z_min)
    return np.ascontiguousarray(np.rint(255 * np.clip(scaled, 0, 1)).astype(np.uint8))


@dataclass
class DetectorTile:
    image: np.ndarray
    x0: int
    height: int
    width: int
    scale_x: float
    scale_y: float
    pad_left: int
    pad_top: int

    def metadata(self):
        return {k: v for k, v in vars(self).items() if k != "image"}

    def to_source(self, boxes):
        """Map xyxy edges; callers can reject zero-area boxes in padding."""
        b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        b[:, [0, 2]] = np.clip((b[:, [0, 2]] - self.pad_left) / self.scale_x,
                               0, self.width) + self.x0
        b[:, [1, 3]] = np.clip((b[:, [1, 3]] - self.pad_top) / self.scale_y,
                               0, self.height)
        return b

    def from_source(self, boxes):
        b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        b[:, [0, 2]] = (b[:, [0, 2]] - self.x0) * self.scale_x + self.pad_left
        b[:, [1, 3]] = b[:, [1, 3]] * self.scale_y + self.pad_top
        return b


def tile_starts(width, tile_width, overlap):
    if tile_width <= 0 or width <= tile_width:
        return [0]
    last = width - tile_width
    starts = list(range(0, last + 1, tile_width - overlap))
    if starts[-1] != last:
        starts.append(last)
    return starts


def render_tiles(data, config, domain="db") -> Iterator[DetectorTile]:
    x = np.asarray(data)
    if x.ndim != 2 or x.size == 0:
        raise ValueError("Expected nonempty frequency/time matrix")
    h, full_width = x.shape
    for x0 in tile_starts(full_width, config.tile_width, config.tile_overlap):
        tile = x[:, x0:x0 + config.tile_width] if config.tile_width else x
        gray = render_gray(tile, config, domain)
        w = gray.shape[1]
        size = config.image_size
        ratio = min(size / h, size / w)
        rw, rh = max(1, round(w * ratio)), max(1, round(h * ratio))
        resized = cv2.resize(gray, (rw, rh), interpolation=cv2.INTER_LINEAR)
        left, top = (size - rw) // 2, (size - rh) // 2
        canvas = np.full((size, size), config.pad_value, dtype=np.uint8)
        canvas[top:top + rh, left:left + rw] = resized
        rgb = np.ascontiguousarray(np.repeat(canvas[:, :, None], 3, axis=2))
        yield DetectorTile(rgb, x0, h, w, rw / w, rh / h, left, top)


def rgb_to_tensor_array(rgb):
    """Exact CHW float32 array expected immediately before YOLO's first layer."""
    x = np.asarray(rgb)
    if x.dtype != np.uint8 or x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("Expected uint8 HWC RGB image")
    return np.ascontiguousarray(x.transpose(2, 0, 1)).astype(np.float32) / 255.0


def nms_indices(boxes, scores, threshold=0.5):
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if rest.size == 0:
            break
        intersection = np.maximum(0, np.minimum(b[i, 2:], b[rest, 2:])
                                  - np.maximum(b[i, :2], b[rest, :2])).prod(axis=1)
        area_i = np.maximum(0, b[i, 2:] - b[i, :2]).prod()
        areas = np.maximum(0, b[rest, 2:] - b[rest, :2]).prod(axis=1)
        iou = intersection / np.maximum(area_i + areas - intersection, 1e-12)
        order = rest[iou <= threshold]
    return keep


def pixel_y_to_freq_mhz(y, height, sample_rate, axis="legacy"):
    if height <= 1:
        return 0.0
    if axis == "fft":
        return (float(y) - height // 2) * float(sample_rate) / height / 1e6
    return (float(y) / (height - 1) - 0.5) * float(sample_rate) / 1e6


def pixel_height_to_bandwidth_mhz(height_px, height, sample_rate, axis="legacy"):
    if height <= 1:
        return 0.0
    denominator = height if axis == "fft" else height - 1
    return abs(float(height_px)) * float(sample_rate) / denominator / 1e6

def apply_frontend_manifest(args):
    """Load the exact frontend and source settings saved by the exporter."""
    path = getattr(args, "frontend_config", "")
    if not path:
        return
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("version") != FRONTEND_VERSION:
        raise ValueError("Unsupported frontend manifest version")
    cfg = FrontendConfig(**manifest["frontend"])
    mapping = {
        "mode": "frontend_mode", "z_min": "frontend_z_min", "z_max": "frontend_z_max",
        "min_scale_db": "frontend_min_scale_db",
        "frequency_quantile": "frontend_frequency_quantile",
        "frequency_smooth": "frontend_frequency_smooth",
        "tile_width": "detector_tile_width", "tile_overlap": "detector_tile_overlap",
        "nms_iou": "detector_tile_nms_iou",
    }
    for field, name in mapping.items():
        setattr(args, name, getattr(cfg, field))
    args.yolo_imgsz_h = args.yolo_imgsz_w = cfg.image_size
    if cfg.pad_value != 114:
        raise ValueError("CLI currently requires pad_value=114")
    source = manifest.get("source", {})
    if source.get("input_type", args.input_type) != args.input_type:
        raise ValueError("Frontend manifest source type does not match --input_type")
    for name in ("iq_sample_rate", "iq_segment_ms", "iq_n_fft", "iq_hop_length",
                 "mat_key", "mat_spec_domain", "frequency_axis", "iq_byte_order"):
        if name in source:
            setattr(args, name, source[name])


def read_mat_matrix(path, mat_key, domain):
    """Read the declared MAT representation, honoring provenance when available."""
    from scipy.io import loadmat
    mat = loadmat(path, variable_names=[mat_key, "spectrogram_domain"])
    if mat_key not in mat:
        raise KeyError(str(path) + ": missing " + mat_key)
    if "spectrogram_domain" in mat:
        declared = str(np.asarray(mat["spectrogram_domain"]).squeeze())
        if declared != domain:
            raise ValueError("MAT provenance declares " + declared + ", but configured domain is " + domain)
    x = np.asarray(mat[mat_key])
    if x.ndim != 2 or x.size == 0:
        raise ValueError("Expected nonempty 2D MAT spectrum")
    return x
