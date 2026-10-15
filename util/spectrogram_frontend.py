"""MATLAB-compatible 50 ms amplitude frontend; native PNGs are always 512x750."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import cv2
import numpy as np
from scipy.fft import fft, fftshift
from scipy.ndimage import median_filter
from scipy.signal import resample_poly
from scipy.signal.windows import hamming

FRONTEND_VERSION = "amplitude-512x750-v2"
TARGET_SAMPLE_RATE = 122_880_000
SEGMENT_MS = 50.0
N_FFT = 512
FRAMES_PER_COLUMN = 16
SPEC_SHAPE = (512, 750)
TARGET_SAMPLES = 6_144_000
AMPLITUDE_LEVELS = 65535.0
AMPLITUDE_FLOOR = 1.0 / AMPLITUDE_LEVELS


@dataclass(frozen=True)
class FrontendConfig:
    mode: str = "global"
    z_min: float = -2.0
    z_max: float = 8.0
    min_scale_db: float = 1.0
    frequency_quantile: float = 0.2
    frequency_smooth: int = 9
    background_band_mhz: float = 50.0
    image_size: int = 640
    pad_value: int = 114

    def __post_init__(self):
        if self.mode not in ("global", "frequency", "none"):
            raise ValueError("frontend mode must be global, frequency or none")
        if not np.isfinite([self.z_min, self.z_max, self.min_scale_db,
                            self.background_band_mhz]).all():
            raise ValueError("Non-finite frontend parameter")
        if self.z_max <= self.z_min or self.min_scale_db <= 0:
            raise ValueError("Need z_max > z_min and min_scale_db > 0")
        if not 0 < self.frequency_quantile < 1:
            raise ValueError("frequency_quantile must be in (0,1)")
        if self.frequency_smooth < 1 or self.frequency_smooth % 2 != 1:
            raise ValueError("frequency_smooth must be a positive odd integer")
        if not 0 < self.background_band_mhz <= TARGET_SAMPLE_RATE / 2e6:
            raise ValueError("Invalid background reference half-bandwidth")
        if self.image_size < 32 or self.image_size % 32:
            raise ValueError("image_size must be a multiple of 32")
        if not 0 <= self.pad_value <= 255:
            raise ValueError("Invalid padding value")

    @classmethod
    def from_config(cls, config):
        mapping = {
            "mode": "frontend_mode", "z_min": "frontend_z_min",
            "z_max": "frontend_z_max", "min_scale_db": "frontend_min_scale_db",
            "frequency_quantile": "frontend_frequency_quantile",
            "frequency_smooth": "frontend_frequency_smooth",
            "background_band_mhz": "frontend_background_band_mhz",
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
    g = parser.add_argument_group("512x750 amplitude frontend")
    g.add_argument("--frontend_config", default="")
    g.add_argument("--mat_spec_domain", choices=["amplitude", "power", "db", "image"],
                   default="amplitude", help="Existing MATLAB uint16 MATs use amplitude.")
    g.add_argument("--frontend_mode", choices=["global", "frequency", "none"], default="global")
    g.add_argument("--frontend_z_min", type=float, default=-2.0)
    g.add_argument("--frontend_z_max", type=float, default=8.0)
    g.add_argument("--frontend_min_scale_db", type=float, default=1.0)
    g.add_argument("--frontend_frequency_quantile", type=float, default=0.2)
    g.add_argument("--frontend_frequency_smooth", type=int, default=9)
    g.add_argument("--frontend_background_band_mhz", type=float, default=50.0,
                   help="Shared background-reference half-band, not an image crop.")
    g.add_argument("--frequency_axis", choices=["fft"], default="fft")


def require_shape(data):
    x = np.asarray(data)
    if x.shape != SPEC_SHAPE:
        raise ValueError(f"Expected frequency/time shape {SPEC_SHAPE}, got {x.shape}; "
                         "do not resize scientific spectra to manufacture this shape")
    return x


def canonical_amplitude(data, domain="amplitude"):
    """Return relative float32 amplitude with MATLAB im2uint16-compatible levels.

    Peak normalization plus 16-bit quantization is idempotent. Existing MATLAB
    uint16 counts and their counts/65535 representation produce identical values.
    """
    x = require_shape(data)
    if np.iscomplexobj(x) or not np.isfinite(x).all():
        raise ValueError("Scientific spectra must be real and finite")
    x = x.astype(np.float64)
    if domain in ("amplitude", "power"):
        if (x < 0).any():
            raise ValueError("Amplitude/power cannot be negative; declare the correct MAT domain")
        if domain == "power":
            x = np.sqrt(x)
        peak = float(x.max())
        relative = x / peak if peak > 0 else np.zeros_like(x)
    elif domain == "db":
        relative = np.power(10.0, (x - float(x.max())) / 20.0)
    else:
        raise ValueError("Expected amplitude, power or db; images bypass scientific processing")
    levels = np.floor(np.clip(relative, 0, 1) * AMPLITUDE_LEVELS + 0.5)
    return np.ascontiguousarray(levels / AMPLITUDE_LEVELS, dtype=np.float32)


def validate_iq_settings(sample_rate, segment_ms=SEGMENT_MS, n_fft=N_FFT, hop_length=N_FFT):
    rate = int(sample_rate)
    if rate != sample_rate or rate not in (100_000_000, TARGET_SAMPLE_RATE):
        raise ValueError("Declare the real IQ rate: supported rates are 100000000 and 122880000")
    if float(segment_ms) != SEGMENT_MS or int(n_fft) != N_FFT or int(hop_length) != N_FFT:
        raise ValueError("This profile requires 50 ms, FFT 512, hop 512 and 16-frame coherent sums")
    return round(rate * SEGMENT_MS / 1000.0)


def resample_iq_segment(iq, sample_rate=100_000_000):
    count = validate_iq_settings(sample_rate)
    x = np.asarray(iq, dtype=np.complex64)
    if x.ndim != 1 or x.size != count or not np.isfinite(x).all():
        raise ValueError(f"Need exactly {count} finite complex samples for a 50 ms segment")
    rate = int(sample_rate)
    if rate != TARGET_SAMPLE_RATE:
        divisor = math.gcd(rate, TARGET_SAMPLE_RATE)
        x = resample_poly(x, TARGET_SAMPLE_RATE // divisor, rate // divisor,
                          window=("kaiser", 5.0), padtype="constant")
    if x.size != TARGET_SAMPLES:
        raise AssertionError("Unexpected rational-resampling output length")
    return np.ascontiguousarray(x, dtype=np.complex64)


def target_iq_to_amplitude(iq, block_columns=32):
    """512-point periodic-Hamming FFTs, sum 16 COMPLEX frames, then abs.

    Input is exactly 50 ms at 122.88 MHz. This preserves the supplied MATLAB
    representation; it is deliberately not a power average or a resized STFT.
    FFT-library roundoff can differ from MATLAB; cross-language equality is not
    assumed. All Python export/runtime paths use this exact implementation.
    """
    x = np.asarray(iq, dtype=np.complex64)
    if x.ndim != 1 or x.size != TARGET_SAMPLES or not np.isfinite(x).all():
        raise ValueError("Need 6144000 finite complex samples at the target rate")
    if block_columns < 1:
        raise ValueError("block_columns must be positive")
    window = hamming(N_FFT, sym=False).astype(np.float32)
    groups = x.reshape(SPEC_SHAPE[1], FRAMES_PER_COLUMN, N_FFT)
    amplitude = np.empty(SPEC_SHAPE, dtype=np.float32)
    for start in range(0, SPEC_SHAPE[1], block_columns):
        stop = min(start + block_columns, SPEC_SHAPE[1])
        spectrum = fftshift(fft(groups[start:stop] * window, axis=2, workers=1), axes=2)
        coherent = np.sum(spectrum, axis=1, dtype=np.complex64)
        amplitude[:, start:stop] = np.abs(coherent).T
    return canonical_amplitude(amplitude)


def iq_to_amplitude(iq, sample_rate=100_000_000):
    return target_iq_to_amplitude(resample_iq_segment(iq, sample_rate))


def amplitude_to_db(amplitude):
    x = require_shape(amplitude)
    if np.iscomplexobj(x) or not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError("Expected canonical relative amplitude in [0,1]")
    return (20.0 * np.log10(np.maximum(x.astype(np.float64), AMPLITUDE_FLOOR))).astype(np.float32)


def to_db(data, domain):
    """Compatibility converter: canonical relative amplitude -> dB, exactly once."""
    return amplitude_to_db(canonical_amplitude(data, domain))


def background_reference_rows(config):
    freq = np.fft.fftshift(np.fft.fftfreq(N_FFT, 1.0 / TARGET_SAMPLE_RATE)) / 1e6
    return np.abs(freq) <= config.background_band_mhz


def background_standardize(db, config):
    """Empirical significance; shared reference band excludes resampling stopband."""
    x = require_shape(db).astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite dB spectrum")
    if config.mode == "none":
        return x.astype(np.float32)
    if config.mode == "frequency":
        floor = np.quantile(x, config.frequency_quantile, axis=1, keepdims=True)
        floor = median_filter(floor, size=(config.frequency_smooth, 1), mode="nearest")
        x = x - floor
    reference = x[background_reference_rows(config)]
    center = float(np.median(reference))
    scale = max(1.4826 * float(np.median(np.abs(reference - center))), config.min_scale_db)
    return ((x - center) / scale).astype(np.float32)


def render_gray(data, config, domain="amplitude"):
    """Native 512x750 raster. No resize, letterbox, axis flip or extra enhancement."""
    x = require_shape(data)
    if domain == "image":
        if np.iscomplexobj(x) or not np.isfinite(x).all() or x.min() < 0 or x.max() > 255:
            raise ValueError("Rendered PNG values must be in [0,255]")
        return np.ascontiguousarray(np.rint(x).astype(np.uint8))
    # Dataset adapters already return canonical amplitude. Conversions for named
    # external power/db inputs remain explicit and pass through the same boundary.
    amplitude = x if domain == "amplitude" else canonical_amplitude(x, domain)
    z = background_standardize(amplitude_to_db(amplitude), config)
    return np.ascontiguousarray(np.rint(255 * np.clip(
        (z - config.z_min) / (config.z_max - config.z_min), 0, 1)).astype(np.uint8))


def render_rgb(data, config, domain="amplitude"):
    return np.ascontiguousarray(np.repeat(render_gray(data, config, domain)[:, :, None], 3, axis=2))


def rgb_to_tensor_array(rgb):
    x = np.asarray(rgb)
    if x.dtype != np.uint8 or x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("Expected uint8 HWC RGB image")
    return np.ascontiguousarray(x.transpose(2, 0, 1)).astype(np.float32) / 255.0


@dataclass
class YoloImage:
    image: np.ndarray
    height: int
    width: int
    scale_x: float
    scale_y: float
    pad_left: float
    pad_top: float

    def metadata(self):
        return {key: value for key, value in vars(self).items() if key != "image"}

    def to_source(self, boxes):
        b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        b[:, [0, 2]] = np.clip((b[:, [0, 2]]-self.pad_left)/self.scale_x, 0, self.width)
        b[:, [1, 3]] = np.clip((b[:, [1, 3]]-self.pad_top)/self.scale_y, 0, self.height)
        return b

    def from_source(self, boxes):
        b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        b[:, [0, 2]] = b[:, [0, 2]]*self.scale_x + self.pad_left
        b[:, [1, 3]] = b[:, [1, 3]]*self.scale_y + self.pad_top
        return b


def resize_yolo_image(image, image_size):
    """Shared first resize used by both actual YOLO dataset and inference."""
    h, w = image.shape[:2]
    ratio = image_size / max(h, w)
    if ratio == 1:
        return image.copy()
    rw, rh = math.ceil(w*ratio), math.ceil(h*ratio)
    return cv2.resize(image, (rw, rh),
                      interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)


def prepare_yolo_image(rgb, config):
    """Network-only square letterbox; never saved as the annotation PNG."""
    h, w = rgb.shape[:2]
    if (h, w) != SPEC_SHAPE or rgb.dtype != np.uint8 or rgb.shape[2:] != (3,):
        raise ValueError("Expected native 512x750 uint8 RGB")
    resized = resize_yolo_image(rgb, config.image_size)
    rh, rw = resized.shape[:2]
    dx, dy = (config.image_size-rw)/2.0, (config.image_size-rh)/2.0
    left, top = round(dx-0.1), round(dy-0.1)
    canvas = np.full((config.image_size, config.image_size, 3),
                     config.pad_value, dtype=np.uint8)
    canvas[top:top+rh, left:left+rw] = resized
    # Floating half-padding follows YOLO's label-coordinate convention.
    return YoloImage(canvas, h, w, rw/w, rh/h, dx, dy)


def pixel_y_to_freq_mhz(y, height, sample_rate, axis="fft"):
    if height <= 1:
        return 0.0
    if axis == "fft":
        return (float(y)-height//2) * float(sample_rate)/height/1e6
    return (float(y)/(height-1)-0.5)*float(sample_rate)/1e6


def pixel_height_to_bandwidth_mhz(height_px, height, sample_rate, axis="fft"):
    return abs(float(height_px))*float(sample_rate)/(height if axis == "fft" else height-1)/1e6


def apply_frontend_manifest(args):
    path = getattr(args, "frontend_config", "")
    if not path:
        return
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("version") != FRONTEND_VERSION or manifest.get("spectrogram") != spectrogram_contract():
        raise ValueError("Frontend profile changed; regenerate v2 images/configuration")
    cfg = FrontendConfig(**manifest["frontend"])
    for field, name in {
        "mode": "frontend_mode", "z_min": "frontend_z_min", "z_max": "frontend_z_max",
        "min_scale_db": "frontend_min_scale_db", "frequency_quantile": "frontend_frequency_quantile",
        "frequency_smooth": "frontend_frequency_smooth",
        "background_band_mhz": "frontend_background_band_mhz",
    }.items():
        setattr(args, name, getattr(cfg, field))
    args.yolo_imgsz_h = args.yolo_imgsz_w = cfg.image_size
    if cfg.pad_value != 114:
        raise ValueError("CLI currently requires pad_value=114")
    # Scientific preprocessing can be shared across MAT and IQ. Do not overwrite
    # the caller's input_type, MAT units or true IQ source rate from another source.
    source = manifest.get("source", {})
    if source.get("input_type") == getattr(args, "input_type", None):
        for name in ("iq_sample_rate", "iq_segment_ms", "iq_n_fft", "iq_hop_length",
                     "mat_key", "mat_spec_domain", "frequency_axis", "iq_byte_order"):
            if name in source:
                setattr(args, name, source[name])


def spectrogram_contract():
    return {"sample_rate": TARGET_SAMPLE_RATE, "segment_ms": SEGMENT_MS,
            "shape": list(SPEC_SHAPE), "n_fft": N_FFT, "hop_length": N_FFT,
            "window": "hamming_periodic", "frames_per_column": FRAMES_PER_COLUMN,
            "aggregation": "complex_sum_then_abs", "amplitude": "peak_normalized_uint16_levels",
            "resampler": "scipy_resample_poly_kaiser5_constant", "png": "native_512x750"}


def read_mat_matrix(path, mat_key, domain):
    from scipy.io import loadmat
    mat = loadmat(path, variable_names=[mat_key, "spectrogram_domain"])
    if mat_key not in mat:
        raise KeyError(str(path)+": missing "+mat_key)
    if "spectrogram_domain" in mat:
        declared = str(np.asarray(mat["spectrogram_domain"]).squeeze())
        if declared != domain:
            raise ValueError(f"MAT declares {declared}, configured domain is {domain}")
    return require_shape(mat[mat_key])


def read_mat_spectrum(path, mat_key="summed_submatrices", domain="amplitude"):
    matrix = read_mat_matrix(path, mat_key, domain)
    if domain == "image":
        return render_gray(matrix, FrontendConfig(), "image")
    return canonical_amplitude(matrix, domain)
