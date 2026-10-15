#!/usr/bin/env python3
"""Synthetic gain/SNR diagnostics, not detector accuracy or real-data validation."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from util.spectrogram_frontend import (
    FrontendConfig, iq_to_db, render_gray, background_standardize,
)


def diagnostics(output):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n_fft, hop, frames, tone_bin = 128, 64, 384, 13
    count = n_fft + (frames-1)*hop
    noise = (rng.normal(size=count) + 1j*rng.normal(size=count)) / np.sqrt(2)
    time = np.arange(count)
    active = (time >= count//4) & (time < count//2)
    tone = np.exp(2j*np.pi*tone_bin*time/n_fft) * active
    report, panels = [], []
    for mode in ("global", "frequency"):
        cfg = FrontendConfig(mode=mode)
        mode_rows = []
        for snr in (-20, -10, 0, 10):
            iq = (noise + np.sqrt(10**(snr/10))*tone).astype(np.complex64)
            db = iq_to_db(iq, n_fft, hop)
            image = render_gray(db, cfg)
            gain_errors = {}
            for gain in (0.01, 0.1, 10.0, 100.0):
                altered = render_gray(iq_to_db(iq*gain, n_fft, hop), cfg)
                error = int(np.max(np.abs(altered.astype(int)-image.astype(int))))
                gain_errors[str(gain)] = error
                if error > 1:
                    raise AssertionError("Gain invariance exceeded one quantization level")
            z = background_standardize(db, cfg)
            row = n_fft//2 + tone_bin
            signal = z[row-1:row+2, frames//4+2:frames//2-2]
            background = z[row-1:row+2, :frames//8]
            report.append({"mode": mode, "active_sample_snr_db": snr,
                           "standardized_signal_minus_background": float(signal.mean()-background.mean()),
                           "gain_max_pixel_error": gain_errors})
            color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            color = cv2.resize(color, (768, 256), interpolation=cv2.INTER_NEAREST)
            cv2.putText(color, mode+"  active sample SNR "+str(snr)+" dB", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            mode_rows.append(color)
        panels.append(np.hstack(mode_rows))
    # Deliberately include persistent occupancy: frequency subtraction can erase it.
    persistent = (noise + np.exp(2j*np.pi*tone_bin*time/n_fft)).astype(np.complex64)
    persistent_db = iq_to_db(persistent, n_fft, hop)
    occupancy = {}
    for mode in ("global", "frequency"):
        z = background_standardize(persistent_db, FrontendConfig(mode=mode))
        row = n_fft//2 + tone_bin
        occupancy[mode] = float(z[row-1:row+2].mean())
    result = {"synthetic_only": True, "gain_test": "passed", "rows": report,
              "persistent_signal_mean_z": occupancy,
              "interpretation": "SNR is at active IQ samples; STFT processing gain differs. "
                                "No mAP, recall or real recording conclusions are implied."}
    (output / "diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not cv2.imwrite(str(output / "comparison.png"), np.vstack(panels)):
        raise IOError("Cannot save comparison")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(diagnostics(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
