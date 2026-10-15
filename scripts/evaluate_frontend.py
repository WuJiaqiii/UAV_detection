#!/usr/bin/env python3
"""Native amplitude gain/SNR rendering diagnostics (synthetic spectral cells)."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from util.spectrogram_frontend import (
    FrontendConfig, SPEC_SHAPE, canonical_amplitude, amplitude_to_db,
    render_gray, background_standardize, spectrogram_contract,
)


def diagnostics(output):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    noise = (rng.normal(size=SPEC_SHAPE) + 1j*rng.normal(size=SPEC_SHAPE))/np.sqrt(2)
    signal = np.zeros(SPEC_SHAPE, np.complex64)
    signal[220:240, 200:400] = 1
    report, panels = [], []
    for mode in ("global", "frequency"):
        cfg = FrontendConfig(mode=mode)
        images = []
        for snr in (-20, -10, 0, 10):
            raw = np.abs(noise + np.sqrt(10**(snr/10))*signal)
            amplitude = canonical_amplitude(raw)
            image = render_gray(amplitude, cfg)
            errors = {}
            for gain in (.01, .1, 10., 100.):
                altered = render_gray(canonical_amplitude(raw*gain), cfg)
                error = int(np.abs(altered.astype(int)-image.astype(int)).max())
                errors[str(gain)] = error
                if error > 1:
                    raise AssertionError("Unexpected gain sensitivity")
            z = background_standardize(amplitude_to_db(amplitude), cfg)
            report.append({"mode": mode, "spectral_cell_snr_db": snr,
                           "signal_minus_background_z": float(z[220:240,200:400].mean()-z[220:240,:100].mean()),
                           "gain_max_pixel_error": errors})
            view = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.putText(view, mode+" / cell SNR "+str(snr)+" dB", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,0), 1)
            images.append(view)
        panels.append(np.hstack(images))
    result = {"synthetic_only": True, "gain_test": "passed",
              "spectrogram": spectrogram_contract(), "rows": report,
              "interpretation": "Noise and signal are constructed in spectral cells. "
                                "These are rendering diagnostics, not IQ SNR, mAP or real-data accuracy."}
    (output/"diagnostics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not cv2.imwrite(str(output/"comparison.png"), np.vstack(panels)):
        raise IOError("Cannot save diagnostics")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(diagnostics(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
