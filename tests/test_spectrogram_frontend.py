import json
import logging
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from scipy.fft import fft, fftshift
from scipy.io import savemat
from scipy.signal.windows import hamming
from util.spectrogram_frontend import (
    FrontendConfig, FRONTEND_VERSION, SPEC_SHAPE, TARGET_SAMPLE_RATE, TARGET_SAMPLES,
    AMPLITUDE_LEVELS, canonical_amplitude, amplitude_to_db, background_standardize,
    render_gray, render_rgb, iq_to_amplitude, target_iq_to_amplitude, resample_iq_segment,
    prepare_yolo_image, rgb_to_tensor_array, validate_iq_settings,
    pixel_y_to_freq_mhz, spectrogram_contract, apply_frontend_manifest,
)


def amplitude_fixture():
    rows, cols = np.indices(SPEC_SHAPE)
    x = (0.02 + 0.004*np.sin(rows*0.71+cols*0.09)).astype(np.float32)
    x[180:200, 100:200] += 0.3
    x[280:300, 400:500] += 0.6
    return canonical_amplitude(x)


def write_iq(path, extra=0, byte_order="<"):
    # 50 ms at the real 100 MHz capture rate, with a reproducible repeating IQ pattern.
    base = np.random.default_rng(7).integers(-5000, 5000, 2048, dtype=np.int16)
    values = np.resize(base, 2*(5_000_000+extra)).astype(byte_order+"i2")
    path.write_bytes(values.tobytes())


class FrontendTests(unittest.TestCase):
    def test_uint16_amplitude_and_normalized_float_are_identical(self):
        amplitude = amplitude_fixture()
        counts = np.rint(amplitude*AMPLITUDE_LEVELS).astype(np.uint16)
        np.testing.assert_array_equal(canonical_amplitude(counts), amplitude)
        np.testing.assert_array_equal(canonical_amplitude(amplitude), amplitude)
        np.testing.assert_array_equal(render_rgb(canonical_amplitude(counts), FrontendConfig()),
                                      render_rgb(amplitude, FrontendConfig()))

    def test_wrong_shape_and_units_fail_instead_of_resizing(self):
        with self.assertRaisesRegex(ValueError, "512"):
            canonical_amplitude(np.ones((512, 749)))
        with self.assertRaises(ValueError):
            canonical_amplitude(-np.ones(SPEC_SHAPE))
        with self.assertRaises(ValueError):
            render_gray(np.ones(SPEC_SHAPE)*65535, FrontendConfig())
        with self.assertRaises(ValueError):
            validate_iq_settings(100_000_000, 100, 512, 256)

    def test_zero_floor_and_constant_background_are_finite(self):
        zeros = np.zeros(SPEC_SHAPE, np.float32)
        db = amplitude_to_db(zeros)
        self.assertTrue(np.isfinite(db).all())
        self.assertAlmostEqual(float(db[0,0]), -96.3295, places=3)
        self.assertTrue(np.isfinite(background_standardize(db, FrontendConfig())).all())

    def test_gain_invariance_before_quantization(self):
        raw = amplitude_fixture()
        for mode in ("global", "frequency"):
            cfg = FrontendConfig(mode=mode)
            reference = render_gray(canonical_amplitude(raw), cfg)
            for gain in (.01, .1, 10., 100.):
                candidate = render_gray(canonical_amplitude(raw*gain), cfg)
                self.assertLessEqual(np.abs(reference.astype(int)-candidate.astype(int)).max(), 1)

    def test_background_statistics_exclude_resampling_stopband(self):
        db = np.full(SPEC_SHAPE, -30, np.float32)
        cfg = FrontendConfig()
        from util.spectrogram_frontend import background_reference_rows
        mask = background_reference_rows(cfg)
        changed = db.copy()
        changed[~mask] = -90
        np.testing.assert_array_equal(background_standardize(db, cfg)[mask],
                                      background_standardize(changed, cfg)[mask])

    def test_png_is_native_and_not_enhanced_again(self):
        gray = render_gray(amplitude_fixture(), FrontendConfig())
        self.assertEqual(gray.shape, (512, 750))
        np.testing.assert_array_equal(render_gray(gray, FrontendConfig(), "image"), gray)

    def test_yolo_geometry_matches_resize_then_standard_letterbox(self):
        rgb = render_rgb(amplitude_fixture(), FrontendConfig())
        cfg = FrontendConfig()
        prepared = prepare_yolo_image(rgb, cfg)
        # Independent reconstruction of the actual no-augmentation YOLO recipe.
        resized = cv2.resize(rgb, (640, 437), interpolation=cv2.INTER_AREA)
        reference = cv2.copyMakeBorder(resized, 101, 102, 0, 0,
                                       cv2.BORDER_CONSTANT, value=(114,114,114))
        np.testing.assert_array_equal(prepared.image, reference)
        self.assertEqual(rgb.shape, (512, 750, 3))
        boxes = np.array([[10, 20, 740, 500]], np.float32)
        np.testing.assert_allclose(prepared.to_source(prepared.from_source(boxes)), boxes, atol=1e-4)

    def test_detector_restores_native_coordinates(self):
        import torch
        from util.detector import YoloV5Detector
        spec = amplitude_fixture()
        cfg = FrontendConfig()
        expected_rgb = render_rgb(spec, cfg)
        model_input = prepare_yolo_image(expected_rgb, cfg)
        box = np.array([[100, 40, 200, 100]], np.float32)

        class FakeModel:
            def __call__(self, image, size):
                np.testing.assert_array_equal(image, model_input.image)
                det = np.column_stack([model_input.from_source(box), [0.9], [0]])
                return SimpleNamespace(xyxy=[torch.tensor(det)])

        detector = YoloV5Detector.__new__(YoloV5Detector)
        detector.frontend, detector.input_domain = cfg, "amplitude"
        detector.model, detector.imgsz = FakeModel(), 640
        self.assertEqual(detector.detect(spec), [[100,40,200,100]])

    def test_fft_axis_uses_target_rate(self):
        expected = np.fft.fftshift(np.fft.fftfreq(512, 1/TARGET_SAMPLE_RATE))/1e6
        actual = [pixel_y_to_freq_mhz(i, 512, TARGET_SAMPLE_RATE) for i in range(512)]
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_manifest_allows_mat_to_iq_shared_frontend_without_changing_capture_rate(self):
        from dataclasses import asdict
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"frontend.json"
            path.write_text(json.dumps({"version": FRONTEND_VERSION,
                "spectrogram": spectrogram_contract(), "frontend": asdict(FrontendConfig()),
                "source": {"input_type": "mat", "mat_spec_domain": "amplitude",
                           "iq_sample_rate": 122880000}}))
            args = SimpleNamespace(frontend_config=str(path), input_type="iq_dat", iq_sample_rate=100000000)
            apply_frontend_manifest(args)
            self.assertEqual(args.iq_sample_rate, 100000000)
            self.assertEqual(args.yolo_imgsz_h, 640)
            path.write_text(json.dumps({"version": "background-v1"}))
            with self.assertRaises(ValueError):
                apply_frontend_manifest(args)


class SpectrumTests(unittest.TestCase):
    def test_matlab_column_major_formula_matches_blocked_python(self):
        # Independent translation of the supplied MATLAB reshapes and sum order.
        rng = np.random.default_rng(2)
        iq = (rng.standard_normal(TARGET_SAMPLES, dtype=np.float32)
              + 1j*rng.standard_normal(TARGET_SAMPLES, dtype=np.float32)).astype(np.complex64)
        frames = iq.reshape((512, 12000), order="F")
        window = hamming(512, sym=False).astype(np.float32)[:,None]
        spectrum = fftshift(fft(frames*window, axis=0, workers=1), axes=0)
        grouped = spectrum.reshape((512, 16, 750), order="F")
        reference = canonical_amplitude(np.abs(np.sum(grouped, axis=1, dtype=np.complex64)))
        actual = target_iq_to_amplitude(iq)
        self.assertEqual(actual.shape, (512,750))
        # FFT libraries/strides can differ by a quantization level, not a semantic transform.
        self.assertLessEqual(float(np.abs(actual-reference).max()), 2/65535)
        noncoherent = canonical_amplitude(np.mean(np.abs(grouped), axis=1))
        self.assertGreater(float(np.abs(actual-noncoherent).mean()), 0.01)

    def test_100mhz_resampling_and_public_time_frequency_coordinates(self):
        from data.iq_dat_loader import read_iq_spectrogram, read_iq_dat
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"record.dat"
            write_iq(path)
            iq = read_iq_dat(path, 0, 5_000_000)
            upsampled = resample_iq_segment(iq, 100_000_000)
            self.assertEqual(upsampled.size, 6_144_000)
            expected = target_iq_to_amplitude(upsampled)
            spec, times, frequencies = read_iq_spectrogram(path)
            np.testing.assert_array_equal(spec, expected)
            self.assertEqual(spec.shape, (512,750))
            self.assertEqual(len(times), 750)
            self.assertAlmostEqual(times[1]-times[0], 1000/15000, places=10)
            self.assertAlmostEqual(frequencies[0], -61.44)
            self.assertAlmostEqual(frequencies[-1], 61.20)

    def test_122mhz_path_and_zero_signal(self):
        iq = np.zeros(TARGET_SAMPLES, np.complex64)
        out = iq_to_amplitude(iq, TARGET_SAMPLE_RATE)
        self.assertEqual(out.shape, SPEC_SHAPE)
        self.assertFalse(out.any())
        with self.assertRaises(ValueError):
            iq_to_amplitude(iq[:-1], TARGET_SAMPLE_RATE)


class DatasetTests(unittest.TestCase):
    def test_mat_png_filename_dataset_keeps_labels(self):
        from data.data_loader import UAVDataset
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            name = "Test-[0,1,1000,2]-SNR-5"
            amplitude = amplitude_fixture()
            savemat(root/(name+".mat"), {"summed_submatrices":
                    np.rint(amplitude*65535).astype(np.uint16)})
            gray = render_gray(amplitude, FrontendConfig())
            cv2.imwrite(str(root/(name+".png")), gray)
            cfg = SimpleNamespace(input_type="mat", classes={"Test":0}, dataset_path=[str(root)],
                                  mat_spec_domain="amplitude", run_mode="train", train_signal_mode="single")
            x, targets, _, _ = UAVDataset(cfg, logging.getLogger("test"))[0]
            np.testing.assert_array_equal(x.numpy(), amplitude)
            self.assertEqual(targets[0]["center_freq"], 1)
            cfg.input_type = "png"
            x, png_targets, _, _ = UAVDataset(cfg, logging.getLogger("test"))[0]
            np.testing.assert_array_equal(x.numpy(), gray)
            self.assertEqual(targets, png_targets)

    def test_iq_dataset_real_capture_rate_and_center_crop(self):
        from data.iq_dat_loader import RawIQDataset, read_iq_dat
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/"DeviceA"
            folder.mkdir()
            path = folder/"record.dat"
            write_iq(path, extra=200, byte_order=">")
            cfg = SimpleNamespace(classes={"DeviceA":0}, dataset_path=[temp],
                sampling_rate=TARGET_SAMPLE_RATE, iq_sample_rate=100_000_000,
                iq_segment_ms=50, iq_n_fft=512, iq_hop_length=512, iq_byte_order=">")
            x, targets, _, _ = RawIQDataset(cfg, split_name="val")[0]
            expected = iq_to_amplitude(read_iq_dat(path, 100, 5_000_000, byte_order=">"))
            np.testing.assert_array_equal(x.numpy(), expected)
            self.assertFalse(targets[0]["has_frequency_label"])

    def test_one_mat_one_native_png_preview_and_reconstruction(self):
        from scripts.export_iq_yolo_dataset import parser, export, finalize
        from scripts.check_yolo_input_equivalence import check
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp)/"source", Path(temp)/"output"
            root.mkdir()
            for index in range(2):
                savemat(root/f"record-SNR-{index}-Figure-{index+1}.mat",
                        {"summed_submatrices": np.rint(amplitude_fixture()*65535).astype(np.uint16)})
            args = parser().parse_args(["--input_dir",str(root),"--output_dir",str(out),
                                       "--input_type","mat","--split_mode","none"])
            export(args)
            rows = [json.loads(line) for line in (out/"metadata.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["png_shape"] == [512,750] for row in rows))
            self.assertTrue(check(out, root, limit=0)["native_pixels_equal"])
            self.assertFalse((out/"data.yaml").exists())
            with self.assertRaisesRegex(ValueError, "Unassigned"):
                finalize(out)

    def test_iq_export_matches_runtime_adapter_without_padding_png(self):
        from scripts.export_iq_yolo_dataset import parser, export
        from scripts.check_yolo_input_equivalence import check
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp)/"source", Path(temp)/"output"
            (root/"DeviceA").mkdir(parents=True)
            write_iq(root/"DeviceA"/"record.dat", extra=200)
            args = parser().parse_args(["--input_dir",str(root),"--output_dir",str(out),
                "--input_type","iq_dat","--split_mode","none","--segments_per_file","1"])
            export(args)
            rows = [json.loads(line) for line in (out/"metadata.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["segment"]["num_samples"], 5_000_000)
            self.assertEqual(rows[0]["segment"]["start_sample"], 100)
            result = check(out, root, limit=0)
            self.assertTrue(result["native_pixels_equal"])
            self.assertEqual(result["native_png_shape"], [512,750])

    def test_formal_split_and_annotation_gate(self):
        from scripts.export_iq_yolo_dataset import parser, export, finalize
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp)/"source", Path(temp)/"output"
            root.mkdir()
            savemat(root/"recordA.mat", {"summed_submatrices": amplitude_fixture()})
            args = parser().parse_args(["--input_dir",str(root),"--output_dir",str(out),"--input_type","mat"])
            with self.assertRaisesRegex(ValueError, "independent group"):
                export(args)
            self.assertFalse(out.exists())
            savemat(root/"recordB.mat", {"summed_submatrices": amplitude_fixture()})
            export(args)
            with self.assertRaises(ValueError):
                finalize(out)
            rows = [json.loads(line) for line in (out/"metadata.jsonl").read_text().splitlines()]
            for row in rows:
                (out/row["label"]).write_text("0 0.5 0.5 0.25 0.25\n")
            finalize(out)
            self.assertTrue((out/"data.yaml").exists())
            train = {r["group"] for r in rows if r["split"]=="train"}
            val = {r["group"] for r in rows if r["split"]=="val"}
            self.assertFalse(train & val)

    def test_wrong_mat_provenance_rejected(self):
        from util.spectrogram_frontend import read_mat_spectrum
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"x.mat"
            savemat(path, {"summed_submatrices": amplitude_fixture(),
                           "spectrogram_domain": np.array("db")})
            with self.assertRaisesRegex(ValueError, "declares"):
                read_mat_spectrum(path, domain="amplitude")


if __name__ == "__main__":
    unittest.main()
