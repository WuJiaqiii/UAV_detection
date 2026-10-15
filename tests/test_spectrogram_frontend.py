import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.io import savemat
from util.spectrogram_frontend import (
    FrontendConfig, background_standardize, iq_to_db, to_db, render_gray,
    render_tiles, nms_indices, pixel_y_to_freq_mhz, pixel_height_to_bandwidth_mhz,
)


class FrontendTests(unittest.TestCase):
    def test_power_amplitude_db_agree_and_wrong_units_fail(self):
        power = np.linspace(0.001, 2, 64*80).reshape(64, 80)
        np.testing.assert_allclose(to_db(power, "power"),
                                   to_db(np.sqrt(power), "amplitude"), atol=1e-5)
        cfg = FrontendConfig()
        np.testing.assert_array_equal(render_gray(power, cfg, "power"),
                                      render_gray(to_db(power, "power"), cfg, "db"))
        with self.assertRaises(ValueError):
            to_db(-power, "power")

    def test_absolute_db_not_peak_normalized_or_clipped(self):
        rng = np.random.default_rng(2)
        iq = (rng.normal(size=5000)+1j*rng.normal(size=5000)).astype(np.complex64)
        a = iq_to_db(iq, 128, 64)
        b = iq_to_db(iq*0.01, 128, 64)
        np.testing.assert_allclose(b-a, -40, atol=1e-3)
        with self.assertRaises(ValueError):
            iq_to_db(iq, 128, 256)

    def test_blocked_fft_matches_single_batch(self):
        rng = np.random.default_rng(3)
        iq = (rng.normal(size=7000)+1j*rng.normal(size=7000)).astype(np.complex64)
        np.testing.assert_array_equal(iq_to_db(iq, 128, 64, 7),
                                      iq_to_db(iq, 128, 64, 2048))

    def test_gain_invariance_and_empty_background_are_finite(self):
        rng = np.random.default_rng(4)
        db = rng.normal(-60, 4, (128, 300)).astype(np.float32)
        db[60:65, 100:160] += 20
        for mode in ("global", "frequency"):
            cfg = FrontendConfig(mode=mode)
            a = render_gray(db, cfg)
            b = render_gray(db+40, cfg)
            self.assertLessEqual(np.abs(a.astype(int)-b.astype(int)).max(), 1)
            self.assertTrue(np.isfinite(background_standardize(np.zeros((8, 8)), cfg)).all())

    def test_rendered_png_is_not_enhanced(self):
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)
        np.testing.assert_array_equal(render_gray(image, FrontendConfig(), "image"), image)

    def test_tile_coverage_coordinates_and_padding(self):
        cfg = FrontendConfig(image_size=128, tile_width=100, tile_overlap=25)
        tiles = list(render_tiles(np.zeros((64, 257)), cfg, "db"))
        covered = np.zeros(257, bool)
        for tile in tiles:
            covered[tile.x0:tile.x0+tile.width] = True
            b = np.array([[tile.x0+4, 2, tile.x0+80, 60]], np.float32)
            np.testing.assert_allclose(tile.to_source(tile.from_source(b)), b, atol=1e-4)
            self.assertEqual(tile.image.shape, (128, 128, 3))
            outside = tile.to_source([[0, 0, 128, tile.pad_top]])
            self.assertEqual(outside[0, 1], outside[0, 3])
        self.assertTrue(covered.all())
        self.assertEqual(tiles[-1].x0+tiles[-1].width, 257)

    def test_overlap_nms_retains_distinct_frames(self):
        keep = nms_indices([[0, 0, 10, 10], [1, 1, 11, 11], [20, 0, 30, 10]],
                           [0.9, 0.8, 0.7], 0.5)
        self.assertEqual(keep, [0, 2])

    def test_fft_axis_matches_numpy(self):
        for height in (127, 128):
            expected = np.fft.fftshift(np.fft.fftfreq(height, 1/1e6))/1e6
            actual = [pixel_y_to_freq_mhz(i, height, 1e6, "fft") for i in range(height)]
            np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertEqual(pixel_height_to_bandwidth_mhz(128, 128, 1e6, "fft"), 1.0)

    def test_detector_maps_tiles_to_full_matrix_before_nms(self):
        import torch
        from util.detector import YoloV5Detector
        cfg = FrontendConfig(image_size=128, tile_width=100, tile_overlap=25)
        spec = np.zeros((64, 257), dtype=np.float32)
        tiles = list(render_tiles(spec, cfg, "db"))
        source_box = np.array([[80, 10, 90, 30]], dtype=np.float32)

        class FakeModel:
            def __init__(self):
                self.index = 0

            def __call__(self, image, size):
                tile = tiles[self.index]
                self.index += 1
                np.testing.assert_array_equal(image, tile.image)
                if tile.x0 <= 80 and tile.x0+tile.width >= 90:
                    box = tile.from_source(source_box)
                    det = torch.tensor(np.column_stack([box, [0.9], [0]]))
                else:
                    det = torch.zeros((0, 6))
                return SimpleNamespace(xyxy=[det])

        detector = YoloV5Detector.__new__(YoloV5Detector)
        detector.frontend = cfg
        detector.input_domain = "db"
        detector.imgsz = cfg.image_size
        detector.config = SimpleNamespace(yolo_max_det=100)
        detector.model = FakeModel()
        self.assertEqual(detector.detect(spec), [[80, 10, 90, 30]])


class DatasetIntegrationTests(unittest.TestCase):
    def logger(self):
        import logging
        return logging.getLogger("frontend_test")

    def test_legacy_mat_png_filename_dataset_keeps_labels(self):
        from data.data_loader import UAVDataset
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            name = "Test-[0,1,1000,2]-SNR-5"
            raw = np.linspace(.001, 1, 64*80).reshape(64, 80).astype(np.float32)
            savemat(root / (name+".mat"), {"summed_submatrices": raw})
            image = np.arange(256, dtype=np.uint8).reshape(16, 16)
            cv2.imwrite(str(root / (name+".png")), image)
            cfg = SimpleNamespace(input_type="mat", classes={"Test": 0},
                                  dataset_path=[str(root)], mat_spec_domain="power",
                                  run_mode="train", train_signal_mode="single")
            ds = UAVDataset(cfg, self.logger())
            x, targets, snr, fp = ds[0]
            np.testing.assert_array_equal(x.numpy(), to_db(raw, "power"))
            self.assertEqual(targets[0]["center_freq"], 1)
            self.assertEqual(targets[0]["bandwidth"], 2)
            cfg.input_type = "png"
            x, png_targets, _, _ = UAVDataset(cfg, self.logger())[0]
            np.testing.assert_array_equal(x.numpy(), image)
            self.assertEqual(png_targets, targets)
            cfg.input_type, cfg.mat_spec_domain = "mat", "db"
            savemat(root / (name+".mat"), {"summed_submatrices": to_db(raw, "power")})
            x, _, _, _ = UAVDataset(cfg, self.logger())[0]
            np.testing.assert_array_equal(x.numpy(), to_db(raw, "power"))
            savemat(root / (name+".mat"), {"summed_submatrices": to_db(raw, "power"),
                                           "spectrogram_domain": np.array("db")})
            cfg.mat_spec_domain = "power"
            with self.assertRaises(RuntimeError):
                UAVDataset(cfg, self.logger())[0]

    def test_iq_class_folder_dataset_byte_order_crop_and_units(self):
        from data.iq_dat_loader import RawIQDataset, read_iq_dat
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "DeviceA"
            root.mkdir()
            rng = np.random.default_rng(5)
            values = rng.integers(-2000, 2000, 10000, dtype=np.int16)
            path = root / "record.dat"
            path.write_bytes(values.astype(">i2").tobytes())
            cfg = SimpleNamespace(classes={"DeviceA": 0}, dataset_path=[str(root.parent)],
                                  sampling_rate=1_000_000, iq_n_fft=128, iq_hop_length=64,
                                  iq_segment_ms=2, iq_byte_order=">")
            dataset = RawIQDataset(cfg, split_name="val")
            x, targets, _, fp = dataset[0]
            expected_iq = read_iq_dat(path, 1500, 2000, byte_order=">")
            np.testing.assert_array_equal(x.numpy(), iq_to_db(expected_iq, 128, 64))
            self.assertFalse(targets[0]["has_frequency_label"])
            self.assertEqual(targets[0]["class_name"], "DeviceA")

    def test_export_rebuild_pixels_and_manual_annotation_gate_for_both_datasets(self):
        from scripts.export_iq_yolo_dataset import parser, export, finalize
        from scripts.check_yolo_input_equivalence import check
        for kind in ("iq_dat", "mat", "png"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                root, output = base/"source", base/"output"
                (root/"DeviceA").mkdir(parents=True)
                rng = np.random.default_rng(7)
                for index in range(2):
                    folder = root/"DeviceA"
                    if kind == "iq_dat":
                        values = rng.integers(-1000, 1000, 10000, dtype=np.int16)
                        (folder/("record"+str(index)+".dat")).write_bytes(values.astype("<i2").tobytes())
                    elif kind == "mat":
                        savemat(folder/("record"+str(index)+".mat"),
                                {"summed_submatrices": rng.random((128, 80))})
                    else:
                        cv2.imwrite(str(folder/("record"+str(index)+".png")),
                                    rng.integers(0, 256, (128, 80), dtype=np.uint8))
                args = parser().parse_args([
                    "--input_dir", str(root), "--output_dir", str(output), "--input_type", kind,
                    "--iq_sample_rate", "1000000", "--iq_segment_ms", "2",
                    "--iq_n_fft", "128", "--iq_hop_length", "64",
                    "--detector_tile_width", "24", "--detector_tile_overlap", "8",
                    "--yolo_imgsz_h", "64", "--yolo_imgsz_w", "64",
                ])
                export(args)
                result = check(output, root, limit=0)
                self.assertTrue(result["pixel_equal"])
                self.assertFalse((output/"data.yaml").exists())
                with self.assertRaises(ValueError):
                    finalize(output)
                rows = [json.loads(line) for line in (output/"metadata.jsonl").read_text().splitlines()]
                groups_by_split = {split: {r["group"] for r in rows if r["split"] == split}
                                   for split in ("train", "val")}
                self.assertFalse(groups_by_split["train"] & groups_by_split["val"])
                for row in rows:
                    (output/row["label"]).write_text("0 0.5 0.5 0.25 0.25\n")
                finalize(output)
                self.assertTrue((output/"data.yaml").exists())
                with self.assertRaises(ValueError):
                    export(args)

    def test_synthetic_gain_snr_report(self):
        from scripts.evaluate_frontend import diagnostics
        with tempfile.TemporaryDirectory() as temp:
            report = diagnostics(Path(temp))
            self.assertEqual(report["gain_test"], "passed")
            self.assertEqual(len(report["rows"]), 8)
            for mode in ("global", "frequency"):
                contrasts = [row["standardized_signal_minus_background"]
                             for row in report["rows"] if row["mode"] == mode]
                self.assertGreater(contrasts[-1], contrasts[0])
            self.assertTrue((Path(temp)/"comparison.png").is_file())


if __name__ == "__main__":
    unittest.main()
