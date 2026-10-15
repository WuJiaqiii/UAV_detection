import json
from pathlib import Path
import tempfile
import unittest

from scripts.remap_yolo_labels import remap_dataset


class RemapYoloLabelsTests(unittest.TestCase):
    def test_rewrites_only_other_class_tokens_and_backs_up_originals_and_caches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root/"labels"/"train"/"DeviceA"
            val = root/"labels"/"val"
            train.mkdir(parents=True)
            val.mkdir()
            label = train/"example.txt"
            original = (b"\xef\xbb\xbf  0\t0.5 0.5 0.2 0.3\r\n"
                        b"1 0.20 0.30 0.10 0.10\r\n"
                        b"2 0.3 0.4 0.2 0.1\r\n"
                        b" \t3\t0.5000 0.6000 0.2000 0.1000\r\n"
                        b"\r\n"
                        b"12 0.5 0.5 0.2 0.2")
            label.write_bytes(original)
            unchanged = val/"unchanged.txt"
            unchanged.write_bytes(b"2 0.5 0.5 0.25 0.25\n")
            (val/"empty.txt").write_bytes(b"")
            cache = root/"labels"/"train.cache"
            cache.write_bytes(b"cached-old-classes")
            flat_cache = root/"labels.cache"
            flat_cache.write_bytes(b"another-cache")
            (root/"images").mkdir()
            (root/"images"/"untouched.txt").write_text("not a label")
            (root/"data.yaml").write_text("nc: 3\nnames: [a, b, c]\n")

            result = remap_dataset(root)
            expected = original.replace(b" \t3\t", b" \t0\t").replace(b"12 0.5", b"0 0.5")
            self.assertEqual(label.read_bytes(), expected)
            self.assertEqual(unchanged.read_bytes(), b"2 0.5 0.5 0.25 0.25\n")
            self.assertEqual((val/"empty.txt").read_bytes(), b"")
            self.assertEqual(result["files_scanned"], 3)
            self.assertEqual(result["empty_files"], 1)
            self.assertEqual(result["files_changed"], 1)
            self.assertEqual(result["annotations_changed"], 2)
            self.assertEqual(result["classes_before"], {0:1, 1:1, 2:2, 3:1, 12:1})
            self.assertEqual(result["classes_after"], {0:3, 1:1, 2:2})
            backup = Path(result["backup_dir"])
            self.assertEqual((backup/"labels"/"train"/"DeviceA"/"example.txt").read_bytes(), original)
            manifest = json.loads((backup/"manifest.json").read_text())
            self.assertEqual(len(manifest["caches"]), 2)
            self.assertFalse(cache.exists())
            self.assertFalse(flat_cache.exists())
            self.assertEqual((root/"images"/"untouched.txt").read_text(), "not a label")
            self.assertEqual((root/"data.yaml").read_text(), "nc: 3\nnames: [a, b, c]\n")
            # A second run is a no-op and creates no extra backup.
            again = remap_dataset(root/"labels")
            self.assertEqual(again["files_changed"], 0)
            self.assertIsNone(again["backup_dir"])
            self.assertEqual(len(list(root.glob("labels_backup_*"))), 1)

    def test_dry_run_leaves_all_files_and_caches_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/"labels"/"test").mkdir(parents=True)
            label = root/"labels"/"test"/"sample.txt"
            original = b"4 0.5 0.5 0.2 0.2\n"
            label.write_bytes(original)
            cache = root/"labels"/"test.cache"
            cache.write_bytes(b"cache")
            result = remap_dataset(root, dry_run=True)
            self.assertEqual(result["annotations_to_change"], 1)
            self.assertEqual(result["caches_to_invalidate"], 1)
            self.assertEqual(label.read_bytes(), original)
            self.assertEqual(cache.read_bytes(), b"cache")
            self.assertEqual(list(root.glob("labels_backup_*")), [])

    def test_invalid_label_aborts_before_modifying_valid_earlier_files(self):
        for invalid in (b"1 0.5 0.5\n", b"3.5 0.5 0.5 0.2 0.2\n",
                        b"3 nan 0.5 0.2 0.2\n", b"3 bad 0.5 0.2 0.2\n"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root/"labels").mkdir()
                first = root/"labels"/"a.txt"
                first.write_bytes(b"9 0.5 0.5 0.2 0.2\n")
                (root/"labels"/"z.txt").write_bytes(invalid)
                with self.assertRaises(ValueError):
                    remap_dataset(root)
                self.assertEqual(first.read_bytes(), b"9 0.5 0.5 0.2 0.2\n")
                self.assertEqual(list(root.glob("labels_backup_*")), [])

    def test_missing_label_directory_and_empty_dataset_are_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(FileNotFoundError):
                remap_dataset(root)
            (root/"labels").mkdir()
            with self.assertRaises(FileNotFoundError):
                remap_dataset(root)


if __name__ == "__main__":
    unittest.main()
