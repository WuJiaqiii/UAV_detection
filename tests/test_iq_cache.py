from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from util.iq_cache import fixed_spectrum


class FixedSpectrumCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root/"record.dat"
        self.source.write_bytes(b"original")
        self.compute = Mock(return_value=np.full((512, 750), .125, dtype=np.float32))

    def test_repeat_read_is_bitwise_equal_and_does_not_recompute(self):
        first = fixed_spectrum(self.root/"cache", self.source, {"start": 2}, self.compute)
        second = fixed_spectrum(self.root/"cache", self.source, {"start": 2}, self.compute)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(self.compute.call_count, 1)

    def test_source_or_frontend_changes_cannot_reuse_old_cache(self):
        for source, version in ((b"original", 1), (b"changed-source", 1), (b"changed-source", 2)):
            self.source.write_bytes(source)
            fixed_spectrum(self.root/"cache", self.source, {"frontend": version}, self.compute)
        self.assertEqual(self.compute.call_count, 3)

    def test_corrupt_cache_is_rebuilt(self):
        fixed_spectrum(self.root/"cache", self.source, {}, self.compute)
        next((self.root/"cache").glob("*.npy")).write_bytes(b"incomplete")
        result = fixed_spectrum(self.root/"cache", self.source, {}, self.compute)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(self.compute.call_count, 2)

    def test_wrong_shape_is_rejected_without_cache_file(self):
        with self.assertRaises(ValueError):
            fixed_spectrum(self.root/"cache", self.source, {}, lambda: np.ones((10, 10), dtype=np.float32))
        self.assertFalse(list((self.root/"cache").glob("*.npy")))

    def test_unwritable_cache_returns_fresh_spectrum(self):
        with patch("util.iq_cache.tempfile.NamedTemporaryFile", side_effect=OSError("unwritable")):
            with self.assertWarnsRegex(RuntimeWarning, "using fresh conversion"):
                result = fixed_spectrum(self.root/"cache", self.source, {}, self.compute)
        np.testing.assert_array_equal(result, self.compute.return_value)
        self.assertFalse(list((self.root/"cache").glob("*.npy")))

    def test_atomic_replace_failure_cleans_temporary_file(self):
        with patch("util.iq_cache.os.replace", side_effect=OSError("replace denied")):
            with self.assertWarnsRegex(RuntimeWarning, "using fresh conversion"):
                result = fixed_spectrum(self.root/"cache", self.source, {}, self.compute)
        np.testing.assert_array_equal(result, self.compute.return_value)
        self.assertEqual(list((self.root/"cache").iterdir()), [])

    def test_cache_filename_fits_managed_windows_experiment_path(self):
        fixed_spectrum(self.root/"cache", self.source, {}, self.compute)
        filename = next((self.root/"cache").glob("*.npy")).name
        self.assertLessEqual(len(filename), 28)


if __name__ == "__main__":
    unittest.main()
