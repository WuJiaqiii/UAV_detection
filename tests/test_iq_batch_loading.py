from types import MethodType
import time
import unittest

import numpy as np

from data.iq_dat_loader import RawIQDataset


class BatchIQLoadingTests(unittest.TestCase):
    def make_dataset(self, random_crop=True):
        data = RawIQDataset.__new__(RawIQDataset)
        data.samples = [{"num_samples": 100+i*20} for i in range(4)]
        data.segment_samples = 32
        data.random_crop = random_crop
        data.preprocess_threads = 3
        def read(instance, index, start):
            time.sleep((3-index)*.002)
            return np.array([index, start])
        data._read_item = MethodType(read, data)
        return data

    def test_parallel_fetch_preserves_crop_assignment_order_and_rng(self):
        data = self.make_dataset()
        indices = [2, 0, 2, 1, 3]
        np.random.seed(42)
        serial = [data[i] for i in indices]
        next_serial = np.random.randint(10000)
        np.random.seed(42)
        parallel = data.__getitems__(indices)
        next_parallel = np.random.randint(10000)
        np.testing.assert_array_equal(serial, parallel)
        self.assertEqual(next_serial, next_parallel)

    def test_fixed_validation_fetch_and_serial_fallback_match(self):
        data = self.make_dataset(random_crop=False)
        threaded = data.__getitems__([3, 1, 0])
        data.preprocess_threads = 1
        serial = data.__getitems__([3, 1, 0])
        np.testing.assert_array_equal(serial, threaded)

    def test_read_errors_propagate(self):
        data = self.make_dataset()
        def read(instance, index, start):
            if index == 1:
                raise IOError("test short read")
            return index
        data._read_item = MethodType(read, data)
        with self.assertRaisesRegex(IOError, "short read"):
            data.__getitems__([0, 1, 2])


if __name__ == "__main__":
    unittest.main()
