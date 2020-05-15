import unittest

import mne
import torch

from dn3.transforms.basic import ZScore
from tests.dummy_data import create_dummy_dataset, retrieve_underlying_dummy_data, EVENTS


def simple_zscoring(data: torch.Tensor):
    return (data - data.mean()) / data.std()


class TestTransforms(unittest.TestCase):

    def setUp(self) -> None:
        mne.set_log_level(False)
        self.dataset = create_dummy_dataset()
        self.transform = ZScore()
        self.dataset.add_transform(self.transform)

    def test_AddTransform(self):
        self.assertIn(self.transform, self.dataset._transforms)

    def test_ClearTransform(self):
        self.assertIn(self.transform, self.dataset._transforms)
        self.dataset.clear_transforms()
        self.assertNotIn(self.transform, self.dataset._transforms)

    def test_ZScoreTransform(self):
        for i, (x, y) in enumerate(self.dataset):
            with self.subTest(i=i):
                ev_id = i % len(EVENTS)
                self.assertTrue(torch.allclose(x, simple_zscoring(retrieve_underlying_dummy_data(ev_id))))


if __name__ == '__main__':
    unittest.main()
