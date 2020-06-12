import unittest

import mne
import torch

from dn3.utils import min_max_normalize
from dn3.transforms.channels import DEEP_1010_CHS_LISTING
from dn3.transforms.basic import ZScore, MappingDeep1010
from tests.dummy_data import create_dummy_dataset, retrieve_underlying_dummy_data, EVENTS, check_epoch_against_data


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

    def test_TransformAfterLOSO(self):
        for i, (training, validating, testing) in enumerate(self.dataset.loso()):
            with self.subTest(i=i):
                self.assertIn(self.transform, training._transforms)
                self.assertIn(self.transform, validating._transforms)
                self.assertIn(self.transform, testing._transforms)
                train, val, test = testing.split(testing_sess_ids=['sess1'])
                self.assertIn(self.transform, test._transforms)

    def test_TransformAfterLMSO(self):
        for i, (training, validating, testing) in enumerate(self.dataset.lmso()):
            with self.subTest(i=i):
                self.assertIn(self.transform, training._transforms)
                self.assertIn(self.transform, validating._transforms)
                self.assertIn(self.transform, testing._transforms)

    def test_ZScoreTransform(self):
        i = 0
        for x, y in self.dataset:
            i += 1
            with self.subTest(i=i):
                ev_id = (i-1) % len(EVENTS)
                self.assertTrue(torch.allclose(x, simple_zscoring(retrieve_underlying_dummy_data(ev_id))))

    def test_MapDeep1010Channels(self):
        transform = MappingDeep1010(self.dataset)
        self.dataset.add_transform(transform)
        with self.subTest('channel shape'):
            self.assertEqual((len(DEEP_1010_CHS_LISTING), 2), self.dataset.channels.shape)

        i = 0
        for x, y in self.dataset:
            i += 1
            with self.subTest(i=i):
                ev_id = (i - 1) % len(EVENTS)
                self.assertTrue(x.max() == 1)
                self.assertTrue(x.min() == -1)


if __name__ == '__main__':
    unittest.main()
