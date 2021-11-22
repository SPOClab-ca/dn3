import unittest

import mne
import torch

from dn3.utils import min_max_normalize
from dn3.transforms.channels import DEEP_1010_CHS_LISTING, stringify_channel_mapping
from tests.dummy_data import create_dummy_dataset, retrieve_underlying_dummy_data, EVENTS, check_epoch_against_data

from dn3.transforms.instance import ZScore, MappingDeep1010, TemporalInterpolation


def simple_zscoring(data: torch.Tensor):
    return (data - data.mean()) / data.std()


def _check_zscored_trial(event_id):
    return simple_zscoring(retrieve_underlying_dummy_data(event_id))


class TestInstanceTransforms(unittest.TestCase):

    def setUp(self) -> None:
        mne.set_log_level(False)
        self.dataset = create_dummy_dataset()
        self.dataset.return_person_id = True
        self.dataset.return_session_id = True
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
        for data in self.dataset:
            x = data[0]
            i += 1
            with self.subTest(i=i):
                ev_id = (i-1) % len(EVENTS)
                self.assertTrue(torch.allclose(x, _check_zscored_trial(ev_id)))

    def test_TemporalInterpolation(self):
        new_seq_len = self.dataset.sequence_length * 2
        transform = TemporalInterpolation(new_seq_len)
        self.dataset.add_transform(transform)

        with self.subTest(i="initialization"):
            self.assertEqual(self.dataset.sequence_length, new_seq_len)

        i = 0
        for data in self.dataset:
            x = data[0]
            i += 1
            with self.subTest(i=i):
                ev_id = (i - 1) % len(EVENTS)
                # nearest should mean that values are just dumplicated and we can slice to get original
                self.assertTrue(torch.allclose(x[:, slice(0, x.shape[1], 2)], _check_zscored_trial(ev_id)))

    def test_MapDeep1010Channels(self):
        transform = MappingDeep1010(self.dataset, return_mask=True)
        self.dataset.add_transform(transform)
        with self.subTest('channel shape'):
            self.assertEqual((len(DEEP_1010_CHS_LISTING), 2), self.dataset.channels.shape)

        i = 0
        for data in self.dataset:
            x = data[0]
            mask = data[1]
            i += 1
            with self.subTest(i=i):
                ev_id = (i - 1) % len(EVENTS)
                self.assertEqual(x.shape[0], len(MappingDeep1010.channel_listing()))
                self.assertEqual(len(mask), len(MappingDeep1010.channel_listing()))
                self.assertTrue(x.max() == 1)
                self.assertTrue(x.min() == -1)

    def test_MapDeep1010Print(self):
        ch_names = [ch[0] for ch in self.dataset.channels]
        transform = MappingDeep1010(self.dataset)
        stringify_channel_mapping(ch_names, transform.mapping.numpy())


class TestBatchTransforms(unittest.TestCase):

    def setUp(self) -> None:
        mne.set_log_level(False)
        self.dataset = create_dummy_dataset()


if __name__ == '__main__':
    unittest.main()
