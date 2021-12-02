import unittest

import mne
import numpy as np

from dn3.transforms.preprocessors import EuclideanAlignmentPreprocessor, CommonChannelSet
from dn3.transforms.instance import MappingDeep1010, ChannelRemapping
from dn3.transforms.channels import make_map
from tests.dummy_data import create_dummy_dataset, retrieve_underlying_dummy_data, EVENTS, create_dummy_thinker


class TestPreprocessors(unittest.TestCase):

    def setUp(self) -> None:
        mne.set_log_level(False)
        self.dataset = create_dummy_dataset(return_person_id=True, return_session_id=True)

    def test_EApreprocessor(self):
        prep = EuclideanAlignmentPreprocessor(inds=[0, 1, 2, 3])
        self.dataset.preprocess(prep, apply_transform=False)
        self.dataset.add_transform(prep.get_transform())
        for i, data in enumerate(self.dataset):
            with self.subTest(i=i):
                pass
        self.assertTrue(True)

    def test_EApreprocessorMasked(self):
        prep = EuclideanAlignmentPreprocessor()
        self.dataset.add_transform(MappingDeep1010(self.dataset, return_mask=True), deep=True)
        self.dataset.preprocess(prep, apply_transform=False)
        self.dataset.add_transform(prep.get_transform())
        for i, data in enumerate(self.dataset):
            with self.subTest(i=i):
                pass
        self.assertTrue(True)

    def test_CommonSubset(self):
        chs = self.dataset.channels
        SHORT_LENGTH = 4
        self.dataset += create_dummy_thinker(epoched=False, raw_args=dict(num_channels=SHORT_LENGTH), person_id="d0")
        self.dataset += create_dummy_thinker(epoched=False, raw_args=dict(num_channels=SHORT_LENGTH+1), person_id='d1')
        thinkers = self.dataset.thinkers

        prep = CommonChannelSet()
        prep(self.dataset)

        for i, th in enumerate(thinkers):
            with self.subTest(thinker=th):
                self.assertEqual(SHORT_LENGTH, len(thinkers[th].channels))


if __name__ == '__main__':
    unittest.main()
