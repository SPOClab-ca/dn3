import unittest

import mne

from dn3.transforms.preprocessors import EuclideanAlignmentPreprocessor
from dn3.transforms.instance import MappingDeep1010
from tests.dummy_data import create_dummy_dataset, retrieve_underlying_dummy_data, EVENTS


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


if __name__ == '__main__':
    unittest.main()
