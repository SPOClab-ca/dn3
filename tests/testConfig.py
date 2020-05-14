import unittest
import os

from pathlib import Path
from configuratron.config import ExperimentConfig

_DATASET_URL = "https://physionet.org/files/eegmmidb/1.0.0/"
_TEST_DATASET_LOCATION = "./test_dataset/"


def download_test_dataset():
    location = Path(_TEST_DATASET_LOCATION)
    # Warning: just assuming if the directory exists we can proceed with tests
    if location.exists():
        return

    os.system("wget -r -N -c -np -P {} {}".format(_TEST_DATASET_LOCATION, _DATASET_URL))


class TestExperimentConfiguration(unittest.TestCase):

    def setUp(self) -> None:
        download_test_dataset()
        self.experiment_config = ExperimentConfig('./test_dataset_config.yml')
        pass

    def test_DatasetAvailable(self):
        self.assertTrue(Path(_TEST_DATASET_LOCATION).exists())

    def test_BasicParse(self):
        pass

    def test_DatasetsRetrieved(self):
        self.assertEqual(3, len(self.experiment_config.datasets))

    def test_Auxiliaries(self):
        self.assertTrue(hasattr(self.experiment_config, "another_extra"))
        self.assertFalse(hasattr(self.experiment_config, "an_extra_param"))


class TestDatasetConfiguration(unittest.TestCase):

    NUM_SUBJECTS = 109
    RAW_TRIALS = 10000
    EPOCH_TRIALS = 100
    SFREQ = 160

    def setUp(self) -> None:
        self.experiment_config = ExperimentConfig('./test_dataset_config.yml')
        self.minimal_raw = self.experiment_config.datasets['mmidb_minimally_specified_raw']
        self.minimal_epoch = self.experiment_config.datasets['mmidb_minimally_specified_epoch']
        self.fully = self.experiment_config.datasets['mmidb_fully_specified']

    def test_AllSpecificationsFindFiles(self):
        three_values = self.minimal_raw.auto_mapping() == self.minimal_epoch.auto_mapping() == self.fully.auto_mapping()
        self.assertTrue(three_values)

    def test_MinimallySpecifiedRawConstruct(self):
        dataset = self.minimal_raw.auto_construct_dataset()
        self.assertEqual(self.NUM_SUBJECTS, len(dataset.get_thinkers()))

    def test_MinimallySpecifiedEpochConstruct(self):
        dataset = self.minimal_epoch.auto_construct_dataset()
        self.assertEqual(self.NUM_SUBJECTS, len(dataset.get_thinkers()))

    def test_FullySpecifiedConstruct(self):
        dataset = self.fully.auto_construct_dataset()
        self.assertEqual(self.NUM_SUBJECTS, len(dataset.get_thinkers()))
        self.assertEqual(dataset.sfreq, )


if __name__ == '__main__':
    unittest.main()
