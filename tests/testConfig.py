import mne
import unittest
import os

from pathlib import Path
from dn3.configuratron.config import ExperimentConfig

_DATASET_URL = "https://physionet.org/files/eegmmidb/1.0.0/"
_TEST_DATASET_LOCATION = "./test_dataset/"


def download_test_dataset():
    location = Path(_TEST_DATASET_LOCATION)
    # Warning: just assuming if the directory exists we can proceed with tests
    if location.exists():
        return

    os.system("wget -r -N -c -np -P {} {}".format(_TEST_DATASET_LOCATION, _DATASET_URL))


def create_mmi_dataset_from_config():
    experiment = ExperimentConfig('./test_dataset_config.yml')
    return experiment.datasets['mmi_fully_specified'].auto_construct_dataset()


class TestExperimentConfiguration(unittest.TestCase):

    def setUp(self) -> None:
        mne.set_log_level(False)
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
    SFREQ = 160.0
    ALT_SFREQ = 128.0

    def setUp(self) -> None:
        mne.set_log_level(False)
        self.experiment_config = ExperimentConfig('./test_dataset_config.yml')
        self.minimal_raw = self.experiment_config.datasets['mmidb_minimally_specified_raw']
        self.minimal_epoch = self.experiment_config.datasets['mmidb_minimally_specified_epoch']
        self.fully = self.experiment_config.datasets['mmidb_fully_specified']

    def test_MinimallySpecifiedRawConstruct(self):
        dataset = self.minimal_raw.auto_construct_dataset()
        self.assertEqual(self.NUM_SUBJECTS, len(dataset.get_thinkers()))

    def test_MinimallySpecifiedEpochConstruct(self):
        dataset = self.minimal_epoch.auto_construct_dataset()
        self.assertEqual(self.NUM_SUBJECTS, len(dataset.get_thinkers()))
        self.assertSetEqual(set(dataset.sfreq), {self.SFREQ, self.ALT_SFREQ})

    def test_FullySpecifiedConstruct(self):
        dataset = self.fully.auto_construct_dataset()
        # Check the exclusion worked
        self.assertEqual(self.NUM_SUBJECTS - 4, len(dataset.get_thinkers()))
        # After exclusion, should have single SFREQ
        self.assertEqual(self.SFREQ / self.fully.decimate, dataset.sfreq)


if __name__ == '__main__':
    unittest.main()
