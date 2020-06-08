import mne
import unittest
import os
import yaml
import json

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


_TMP_DIR = './test-tmp/'
_TMP_FN = 'test_Includes'


def _clear_include_files():
    os.remove(os.path.join(_TMP_DIR, _TMP_FN + '.yml'))
    os.rmdir(_TMP_DIR)
    os.remove(_TMP_FN + '.yml')
    os.remove(_TMP_FN + '.json')


def _generate_include_files():
    os.mkdir(_TMP_DIR)
    basic_include = dict(eyes=5, feet=False)
    glob_include = dict(events=[1, 4, 'f45'])
    json_include = dict(layers=2, dropout=0.5, activation='relu')
    with open(_TMP_FN + '.yml', 'w') as stream:
        yaml.dump(basic_include, stream)
    with open(os.path.join(_TMP_DIR, _TMP_FN + '.yml'), 'w') as stream:
        yaml.dump(glob_include, stream)
    with open(_TMP_FN + '.json', 'w') as stream:
        json.dump(json_include, stream)

    return basic_include, glob_include, json_include


class TestExperimentConfiguration(unittest.TestCase):

    def setUp(self) -> None:
        mne.set_log_level(False)
        download_test_dataset()
        self.basic_include, self.glob_include, self.json_include = _generate_include_files()
        self.experiment_config = ExperimentConfig('./test_dataset_config.yml')

    def test_DatasetAvailable(self):
        self.assertTrue(Path(_TEST_DATASET_LOCATION).exists())

    def test_BasicParse(self):
        pass

    def test_DatasetsRetrieved(self):
        self.assertEqual(3, len(self.experiment_config.datasets))

    def test_Auxiliaries(self):
        self.assertTrue(hasattr(self.experiment_config, "another_extra"))
        self.assertFalse(hasattr(self.experiment_config, "an_extra_param"))

    def test_IncludesBasic(self):
        self.assertTrue(hasattr(self.experiment_config, "basic_include"))
        self.assertEqual(self.experiment_config.basic_include.eyes, self.basic_include['eyes'])
        self.assertEqual(self.experiment_config.basic_include.feet, self.basic_include['feet'])

    def test_IncludesGlob(self):
        self.assertTrue(hasattr(self.experiment_config, "glob_include"))
        self.assertSetEqual(set(self.experiment_config.glob_include[0].events), set(self.glob_include['events']))

    def test_IncludesNonYAML(self):
        self.assertTrue(hasattr(self.experiment_config, "non_yaml"))
        for k in self.json_include.keys():
            with self.subTest(msg=k):
                self.assertEqual(self.json_include[k], self.experiment_config.non_yaml.__dict__[k])

    def tearDown(self):
        _clear_include_files()


class TestDatasetConfiguration(unittest.TestCase):

    NUM_SUBJECTS = 109
    RAW_TRIALS = 10000
    EPOCH_TRIALS = 100
    SFREQ = 160.0
    ALT_SFREQ = 128.0

    def setUp(self) -> None:
        mne.set_log_level(False)
        self.basic_include, self.glob_include, self.json_include = _generate_include_files()
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

    def tearDown(self) -> None:
        _clear_include_files()


if __name__ == '__main__':
    unittest.main()
