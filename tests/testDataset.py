import unittest

from copy import deepcopy
from dn3.data.utils import MultiDatasetContainer
from tests.dummy_data import *


class NaNTransform(BaseTransform):

    def __call__(self, x):
        return torch.zeros_like(x) / 0


class TestSessionsDummyData(unittest.TestCase):

    def setUp(self):
        mne.set_log_level(False)
        self.data = torch.from_numpy(create_basic_data())
        self.raw = create_dummy_raw()
        events = mne.find_events(self.raw)
        self.epochs = mne.Epochs(self.raw, events, tmin=TMIN, tmax=TLEN + TMIN - 1 / SFREQ, baseline=None)

    def make_raw_recording(self, **kwargs):
        return RawTorchRecording(self.raw, TLEN, **kwargs)

    def make_epoch_recording(self, **kwargs):
        return EpochTorchRecording(self.epochs, **kwargs)

    def test_RawRecordingCreate(self):
        recording = self.make_raw_recording()
        sample_len = int(TLEN * SFREQ)
        self.assertEqual(len(recording), SFREQ * (END_POINT - START_POINT) - sample_len)

    def test_RawRecordingGet(self):
        recording = self.make_raw_recording(picks=[0, 1])
        for i in (0, 10, -1):
            data_offset = list(range(len(recording)))[i]
            self.assertTrue(check_raw_against_data(recording[i][0], data_offset))

    def test_StridedRawRecording(self):
        stride = 257
        recording = self.make_raw_recording(picks=[0, 1], stride=stride)
        for i in (0, 5, -1):
            data_offset = list(range(len(recording)))[i]
            self.assertTrue(check_raw_against_data(recording[i][0], data_offset*stride))

    def test_DecimatedRawRecording(self):
        decimate = 3
        recording = self.make_raw_recording(picks=[0, 1], decimate=decimate)
        for i in (0, 5, -1):
            data_offset = list(range(len(recording)))[i]
            self.assertTrue(check_raw_against_data(recording[i][0], data_offset, decimate=decimate))

    def test_EpochRecordingCreate(self):
        recording = self.make_epoch_recording()
        self.assertEqual(len(recording), len(EVENTS))
        
    def test_EpochRecordingGet(self):
        recording = self.make_epoch_recording()
        for i, (sample, label) in enumerate(EVENTS):
            x, y = recording[i]
            self.assertTrue(check_epoch_against_data(x, i))
            self.assertEqual(torch.tensor(label) - 1, y)


class TestThinkersDummyData(unittest.TestCase):

    def setUp(self) -> None:
        mne.set_log_level(False)
        self.raw_session = create_dummy_session(epoched=False)
        self.epoch_session = create_dummy_session()
        self.thinker = create_dummy_thinker(sessions_per_thinker=NUM_SESSIONS_PER_THINKER)

    def test_MakeThinkers(self):
        self.assertEqual(len(self.thinker), len(self.epoch_session) * NUM_SESSIONS_PER_THINKER)

    def test_ThinkersGet(self):
        for i, (x, sess_id, y) in enumerate(self.thinker):
            if i < len(EVENTS):
                self.assertEqual(sess_id, 0)
            else:
                self.assertEqual(sess_id, 1)
            self.assertTrue(check_epoch_against_data(x, i % len(self.epoch_session)))
            self.assertEqual(torch.tensor(EVENTS[i % len(self.epoch_session)][1]) - 1, y)

    def test_ThinkerSplitFractions(self):
        training, validating, testing = self.thinker.split(test_frac=0.5, validation_frac=0.5)
        self.assertEqual(len(training), len(EVENTS) // 2)
        self.assertEqual(len(validating), len(EVENTS) // 2)
        self.assertEqual(len(testing), len(EVENTS))

    def test_ThinkerSplitByID(self):
        training, validating, testing = self.thinker.split(testing_sess_ids=['sess1'], validation_frac=0.25)
        self.assertEqual(len(training), 3 * len(EVENTS) // 4)
        self.assertEqual(len(validating), len(EVENTS) // 4 + int(len(EVENTS) % 4 > 0))
        self.assertEqual(len(testing), len(EVENTS))
        for _, (x, sess, y) in enumerate(testing):
            self.assertEqual(sess, 0)


class TestDatasetDummyData(unittest.TestCase):

    # Some unique ids that won't be confused with sessions/people
    _DATASET_ID = 105
    _TASK_ID = 900

    def setUp(self) -> None:
        mne.set_log_level(False)
        self.dataset = create_dummy_dataset(dataset_id=self._DATASET_ID, task_id=self._TASK_ID, return_person_id=True,
                                            return_session_id=True, return_dataset_id=True, return_task_id=True)
        # Test return trial_id using update
        self.dataset.update_id_returns(trial=True)

    def test_MakeDataset(self):
        self.assertEqual(len(self.dataset), len(EVENTS) * 2 * THINKERS_IN_DATASETS)

    def test_DatasetGet(self):
        sess_count, person_count = -1, -1
        for i, (x, task_id, ds_id, person_id, sess_id, trial_id, y) in enumerate(self.dataset):
            with self.subTest(i=i):
                if i % len(EVENTS) == 0:
                    sess_count = (sess_count + 1) % NUM_SESSIONS_PER_THINKER
                if i % (NUM_SESSIONS_PER_THINKER * len(EVENTS)) == 0:
                    person_count += 1
                self.assertEqual(sess_id, sess_count)
                self.assertEqual(person_id, person_count)
                self.assertEqual(ds_id, self._DATASET_ID)
                self.assertEqual(task_id, self._TASK_ID)
                self.assertEqual(trial_id, i % len(EVENTS))
                self.assertTrue(check_epoch_against_data(x, i % len(EVENTS)))
                self.assertEqual(torch.tensor(EVENTS[i % len(EVENTS)][1])-1, y)

    def test_DatasetLOSO(self):
        people = self.dataset.get_thinkers()
        for i, (training, validation, testing) in enumerate(self.dataset.loso()):
            val, test = people[i-1], people[i]
            for p1, p2 in zip((people[j] for j in range(len(people)) if people[j] not in (val, test)),
                              training.get_thinkers()):
                self.assertEqual(p1, p2)
            self.assertEqual(val, validation.person_id)
            self.assertEqual(test, testing.person_id)

    def test_DatasetLOSOIds(self):
        vids = list()
        people = self.dataset.get_thinkers()
        training_people = set(self.dataset.get_thinkers()).difference(['p1'])
        for i, (training, validation, testing) in enumerate(self.dataset.loso(test_person_id='p1')):
            for p1, p2 in zip((people[j] for j in range(len(people)) if people[j] not in (validation.person_id, 'p1')),
                              training.get_thinkers()):
                self.assertEqual(p1, p2)
            self.assertEqual('p1', testing.person_id)
            vids.append(validation.person_id)
        self.assertEqual(len(vids), len(training_people))
        self.assertEqual(training_people, set(vids))

    def test_DatasetLOSOSameValTest(self):

        def check():
            for _ in self.dataset.loso(validation_person_id='p1', test_person_id='p1'):
                pass
        self.assertRaises(ValueError, check)

    def test_DatasetLMSO(self):
        people = self.dataset.get_thinkers()
        for i, (training, validation, testing) in enumerate(self.dataset.lmso(folds=NUM_FOLDS)):
            training_people = set(people).difference(testing.get_thinkers()).difference(
                validation.get_thinkers())
            self.assertSetEqual(set(training.get_thinkers()), training_people)
            self.assertEqual(len(set(testing.get_thinkers()).intersection(validation.get_thinkers())), 0)

    def test_DatasetLMSOTestIds(self):
        people = self.dataset.get_thinkers()
        test_people = people[15:]
        for i, (training, validation, testing) in enumerate(self.dataset.lmso(folds=NUM_FOLDS, test_splits=test_people)):
            training_people = set(people).difference(testing.get_thinkers()).difference(
                validation.get_thinkers())
            self.assertSetEqual(set(training.get_thinkers()), training_people)
            self.assertSetEqual(set(testing.get_thinkers()), set(test_people))
            self.assertEqual(len(set(testing.get_thinkers()).intersection(validation.get_thinkers())), 0)

    def test_DatasetGetTargets(self):
        targets = self.dataset.get_targets()
        for i, y in enumerate(targets):
            with self.subTest(i=i):
                label = EVENTS[i % len(EVENTS)][1]
                self.assertEqual(label - 1, y)

    def test_SafeMode(self):
        self.dataset._safe_mode = True
        person = self.dataset.get_thinkers()[0]
        # Add it to a person to make sure the NaN propagates nicely.
        self.dataset.thinkers[person].add_transform(NaNTransform())
        self.assertRaises(DN3atasetNanFound, lambda: [_ for _ in self.dataset])


class TestDatasetUtils(unittest.TestCase):

    DS_ID = 0

    def setUp(self) -> None:
        mne.set_log_level(False)
        self.dataset = create_dummy_dataset(dataset_id=self.DS_ID, task_id=0, return_person_id=True,
                                            return_session_id=True, return_dataset_id=True, return_task_id=True)

    def test_MultiDatasetContainer(self):
        with self.subTest("Make container"):
            multi = MultiDatasetContainer(self.dataset, self.dataset)
            self.assertEqual(2 * len(self.dataset), len(multi))

        with self.subTest("Oversample"):
            multi = MultiDatasetContainer(self.dataset, ConcatDataset([self.dataset, self.dataset]), oversample=True)
            self.assertEqual(4 * len(self.dataset), len(multi))

        with self.subTest("Oversample w/ Max"):
            multi = MultiDatasetContainer(self.dataset, ConcatDataset([self.dataset, self.dataset]), oversample=True,
                                          max_artificial_size=int(1.5 * len(self.dataset)))
            self.assertEqual(int(3.5 * len(self.dataset)), len(multi))

        with self.subTest("DS ids"):
            ds_copy = deepcopy(self.dataset)
            ds_copy.dataset_id = self.DS_ID + 1
            multi = MultiDatasetContainer(self.dataset, ds_copy, return_dataset_ids=True)
            self.assertEqual(multi[0][-1], self.DS_ID)
            self.assertEqual(multi[len(self.dataset)][-1], self.DS_ID+1)


if __name__ == '__main__':
    unittest.main()
