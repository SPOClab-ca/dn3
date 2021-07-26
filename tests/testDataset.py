import unittest
import os

from copy import deepcopy
from dn3.data.utils import MultiDatasetContainer, SingleStatisticSpanRejection
from dn3.transforms.instance import ZScore, MappingDeep1010
from tests.dummy_data import *


class NaNTransform(InstanceTransform):

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

    def test_RawRecordingSkipGet(self):
        num_skips = 2
        recording = self.make_raw_recording(picks=[0, 1], bad_spans=[(0, num_skips / SFREQ)])
        for i, offset in ((0, num_skips), (5, 5 + num_skips), (-1, (END_POINT - 1) * SFREQ - 1)):
            self.assertTrue(check_raw_against_data(recording[i][0], int(offset)))

    def test_RawRecordingSkipGetDecimated(self):
        decimate = 3
        num_skips = 2
        decimated_sfreq = SFREQ / decimate
        recording = self.make_raw_recording(picks=[0, 1], decimate=decimate, bad_spans=[(0, num_skips/decimated_sfreq)])
        for i, offset in ((0, num_skips), (5, 5+num_skips), (-1, (END_POINT - 1) * decimated_sfreq - 1)):
            self.assertTrue(check_raw_against_data(recording[i][0], int(offset), decimate=decimate))

    def test_EpochRecordingSkipGet(self):
        to_skip = [0, 3]
        new_events = [(i, ev) for i, ev in enumerate(EVENTS) if i not in to_skip]
        recording = self.make_epoch_recording(skip_epochs=to_skip)

        for modified_index, (true_index, (sample, label)) in enumerate(new_events):
            x, y = recording[modified_index]
            self.assertTrue(check_epoch_against_data(x, true_index))
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

    def test_ToNumpy(self):
        # Make sure it works with transforms
        self.dataset.add_transform(ZScore())
        np_data = self.dataset.to_numpy(num_workers=0)

        # data + 4 ids + target
        self.assertEqual(len(np_data), 1 + 4 + 1)
        self.assertTrue(np.all([arr.shape[0] == len(self.dataset) for arr in np_data]))

    def test_DumpDataset(self):
        directory = Path('./dumped-ds/')

        with self.subTest("Save to disk"):
            self.dataset.dump_dataset(directory, chunksize=10)
            self.assertTrue(directory.exists())
            self.assertEqual(len([_ for _ in directory.iterdir()]), round(len(self.dataset) / 10) + 1)

        dumped = DumpedDataset(directory)

        with self.subTest("Re-loaded stats"):
            self.assertEqual(len(dumped), len(self.dataset))
            self.assertEqual(dumped.sfreq, self.dataset.sfreq)
            # self.assertTrue(np.all(dumped.channels, self.dataset.channels))
            self.assertEqual(dumped.sequence_length, self.dataset.sequence_length)

        def check_data():
            for i in range(len(dumped)):
                with self.subTest(ds_idx=i):
                    orig_x = self.dataset[i]
                    dump_x = dumped[i]
                    self.assertEqual(len(dump_x), len(orig_x))
                    for j in range(len(dump_x)):
                        with self.subTest(x_idx=j):
                            self.assertTrue(torch.allclose(dump_x[j], orig_x[j]))

        with self.subTest("Verify Loadable"):
            check_data()

        with self.subTest("Verify cache"):
            check_data()

    def test_DeviationSpanRejection(self):
        # Less people to speed it up
        raw_dataset = create_dummy_dataset(epoched=False, num_thinkers=2)
        bad_raw = create_dummy_raw()
        bad_spans = [(1, 3), (7, 8)]
        for bad_span in bad_spans:
            # First 4 are "EEG" channels
            bad_raw._data[:4, bad_span[0]*SFREQ:bad_span[1]*SFREQ] = 0
        bad_session = create_dummy_session(epoched=False, raw=bad_raw, session_id='bad', stride=SFREQ)

        raw_dataset.thinkers['p1'] + bad_session
        raw_dataset.add_transform(MappingDeep1010(raw_dataset, return_mask=True))

        save_file = 'exclude_test.yml'

        rejector = SingleStatisticSpanRejection(raw_dataset, mask_ind=1)
        rejector.collect_statistic()
        self.assertEqual(0, len(rejector.rejected_stats))

        rejector.deviation_threshold_rejection()
        excluded = rejector.get_configuratron_exclusions(save_to_file=save_file)
        for i, bad_span in enumerate(bad_spans):
            with self.subTest(bad_span=bad_span):
                self.assertTupleEqual(bad_span, tuple(excluded['p1']['bad'][i]))
        self.assertTrue(os.path.exists(save_file))
        os.remove(save_file)


if __name__ == '__main__':
    unittest.main()
