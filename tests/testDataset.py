import unittest
from data.utils import min_max_normalize

from data.dataset import *

_START_POINT = 0
_END_POINT = 10
_SFREQ = 1000
_EVENTS = ((2, 3), (60, 2), (500, 1), (700, 3), (1200, 2), (2000, 1))

_SAMPLE_LENGTH = 128

_TMIN = 0
_TLEN = 1.0

_THINKERS_IN_DATASETS = 20
_NUM_FOLDS = 5


def create_basic_data():
    sinx = np.sin(np.arange(_START_POINT, _END_POINT, 1 / _SFREQ) * 10).astype('float')
    cosx = np.cos(np.arange(_START_POINT, _END_POINT, 1 / _SFREQ) * 10).astype('float')
    events = np.zeros_like(sinx)
    for ev_sample, label in _EVENTS:
        events[ev_sample] = label
    return np.array([sinx, cosx, events])


def create_dummy_raw():
    """
    Creates a Raw instance from `create_basic_data`
    Returns:
    -------
    raw : mne.io.Raw
    """
    data = create_basic_data()
    ch_names = [str(s) for s in range(2)] + ['STI 014']
    ch_types = ['eeg', 'eeg', 'stim']

    info = mne.create_info(ch_names=ch_names, sfreq=_SFREQ, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    return raw


class TestDatasetWIthDummyData(unittest.TestCase):

    def setUp(self):
        self.data = torch.from_numpy(create_basic_data())
        self.raw = create_dummy_raw()
        events = mne.find_events(self.raw)
        self.epochs = mne.Epochs(self.raw, events, tmin=_TMIN, tmax=_TLEN + _TMIN - 1 / _SFREQ, baseline=None)

    def make_raw_recording(self, **kwargs):
        return RawTorchRecording(self.raw, sample_len=_SAMPLE_LENGTH, **kwargs)

    def make_epoch_recording(self, **kwargs):
        return EpochTorchRecording(self.epochs, **kwargs)

    def check_raw_against_data(self, retrieved, index):
        return torch.allclose(retrieved, min_max_normalize(self.data[:2, index:index+_SAMPLE_LENGTH]).float())

    def check_epoch_against_data(self, retrieved, event_index):
        sample = _EVENTS[event_index][0]
        window = slice(int(sample - _TMIN * _SFREQ), int(sample + (_TLEN + _TMIN) * _SFREQ))
        return torch.allclose(retrieved, min_max_normalize(self.data[:, window]).float())

    def test_RawRecordingCreate(self):
        recording = self.make_raw_recording()
        self.assertEqual(len(recording), _SFREQ * (_END_POINT - _START_POINT) - _SAMPLE_LENGTH)

    def test_RawRecordingGet(self):
        recording = self.make_raw_recording(picks=[0, 1])
        for i in (0, 10, -1):
            data_offset = list(range(len(recording)))[i]
            self.assertTrue(self.check_raw_against_data(recording[i], data_offset))

    def test_EpochRecordingCreate(self):
        recording = self.make_epoch_recording()
        self.assertEqual(len(recording), len(_EVENTS))
        
    def test_EpochRecordingGet(self):
        recording = self.make_epoch_recording()
        for i, (sample, label) in enumerate(_EVENTS):
            x, y = recording[i]
            self.assertTrue(self.check_epoch_against_data(x, i))
            self.assertEqual(torch.tensor(label), y)

    def test_MakeThinkers(self):
        raw_session = self.make_raw_recording()
        thinker = Thinker(dict(sess1=raw_session, sess2=raw_session), return_session_id=False)
        self.assertEqual(len(thinker), len(raw_session)*2)

    def test_ThinkersGet(self):
        epoch_session = self.make_epoch_recording()
        thinker = Thinker(dict(sess1=epoch_session, sess2=epoch_session), return_session_id=True)
        for i, (x, sess_id, y) in enumerate(thinker):
            if i < len(_EVENTS):
                self.assertEqual(sess_id, 0)
            else:
                self.assertEqual(sess_id, 1)
            self.assertTrue(self.check_epoch_against_data(x, i % len(epoch_session)))
            self.assertEqual(torch.tensor(_EVENTS[i % len(epoch_session)][1]), y)

    def test_ThinkerSplitFractions(self):
        epoch_session = self.make_epoch_recording()
        thinker = Thinker(dict(sess1=epoch_session, sess2=epoch_session), return_session_id=True)
        training, validating, testing = thinker.split(test_frac=0.5, validation_frac=0.5)
        self.assertEqual(len(training), len(_EVENTS) // 2)
        self.assertEqual(len(validating), len(_EVENTS) // 2)
        self.assertEqual(len(testing), len(_EVENTS))

    def test_ThinkerSplitByID(self):
        epoch_session = self.make_epoch_recording()
        thinker = Thinker(dict(sess1=epoch_session, sess2=epoch_session), return_session_id=True)
        training, validating, testing = thinker.split(testing_sess_ids=['sess1'], validation_frac=0.25)
        self.assertEqual(len(training), 3 * len(_EVENTS) // 4)
        self.assertEqual(len(validating), len(_EVENTS) // 4 + int(len(_EVENTS) % 4 > 0))
        self.assertEqual(len(testing), len(_EVENTS))
        for _, (x, sess, y) in enumerate(testing):
            self.assertEqual(sess, 0)

    def make_dataset_duplicated_thinkers(self, **dsargs):
        epoch_session = self.make_epoch_recording()
        thinker = Thinker(dict(sess1=epoch_session.clone(), sess2=epoch_session.clone()), return_session_id=True)
        return Dataset({"p{}".format(i): thinker.clone() for i in range(20)}, **dsargs)

    def test_MakeDataset(self):
        dataset = self.make_dataset_duplicated_thinkers()
        self.assertEqual(len(dataset), len(_EVENTS) * 2 * _THINKERS_IN_DATASETS)

    def test_DatasetGet(self):
        dataset = self.make_dataset_duplicated_thinkers(dataset_id=0, task_id=1, return_person_id=True,
                                                        return_session_id=True)
        sess_count, person_count = -1, -1
        for i, (x, person_id, sess_id, y) in enumerate(dataset):
            if i % len(_EVENTS) == 0:
                sess_count = (sess_count + 1) % 2
            if i % (2 * len(_EVENTS)) == 0:
                person_count += 1
            self.assertEqual(sess_id, sess_count)
            self.assertEqual(person_id, person_count)
            self.assertTrue(self.check_epoch_against_data(x, i % len(_EVENTS)))
            self.assertEqual(torch.tensor(_EVENTS[i % len(_EVENTS)][1]), y)

    def test_DatasetLOSO(self):
        dataset = self.make_dataset_duplicated_thinkers(dataset_id=0, task_id=1, return_person_id=True,
                                                        return_session_id=True, return_dataset_id=True)
        people = dataset.get_thinkers()
        for i, (training, validation, testing) in enumerate(dataset.loso()):
            val, test = people[i-1], people[i]
            for p1, p2 in zip((people[j] for j in range(len(people)) if people[j] not in (val, test)),
                              training.get_thinkers()):
                self.assertEqual(p1, p2)
            self.assertEqual(val, validation.person_id)
            self.assertEqual(test, testing.person_id)

    def test_DatasetLOSOIds(self):
        dataset = self.make_dataset_duplicated_thinkers(dataset_id=0, task_id=1, return_person_id=True,
                                                        return_session_id=True, return_dataset_id=True)
        vids = list()
        people = dataset.get_thinkers()
        training_people = set(dataset.get_thinkers()).difference(['p1'])
        for i, (training, validation, testing) in enumerate(dataset.loso(test_person_id='p1')):
            for p1, p2 in zip((people[j] for j in range(len(people)) if people[j] not in (validation.person_id, 'p1')),
                              training.get_thinkers()):
                self.assertEqual(p1, p2)
            self.assertEqual('p1', testing.person_id)
            vids.append(validation.person_id)
        self.assertEqual(len(vids), len(training_people))
        self.assertEqual(training_people, set(vids))

    def test_DatasetLOSOSameValTest(self):
        dataset = self.make_dataset_duplicated_thinkers(dataset_id=0, task_id=1, return_person_id=True,
                                                        return_session_id=True, return_dataset_id=True)

        def check():
            for _ in dataset.loso(validation_person_id='p1', test_person_id='p1'):
                pass
        self.assertRaises(ValueError, check)

    def test_DatasetLMSO(self):
        dataset = self.make_dataset_duplicated_thinkers(dataset_id=0, task_id=1, return_person_id=True,
                                                        return_session_id=True, return_dataset_id=True)
        people = dataset.get_thinkers()
        for i, (training, validation, testing) in enumerate(dataset.lmso(folds=_NUM_FOLDS)):
            training_people = set(people).difference(testing.get_thinkers()).difference(
                validation.get_thinkers())
            self.assertSetEqual(set(training.get_thinkers()), training_people)
            self.assertEqual(len(set(testing.get_thinkers()).intersection(validation.get_thinkers())), 0)

    def test_DatasetLMSOTestIds(self):
        dataset = self.make_dataset_duplicated_thinkers(dataset_id=0, task_id=1, return_person_id=True,
                                                        return_session_id=True, return_dataset_id=True)
        people = dataset.get_thinkers()
        test_people = people[15:]
        for i, (training, validation, testing) in enumerate(dataset.lmso(folds=_NUM_FOLDS, test_splits=test_people)):
            training_people = set(people).difference(testing.get_thinkers()).difference(
                validation.get_thinkers())
            self.assertSetEqual(set(training.get_thinkers()), training_people)
            self.assertSetEqual(set(testing.get_thinkers()), set(test_people))
            self.assertEqual(len(set(testing.get_thinkers()).intersection(validation.get_thinkers())), 0)


if __name__ == '__main__':
    unittest.main()
