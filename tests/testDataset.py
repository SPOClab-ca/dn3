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


class DummyCases(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
