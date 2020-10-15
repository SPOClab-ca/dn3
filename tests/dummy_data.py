from dn3.data.dataset import *
from dn3.transforms.channels import DEEP_1010_CHS_LISTING
from dn3.utils import min_max_normalize


START_POINT = 0
END_POINT = 10
SFREQ = 256
EVENTS = ((2, 3), (60, 2), (500, 1), (700, 3), (1200, 2), (2000, 1))

TMIN = 0
TLEN = 1.0

NUM_SESSIONS_PER_THINKER = 2
THINKERS_IN_DATASETS = 20
NUM_FOLDS = 5

# Creation Functions
# ------------------


def create_basic_data():
    sinx = 0.5 * np.sin(np.arange(START_POINT, END_POINT, 1 / SFREQ) * 10).astype('float')
    cosx = 0.5 * np.cos(np.arange(START_POINT, END_POINT, 1 / SFREQ) * 10).astype('float')
    events = np.zeros_like(sinx)
    for ev_sample, label in EVENTS:
        events[ev_sample] = label
    return np.array([*([sinx, cosx] * 5), events])


def create_dummy_raw():
    """
    Creates a Raw instance from `create_basic_data`
    Returns:
    -------
    raw : mne.io.Raw
    """
    data = create_basic_data()
    ch_names = DEEP_1010_CHS_LISTING[:8] + [' V-EOG L', 'V-EOG-R'] + ['STI 014']
    ch_types = (['eeg'] * 8) + (['eog'] * 2) + ['stim']

    info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    return raw


def create_dummy_session(epoched=True, raw=None, **kwargs):
    raw = create_dummy_raw() if raw is None else raw
    if epoched:
        events = mne.find_events(raw)
        epochs = mne.Epochs(raw, events, tmin=TMIN, tmax=TLEN + TMIN - 1 / SFREQ, baseline=None)
        return EpochTorchRecording(epochs, **kwargs)
    return RawTorchRecording(raw, TLEN, **kwargs)


def create_dummy_thinker(epoched=True, sessions_per_thinker=2, sess_args=dict(), **kwargs):
    session = create_dummy_session(epoched=epoched, **sess_args)
    return Thinker({'sess{}'.format(i): session.clone() for i in range(1, sessions_per_thinker + 1)},
                   return_session_id=True, **kwargs)


def create_dummy_dataset(epoched=True, sessions_per_thinker=2, num_thinkers=THINKERS_IN_DATASETS,
                         sess_args=dict(), thinker_args=dict(), **dataset_args):
    thinker = create_dummy_thinker(epoched=epoched, sessions_per_thinker=sessions_per_thinker, sess_args=sess_args,
                                   **thinker_args)
    info = DatasetInfo('Test dataset', data_max=1.0, data_min=-1.0, targets=3)
    dataset_args.setdefault('dataset_info', info)
    return Dataset({"p{}".format(i): thinker.clone() for i in range(num_thinkers)}, **dataset_args)


# Check functions
# ---------------

def check_raw_against_data(retrieved, index, normalizer=lambda x: x, decimate=1):
    data = torch.from_numpy(create_basic_data())
    sample_len = int(TLEN * (SFREQ // decimate))
    d = data[:2, ::decimate]
    return torch.allclose(retrieved, normalizer(d[:, index:index+sample_len].float()))


def retrieve_underlying_dummy_data(event_index):
    data = torch.from_numpy(create_basic_data())
    sample = EVENTS[event_index][0]
    window = slice(int(sample - TMIN * SFREQ), int(sample + (TLEN + TMIN) * SFREQ))
    return data[:, window].float()


def check_epoch_against_data(retrieved, event_index, normalizer=lambda x: x):
    return torch.allclose(retrieved, normalizer(retrieve_underlying_dummy_data(event_index)))
