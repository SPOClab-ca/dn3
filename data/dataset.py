import mne
import torch
import numpy as np

from transform.preprocess import Preprocessor
from transform.transform import BaseTransform
from .utils import min_max_normalize as _min_max_normalize

from abc import ABC
from collections import OrderedDict
from collections.abc import Iterable
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import ConcatDataset


class DNPTDataset(TorchDataset):
    """
    Base class for all DNPT objects that can be used as pytorch-compatible datasets. Ensuring that DNPT extension is
    conformed to.
    """
    def __init__(self):
        self._transforms = list()

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError

    def add_transform(self, transform):
        """
        Add a transformation that is applied to every fetched item in the dataset
        Parameters
        ----------
        transform : BaseTransform
                    For each item retrieved by __getitem__, transform is called to modify that item.
        Returns
        -------
        The number of transforms currently associated with this dataset
        """
        if isinstance(transform, BaseTransform):
            self._transforms.append(transform)
        return len(self._transforms)

    def clear_transforms(self):
        self._transforms = list()

    def preprocess(self, preprocessor: Preprocessor, apply_transform=True):
        raise NotImplementedError()


class _Recording(DNPTDataset, ABC):
    """
    Abstract base class for any supported recording
    """
    def __init__(self, info, session_id, person_id):
        super().__init__()
        self.info = info
        self.sfreq = info['sfreq']
        assert self.sfreq is not None
        self.session_id = session_id
        self.person_id = person_id


class RawTorchRecording(_Recording):
    """
    Interface for bridging mne Raw instances as PyTorch compatible "Dataset".

    Parameters
    ----------
    raw : mne.io.Raw
          Raw data, data does not need to be preloaded.
    session_id : (int, str, optional)
          A unique (with respect to a thinker within an eventual dataset) identifier for the current recording
          session. If not specified, defaults to '0'.
    person_id : (int, str, optional)
          A unique (with respect to an eventual dataset) identifier for the particular person being recorded.
    sample_len : int
          Number of samples to be loaded each index.
    tlen : float
          Alternative to sample_len specified in seconds, overrides `sample_len`.
    stride : int
          The number of samples to skip between each starting offset of loaded samples.
    """

    def __init__(self, raw: mne.io.Raw, session_id=0, person_id=0, sample_len=512, tlen=None, stride=1, **kwargs):
        """
        Interface for bridging mne Raw instances as PyTorch compatible "Dataset".

        Parameters
        ----------
        raw : mne.io.Raw
              Raw data, data does not need to be preloaded.
        session_id : (int, str, optional)
              A unique (with respect to a thinker within an eventual dataset) identifier for the current recording
              session. If not specified, defaults to '0'.
        person_id : (int, str, optional)
              A unique (with respect to an eventual dataset) identifier for the particular person being recorded.
        sample_len : int
              Number of samples to be loaded each index.
        tlen : float
              Alternative to sample_len specified in seconds, overrides `sample_len`.
        stride : int
              The number of samples to skip between each starting offset of loaded samples.
        """
        super().__init__(raw.info, session_id, person_id)
        self.raw = raw
        self.stride = stride
        self.max = kwargs.get('max', None)
        self.min = kwargs.get('min', 0)
        self.normalizer = kwargs.get('normalize', _min_max_normalize)
        self.sample_len = sample_len if tlen is None else int(self.sfreq * tlen)

    def __getitem__(self, index):
        index *= self.stride

        # Get the actual signals
        x = self.raw.get_data(self.picks, start=index, stop=index+self.sample_len)
        scale = 1 if self.max is None else (x.max() - x.min()) / (self.max - self.min)
        if scale > 1 or np.isnan(scale):
            print('Warning: scale exeeding 1')
        x = _min_max_normalize(torch.from_numpy(x))
        if torch.any(x > 1):
            print("Somehow larger than 1.. {}, index {}".format(self.raw.filenames[0], index))
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print("Nan found: raw {}, index {}".format(self.raw.filenames[0], index))
            print("Replacing with appropriate random values for now...")
            x = torch.rand_like(x)

        for t in self._transforms:
            x = t(x)

        return x

    def __len__(self):
        return (self.raw.n_times - self.seq_len) // self.stride

    def preprocess(self, preprocessor: Preprocessor, apply_transform=True):
        self.raw = preprocessor(recording=self)
        if apply_transform:
            self.add_transform(preprocessor.get_transform())


class EpochTorchRecording(_Recording):

    def __init__(self, epochs: mne.Epochs, session_id, person_id, force_label=None, cached=False):
        super().__init__(epochs, session_id, person_id)
        self.epochs = epochs
        self._t_len = epochs.tmax - epochs.tmin
        self._cache = [None for _ in range(len(epochs.events))] if cached else None
        self.force_label = force_label if force_label is None else torch.tensor(force_label)

    @property
    def channels(self):
        if self.picks is None:
            return len(self.epochs.ch_names)
        else:
            return len(self.picks)

    def __getitem__(self, index):
        ep = self.epochs[index]

        if self._cache is None or self._cache[index] is None:
            x = ep.get_data()
            if len(x.shape) != 3 or 0 in x.shape:
                print("I don't know why: {} index{}/{}".format(self.epochs.filename, index, len(self)))
                print(self.epochs.info['description'])
                print("Using trial {} in place for now...".format(index-1))
                return self.__getitem__(index - 1)
            x = torch.from_numpy(self.normalizer(x).astype('float32')).squeeze(0)
            if self._cache is not None:
                self._cache[index] = x
        else:
            x = self._cache[index]

        y = torch.from_numpy(ep.events[..., -1]).long() if self.force_label is None else self.force_label

        for t in self._transforms:
            x = t(x)

        return x, y

    def __len__(self):
        events = self.epochs.events[:, 0].tolist()
        return len(events)

    def preprocess(self, preprocessor: Preprocessor, apply_transform=True):
        self.epochs = preprocessor(recording=self)
        if apply_transform:
            self.add_transform(preprocessor.get_transform())


class Thinker(DNPTDataset):
    """
    Collects multiple recordings of the same person, intended to be of the same task, at different times or conditions.
    """

    def __init__(self, sessions, person_id="auto"):
        """
        Collects multiple recordings of the same person, intended to be of the same task, at different times or
        conditions.
        Parameters
        ----------
        sessions : (iterable, dict)
                   Either a sequence of recordings, or a mapping of session_ids to recordings. If the former, the
                   recording's session_id is preserved. If the
        person_id : (int, str)
                   Label to be used for the thinker. If set to "auto" (default), will automatically pick the person_id
                   using the most common person_id in the recordings.
        """
        super().__init__()
        if isinstance(sessions, Iterable):
            self.sessions = OrderedDict()
            for r in sessions:
                self.__add__(r)
        elif isinstance(sessions, dict):
            self.sessions = OrderedDict(sessions)
        else:
            raise TypeError("Recordings must be iterable or already processed dict.")
        if person_id == 'auto':
            ids = [r.person_id for sess in self.sessions.values() for r in sess]
            person_id = max(set(ids), key=ids.count)
        self.person_id = person_id

        for recording in [r for sess in self.sessions.values() for r in sess]:
            recording.person_id = person_id

    def __add__(self, recording):
        assert isinstance(recording, _Recording)
        if recording.session_id in self.sessions.keys():
            self.sessions[recording.session_id] += [recording]
        else:
            self.sessions[recording.session_id] = [recording]

    def __getitem__(self, item):
        pass

    def __len__(self):
        return sum(len(s) for s in self.sessions)

    def preprocess(self, preprocessor: Preprocessor, apply_transform=True):
        pass


class Dataset(DNPTDataset):
    """
    Collects thinkers, each of which may collect multiple recording sessions of the same tasks, into a dataset with
    (largely) consistent:
      - hardware:
        - channel number/labels
        - sampling frequency
      - annotation paradigm:
        - consistent event types
    """
    def __init__(self, thinkers, dataset_id=None, task_id=None, return_person_id=False, return_session_id=False,
                 return_dataset_id=False, return_task_id=False):
        """
        Collects recordings from multiple people, intended to be of the same task, at different times or
        conditions.
        Parameters
        ----------
        thinkers : (iterable, dict)
                   Either a sequence of `Thinker`, or a mapping of person_id to `Thinker`. If the latter, id's are
                   overwritten by the mapping id's.
        dataset_id : (int, str)
                     An identifier associated with data from the entire dataset
        task_id : (int, str)
                  An identifier associated with data from the entire dataset, and potentially others of the same task
        return_person_id :
                 Whether to
        """
        super().__init__()
        if isinstance(thinkers, Iterable):
            self.thinkers = dict()
            for t in thinkers:
                self.__add__(t)

        elif isinstance(thinkers, dict):
            self.thinkers = thinkers
        else:
            raise TypeError("Thinkers must be iterable or already processed dict.")
        self.thinkers = thinkers

    def __add__(self, thinker):
        assert isinstance(thinker, Thinker)
        if thinker.person_id in self.thinkers.keys():
            self.thinkers[thinker.person_id] += thinker
        else:
            self.thinkers[thinker.person_id] = thinker

    @property
    def sfreq(self):
        raise NotImplementedError()

    @property
    def channels(self):
        raise NotImplementedError()

    def get_thinkers(self):
        pass

    def get_sessions(self):
        pass

    def __len__(self):
        return sum(len(r) for r in self.recordings)

    def thinker_specific(self, training_sessions=None, validation_sessions=None, testing_sessions=None,
                         split=(0.5, 0.25, 0.25)):
        pass

    def loso(self, validation_id=None, test_id=None):
        pass

    def lmso(self, folds=10, split=None):
        pass

    def apply_preprocessor(self, apply_transform=True):
        pass

    def add_transform(self, transform):
        pass

    def clear_transforms(self):
        pass
