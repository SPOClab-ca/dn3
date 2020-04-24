import mne
import torch
import numpy as np

from transform.preprocess import Preprocessor
from transform.transform import BaseTransform
from .utils import min_max_normalize as _min_max_normalize

from abc import ABC
from collections import OrderedDict
from collections.abc import Iterable, Sized
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
        """
        if isinstance(transform, BaseTransform):
            self._transforms.append(transform)

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


class Thinker(DNPTDataset, ConcatDataset):
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
        DNPTDataset.__init__(self)
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

        ConcatDataset.__init__(self, self.sessions)

    def __add__(self, recording):
        assert isinstance(recording, _Recording)
        if recording.session_id in self.sessions.keys():
            self.sessions[recording.session_id] += recording
        else:
            self.sessions[recording.session_id] = recording

    def __getitem__(self, item):
        ConcatDataset.__getitem__(self, item)

    def __len__(self):
        ConcatDataset.__len__(self)

    def split(self, training_sess_ids=None, validation_sess_ids=None, testing_sess_ids=None, split=(0.5, 0.25, 0.25)):
        pass

    def preprocess(self, preprocessor: Preprocessor, apply_transform=True):
        pass

    def clear_transforms(self):
        for s in self.sessions.values():
            s.clear_transforms()

    def add_transform(self, transform):
        for s in self.sessions.values():
            s.add_transform(transform)


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
        Optionally, can specify whether to return person, session, dataset and task labels. Person and session ids will
        be converted to an enumerated set of integer ids, rather than those provided during creation of those datasets
        in order to make a minimal set of labels. e.g. if there are 3 thinkers, {A01, A02, and A05}, specifying
        `return_person_id` will return an additional tensor with 0 for A01, 1 for A02 and 2 for A05 respectively. To
        recover any original identifier, get_thinkers() returns a list of the original thinker ids such that the
        enumerated offset recovers the original identity. Extending the example above:
        ``self.get_thinkers()[1] == "A02"``

        .. warning:: The enumerated ids above are only ever used in the construction of model input tensors,
                     otherwise, anywhere where ids are required as API, the *human readable* version is uesd
                     (e.g. in our example above A02)
        Parameters
        ----------
        thinkers : (iterable, dict)
                   Either a sequence of `Thinker`, or a mapping of person_id to `Thinker`. If the latter, id's are
                   overwritten by the mapping id's.
        dataset_id : int
                     An identifier associated with data from the entire dataset. Unlike person and sessions, this should
                     simply be an integer for the sake of returning labels that can functionally be used for learning.
        task_id : int
                  An identifier associated with data from the entire dataset, and potentially others of the same task.
                  Like dataset_idm this should simply be an integer.
        return_person_id : bool
                           Whether to return (enumerated - see above) person_ids with the data itself.
        return_session_id : bool
                           Whether to return (enumerated - see above) session_ids with the data itself.
        return_dataset_id : bool
                           Whether to return the dataset_id with the data itself.
        return_task_id : bool
                           Whether to return the dataset_id with the data itself.
        """
        super().__init__()
        if isinstance(thinkers, Iterable):
            self._thinkers = OrderedDict()
            for t in thinkers:
                self.__add__(t)

        elif isinstance(thinkers, dict):
            self._thinkers = OrderedDict(thinkers)
        else:
            raise TypeError("Thinkers must be iterable or already processed dict.")
        self._thinkers = thinkers
        self._sessions = list()

        self.dataset_id = dataset_id
        self.task_id = task_id

        self.return_person_id = return_person_id
        self.return_session_id = return_session_id
        self.return_task_id = return_task_id
        self.return_dataset_id = return_dataset_id

    def __add__(self, thinker):
        assert isinstance(thinker, Thinker)
        if thinker.person_id in self._thinkers.keys():
            self._thinkers[thinker.person_id] += thinker
        else:
            self._thinkers[thinker.person_id] = thinker

    @property
    def sfreq(self):
        raise NotImplementedError()

    @property
    def channels(self):
        raise NotImplementedError()

    def get_thinkers(self):
        return list(self._thinkers.keys())

    def get_sessions(self):
        return self._sessions

    def __len__(self):
        return sum(len(r) for r in self._sessions)

    def _make_like_me(self, people: list):
        Dataset({p: self._thinkers[p] for p in people}, self.dataset_id, self.task_id, self.return_person_id,
                self.return_session_id, self.return_dataset_id, self.return_task_id)

    def _generate_splits(self, validation, testing):
        for val, test in zip(validation, testing):
            training = list(self._thinkers.keys())
            for v in val:
                training.remove(v)
            for t in test:
                training.remove(t)
            training = self._make_like_me(training)
            validating = self._make_like_me(val) if len(val) > 1 else self._thinkers[val[0]]
            testing = self._make_like_me(test) if len(test) > 1 else self._thinkers[test[0]]
            yield training, validating, testing

    def loso(self, validation_person_id=None, test_person_id=None):
        """
        This *generates* a "Leave-one-subject-out" (LOSO) split. Tests each person one-by-one, and validates on the
        previous (the first is validated with the last).
        Parameters
        ----------
        validation_person_id : (int, str, list, optional)
                               If specified, and corresponds to one of the person_ids in this dataset, the loso cross
                               validation will consistently generate this thinker as `validation`. If *list*, must
                               be the same length as `test_person_id`, say a length N. If so, will yield N
                               each in sequence, and use remainder for test.
        test_person_id : (int, str, list, optional)
                         Same as `validation_person_id`, but for testing. However, testing may be an list when
                         validation is a single value. Thus if testing is N ids, will yield N values, with a consistent
                         single validation person.

        Yields
        -------
        training : Dataset
                   Another dataset that represents the training set
        validation : Thinker
                     The validation thinker
        test : Thinker
               The test thinker
        """
        if isinstance(validation_person_id, list):
            if not isinstance(test_person_id, list) or len(test_person_id) != len(validation_person_id):
                raise TypeError("Test ids must be same length iterable as validation ids.")
            yield from self._generate_splits([[v] for v in validation_person_id], [[v] for v in test_person_id])

        if isinstance(test_person_id, list):
            test_person_id = [[t] for t in test_person_id]
        elif test_person_id is not None:
            assert isinstance(validation_person_id, (int, str))
            test_person_id = [[test_person_id] for _ in range(len(self.get_thinkers())-1)]
        else:
            test_person_id = [[t] for t in self.get_thinkers()]

        if validation_person_id is not None:
            validation_person_id = [[validation_person_id] for _ in range(len(test_person_id))]
        else:
            validation_person_id = [self.get_thinkers()[self.get_thinkers().index(t[0])-1] for t in test_person_id]

        yield from self._generate_splits(validation_person_id, test_person_id)

    def lmso(self, folds=10, splits=None):
        """
        This *generates* a "Leave-multiple-subject-out" (LMSO) split. In other words X-fold cross-validation, with
        boundaries enforced at thinkers (each person's data is not split into different folds).
        Parameters
        ----------
        folds : int
                If this is specified and `splits` is None, will split the subjects into this many folds, and then use
                each fold as a test set in turn (and the previous fold - starting with the last - as validation).
        splits : list
                If this is not None, folds is ignored. Instead, this should be a list of tuples/lists that represent
                each fold. Each element of the list then has two sub-lists, first the ids for validation and second
                the ids for testing.

        Yields
        -------
        training : Dataset
                   Another dataset that represents the training set
        validation : Dataset
                     The validation people as a dataset
        test : Thinker
               The test people as a dataset
        """
        if splits is None:
            folds = [list(x) for x in np.array_split(self.get_thinkers(), folds)]
            splits = [(folds[i-1], folds[i]) for i in range(len(folds))]
        yield from self._generate_splits(*zip(*splits))

    def apply_preprocessor(self, preprocessor: Preprocessor, apply_transform=True, thinkers=None, sessions=None):
        for thinker in self._thinkers.values():
            thinker.apply_preprocessor(preprocessor)
        if apply_transform:
            self.add_transform(preprocessor.get_transform())

    def add_transform(self, transform, thinkers=None, sessions=None):
        for thinker in self._thinkers.values():
            thinker.add_transform(transform)

    def clear_transforms(self):
        for thinker in self._thinkers.values():
            thinker.clear_transforms()
