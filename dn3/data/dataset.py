import mne
import torch
import copy
import bisect
import numpy as np

from dn3.transforms.preprocessors import Preprocessor
from dn3.transforms.basic import BaseTransform
from dn3.utils import min_max_normalize as _min_max_normalize
from dn3.utils import rand_split, unfurl

from abc import ABC
from collections import OrderedDict
from collections.abc import Iterable
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import ConcatDataset


class DN3ataset(TorchDataset):
    """
    Base class for that specifies the interface for DN3 datasets.
    """
    def __init__(self):
        self._transforms = list()

    def __getitem__(self, item):
        raise NotImplementedError

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __len__(self):
        raise NotImplementedError

    @property
    def sfreq(self):
        raise NotImplementedError

    @property
    def channels(self):
        raise NotImplementedError

    @property
    def sequence_length(self):
        raise NotImplementedError

    def clone(self):
        """
        A copy of this object to allow the repetition of recordings, thinkers, etc. that load data from
        the same memory/files but have their own tracking of ids.

        Returns
        -------
        cloned : DN3ataset
                 New copy of this object.
        """
        return copy.deepcopy(self)

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

    def _execute_transforms(self, *x):
        for transform in self._transforms:
            assert isinstance(transform, BaseTransform)
            if transform.only_trial_data:
                new_x = transform(x[0])
                if isinstance(new_x, (list, tuple)):
                    x = (*new_x, *x[1:])
                else:
                    x = (new_x, *x[1:])
            else:
                x = transform(*x)
        return x

    def clear_transforms(self):
        """
        Remove all added transforms from dataset.
        """
        self._transforms = list()

    def preprocess(self, preprocessor: Preprocessor, apply_transform=True):
        """
        Applies a preprocessor to the dataset

        Parameters
        ----------
        preprocessor : Preprocessor
                       A preprocessor to be applied
        apply_transform : bool
                          Whether to apply the transform to this dataset (and all members e.g thinkers or sessions)
                          after preprocessing them. Alternatively, the preprocessor is returned for manual application
                          of its transform through :meth:`Preprocessor.get_transform()`

        Returns
        ---------
        preprocessor : Preprocessor
                       The preprocessor after application to all relevant thinkers
        """
        raise NotImplementedError


class _Recording(DN3ataset, ABC):
    """
    Abstract base class for any supported recording
    """
    def __init__(self, info, session_id, person_id, tlen):
        super().__init__()
        self.info = info
        self._recording_channels = [ch['ch_name'] for ch in info['chs']]
        self._recording_sfreq = info['sfreq']
        self._recording_len = int(self._recording_sfreq * tlen)
        assert self._recording_sfreq is not None
        self.session_id = session_id
        self.person_id = person_id

    def get_all(self):
        all_recordings = [x for x in self]
        return [torch.stack(t) for t in zip(*all_recordings)]

    @property
    def sfreq(self):
        sfreq = self._recording_sfreq
        for xform in self._transforms:
            sfreq = xform.new_sfreq(sfreq)
        return sfreq

    @property
    def channels(self):
        channels = np.array(self._recording_channels)
        for xform in self._transforms:
            channels = xform.new_channels(channels)
        return channels

    @property
    def sequence_length(self):
        sequence_length = self._recording_len
        for xform in self._transforms:
            sequence_length = xform.new_sequence_length(sequence_length)
        return sequence_length


class RawTorchRecording(_Recording):
    """
    Interface for bridging mne Raw instances as PyTorch compatible "Dataset".

    Parameters
    ----------
    raw : mne.io.Raw
          Raw data, data does not need to be preloaded.
    tlen : float
          Length of recording specified in seconds.
    session_id : (int, str, optional)
          A unique (with respect to a thinker within an eventual dataset) identifier for the current recording
          session. If not specified, defaults to '0'.
    person_id : (int, str, optional)
          A unique (with respect to an eventual dataset) identifier for the particular person being recorded.
    stride : int
          The number of samples to skip between each starting offset of loaded samples.
    """

    def __init__(self, raw: mne.io.Raw, tlen, session_id=0, person_id=0, stride=1, picks=None,
                 normalizer=_min_max_normalize, **kwargs):

        """
        Interface for bridging mne Raw instances as PyTorch compatible "Dataset".

        Parameters
        ----------
        raw : mne.io.Raw
              Raw data, data does not need to be preloaded.
        tlen : float
              Length of recording specified in seconds.
        session_id : (int, str, optional)
              A unique (with respect to a thinker within an eventual dataset) identifier for the current recording
              session. If not specified, defaults to '0'.
        person_id : (int, str, optional)
              A unique (with respect to an eventual dataset) identifier for the particular person being recorded.
        stride : int
              The number of samples to skip between each starting offset of loaded samples.
        """
        super().__init__(raw.info, session_id, person_id, tlen)
        self.raw = raw
        self.stride = stride
        self.max = kwargs.get('max', None)
        self.min = kwargs.get('min', 0)
        self.normalizer = normalizer
        self.picks = picks
        self.__dict__.update(kwargs)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        index *= self.stride

        # Get the actual signals
        x = self.raw.get_data(self.picks, start=index, stop=index+self.sequence_length)
        scale = 1 if self.max is None else (x.max() - x.min()) / (self.max - self.min)
        if scale > 1 or np.isnan(scale):
            print('Warning: scale exeeding 1')
        x = self.normalizer(torch.from_numpy(x)).float()
        if torch.any(x > 1):
            print("Somehow larger than 1.. {}, index {}".format(self.raw.filenames[0], index))
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print("Nan found: raw {}, index {}".format(self.raw.filenames[0], index))
            print("Replacing with appropriate random values for now...")
            x = torch.rand_like(x)

        return self._execute_transforms(x)

    def __len__(self):
        return (self.raw.n_times - self.sequence_length) // self.stride

    def preprocess(self, preprocessor: Preprocessor, apply_transform=True):
        self.raw = preprocessor(recording=self)
        if apply_transform:
            self.add_transform(preprocessor.get_transform())


class EpochTorchRecording(_Recording):
    def __init__(self, epochs: mne.Epochs, session_id=0, person_id=0, force_label=None, cached=False,
                 normalizer=_min_max_normalize, picks=None, event_mapping=None):
        """
        Wraps :any:`Epoch` objects so that they conform to the :any:`Recording` structure.
        Parameters
        ----------
        epochs
        session_id
        person_id
        force_label
        cached
        normalizer
        picks
        event_mapping : dict, Optional
                        Mapping of human-readable names to numeric codes used by `epochs`.
        """
        super().__init__(epochs.info, session_id, person_id, epochs.tmax - epochs.tmin + 1 / epochs.info['sfreq'])
        self.epochs = epochs
        self._cache = [None for _ in range(len(epochs.events))] if cached else None
        self.force_label = force_label if force_label is None else torch.tensor(force_label)
        self.normalizer = normalizer
        self.picks = picks
        if event_mapping is None:
            # mne parses this for us
            event_mapping = epochs.event_id
        reverse_mapping = {v: k for k, v in event_mapping.items()}
        self.epoch_codes_to_class_labels = {v: i for i, v in enumerate(sorted(reverse_mapping.keys()))}

    def __getitem__(self, index):
        ep = self.epochs[index]

        if self._cache is None or self._cache[index] is None:
            # TODO Could have a speedup if not using ep, but items, but would need to drop bads?
            x = ep.get_data(picks=self.picks)
            if len(x.shape) != 3 or 0 in x.shape:
                print("I don't know why: {} index{}/{}".format(self.epochs.filename, index, len(self)))
                print(self.epochs.info['description'])
                print("Using trial {} in place for now...".format(index-1))
                return self.__getitem__(index - 1)
            x = self.normalizer(torch.from_numpy(x.squeeze(0))).float()
            if self._cache is not None:
                self._cache[index] = x
        else:
            x = self._cache[index]

        y = torch.tensor(self.epoch_codes_to_class_labels[ep.events[0, -1]]).squeeze().long() if \
            self.force_label is None else self.force_label

        return self._execute_transforms(x, y)

    def __len__(self):
        return len(self.epochs.events)

    def preprocess(self, preprocessor: Preprocessor, apply_transform=True):
        self.epochs = preprocessor(recording=self)
        if apply_transform:
            self.add_transform(preprocessor.get_transform())

    def event_mapping(self):
        """
        Maps the labels returned by this to the events as recorded in the original annotations or stim channel.

        Returns
        -------
        mapping : dict
                  Keys are the class labels used by this object, values are the original event signifier.
        """
        return self.event_labels_to_epoch_codes


class Thinker(DN3ataset, ConcatDataset):
    """
    Collects multiple recordings of the same person, intended to be of the same task, at different times or conditions.
    """

    def __init__(self, sessions, person_id="auto", return_session_id=False, propagate_kwargs=False):
        """
        Collects multiple recordings of the same person, intended to be of the same task, at different times or
        conditions.

        Parameters
        ----------
        sessions : Iterable, dict
                   Either a sequence of recordings, or a mapping of session_ids to recordings. If the former, the
                   recording's session_id is preserved. If the
        person_id : int, str
                    Label to be used for the thinker. If set to "auto" (default), will automatically pick the person_id
                    using the most common person_id in the recordings.
        return_session_id : bool
                           Whether to return (enumerated - see `Dataset`) session_ids with the data itself. Overridden
                           by `propagate_kwargs`, with key `session_id`
        propagate_kwargs : bool
                           If True, items are returned additional tensors generated by transforms, and session_id as
        """
        DN3ataset.__init__(self)
        if not isinstance(sessions, dict) and isinstance(sessions, Iterable):
            self.sessions = OrderedDict()
            for r in sessions:
                self.__add__(r)
        elif isinstance(sessions, dict):
            self.sessions = OrderedDict(sessions)
        else:
            raise TypeError("Recordings must be iterable or already processed dict.")
        if person_id == 'auto':
            ids = [sess.person_id for sess in self.sessions.values()]
            person_id = max(set(ids), key=ids.count)
        self.person_id = person_id

        for sess in self.sessions.values():
            sess.person_id = person_id

        self._reset_dataset()
        self.return_session_id = return_session_id

    def _reset_dataset(self):
        for _id in self.sessions:
            self.sessions[_id].session_id = _id
        ConcatDataset.__init__(self, self.sessions.values())

    @property
    def sfreq(self):
        sfreq = set(self.sessions[s].sfreq for s in self.sessions)
        if len(sfreq) > 1:
            print("Warning: Multiple sampling frequency values found. Over/re-sampling may be necessary.")
            return unfurl(sfreq)
        sfreq = sfreq.pop()
        for xform in self._transforms:
            sfreq = xform.new_sfreq(sfreq)
        return sfreq

    @property
    def channels(self):
        channels = set(tuple(self.sessions[s].channels) for s in self.sessions)
        if len(channels) > 1:
            print("Warning: Multiple channel sets found. A consistent mapping like Deep1010 may be prudent.")
            return unfurl(channels)
        channels = channels.pop()
        for xform in self._transforms:
            channels = xform.new_sfreq(channels)
        return channels

    @property
    def sequence_length(self):
        sequence_length = set(self.sessions[s].sequence_length for s in self.sessions)
        if len(sequence_length) > 1:
            print("Warning: Multiple sequence lengths found. A cropping transformation may be in order.")
            return unfurl(sequence_length)
        sequence_length = sequence_length.pop()
        for xform in self._transforms:
            sequence_length = xform.new_sfreq(sequence_length)
        return sequence_length

    def __add__(self, sessions):
        assert isinstance(sessions, (_Recording, Thinker))
        if isinstance(sessions, Thinker):
            if sessions.person_id != self.person_id:
                print("Person IDs don't match: adding {} to {}. Assuming latter...")
            sessions = sessions.sessions

        if sessions.session_id in self.sessions.keys():
            self.sessions[sessions.session_id] += sessions
        else:
            self.sessions[sessions.session_id] = sessions

        self._reset_dataset()

    def __getitem__(self, item, return_id=False):
        x = ConcatDataset.__getitem__(self, item)
        if self.return_session_id:
            session_idx = torch.tensor(bisect.bisect_right(self.cumulative_sizes, item))
            return self._execute_transforms(x[0], session_idx, *x[1:])
        return self._execute_transforms(*x)

    def __len__(self):
        return ConcatDataset.__len__(self)

    def _make_like_me(self, sessions: Iterable):
        if not isinstance(sessions, dict):
            sessions = {s: self.sessions[s] for s in sessions}
        like_me = Thinker(sessions, self.person_id, self.return_session_id)
        for x in self._transforms:
            like_me.add_transform(x)
        return like_me

    def split(self, training_sess_ids=None, validation_sess_ids=None, testing_sess_ids=None, test_frac=0.25,
              validation_frac=0.25):
        """
        Split the thinker's data into training, validation and testing sets.

        Parameters
        ----------
        test_frac : float
                    Proportion of the total data to use for testing, this is overridden by `testing_sess_ids`.
        validation_frac : float
                          Proportion of the data remaining - after removing test proportion/sessions - to use as
                          validation data. Likewise, `validation_sess_ids` overrides this value.
        training_sess_ids : : (Iterable, None)
                            The session ids to be explicitly used for training.
        validation_sess_ids : (Iterable, None)
                            The session ids to be explicitly used for validation.
        testing_sess_ids : (Iterable, None)
                           The session ids to be explicitly used for testing.

        Returns
        -------
        training : DN3ataset
                   The training dataset
        validation : DN3ataset
                   The validation dataset
        testing : DN3ataset
                   The testing dataset
        """
        training_sess_ids = set(training_sess_ids) if training_sess_ids is not None else set()
        validation_sess_ids = set(validation_sess_ids) if validation_sess_ids is not None else set()
        testing_sess_ids = set(testing_sess_ids) if testing_sess_ids is not None else set()
        duplicated_ids = training_sess_ids.intersection(validation_sess_ids).intersection(testing_sess_ids)
        if len(duplicated_ids) > 0:
            print("Ids duplicated across train/val/test split: {}".format(duplicated_ids))
        use_sessions = self.sessions.copy()
        training, validating, testing = (
            self._make_like_me({s_id: use_sessions.pop(s_id) for s_id in ids}) if len(ids) else None
            for ids in (training_sess_ids, validation_sess_ids, testing_sess_ids)
        )
        if training is not None and validating is not None and testing is not None:
            if len(use_sessions) > 0:
                print("Warning: sessions specified do not span all sessions. Skipping {} sessions.".format(
                    len(use_sessions)))
                return training, validating, testing

        remainder = self._make_like_me(use_sessions.keys())
        if testing is None:
            assert test_frac is not None and 0 < test_frac < 1
            remainder, testing = rand_split(remainder, frac=test_frac)
        if validating is None:
            assert validation_frac is not None and 0 <= test_frac < 1
            if validation_frac > 0:
                validating, remainder = rand_split(remainder, frac=validation_frac)

        training = remainder if training is None else training

        return training, validating, testing

    def preprocess(self, preprocessor: Preprocessor, apply_transform=True, sessions=None):
        """
        Applies a preprocessor to the dataset

        Parameters
        ----------
        preprocessor : Preprocessor
                       A preprocessor to be applied
        sessions : (None, Iterable)
                   If specified (default is None), the sessions to use for preprocessing calculation
        apply_transform : bool
                          Whether to apply the transform to this dataset (all sessions, not just those specified for
                          preprocessing) after preprocessing them. Exclusive application to select sessions can be
                          done using the return value and a separate call to `add_transform` with the same `sessions`
                          list.

        Returns
        ---------
        preprocessor : Preprocessor
                       The preprocessor after application to all relevant thinkers
        """
        sessions = list(self.sessions.values()) if sessions is None else sessions
        for session in sessions:
            session.preprocess(preprocessor)
        if apply_transform:
            self.add_transform(preprocessor.get_transform())
        return preprocessor

    def clear_transforms(self, deep_clear=False):
        self._transforms = list()
        if deep_clear:
            for s in self.sessions.values():
                s.clear_transforms()

    def add_transform(self, transform):
        self._transforms.append(transform)


class Dataset(DN3ataset, ConcatDataset):
    """
    Collects thinkers, each of which may collect multiple recording sessions of the same tasks, into a dataset with
    (largely) consistent:
      - hardware:
        - channel number/labels
        - sampling frequency
      - annotation paradigm:
        - consistent event types
    """
    def __init__(self, thinkers, dataset_id=None, task_id=None, return_session_id=False, return_person_id=False,
                 return_dataset_id=False, return_task_id=False, dataset_name=None):
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
        thinkers : Iterable, dict
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
        dataset_name : str, Optional
                       A human-readable string that is not used for anything critical, just nice printing.

        Notes
        -----------
        When getting items from a dataset, the id return order is returned most general to most specific, wrapped by
        the actual raw data and (optionally, if epoch-variety recordings) the label for the raw data, thus:
        raw_data, task_id, dataset_id, person_id, session_id, *label
        """
        super().__init__()
        self.update_id_returns(return_session_id, return_person_id, return_dataset_id, return_task_id)
        self.dataset_name = dataset_name

        if not isinstance(thinkers, dict) and isinstance(thinkers, Iterable):
            self._thinkers = OrderedDict()
            for t in thinkers:
                self.__add__(t, return_session_id=return_session_id)

        elif isinstance(thinkers, dict):
            self._thinkers = OrderedDict(thinkers)
            self._reset_dataset()
        else:
            raise TypeError("Thinkers must be iterable or already processed dict.")

        def set_sess_return(name, thinker: Thinker):
            thinker.return_session_id = return_session_id

        self._apply(set_sess_return)

        self.dataset_id = torch.tensor(dataset_id) if dataset_id is not None else None
        self.task_id = torch.tensor(task_id) if task_id is not None else None

    def update_id_returns(self, session=None, person=None, task=None, dataset=None):
        """
        Updates which ids are to be returned by the dataset. If any argument is `None` it preserves the previous value.

        Parameters
        ----------
        session : None, bool
                  Whether to return session ids.
        person : None, bool
                 Whether to return person ids.
        task    : None, bool
                  Whether to return task ids.
        dataset : None, bool
                 Whether to return dataset ids.
        """
        self.return_person_id = self.return_person_id if person is None else person
        self.return_session_id = self.return_session_id if session is None else session
        self.return_dataset_id = self.return_dataset_id if dataset is None else dataset
        self.return_task_id = self.return_task_id if task is None else task

    def _reset_dataset(self):
        for p_id in self._thinkers:
            self._thinkers[p_id].person_id = p_id
            for s_id in self._thinkers[p_id].sessions:
                self._thinkers[p_id].sessions[s_id].session_id = s_id
                self._thinkers[p_id].sessions[s_id].person_id = p_id
        ConcatDataset.__init__(self, self._thinkers.values())

    def _apply(self, lam_fn):
        for th_id, thinker in self._thinkers.items():
            lam_fn(th_id, thinker)

    def __add__(self, thinker, return_session_id=None):
        assert isinstance(thinker, Thinker)
        return_session_id = self.return_session_id if return_session_id is None else return_session_id
        thinker.return_session_id = return_session_id

        if thinker.person_id in self._thinkers.keys():
            print("Warning. Person {} already in dataset... Merging sessions.".format(thinker.person_id))
            self._thinkers[thinker.person_id] += thinker
        else:
            self._thinkers[thinker.person_id] = thinker
        self._reset_dataset()

    def __getitem__(self, item):
        person_id = bisect.bisect_right(self.cumulative_sizes, item)
        if person_id == 0:
            sample_idx = item
        else:
            sample_idx = item - self.cumulative_sizes[person_id - 1]
        x = list(self._thinkers[self.get_thinkers()[person_id]].__getitem__(sample_idx))

        if self.return_person_id:
            x.insert(1, torch.tensor(person_id))

        if self.return_dataset_id:
            x.insert(1, self.dataset_id)

        if self.return_task_id:
            x.insert(1, self.task_id)

        return self._execute_transforms(*x)

    def preprocess(self, preprocessor: Preprocessor, apply_transform=True, thinkers=None):
        """
        Applies a preprocessor to the dataset

        Parameters
        ----------
        preprocessor : Preprocessor
                       A preprocessor to be applied
        thinkers : (None, Iterable)
                   If specified (default is None), the thinkers to use for preprocessing calculation
        apply_transform : bool
                          Whether to apply the transform to this dataset (all thinkers, not just those specified for
                          preprocessing) after preprocessing them. Exclusive application to specific thinkers can be
                          done using the return value and a separate call to `add_transform` with the same `thinkers`
                          list.

        Returns
        ---------
        preprocessor : Preprocessor
                       The preprocessor after application to all relevant thinkers
        """
        thinkers = self.get_thinkers() if thinkers is None else thinkers
        for thinker in thinkers:
            thinker.preprocess(preprocessor)
        if apply_transform:
            self.add_transform(preprocessor.get_transform())
        return preprocessor

    @property
    def sfreq(self):
        sfreq = set(self._thinkers[t].sfreq for t in self._thinkers)
        if len(sfreq) > 1:
            print("Warning: Multiple sampling frequency values found. Over/re-sampling may be necessary.")
            return unfurl(sfreq)
        sfreq = sfreq.pop()
        for xform in self._transforms:
            sfreq = xform.new_sfreq(sfreq)
        return sfreq

    @property
    def channels(self):
        channels = set(self._thinkers[t].channels for t in self._thinkers)
        if len(channels) > 1:
            print("Warning: Multiple channel sets found. A consistent mapping like Deep1010 may be prudent.")
            return unfurl(channels)
        channels = channels.pop()
        for xform in self._transforms:
            channels = xform.new_channels(channels)
        return channels

    @property
    def sequence_length(self):
        sequence_length = set(self._thinkers[t].sequence_length for t in self._thinkers)
        if len(sequence_length) > 1:
            print("Warning: Multiple sequence lengths found. A cropping transformation may be in order.")
            return unfurl(sequence_length)
        sequence_length = sequence_length.pop()
        for xform in self._transforms:
            sequence_length = xform.new_sfreq(sequence_length)
        return sequence_length

    def get_thinkers(self):
        return list(self._thinkers.keys())

    def __len__(self):
        return self.cumulative_sizes[-1]

    def _make_like_me(self, people: list):
        if len(people) == 1:
            like_me = self._thinkers[people[0]].clone()
        else:
            dataset_id = self.dataset_id.item() if self.dataset_id is not None else None
            task_id = self.task_id.item() if self.dataset_id is not None else None

            like_me = Dataset({p: self._thinkers[p] for p in people}, dataset_id, task_id, self.return_person_id,
                              self.return_session_id, self.return_dataset_id, self.return_task_id)
        for x in self._transforms:
            like_me.add_transform(x)
        return like_me

    def _generate_splits(self, validation, testing):
        for val, test in zip(validation, testing):
            training = list(self._thinkers.keys())
            for v in val:
                training.remove(v)
            for t in test:
                training.remove(t)

            training = self._make_like_me(training)

            validating = self._make_like_me(val)
            _val_set = set(validating.get_thinkers()) if len(val) > 1 else {validating.person_id}

            testing = self._make_like_me(test)
            _test_set = set(testing.get_thinkers()) if len(test) > 1 else {testing.person_id}

            if len(_val_set.intersection(_test_set)) > 0:
                raise ValueError("Validation and test overlap with ids: {}".format(_val_set.intersection(_test_set)))

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
                         Same as `validation_person_id`, but for testing. However, testing may be a list when
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
        if test_person_id is not None and not isinstance(test_person_id, Iterable) or isinstance(test_person_id, str):
            if validation_person_id is None:
                validation_person_id = [pid for pid in self.get_thinkers() if pid != test_person_id]
            test_person_id = [test_person_id for _ in range(len(self.get_thinkers())-1)]
        elif test_person_id is None:
            test_person_id = [t for t in self.get_thinkers()]

        if validation_person_id is not None and not isinstance(validation_person_id, Iterable) or \
                isinstance(validation_person_id, str):
            validation_person_id = [validation_person_id for _ in range(len(test_person_id))]
        elif validation_person_id is None:
            validation_person_id = [test_person_id[i-1] for i in range(len(test_person_id))]

        if not isinstance(test_person_id, list) or len(test_person_id) != len(validation_person_id):
            raise ValueError("Test ids must be same length iterable as validation ids.")

        yield from self._generate_splits([[v] for v in validation_person_id], [[t] for t in test_person_id])

    def lmso(self, folds=10, test_splits=None, validation_splits=None):
        """
        This *generates* a "Leave-multiple-subject-out" (LMSO) split. In other words X-fold cross-validation, with
        boundaries enforced at thinkers (each person's data is not split into different folds).

        Parameters
        ----------
        folds : int
                If this is specified and `splits` is None, will split the subjects into this many folds, and then use
                each fold as a test set in turn (and the previous fold - starting with the last - as validation).
        test_splits : list, tuple
                This should be a list of tuples/lists of either:
                  - The ids of the consistent test set. In which case, folds must be specified, or validation_splits
                    is a nested list that .
                  - Two sub lists, first testing, second validation ids

        Yields
        -------
        training : Dataset
                   Another dataset that represents the training set
        validation : Dataset
                     The validation people as a dataset
        test : Thinker
               The test people as a dataset
        """

        def is_nested(split: list):
            should_be_nested = isinstance(split[0], (list, tuple))
            for x in split[1:]:
                if (should_be_nested and not isinstance(x, (list, tuple))) or (isinstance(x, (list, tuple))
                                                                               and not should_be_nested):
                        raise ValueError("Can't mix list/tuple and other elements when specifying ids.")
            if not should_be_nested and folds is None:
                raise ValueError("Can't infer folds from non-nested list. Specify folds, or nest ids")
            return should_be_nested

        def calculate_from_remainder(known_split):
            _folds = len(known_split) if is_nested(list(known_split)) else folds
            if folds is None:
                print("Inferred {} folds from test split.".format(_folds))
            remainder = list(set(self.get_thinkers()).difference(known_split))
            return [list(x) for x in np.array_split(remainder, _folds)], [known_split for _ in range(_folds)]

        if test_splits is None and validation_splits is None:
            if folds is None:
                raise ValueError("Must specify <folds> if not specifying ids.")
            folds = [list(x) for x in np.array_split(self.get_thinkers(), folds)]
            test_splits, validation_splits = zip(*[(folds[i], folds[i-1]) for i in range(len(folds))])
        elif validation_splits is None:
            validation_splits, test_splits = calculate_from_remainder(test_splits)
        elif test_splits is None:
            test_splits, validation_splits = calculate_from_remainder(validation_splits)

        yield from self._generate_splits(validation_splits, test_splits)

    def add_transform(self, transform, thinkers=None):
        self._transforms.append(transform)

    def clear_transforms(self):
        self._transforms = list()

# TODO Convenience functions or classes for leave one and leave multiple datasets out.
