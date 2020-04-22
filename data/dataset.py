import mne
import torch
import numpy as np

from .config import DatasetConfig
from .utils import min_max_normalize as _min_max_normalize
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import ConcatDataset


class TorchRecording(TorchDataset):
    """
    Base class for any mne dataloader being bridged into an iterable pytorch-compatible dataset
    """
    def __init__(self, raw: mne.io.Raw, session_id, person_id):
        self.sfreq = raw.info['sfreq']
        assert self.sfreq is not None
        self.session_id = session_id
        self.person_id = person_id


class RawTorchRecording(TorchRecording):
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
        super().__init__(raw, session_id, person_id)
        self.raw = raw
        self.stride = stride
        self.max = kwargs.get('max', None)
        self.min = kwargs.get('min', 0)
        self.normalizer = kwargs.get('normalize', _min_max_normalize)
        self.sample_len = sample_len if tlen is None else int(raw.info['sfreq'] * tlen)

    def __getitem__(self, index):
        index *= self.stride

        # Get the actual signals
        x = self.raw.get_data(self.picks, start=index, stop=index+self.sample_len)
        scale = 1 if self.max is None else (x.max() - x.min()) / (self.max - self.min)
        if scale > 1 or np.isnan(scale):
            print('Warning: scale exeeding 1')
        x = _min_max_normalize(torch.from_numpy(x)).transpose(1, 0)
        if torch.any(x > 1):
            print("Somehow larger than 1.. {}, index {}".format(self.raw.filenames[0], index))
            raise ValueError()

        if torch.any(torch.isnan(x)):
            print("Nan found: raw {}, index {}".format(self.raw.filenames[0], index))
            print("Replacing with appropriate random values for now...")
            x = torch.rand_like(x)

        # Construct meta-information
        pos = index % self.max_positions
        position_ids = torch.arange(pos, pos+self.seq_len).unsqueeze(-1)
        next_set = position_ids >= self.max_positions
        position_ids[next_set] -= self.max_positions

        dataset_ids = torch.ones(x.shape[0], 1).long() * int(self.config['dataset_id'])
        person_ids = torch.ones(1).long() * int(self.person_id)

        return x, position_ids, dataset_ids, person_ids, self.mapping.sum(dim=0)

    def __len__(self):
        return (self.raw.n_times - self.seq_len + int(self.cls_token)) // self.stride


class EpochTorchRecording(TorchRecording):

    def __init__(self, epochs: mne.Epochs, force_label=None, picks=None, preproccesors=None, normalizer=zscore,
                 runs=None, train_mode=False):
        self.mode = train_mode
        self.epochs = epochs
        self._t_len = epochs.tmax - epochs.tmin
        self.loaded_x = [None for _ in range(len(epochs.events))]
        self.runs = runs
        self.picks = picks
        self.force_label = force_label if force_label is None else torch.tensor(force_label)
        self.normalizer = normalizer
        self.preprocessors = preproccesors if isinstance(preproccesors, (list, tuple)) else [preproccesors]
        for i, p in enumerate(self.preprocessors):
            self.preprocessors[i] = p(self.epochs)

    @property
    def channels(self):
        if self.picks is None:
            return len(self.epochs.ch_names)
        else:
            return len(self.picks)

    @property
    def sfreq(self):
        return self.epochs.info['sfreq']

    def __getitem__(self, index):
        ep = self.epochs[index]
        if self.loaded_x[index] is None:
            x = ep.get_data()
            if len(x.shape) != 3 or 0 in x.shape:
                print("I don't know why: {} index{}/{}".format(self.epochs, index, len(self)))
                print(self.epochs.info['description'])
                # raise AttributeError()
                return self.__getitem__(index - 1)
            x = x[0, self.picks, :]
            for p in self.preprocessors:
                x = p(x)
            x = torch.from_numpy(self.normalizer(x).astype('float32')).squeeze(0)
            self.loaded_x[index] = x
        else:
            x = self.loaded_x[index]

        y = torch.from_numpy(ep.events[..., -1]).long() if self.force_label is None else self.force_label

        if self.runs is not None:
            return x, y, one_hot(torch.tensor(self.runs * index / len(self)).long(), self.runs)

        return x, y

    def __len__(self):
        events = self.epochs.events[:, 0].tolist()
        return len(events)



class Thinker(object):
    """
    Collects multiple recordings of the same person, intended to be of the same task, at different times or conditions.
    """
    def __init__(self, recordings, person_id="auto"):
        """
        Collects multiple recordings of the same person, intended to be of the same task, at different times or
        conditions.
        Parameters
        ----------
        recordings : (iterable, dict)
                   Either a sequence of recordings, or a mapping of session_ids to recordings. If the former, the
                   recording's session_id is preserved. If the
        person_id : (int, str)
                   Label to be used for the thinker. If set to "auto" (default), will automatically pick the person_id
                   using the most common person_id in the recordings.
        """


class Dataset(object):
    """
    Collects thinkers, each of which may collect multiple recording sessions of the same tasks, into a dataset with
    (largely) consistent:
      - hardware:
        - channel number/labels
        - sampling frequency
      - annotation paradigm:
        - consistent event types
    """
    def __init__(self, *thinkers, **kwargs):
        self.thinkers = thinkers

    def _init_thinkers(self, mapping=None):
        if mapping is None:

    @classmethod
    def from_config(cls, config: DatasetConfig):
        """
        This creates a dataset using a config that points to a collection of files in a standard directory layout:
        Dataset/(*optional - <version>)/<person-id>/<session-id>.{ext}
        The {ext} supported are: edf/gdf, fif

        See `.config.DatasetConfiguration` for how a configuration is constructed.
        Parameters
        ----------
        config : DatasetConfig


        Returns
        -------

        """
        # TODO
        pass

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

    def add_preprocessor(self, apply_transform=True):
        pass

    def add_transform(self, transform):
        pass


class MultiDomainDataset(ConcatDataset):
    pass

