import mne
import tensorflow as tf
from utils import *


class EpochsDataLoader:
    """
    This is effectively a factory class generating tensorflow Dataset classes. This is because this class is difficult
    to extend, and is not strictly necessary. Raw mne structures will be internally converted to Epoch structures and
    individual epochs loaded by the tensorflow graph session.

    This structure supports both preloaded and not preloaded raw data. Allowing the disk i/o to be performed
    simultaneously to training. Optionally the *cached* argument can be used to cahe these operations if there is
    available memory and the raw is not preloaded (e.g. for lazy loading).

    The normalizer should be tf operations only, if not, see tf.py_function to wrap arbitrary logic.
    """

    def __init__(self, raw: mne.io.Raw, events, tmin: float, tlen: float, baseline=(None, 0), normalizer=zscore,
                 num_parallel_calls=tf.data.experimental.AUTOTUNE, **kwargs):
        self.epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmin + tlen - 1 / raw.info['sfreq'], baseline=baseline)
        self._dataset = tf.data.Dataset.from_tensor_slices(
            (tf.range(len(self.epochs.events)), self.epochs.events[:, -1])
        )
        self._dataset = self._dataset.map(
            lambda index, label: tuple(tf.py_function(
                lambda ind, lab: self._retrieve_epoch(ind, lab), [index, label], [tf.float32, tf.uint16])),
            num_parallel_calls=num_parallel_calls)

        self._dataset = self._dataset.map(
            lambda data, label : tuple((normalizer(data), label)), num_parallel_calls=num_parallel_calls)
        self._train_dataset = self._dataset
        self.transforms = [normalizer]

    def _retrieve_epoch(self, run, label):
        return self.epochs.get_data()[run].astype('float32'), tf.cast(label, tf.uint16)

    def train_dataset(self):
        return self._train_dataset

    def eval_dataset(self):
        return self._dataset

    def add_transform(self, map_fn, apply_train=True, apply_eval=False):
        assert apply_train or apply_eval
        self.transforms.append(map_fn)
        if apply_train:
            print('here')
            self._train_dataset = self._train_dataset.map(
                lambda data, label: tuple((map_fn(data), label)))

        if apply_eval:
            self._dataset = self._dataset.map(lambda data, label: tuple((map_fn(data), label)))


def multi_subject(*datasets):
    """
    Concatenates all the datasets provided into one Dataset with an additional label corresponding to its original
    source dataset e.g. which subject. So if the datasets were previously (x_i, label_i) for all i, they now load (x_i,
    label_i, subject_j) for all J subjects with I epochs each.
    :param datasets:
    :return:
    """
    for i, ds in enumerate(datasets):
        assert isinstance(ds, tf.data.Dataset)
        ds.map(lambda *x: (*x, tf.constant(i)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for ds in datasets[1:]:
        datasets[0].concatenate(ds)
