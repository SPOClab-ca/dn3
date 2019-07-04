import mne
import tensorflow as tf


class EpochsDataLoader:

    def __init__(self, raw: mne.io.Raw, events, tmin: float, tlen: float, **kwargs):
        self.epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmin + tlen - 1 / raw.info['sfreq'])
        self._dataset = tf.data.Dataset.from_tensor_slices(
            (tf.range(len(self.epochs.events), self.epochs.events[:, -1]))
        )
        self._dataset = self._dataset.map(
            lambda filename, label: tuple(tf.py_function(
                lambda ind, lab: self._retrieve_epoch(ind, lab), [filename, label], [tf.float32, tf.uint16])))
        self._train_dataset = self._dataset

    def _retrieve_epoch(self, run, label):
        return self.epochs[run].get_data().astype('float32'), tf.cast(label, tf.uint16)

    def train_dataset(self):
        return self._train_dataset

    def eval_dataset(self):
        return self._dataset
