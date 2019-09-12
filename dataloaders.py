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

    def __init__(self, raw: mne.io.Raw, events, tmin: float, tlen: float, baseline=None, preprocessings=None,
                 normalizer=zscore, num_parallel_calls=tf.data.experimental.AUTOTUNE, picks=None, **kwargs):
        self.epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmin + tlen - 1 / raw.info['sfreq'], baseline=baseline)
        self.picks = picks

        self._dataset = tf.data.Dataset.from_tensor_slices(
            (tf.range(len(self.epochs.events)), self.epochs.events[:, -1]))
        # TODO -- find a way to allow parallel epoch loading, currently fails if more than 1
        self._dataset = self._dataset.map(
            lambda index, label: tuple(tf.py_function(
                lambda ind, lab: self._retrieve_epoch(ind, lab), [index, label], [tf.float32, tf.uint16])),
            num_parallel_calls=1)

        self.transforms_in_queue = []
        # preprocessing if necessary
        if preprocessings:
            for preprocessing in preprocessings:
                preprocessing(self.epochs)
                transform = preprocessing.get_transform()
                # to later apply to tf.data.dataset optionally
                self.transforms_in_queue.append(transform)
        self._dataset = self._dataset.map(
            lambda data, label: tuple((normalizer(data), label)), num_parallel_calls=num_parallel_calls)
        self._train_dataset = self._dataset
        self.transforms = [normalizer]

    def _retrieve_epoch(self, ep, label):
        x = self.epochs[ep].get_data(picks=self.picks).astype('float32').squeeze()
        assert len(x.shape) == 2
        return x, tf.cast(label, tf.uint16)

    def train_dataset(self):
        return self._train_dataset

    def eval_dataset(self):
        return self._dataset

    def get_transform_in_queue(self):
        return self.transforms_in_queue

    def add_transform(self, map_fn, apply_train=True, apply_eval=False):
        # fixme parallel calls for transforms
        assert apply_train or apply_eval
        self.transforms.append(map_fn)
        if apply_train:
            self._train_dataset = self._train_dataset.map(
                lambda data, label: tuple((map_fn(data), label)))
        if apply_eval:
            self._dataset = self._dataset.map(lambda data, label: tuple((map_fn(data), label)))


