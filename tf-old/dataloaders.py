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
        self.epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmin + tlen - 1 / raw.info['sfreq'], baseline=baseline,
                                 picks=picks)
        self.num_parallel_calls = num_parallel_calls

        self.num_channels = len(self.epochs.ch_names)
        self.samples_t = self.epochs.times.shape[0]

        self._dataset = tf.data.Dataset.from_tensor_slices(
            (tf.range(len(self.epochs.events)), self.epochs.events[:, -1] - 1))

        # TODO -- find a way to allow parallel epoch loading, currently fails if more than 1
        def tf_retrieve(ind, label):
            [x, ] = tf.py_function(self._retrieve_epoch, [ind], [tf.float32])
            x.set_shape((self.num_channels, self.samples_t))
            return x, label
        self._dataset = self._dataset.map(tf_retrieve, num_parallel_calls=1)

        self.transforms = [normalizer]

        self.transforms_in_queue = []
        # TODO ask Zhihuan what is going on with transform queue
        # preprocessing if necessary
        if preprocessings:
            for preprocessing in preprocessings:
                preprocessing(self.epochs)
                transform = preprocessing.get_transform()
                self._dataset.map(transform, num_parallel_calls=num_parallel_calls)
                # to later apply to tf.data.dataset optionally
                self.transforms_in_queue.append(transform)
        self._dataset = self._dataset.map(normalizer, num_parallel_calls=num_parallel_calls)
        self._train_dataset = self._dataset

    def _retrieve_epoch(self, ep):
        x = self.epochs[ep.numpy()].get_data().astype('float32').squeeze()
        assert len(x.shape) == 2
        return x

    def train_dataset(self):
        return self._train_dataset

    def eval_dataset(self):
        return self._dataset

    def get_transform_in_queue(self):
        return self.transforms_in_queue

    def add_transform(self, map_fn, apply_train=True, apply_eval=False):
        assert apply_train or apply_eval
        self.transforms.append(map_fn)
        if apply_train:
            self._train_dataset = self._train_dataset.map(map_fn, num_parallel_calls=self.num_parallel_calls)
        if apply_eval:
            self._dataset = self._dataset.map(map_fn, num_parallel_calls=self.num_parallel_calls)

    @property
    def targets(self):
        target = self.epochs.events[:, -1]
        # return np.unique(target).shape[0]
        return int(np.max(target) + 1)