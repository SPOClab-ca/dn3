import tensorflow as tf
import mne


def zscore(data, axis=-1):
    return tf.math.divide_no_nan(
        tf.math.subtract(data, tf.reduce_mean(data, axis=axis, keepdims=True)),
        tf.math.reduce_std(data, axis=axis, keepdims=True)
    )


class Preprocessing:
    """
        This is the class for operations which are passed into EpochsDataLoader to perform preprocessing on
        mne.Epoches.
     """
    def __call__(self, data: mne.epochs, *args, **kwargs):
        raise NotImplementedError()


class ICAPreprocessor(Preprocessing):

    def __init__(self, n_components=None, method='fastica'):
        self.data = None
        self.components = None
        self.ica = mne.preprocessing.ICA(n_components=n_components, method=method)

    def __call__(self, data: mne.epochs, *args, **kwargs):
        self.data = data
        self.ica = self.ica.fit(data)
        self.components = self.ica.get_components()

    def get_components(self):
        return self.components

    def get_transform(self):
        ica_data = self.ica.get_sources(self.data)
        return ica_data



