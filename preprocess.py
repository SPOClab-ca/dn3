import mne
import numpy as np
from transforms import *


class Preprocessing:
    """
    Base class for various preprocessing actions. Sub-classes can operate in-place on an mne.epochs object using the
    __call__ method and return None for the transform.
    Or, create a transform that is subsequently added to the execution graph (and performed just before training).
    """
    def __call__(self, data: mne.epochs, *args, **kwargs):
        raise NotImplementedError()

    def get_transform(self):
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
        transform = ICATransform(self.components.T)
        return transform


class EApreprocessor(Preprocessing):
    """

    """
    def __init__(self):
        self.reference_matrix = None

    def __call__(self, epochs: mne.epochs, *args, **kwargs):
        data = epochs.get_data()
        for epoch in data:
            self.reference_matrix += tf.matmul(epoch, epoch, transpose_b=True)
        self.reference_matrix = self.reference_matrix / len(data)
        self.reference_matrix = tf.constant(tf.linalg.inv(tf.linalg.sqrtm(self.reference_matrix)), dtype=tf.float32)

    def get_transform(self):
        if self.reference_matrix is None:
            raise ReferenceError('Preprocessor must be executed before the transform can be retrieved.')
        transform = EATransform(self.reference_matrix)
        return transform