import mne
import numpy as np

from transform import *


class Preprocessor:
    """
    Base class for various preprocessing actions. Sub-classes are called with a subclass of :any:`_Recording`
    and operate on these instances in-place.

    Any modifications to data specifically should be implemented through a subclass of :any:`BaseTransform`, and
    returned by the method :any:`get_transform()`
    """
    def __call__(self, recording, **kwargs):
        raise NotImplementedError()

    def get_transform(self):
        raise NotImplementedError()

# TODO Fix this tf implementation
# class EApreprocessor(Preprocessor):
#     """
#
#     """
#     def __init__(self):
#         self.reference_matrix = None
#
#     def __call__(self, epochs: mne.Epochs, *args, **kwargs):
#         data = epochs.get_data().astype(np.float32)
#         self.reference_matrix = tf.reduce_mean(tf.matmul(data, data, transpose_b=True), axis=0)
#         self.reference_matrix = tf.constant(tf.linalg.inv(tf.linalg.sqrtm(self.reference_matrix)), dtype=tf.float32)
#
#     def get_transform(self):
#         if self.reference_matrix is None:
#             raise ReferenceError('Preprocessor must be executed before the transform can be retrieved.')
#         transform = EATransform(self.reference_matrix)
#         return transform