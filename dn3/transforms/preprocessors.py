from dn3.transforms.instance import InstanceTransform


class Preprocessor:
    """
    Base class for various preprocessing actions. Sub-classes are called with a subclass of `_Recording`
    and operate on these instances in-place.

    Any modifications to data specifically should be implemented through a subclass of :any:`BaseTransform`, and
    returned by the method :meth:`get_transform()`
    """
    def __call__(self, recording, **kwargs):
        """
        Preprocess a particular recording. This is allowed to modify aspects of the recording in-place, but is not
        strictly advised.

        Parameters
        ----------
        recording :
        kwargs : dict
                 New :any:`_Recording` subclasses may need to provide additional arguments. This is here for support of
                 this.
        """
        raise NotImplementedError()

    def get_transform(self):
        """
        Generate and return any transform associated with this preprocessor. Should be used after applying this
        to a dataset, i.e. through :meth:`DN3ataset.preprocess`

        Returns
        -------
        transform : BaseTransform
        """
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