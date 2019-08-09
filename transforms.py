import tensorflow as tf


class BaseTransform(object):

    def __call__(self, x: tf.Tensor, *args, **kwargs):
        raise NotImplementedError()


class ChannelShuffle(BaseTransform):

    def __init__(self, p=0.1, channel_axis=1):
        self.p = p

    def __call__(self, x: tf.Tensor, *args, **kwargs):
        pass


class DummyTransform(BaseTransform):

    def __call__(self, x: tf.Tensor, *args, **kwargs):
        return tf.multiply(x, 2)


class ICATransform(BaseTransform):

    def __init__(self, input):
        self.input = input

    def __call__(self, x: tf.Tensor, *args, **kwargs):
        return tf.linalg.matmul(tf.cast(self.input, tf.float32), x)


class EATransform(BaseTransform):

    def __init__(self, ref_matrix):
        self.ref_matrix = ref_matrix

    def __call__(self, x: tf.Tensor, *args, **kwargs):
        return tf.linalg.matmul(tf.cast(self.ref_matrix, tf.float32), x)
