import tensorflow as tf


class BaseTransform(object):

    def __call__(self, x: tf.Tensor, *args, **kwargs):
        raise NotImplementedError()


class ChannelShuffle(BaseTransform):

    def __init__(self, p=0.1, channel_axis=1):
        self.p = p

    def __call__(self, x: tf.Tensor, *args, **kwargs):
        pass

