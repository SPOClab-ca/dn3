import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class BaseTransform(object):
    """
    Each transform is simply a callable object.
    __call__ should be implemented with tensorflow operations or wrapped in tf.py_function
    When called through a Dataloader, the x and label (and other args) will be a single instance (no batch zeroth batch
    dimension, the channels will be the first dimension typically). This will not be the case if used with
    """

    def __call__(self, x: tf.Tensor, label: tf.Tensor, *args, **kwargs):
        raise NotImplementedError()


class ChannelShuffle(BaseTransform):

    def __init__(self, p=0.1):
        assert 0 <= p <= 1
        self.p = tf.constant(p, dtype=tf.float16)

    def __call__(self, x: tf.Tensor, label: tf.Tensor, *args, **kwargs):
        return (tf.cond(tf.random.uniform(1, dtype=tf.float16) < self.p, tf.random.shuffle(x), x), label, *args)


class LabelSmoothing(BaseTransform):
    """
    Smooth the target labels, in effect adding noise to the targets.
    """
    def __init__(self, targets, gamma=0.1):
        self.targets = tf.constant(targets)
        self.gamma = tf.constant(gamma, dtype=tf.float32)

    def __call__(self, x: tf.Tensor, label: tf.Tensor, *args, **kwargs):
        label = tf.one_hot(label, self.targets, dtype=tf.float32)
        label = tf.subtract(label, self.gamma * (label - 1 / (tf.cast(self.targets, tf.float32) + 1)))
        return (x, label, *args)


class Mixup(BaseTransform):
    """
    Implements the regularization proposed in the article
    'mixup: BEYOND EMPIRICAL RISK MINIMIZATION'
    Zhang et al. ICLR 2018,

    Warnings: This transform is only effective once the dataset has been batched, thus it will be ineffective if
    applied to a DataLoader, it needs to be mapped to the batched (and ideally shuffled) datasets e.g.
    training_data.shuffle(BUF_SIZE).batch(BATCH_SIZE).map(MixupInstance(...))
    """
    def __init__(self, rate=0):
        self.rate = rate

        lam_mu = np.random.beta(self.rate, self.rate)


    def __call__(self, *args, **kwargs):

        for arg in args:
            pass
            #linearly interpolate as x and label
        pass


class DummyTransform(BaseTransform):

    def __call__(self, x: tf.Tensor, label: tf.Tensor, *args, **kwargs):
        return tf.multiply(x, 2), label


class ICATransform(BaseTransform):

    def __init__(self, input):
        self.input = input

    def __call__(self, x: tf.Tensor, label: tf.Tensor, *args, **kwargs):
        return tf.linalg.matmul(tf.cast(self.input, tf.float32), x), label


class EATransform(BaseTransform):

    def __init__(self, ref_matrix):
        self.ref_matrix = ref_matrix

    def __call__(self, x: tf.Tensor, label: tf.Tensor, *args, **kwargs):
        return tf.linalg.matmul(tf.cast(self.ref_matrix, tf.float32), x), label
