import tensorflow as tf
import mne
import numpy as np
from transforms import *


def zscore(data, axis=-1):
    return tf.math.divide_no_nan(
        tf.math.subtract(data, tf.reduce_mean(data, axis=axis, keepdims=True)),
        tf.math.reduce_std(data, axis=axis, keepdims=True)
    )






