import tensorflow as tf


def zscore(data, axis=-1):
    return tf.math.divide_no_nan(
        tf.math.subtract(data, tf.reduce_mean(data, axis=axis, keepdims=True)),
        tf.math.reduce_std(axis=axis, keepdims=True)
    )

