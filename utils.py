import tensorflow as tf
import mne
import numpy as np
from transforms import *


def zscore(data, labels=None, axis=-1):
    data = tf.math.divide_no_nan(
        tf.math.subtract(data, tf.reduce_mean(data, axis=axis, keepdims=True)),
        tf.math.reduce_std(data, axis=axis, keepdims=True)
    )
    return (data, labels) if labels is not None else data


def dataset_concat(*ds):
    dataset = ds[0]
    assert isinstance(dataset, tf.data.Dataset)
    for d in ds[1:]:
        assert isinstance(d, tf.data.Dataset)
        dataset = dataset.concatenate(d)
    return dataset


class CosineScheduleLR:
    def __init__(self, max_lr, warmup=10, total_epochs=100):
        self.max_lr = max_lr
        self.warmup = warmup
        self.real_eps = total_epochs - warmup

    def __call__(self, epoch):
        return min(self.max_lr * epoch / self.warmup, self.max_lr * 0.5 * (1 + np.cos(epoch * np.pi / self.real_eps)))


def labelled_dataset_concat(*datasets):
    """
    Concatenates all the datasets provided into one Dataset with an additional label corresponding to its original
    source dataset. Can be used for multi-subject and multi-run datasets, providing concatenation with identification.
    For example: datasets by subject. If the datasets were previously (x_i, label_i) for all i, they now load (x_i,
    label_i, subject_j) for all J subjects with I epochs each.
    :param datasets:
    :return: concatenated datasets;
    """
    new_datasets = []
    for i, ds in enumerate(datasets):
        assert isinstance(ds, tf.data.Dataset)
        new_datasets.append(ds.map(
            lambda *x: (*x, tf.constant(i)), num_parallel_calls=tf.data.experimental.AUTOTUNE))
    for ds in new_datasets[1:]:
        new_datasets[0] = new_datasets[0].concatenate(ds)
    return new_datasets[0]





