import unittest
import numpy as np
import tensorflow as tf
from dataloaders import EpochsDataLoader
from transforms import LabelSmoothing, Mixup
from tests.testDataloaders import create_dummy_raw


def numpy_label_smooth(labels, targets, gamma):
    one_hot = np.eye(targets)[labels]
    one_hot -= gamma * (one_hot - 1 / (targets + 1))
    return one_hot


class TestLabelSmoothing(unittest.TestCase):

    def setUp(self) -> None:
        self.raw, self.events = create_dummy_raw()
        self.loader = EpochsDataLoader(self.raw, self.events, -0.002, 0.005)
        self.gamma = 0.1
        self.label_smoother = LabelSmoothing(self.loader.targets, self.gamma)
        self.loader.add_transform(self.label_smoother)

    def test_dims_grow(self):
        for _, label in self.loader.train_dataset():
            with self.subTest():
                self.assertEqual(self.loader.targets, label.shape[-1])

    def test_value_range(self):
        for _, label in self.loader.train_dataset():
            with self.subTest():
                self.assertEqual(False, np.any(label < 0))
                self.assertEqual(False, np.any(label > 1))

    def test_exact_train(self):
        train = iter(self.loader.train_dataset())
        eval = iter(self.loader.eval_dataset())
        for i in range(self.loader.epochs.events.shape[0]):
            with self.subTest(i=i):
                _, xform_smooth = next(train)
                _, labels = next(eval)
                self.assertEqual(np.allclose(
                    xform_smooth.numpy(), numpy_label_smooth(labels, self.loader.targets, self.gamma)), True)


class TestMixup(unittest.TestCase):

    def setUp(self) -> None:
        self.raw, self.events = create_dummy_raw()
        self.loader = EpochsDataLoader(self.raw, self.events, -0.002, 0.005)
        self.rate = 1.0
        self.batch_size = 4
        self.mixup = Mixup(self.rate)

    def test_add_batched(self):
        ds = self.loader.train_dataset().repeat(10).batch(
            self.batch_size, drop_remainder=True).map(self.mixup, num_parallel_calls=1)
        for _, label in ds:
            pass

    def test_add_batched_multiproc(self):
        ds = self.loader.train_dataset().repeat(10).batch(self.batch_size, drop_remainder=True).map(
            self.mixup, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        for _, label in ds:
            pass


if __name__ == '__main__':
    unittest.main()
