import mne
import tensorflow as tf
import numpy as np
from mne.datasets import sample
from datasets import *
from transforms import *
import unittest





class TestEpochLoader(unittest.TestCase):

    def setUp(self):
        # create dummy raw data for unit test
        sinx = np.sin(np.arange(0, 10, 0.001) * 10).astype('float')
        cosx = np.cos(np.arange(0, 10, 0.001) * 10).astype('float')
        data = np.array([sinx, cosx])
        ch_names = [str(s) for s in range(2)]
        ch_types = ['mag', 'mag']
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types=ch_types)
        self.raw = mne.io.RawArray(data, info)
        self.epochs_data = np.array(
            [[sinx[:5], cosx[:5]], [sinx[1198:1203], cosx[1198:1203]], [sinx[1998:2003], cosx[1998:2003]]])
        self.events = np.array([[2, 0, 5], [1200, 0, 3], [2000, 0, 1]])

    def test_epochs(self):
        loader = EpochsDataLoader(self.raw, self.events, -0.002, 0.005, baseline=None)

        self.assertEqual(np.any(loader.epochs.get_data() - self.epochs_data), False)

    def test_trainDataAfterZscore(self):
        loader = EpochsDataLoader(self.raw, self.events, -0.002, 0.005, baseline=None)
        # test the first 3 iterations
        dataset = loader.train_dataset()
        it = iter(dataset)
        for i in range(3):
            with self.subTest(i=i):
                x, y = next(it)
                # print(x.numpy())
                tensor = tf.cast(tf.constant(loader.epochs.get_data()[i]), tf.float32)
                transform = zscore(tensor).numpy()
                self.assertEqual(np.any(x - transform), False)

    def test_ZscoreFunction(self):
        loader = EpochsDataLoader(self.raw, self.events, -0.002, 0.005, baseline=None)
        dummy = np.array([[1.0, 2.0, 3.0], [1., 1., 2.]])
        result = np.around((dummy - np.mean(dummy, axis=-1, dtype='float32', keepdims=True)) / \
                 np.std(dummy, axis=-1, dtype='float32', keepdims=True), decimals=5)
        zscore_transform = np.around(zscore(tf.cast(tf.constant(dummy), tf.float32)).numpy(), decimals=5)
        self.assertEqual(np.allclose(result, zscore_transform), True)

    # test add_transform method using dummy transform functions
    def test_addTransformForTrainData(self):
        loader = EpochsDataLoader(self.raw, self.events, -0.002, 0.005, baseline=None)
        dummy_transform = DummyTransform()
        loader.add_transform(dummy_transform, apply_train=True, apply_eval=False)
        dataset = loader.train_dataset()
        it = iter(dataset)
        # test the first 3 iterations
        for i in range(3):
            with self.subTest(i=i):
                x, y = next(it)
                tensor = tf.cast(tf.constant(loader.epochs.get_data()[i]), tf.float32)
                transform = dummy_transform(zscore(tensor))
                self.assertEqual(np.any(x - transform), False)


if __name__ == '__main__':
    unittest.main()
