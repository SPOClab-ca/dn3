import mne
import tensorflow as tf
import numpy as np
from mne.datasets import sample
from datasets import *
from transforms import *
from pmodel import *
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

    def test_evaluateDataAfterZscore(self):
        loader = EpochsDataLoader(self.raw, self.events, -0.002, 0.005, baseline=None)
        # test the first 3 iterations
        dataset = loader.eval_dataset()
        it = iter(dataset)
        for i in range(3):
            with self.subTest(i=i):
                x, y = next(it)
                # print(x.numpy())
                tensor = tf.cast(tf.constant(loader.epochs.get_data()[i]), tf.float32)
                transform = zscore(tensor).numpy()
                self.assertEqual(np.any(x - transform), False)

    def test_ZscoreFunction(self):
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

    def test_addTransformForEvaluateData(self):
        loader = EpochsDataLoader(self.raw, self.events, -0.002, 0.005, baseline=None)
        dummy_transform = DummyTransform()
        loader.add_transform(dummy_transform, apply_train=False, apply_eval=True)
        dataset = loader.eval_dataset()
        it = iter(dataset)
        # test the first 3 iterations
        for i in range(3):
            with self.subTest(i=i):
                x, y = next(it)
                tensor = tf.cast(tf.constant(loader.epochs.get_data()[i]), tf.float32)
                transform = dummy_transform(zscore(tensor))
                self.assertEqual(np.any(x - transform), False)

    def test_MultiSubject(self):
        loader = EpochsDataLoader(self.raw, self.events, -0.002, 0.005, baseline=None)
        ds_1 = loader.train_dataset()
        # create a new dataset
        ds_2 = ds_1.map(lambda x, y: tuple((tf.subtract(x, 2), y-1)))
        test = multi_subject(ds_1, ds_2).__iter__()
        result = ds_1.map(
            lambda x, y: (x, y, tf.constant(0))).concatenate(
            ds_2.map(lambda data, label: (data, label, tf.constant(1)))).__iter__()
        for i in range(6):
            with self.subTest(i=i):
                x1, y1, z1 = next(result)
                x2, y2, z2 = next(test)
                self.assertEqual(np.any(x1.numpy() - x2.numpy()), False)
                self.assertEqual(y1.numpy(), y2.numpy())
                self.assertEqual(z1.numpy(), z2.numpy())

    def test_ModelFit(self):
        positive = np.random.uniform(0, 5, (10, 10000)).astype('float')
        ch_names = [str(s) for s in range(10)]
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='mag')
        raw = mne.io.RawArray(positive, info)
        events = np.array([[400*(i+1), 0, 1] for i in range(20)])
        loader = EpochsDataLoader(raw, events, -0.050, 0.050, baseline=None)
        positive_ds = loader.train_dataset().map(lambda x, y: tuple((tf.math.abs(x), [0, 1])))
        negative = - np.random.uniform(0, 5, (10, 10000)).astype('float')
        raw = mne.io.RawArray(negative, info)
        events = np.array([[400 * (i + 1), 0, 0] for i in range(20)])
        loader = EpochsDataLoader(raw, events, -0.050, 0.050, baseline=None)
        negative_ds = loader.train_dataset().map(lambda x, y: tuple((-tf.math.abs(x), [1, 0])))
        final_ds = positive_ds.concatenate(negative_ds)
        final_ds = final_ds.map(lambda x, y: tuple((tf.reshape(x, [500, ]), y)))
        final_ds = final_ds.shuffle(buffer_size=40).batch(10).repeat(3)
        input_shape = (500, )
        output_shape = 2
        model = get_dummy_model_tofit(input_shape, output_shape)
        model.fit(final_ds, epochs=2)
        test_x_positive = np.random.uniform(0, 5, (50, 1, 500)).astype('float')
        test_x_nagative = - np.random.uniform(0, 5, (50, 1, 500)).astype('float')
        count = 0
        for i in range(50):
            if model.predict_classes(test_x_positive[i])[0] == 1:
                count += 1
            if model.predict_classes(test_x_nagative[i])[0] == 0:
                count += 1
        self.assertEqual(count, 100)

    def test_ICApreprocessingReturnShape(self):
        data = np.random.uniform(0, 5, (100, 10000)).astype('float')
        ch_names = [str(s) for s in range(100)]
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='mag')
        raw = mne.io.RawArray(data, info)
        events = np.array([[400 * (i + 1), 0, 1] for i in range(20)])
        epochs = mne.Epochs(raw, events, tmin=-0.050, tmax=0.049, baseline=None)
        ica = ICAPreprocessor(n_components=3)
        ica(epochs)
        shape = ica.get_transform().get_data().shape
        self.assertEqual(shape, (20, 3, 100))

    def test_ICApreprocessShapeInEpochloader(self):
        data = np.random.uniform(0, 5, (100, 10000)).astype('float')
        ch_names = [str(s) for s in range(100)]
        info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='mag')
        raw = mne.io.RawArray(data, info)
        events = np.array([[400 * (i + 1), 0, 1] for i in range(20)])
        ica = ICAPreprocessor(n_components=3)
        loader = EpochsDataLoader(raw, events, -0.025, 0.100, baseline=None, preprocessing=ica)
        dataset = loader.train_dataset()
        x, y = next(dataset.__iter__())
        self.assertEqual(x.numpy().shape, (3, 100))

if __name__ == '__main__':
    unittest.main()
