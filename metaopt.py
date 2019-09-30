import tensorflow as tf
import numpy as np
import tqdm
from tensorflow import keras
from metamodel import MetaModel
from utils import dataset_concat

class MetaOptimizer:
    """
    Interface for all meta optimizers so that they behave nicely with meta models.

    Otherwise, training can proceed using the train method.
    """
    def train(self, *args, **kwargs):
        raise NotImplementedError()

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError()


class Reptile(MetaOptimizer):

    # TODO properly turn this into a tf.function for optimizations
    def train(self, loader_dict, model, batch_sizes, outer_lr, inner_iterations=20,
              inner_optimizer=keras.optimizers.Adam(),
              minimize_op=None, shuffle_size=None):
        """
        A single training iteration (single outermost loop, multiple inner loops)
        Parameters
        ----------
        loader_dict:
            The dictionary structure of {subjects{sessions{runs:EpochDataLoader}}}
        model:
            A meta-model or standard keras model.
        batch_sizes:
            An integer or list of integers to be used for each layer of meta-loops, the list can not exceed the
            depth of the loader_dict, the last batch element is used for the inner loop (standard training).
        inner_optimizer:
            The optimizer to use for the inner-most training loop
        minimize_op:
            Optional specific tensor to use for minimization, otherwise all losses associated with the model
            (found using model.losses) are minimized.

        Returns
        -------

        """
        if isinstance(batch_sizes, int):
            batch_sizes = 3 * [batch_sizes]
        elif len(batch_sizes) == 1:
            batch_sizes = 3 * batch_sizes
        elif len(batch_sizes) == 2:
            batch_sizes = [batch_sizes[0], *batch_sizes]
        else:
            raise ValueError(f'Batch size must be single integer or list of 1-3 ints, got {batch_sizes}')

        # Only need a shuffle buffer the size of the inner most loop
        if shuffle_size == None:
            shuffle_size = inner_iterations * batch_sizes[-1]

        start_weights = model.get_weights()
        new_subject_weights = []
        meta_loop = tqdm.tqdm(np.random.permutation(list(loader_dict.keys()))[:batch_sizes[0]], unit='subject')
        metrics = []
        for subject in meta_loop:
            # model.set_weights(start_weights)
            train_set = loader_dict[subject].shuffle(shuffle_size).take(
                batch_sizes[2]*inner_iterations).batch(batch_sizes[2])
            metrics.append(model.train_on_batch(train_set))
            meta_loop.set_postfix(dict(zip(model.metrics_names, np.mean(metrics, axis=0))))
            new_subject_weights.append(model.get_weights())
        model.set_weights(self._update_weights(start_weights, new_subject_weights, outer_lr))

    def few_shot_evaluation(self, test_set, model: keras.Model, num_targets, num_shots, batch_size=None, iterations=50):
        #FIXME Currently assumes for every _num_targets_ training points, there will be one of each class present
        shots = num_targets * num_shots
        batch_size = batch_size if batch_size is not None else num_shots
        train = test_set.take(shots).shuffle(shots).batch(batch_size)
        test = test_set.skip(shots).batch(batch_size)
        model.fit(x=train, epochs=iterations)
        return model.evaluate(test)

    @staticmethod
    def add_set(x1, x2):
        return [v1 + v2 for (v1, v2) in zip(x1, x2)]

    @staticmethod
    def subtract_set(x1, x2):
        return [v1 - v2 for (v1, v2) in zip(x1, x2)]

    @staticmethod
    def mean_set(var_list):
        return [np.mean(v, axis=0) for v in zip(*var_list)]

    @staticmethod
    def scale_set(x1, scale):
        return [scale * x for x in x1]

    @staticmethod
    def _update_weights(old_weights, new_weights, lr):
        return Reptile.add_set(old_weights, Reptile.scale_set(
            Reptile.subtract_set(
                Reptile.mean_set(new_weights), old_weights),
            lr))
