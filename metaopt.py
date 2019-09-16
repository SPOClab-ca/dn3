import tensorflow as tf
import tensorflow.python.keras as keras
from metamodel import MetaModel

class MetaOptimizer:
    """
    Interface for all meta optimizers so that they behave nicely with meta models.
    """
    def train(self, *args, **kwargs):
        raise NotImplementedError()

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError()


class Reptile(MetaOptimizer):
    def train(self, loader_dict, model, inner_batch, minimize_op=None):
        """
        
        Parameters
        ----------
        loader_dict
        model
        inner_batch
        minimize_op

        Returns
        -------

        """
        pass

    def evaluate(self):
        pass
