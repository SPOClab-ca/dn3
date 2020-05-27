import torch

# Swap these two for Ipython/Jupyter
import tqdm
# import tqdm.notebook as tqdm

from pandas import DataFrame
from collections import OrderedDict
from torch.utils.data import DataLoader


class BaseProcess(object):

    def __init__(self, optimizer=None, scheduler=None, cuda=False, metrics=None, **kwargs):
        """
        Initialization of the Base Trainable object. Any learning procedure that leverages DN3atasets should subclass
        this base class.

        Parameters
        ----------
        optimizer
        scheduler
        cuda
        """
        self.cuda = cuda
        if isinstance(cuda, bool):
            cuda = "cuda" if cuda else "cpu"
        assert isinstance(cuda, str)
        self.device = torch.device(cuda)
        self.scheduler = scheduler
        self.metrics = OrderedDict()
        if metrics is not None:
            if isinstance(metrics, (list, tuple)):
                metrics = {m.__class__.__name__: m for m in metrics}
            if isinstance(metrics, dict):
                self.add_metrics(metrics)

        _before_members = set(self.__dict__.keys())
        self.build_network(**kwargs)
        new_members = _before_members.difference(self.__dict__.keys())
        for member in new_members:
            if isinstance(self.__dict__[member], torch.nn.Module):
                self.__dict__[member] = self.__dict__[member].to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters()) if optimizer is None else optimizer

    def add_metrics(self, metrics: dict):
        self.metrics.update(**metrics)

    def build_network(self, **kwargs):
        """
        This method is used to add trainable parameters to the trainable. Rather than placing objects for training
        in the __init__ method, they should be placed here.
        """
        raise NotImplementedError

    def parameters(self):
        """
        All the trainable parameters in the Trainable. This includes any architecture parameters and meta-parameters.

        Returns
        -------
        params :
                 An iterator of parameters
        """
        raise NotImplementedError

    def forward(self, *inputs):
        """
        Given a batch of inputs, return the outputs produced by the trainable module.
        Parameters
        ----------
        inputs :
               Tensors needed for underlying module.

        Returns
        -------
        outputs :
                Outputs of module

        """
        raise NotImplementedError

    def calculate_loss(self, intputs, outputs):
        """
        Given the inputs to and outputs from underlying modules, calculate the loss.
        Parameters
        ----------
        Returns
        -------
        Loss :
             Single loss quantity to be minimized.
        """
        raise NotImplementedError

    def calculate_metrics(self, inputs, outputs):
        """
        Given the inputs to and outputs from the underlying module. Return tracked metrics.
        Parameters
        ----------
        inputs :
               Input tensors.
        outputs :
                Output tensors.

        Returns
        -------
        metrics : OrderedDict, None
                  Dictionary of metric quantities.
        """
        metrics = OrderedDict()
        for met_name, met_fn in self.metrics.items():
            metrics[met_name] = met_fn(inputs, outputs)
        return metrics

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

    def train_step(self, *inputs):
        outputs = self.forward(*inputs)
        self.backward(self.calculate_loss(inputs, outputs))

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return self.calculate_metrics(inputs, outputs)

    def evaluate(self, dataset: DataLoader):
        """
        Calculate and return metrics for a dataset

        Parameters
        ----------
        dataset

        Returns
        -------
        metrics : OrderedDict
                Metric scores for the entire
        """
        pbar = tqdm.trange(len(dataset), desc="Iteration")
        data_iterator = iter(dataset)
        metrics = OrderedDict()

        def update_metrics(new_metrics: dict, iterations):
            if len(metrics) == 0:
                return metrics.update(new_metrics)
            else:
                for m in new_metrics:
                    metrics[m] = (metrics[m] * (iterations - 1) + new_metrics[m]) / iterations

        with torch.no_grad():
            for iteration in pbar:
                inputs = next(data_iterator)
                outputs = self.forward(*inputs)
                update_metrics(self.calculate_metrics(inputs, outputs), iteration+1)
                pbar.set_postfix(metrics)

        return metrics

    @classmethod
    def standard_logging(cls, metrics: dict, start_message="End of Epoch"):
        if start_message.rstrip()[-1] != '|':
            start_message = start_message.rstrip() + " |"
        for m in metrics:
            if 'acc' in m or 'pct' in m:
                start_message += " {}: {:.2%} |".format(m, metrics[m])
            else:
                start_message += " {}: {:.2f} |".format(m, metrics[m])
        tqdm.tqdm.write(start_message)


def _check_make_dataloader(dataset, **loader_kwargs):
    # Any args that make more sense as a convenience function to be set
    loader_kwargs.setdefault('shuffle', True)

    if isinstance(dataset, DataLoader):
        return dataset
    return DataLoader(dataset, **loader_kwargs)


class StandardClassification(BaseProcess):

    def __init__(self, classifier: torch.nn.Module, loss_fn=None, cuda=False, metrics=None):
        if isinstance(metrics, dict):
            metrics.setdefault('Accuracy', self._simple_accuracy)
        else:
            metrics = dict(Accuracy=self._simple_accuracy)
        super().__init__(cuda=cuda, classifier=classifier, metrics=metrics)
        self.loss = torch.nn.CrossEntropyLoss().to(self.device) if loss_fn is None else loss_fn.to(self.device)

    @staticmethod
    def _simple_accuracy(inputs, outputs: torch.Tensor):
        return (inputs[-1] == outputs.argmax(dim=-1)).float().mean().item()

    def build_network(self, classifier=None):
        assert classifier is not None
        self.classifier = classifier

    def parameters(self):
        return self.classifier.parameters()

    def train_step(self, *inputs):
        self.classifier.train(True)
        return super(StandardClassification, self).train_step(*inputs)

    def evaluate(self, dataset, **loader_kwargs):
        loader_kwargs.setdefault('batch_size', 1)
        self.classifier.train(False)
        dataset = _check_make_dataloader(dataset)
        return super(StandardClassification, self).evaluate(dataset)

    def forward(self, *inputs):
        return self.classifier(inputs[0])

    def calculate_loss(self, inputs, outputs):
        return self.loss(outputs, inputs[-1])

    def fit(self, training_dataset, epochs=1, validation_dataset=None, step_callback=None,
            epoch_callback=None, batch_size=8, **loader_kwargs):
        """
        sklearn/keras-like convenience method to simply proceed with training across multiple epochs of the provided
        dataset

        Parameters
        ----------
        training_dataset : DN3ataset, DataLoader
        validation_dataset : DN3ataset, DataLoader
        epochs : int
        step_callback : callable
                        Function to run after every training step that has signature: fn(train_metrics) -> None
        epoch_callback : callable
                        Function to run after every epoch that has signature: fn(validation_metrics) -> None
        batch_size : int
                     The batch_size to be used for the training and validation datasets. This is ignored if they are
                     provided as `DataLoader`.
        loader_kwargs :
                      Any remaining keyword arguments will be passed as such to any DataLoaders that are automatically
                      constructed. If both training and validation datasets are provided as `DataLoaders`, this will be
                      ignored.

        Returns
        -------
        train_log : Dataframe
                    Metrics after each iteration of training as a pandas dataframe
        validation_log : Dataframe
                         Validation metrics after each epoch of training as a pandas dataframe
        """
        loader_kwargs.setdefault('batch_size', batch_size)
        training_dataset = _check_make_dataloader(training_dataset, **loader_kwargs)
        # validation_dataset = _check_make_dataloader(validation_dataset, **loader_kwargs)

        def get_batch(iterator):
            return [x.to(self.device) for x in next(iterator)]

        validation_log = list()
        train_log = list()

        epoch_bar = tqdm.trange(1, epochs+1, desc="Epoch")
        for epoch in epoch_bar:
            pbar = tqdm.trange(1, len(training_dataset)+1, desc="Iteration")
            data_iterator = iter(training_dataset)
            for iteration in pbar:
                inputs = get_batch(data_iterator)
                outputs = self.forward(*inputs)
                loss = self.calculate_loss(inputs, outputs)
                self.backward(loss)
                train_metrics = self.calculate_metrics(inputs, outputs)
                train_metrics.setdefault('loss', loss.item())
                pbar.set_postfix(train_metrics)
                train_metrics['epoch'] = epoch
                train_metrics['iteration'] = iteration
                train_metrics['lr'] = self.optimizer.param_groups[0]['lr']
                train_log.append(train_metrics)
                if callable(step_callback):
                    step_callback(train_metrics)

            if validation_dataset is not None:
                val_metrics = self.evaluate(validation_dataset)

                self.standard_logging(val_metrics, "End of Epoch {}".format(epoch))

                val_metrics['epoch'] = epoch
                validation_log.append(val_metrics)
                if callable(epoch_callback):
                    epoch_callback(val_metrics)

        return DataFrame(train_log), DataFrame(validation_log)
