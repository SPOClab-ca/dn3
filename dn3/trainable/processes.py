import re
from sys import gettrace

from dn3.trainable.models import DN3BaseModel

# Swap these two for Ipython/Jupyter
import tqdm
# import tqdm.notebook as tqdm

import torch
from pandas import DataFrame
from collections import OrderedDict
from torch.utils.data import DataLoader


class BaseProcess(object):

    def __init__(self, lr=0.001, warmup=None, l2_weight_decay=0.001, cuda=None, metrics=None,
                 **kwargs):
        """
        Initialization of the Base Trainable object. Any learning procedure that leverages DN3atasets should subclass
        this base class.

        Parameters
        ----------
        cuda : bool, string, None
               If boolean, sets whether to enable training on the GPU, if a string, specifies can be used to specify
               which device to use. If None (default) figures it out automatically.
        metrics : dict, list
                  A dictionary of named (keys) metrics (values) or some iterable set of metrics that will be identified
                  by their class names.
        """
        if cuda is None:
            cuda = torch.cuda.is_available()
            if cuda:
                tqdm.tqdm.write("GPU(s) detected: training and model execution will be performed on GPU.")
        if isinstance(cuda, bool):
            cuda = "cuda" if cuda else "cpu"
        assert isinstance(cuda, str)
        self.cuda = cuda
        self.device = torch.device(cuda)
        self.metrics = OrderedDict()
        if metrics is not None:
            if isinstance(metrics, (list, tuple)):
                metrics = {m.__class__.__name__: m for m in metrics}
            if isinstance(metrics, dict):
                self.add_metrics(metrics)

        _before_members = set(self.__dict__.keys())
        self.build_network(**kwargs)
        new_members = set(self.__dict__.keys()).difference(_before_members)
        for member in new_members:
            if isinstance(self.__dict__[member], (torch.nn.Module, torch.Tensor)):
                self.__dict__[member] = self.__dict__[member].to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=l2_weight_decay)
        self.scheduler = None

    def set_optimizer(self, optimizer):
        del self.optimizer
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def add_metrics(self, metrics: dict):
        self.metrics.update(**metrics)

    def _optimize_dataloader_kwargs(self, **loader_kwargs):
        loader_kwargs.setdefault('pin_memory', self.cuda == 'cuda')
        # Use multiple worker processes when NOT DEBUGGING
        if gettrace() is None:
            # Find number of cpus available (taken from second answer):
            # https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python
            m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                          open('/proc/self/status').read())
            nw = bin(int(m.group(1).replace(',', ''), 16)).count('1')
        else:
            # 0 workers means not extra processes are spun up
            nw = 2
        loader_kwargs.setdefault('num_workers', int(nw - 2))
        return loader_kwargs

    def _get_batch(self, iterator):
        return [x.to(self.device) for x in next(iterator)]

    def build_network(self, **kwargs):
        """
        This method is used to add trainable modules to the process. Rather than placing objects for training
        in the __init__ method, they should be placed here.

        By default any arguments that propagate unused from __init__ are included here.
        """
        self.__dict__.update(**kwargs)

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
        loss = self.calculate_loss(inputs, outputs)
        self.backward(loss)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        train_metrics = self.calculate_metrics(inputs, outputs)
        train_metrics.setdefault('loss', loss.item())

        return train_metrics

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
        num_points = 0
        pbar = tqdm.trange(len(dataset), desc="Iteration")
        data_iterator = iter(dataset)
        metrics = OrderedDict()

        def update_metrics(new_metrics: dict, batch_size):
            if len(metrics) == 0:
                return metrics.update(new_metrics)
            else:
                for m in new_metrics:
                    metrics[m] = (metrics[m] * (num_points - batch_size) + new_metrics[m] * batch_size) / num_points

        with torch.no_grad():
            for iteration in pbar:
                inputs = self._get_batch(data_iterator)
                bs = inputs[0].shape[0]
                outputs = self.forward(*inputs)
                calc_metrics = self.calculate_metrics(inputs, outputs)
                calc_metrics['loss'] = self.calculate_loss(inputs, outputs).item()
                num_points += bs
                update_metrics(calc_metrics, bs)
                pbar.set_postfix(metrics)

        return metrics

    @classmethod
    def standard_logging(cls, metrics: dict, start_message="End of Epoch"):
        if start_message.rstrip()[-1] != '|':
            start_message = start_message.rstrip() + " |"
        for m in metrics:
            if 'acc' in m.lower() or 'pct' in m.lower():
                start_message += " {}: {:.2%} |".format(m, metrics[m])
            else:
                start_message += " {}: {:.2f} |".format(m, metrics[m])
        tqdm.tqdm.write(start_message)


def _check_make_dataloader(dataset, **loader_kwargs):
    # Any args that make more sense as a convenience function to be set
    loader_kwargs.setdefault('shuffle', True)
    loader_kwargs.setdefault('drop_last', True)

    if isinstance(dataset, DataLoader):
        return dataset
    return DataLoader(dataset, **loader_kwargs)


class StandardClassification(BaseProcess):

    def __init__(self, classifier: torch.nn.Module, loss_fn=None, cuda=False, metrics=None, learning_rate=None,
                 **kwargs):
        if isinstance(metrics, dict):
            metrics.setdefault('Accuracy', self._simple_accuracy)
        else:
            metrics = dict(Accuracy=self._simple_accuracy)
        super().__init__(cuda=cuda, classifier=classifier, metrics=metrics, **kwargs)
        if isinstance(learning_rate, float):
            # fixme hardcoded weight-decay
            self.set_optimizer(torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.001))
        self.loss = torch.nn.CrossEntropyLoss().to(self.device) if loss_fn is None else loss_fn.to(self.device)

    @staticmethod
    def _simple_accuracy(inputs, outputs: torch.Tensor):
        return (inputs[-1] == outputs.argmax(dim=-1)).float().mean().item()

    def _easy_scheduler(self, scheduler_string: str, epochs, total_iterations):
        if scheduler_string.lower() == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, total_iterations)
        if scheduler_string.lower() == 'cosine_with_restarts':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, )

    def parameters(self):
        return self.classifier.parameters()

    def train_step(self, *inputs):
        self.classifier.train(True)
        return super(StandardClassification, self).train_step(*inputs)

    def evaluate(self, dataset, **loader_kwargs):
        loader_kwargs.setdefault('batch_size', 1)
        self.classifier.train(False)
        loader_kwargs['drop_last'] = False
        dataset = _check_make_dataloader(dataset)
        return super(StandardClassification, self).evaluate(dataset)

    def forward(self, *inputs):
        if isinstance(self.classifier, DN3BaseModel):
            prediction, _ = self.classifier(inputs[0])
        else:
            prediction = self.classifier(inputs[0])
        return prediction

    def calculate_loss(self, inputs, outputs):
        return self.loss(outputs, inputs[-1])

    def fit(self, training_dataset, epochs=1, validation_dataset=None, step_callback=None,
            epoch_callback=None, batch_size=8, scheduler=None, **loader_kwargs):
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

        Notes
        -----
        If the datasets above are provided as DN3atasets, automatic optimizations are performed to speed up loading.
        These include setting the number of workers = to the number of CPUs/system threads - 1, and pinning memory for
        rapid CUDA transfer if leveraging the GPU. Unless you are very comfortable with PyTorch, it's probably better
        to not provide your own DataLoader, and let this be done automatically.

        Returns
        -------
        train_log : Dataframe
                    Metrics after each iteration of training as a pandas dataframe
        validation_log : Dataframe
                         Validation metrics after each epoch of training as a pandas dataframe
        """
        loader_kwargs.setdefault('batch_size', batch_size)
        loader_kwargs = self._optimize_dataloader_kwargs(**loader_kwargs)
        training_dataset = _check_make_dataloader(training_dataset, **loader_kwargs)
        # validation_dataset = _check_make_dataloader(validation_dataset, **loader_kwargs)

        if scheduler is None:
            scheduler = self.scheduler

        validation_log = list()
        train_log = list()

        metrics = OrderedDict()

        def update_metrics(new_metrics: dict, iterations):
            if len(metrics) == 0:
                return metrics.update(new_metrics)
            else:
                for m in new_metrics:
                    metrics[m] = (metrics[m] * (iterations - 1) + new_metrics[m]) / iterations

        epoch_bar = tqdm.trange(1, epochs+1, desc="Epoch")
        for epoch in epoch_bar:
            pbar = tqdm.trange(1, len(training_dataset)+1, desc="Iteration")
            data_iterator = iter(training_dataset)
            for iteration in pbar:
                inputs = self._get_batch(data_iterator)
                train_metrics = self.train_step(*inputs)
                train_metrics['lr'] = self.optimizer.param_groups[0]['lr']
                update_metrics(train_metrics, iteration+1)
                pbar.set_postfix(metrics)
                train_metrics['epoch'] = epoch
                train_metrics['iteration'] = iteration
                train_log.append(train_metrics)
                if callable(step_callback):
                    step_callback(train_metrics)

            if validation_dataset is not None:
                val_metrics = self.evaluate(validation_dataset, **loader_kwargs)

                self.standard_logging(val_metrics, "End of Epoch {}".format(epoch))

                val_metrics['epoch'] = epoch
                validation_log.append(val_metrics)
                if callable(epoch_callback):
                    epoch_callback(val_metrics)

        return DataFrame(train_log), DataFrame(validation_log)
