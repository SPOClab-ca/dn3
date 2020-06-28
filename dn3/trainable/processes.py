import re
from sys import gettrace

from dn3.utils import LabelSmoothedCrossEntropyLoss
from dn3.trainable.models import DN3BaseModel, Classifier

# Swap these two for Ipython/Jupyter
import tqdm
# import tqdm.notebook as tqdm

import torch
import numpy as np
from pandas import DataFrame
from collections import OrderedDict
from torch.utils.data import DataLoader, WeightedRandomSampler


class BaseProcess(object):

    def __init__(self, lr=0.001, metrics=None, l2_weight_decay=0.01, cuda=None, **kwargs):
        """
        Initialization of the Base Trainable object. Any learning procedure that leverages DN3atasets should subclass
        this base class.

        By default uses the SGD with momentum optimization.

        Parameters
        ----------
        cuda : bool, string, None
               If boolean, sets whether to enable training on the GPU, if a string, specifies can be used to specify
               which device to use. If None (default) figures it out automatically.
        lr : float
             The learning rate to use, this will probably something that should be tuned for each application.
             Start with multiplying or dividing by values of 2, 5 or 10 to seek out a good number.
        metrics : dict, list
                  A dictionary of named (keys) metrics (values) or some iterable set of metrics that will be identified
                  by their class names.
        l2_weight_decay : float
                          One of the simplest and most common regularizing techniques. If you find a model rapidly
                          reaching high training accuracy (and not validation) increase this. If having trouble fitting
                          the training data, decrease this.
        kwargs : dict
                 Arguments that will be used by the processes' :py:meth:`BaseProcess.build_network()` method.
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
        self._training = False
        self._trainables = list()
        for member in new_members:
            if isinstance(self.__dict__[member], (torch.nn.Module, torch.Tensor)):
                if not (isinstance(self.__dict__[member], torch.Tensor) and not self.__dict__[member].requires_grad):
                    self._trainables.append(member)
                self.__dict__[member] = self.__dict__[member].to(self.device)

        self.optimizer = torch.optim.SGD(self.parameters(), weight_decay=l2_weight_decay, lr=lr, nesterov=True,
                                         momentum=0.9)
        self.scheduler = None
        self.lr = lr
        self.weight_decay = l2_weight_decay

    def set_optimizer(self, optimizer):
        assert isinstance(optimizer, torch.optim.Optimizer)
        del self.optimizer
        self.optimizer = optimizer

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def add_metrics(self, metrics: dict):
        self.metrics.update(**metrics)

    def _optimize_dataloader_kwargs(self, num_worker_cap=6, **loader_kwargs):
        loader_kwargs.setdefault('pin_memory', self.cuda == 'cuda')
        # Use multiple worker processes when NOT DEBUGGING
        if gettrace() is None:
            # Find number of cpus available (taken from second answer):
            # https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python
            m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$',
                          open('/proc/self/status').read())
            nw = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            # Cap the number of workers at 6 (actually 4) to avoid pummeling disks too hard
            nw = min(num_worker_cap, nw)
        else:
            # 0 workers means not extra processes are spun up
            nw = 2
        loader_kwargs.setdefault('num_workers', int(nw - 2))
        print("Loading data with {} additional workers".format(loader_kwargs['num_workers']))
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
        for member in self._trainables:
            yield from self.__dict__[member].parameters()

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

    def calculate_loss(self, inputs, outputs):
        """
        Given the inputs to and outputs from underlying modules, calculate the loss.

        Returns
        -------
        Loss :
             Single loss quantity to be minimized.
        """
        loss_fn = self.loss.to('cpu')
        return loss_fn(outputs, inputs[-1])

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
            try:
                metrics[met_name] = met_fn(inputs, outputs)
            # I know its super broad, but basically if metrics fail during training, I want to just ignore them...
            except:
                continue
        return metrics

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()

    def train(self, mode=True):
        self._training = mode
        for member in self._trainables:
            self.__dict__[member].train(mode=mode)

    def train_step(self, *inputs):
        self.train(True)
        outputs = self.forward(*inputs)
        loss = self.calculate_loss(inputs, outputs)
        self.backward(loss)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        train_metrics = self.calculate_metrics(inputs, outputs)
        train_metrics.setdefault('loss', loss.item())

        return train_metrics

    def evaluate(self, dataset, **loader_kwargs):
        """
        Calculate and return metrics for a dataset

        Parameters
        ----------
        dataset: DN3ataset, DataLoader
                 The dataset that will be used for evaluation, if not a DataLoader, one will be constructed
        loader_kwargs: dict
                       Args that will be passed to the dataloader, but `shuffle` and `drop_last` will be both be
                       forced to `False`

        Returns
        -------
        metrics : OrderedDict
                Metric scores for the entire
        """
        inputs, outputs = self.predict(dataset, **loader_kwargs)
        metrics = self.calculate_metrics(inputs, outputs)
        metrics['loss'] = self.calculate_loss(inputs, outputs).item()
        return metrics

    def predict(self, dataset, **loader_kwargs):
        """
        Determine the outputs for all loaded data from the dataset

        Parameters
        ----------
        dataset: DN3ataset, DataLoader
                 The dataset that will be used for evaluation, if not a DataLoader, one will be constructed
        loader_kwargs: dict
                       Args that will be passed to the dataloader, but `shuffle` and `drop_last` will be both be
                       forced to `False`

        Returns
        -------
        inputs : Tensor
                 The exact inputs used to calculate the outputs (in case they were stochastic and need saving)
        outputs : Tensor
                  The outputs from each run of :function:`forward`
        """
        self.train(False)
        loader_kwargs.setdefault('batch_size', 1)
        dataset = self._check_make_dataloader(dataset, training=False, **loader_kwargs)

        pbar = tqdm.trange(len(dataset), desc="Predicting")
        data_iterator = iter(dataset)

        inputs = list()
        outputs = list()

        with torch.no_grad():
            for iteration in pbar:
                input_batch = self._get_batch(data_iterator)
                output_batch = self.forward(*input_batch)

                inputs.append([tensor.cpu() for tensor in input_batch])
                if isinstance(output_batch, torch.Tensor):
                    outputs.append(output_batch)
                else:
                    outputs.append([tensor.cpu() for tensor in output_batch])

        def package_multiple_tensors(batches: list):
            if isinstance(batches[0], torch.Tensor):
                return torch.cat(batches)
            elif isinstance(batches[0], (tuple, list)):
                return [torch.cat(b) for b in zip(*batches)]

        return package_multiple_tensors(inputs), package_multiple_tensors(outputs)

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

    def save_best(self):
        """
        Create a snapshot of what is being currently trained for re-laoding with the :py:meth:`load_best()` method.

        Returns
        -------
        best : Any
               Whatever format is needed for :py:meth:`load_best()`, will be the argument provided to it.
        """
        return [self.__dict__[m].state_dict() for m in self._trainables]

    def load_best(self, best):
        """
        Load the parameters as saved by :py:meth:`save_best()`.

        Parameters
        ----------
        best: Any
        """
        for m, state_dict in zip(self._trainables, best):
            self.__dict__[m].load_state_dict(state_dict)

    def _retain_best(self, old_checkpoint, metrics_to_check: dict, retain_string: str):
        if retain_string is None:
            return old_checkpoint
        best_checkpoint = old_checkpoint

        def found_best():
            tqdm.tqdm.write("Best {}. Retaining checkpoint...".format(retain_string))
            self.best_metric = metrics_to_check[retain_string]
            return self.save_best()

        if retain_string not in metrics_to_check.keys():
            tqdm.tqdm.write("No metric {} found in recorded metrics. Not saving best.")
        if self.best_metric is None:
            best_checkpoint = found_best()
        elif retain_string == 'loss' and metrics_to_check[retain_string] <= self.best_metric:
            best_checkpoint = found_best()
        elif retain_string != 'loss' and metrics_to_check[retain_string] >= self.best_metric:
            best_checkpoint = found_best()

        return best_checkpoint

    @staticmethod
    def _check_make_dataloader(dataset, training=False, **loader_kwargs):
        """Any args that make more sense as a convenience function to be set"""
        if isinstance(dataset, DataLoader):
            return dataset

        # Only shuffle and drop last when training
        loader_kwargs.setdefault('shuffle', training)
        loader_kwargs.setdefault('drop_last', training)

        return DataLoader(dataset, **loader_kwargs)

    def fit(self, training_dataset, epochs=1, validation_dataset=None, step_callback=None,
            epoch_callback=None, batch_size=8, warmup_frac=0.2, retain_best='loss', **loader_kwargs):
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
        warmup_frac : float
                      The fraction of iterations that will be spent *increasing* the learning rate under the default
                      1cycle policy (with cosine annealing). Value will be automatically clamped values between [0, 0.5]
        retain_best : (str, None)
                      **If `validation_dataset` is provided**, which model weights to retain. If 'loss' (default), will
                      retain the model at the epoch with the lowest validation loss. If another string, will assume that
                      is the metric to monitor for the *highest score*. If None, the final model is used.
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
        training_dataset = self._check_make_dataloader(training_dataset, training=True, **loader_kwargs)
        # validation_dataset = _check_make_dataloader(validation_dataset, **loader_kwargs)

        _clear_scheduler_after = self.scheduler is None
        if _clear_scheduler_after:
            self.set_scheduler(
                torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.lr, epochs=epochs,
                                                    steps_per_epoch=len(training_dataset),
                                                    pct_start=warmup_frac)
            )

        validation_log = list()
        train_log = list()
        self.best_metric = None
        best_model = self.save_best()

        metrics = OrderedDict()

        def update_metrics(new_metrics: dict, iterations):
            if len(metrics) == 0:
                return metrics.update(new_metrics)
            else:
                for m in new_metrics:
                    try:
                        metrics[m] = (metrics[m] * (iterations - 1) + new_metrics[m]) / iterations
                    except KeyError:
                        metrics[m] = new_metrics[m]

        epoch_bar = tqdm.trange(1, epochs+1, desc="Epoch", unit='epoch')
        for epoch in epoch_bar:
            pbar = tqdm.trange(1, len(training_dataset)+1, desc="Iteration", unit='batches')
            data_iterator = iter(training_dataset)
            for iteration in pbar:
                inputs = self._get_batch(data_iterator)
                train_metrics = self.train_step(*inputs)
                train_metrics['lr'] = self.optimizer.param_groups[0]['lr']
                if 'momentum' in self.optimizer.defaults:
                    train_metrics['momentum'] = self.optimizer.param_groups[0]['momentum']
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
                best_model = self._retain_best(best_model, val_metrics, retain_best)

                val_metrics['epoch'] = epoch
                validation_log.append(val_metrics)
                if callable(epoch_callback):
                    epoch_callback(val_metrics)

        if _clear_scheduler_after:
            self.set_scheduler(None)

        if retain_best is not None and validation_dataset is not None:
            tqdm.tqdm.write("Loading best model...")
            self.load_best(best_model)

        return DataFrame(train_log), DataFrame(validation_log)


class StandardClassification(BaseProcess):

    def __init__(self, classifier: torch.nn.Module, loss_fn=None, cuda=None, metrics=None, learning_rate=0.01,
                 label_smoothing=None, **kwargs):
        if isinstance(metrics, dict):
            metrics.setdefault('Accuracy', self._simple_accuracy)
        else:
            metrics = dict(Accuracy=self._simple_accuracy)
        super(StandardClassification, self).__init__(cuda=cuda, lr=learning_rate, classifier=classifier,
                                                     metrics=metrics, **kwargs)
        if label_smoothing is not None and isinstance(label_smoothing, float) and (0 < label_smoothing < 1):
            self.loss = LabelSmoothedCrossEntropyLoss(self.classifier.targets, smoothing=label_smoothing).\
                to(self.device)
        elif loss_fn is None:
            self.loss = torch.nn.CrossEntropyLoss().to(self.device)
        else:
            self.loss = loss_fn.to(self.device)
        self.best_metric = None

    @staticmethod
    def _simple_accuracy(inputs, outputs: torch.Tensor):
        return (inputs[-1] == outputs.argmax(dim=-1)).float().mean().item()

    def forward(self, *inputs):
        if isinstance(self.classifier, Classifier) and self.classifier.return_features:
            prediction, _ = self.classifier(inputs[0])
        else:
            prediction = self.classifier(inputs[0])
        return prediction

    def fit(self, training_dataset, epochs=1, validation_dataset=None, step_callback=None, epoch_callback=None,
            batch_size=8, warmup_frac=0.2, retain_best='loss', balance_method='undersample', **loader_kwargs):
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
        warmup_frac : float
                      The fraction of iterations that will be spent *increasing* the learning rate under the default
                      1cycle policy (with cosine annealing). Value will be automatically clamped values between [0, 0.5]
        retain_best : (str, None)
                      **If `validation_dataset` is provided**, which model weights to retain. If 'loss' (default), will
                      retain the model at the epoch with the lowest validation loss. If another string, will assume that
                      is the metric to monitor for the *highest score*. If None, the final model is used.
        balance_method : (None, str)
                         If and how to balance training samples when training. `None` will simply randomly
                         sample all training samples equally. 'undersample' (default) will sample each class N_min times
                         where N_min is equal to the number of examples in the minority class. 'oversample' will sample
                         each class N_max times, where N_max is the number of the majority class.
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
        return super(StandardClassification, self).fit(training_dataset, epochs=epochs, step_callback=step_callback,
                                                       epoch_callback=epoch_callback, batch_size=batch_size,
                                                       warmup_frac=warmup_frac, retain_best=retain_best,
                                                       validation_dataset=validation_dataset,
                                                       balance_method=balance_method,
                                                       **loader_kwargs)

    @staticmethod
    def _check_make_dataloader(dataset, training=False, **loader_kwargs):
        if isinstance(dataset, DataLoader):
            return dataset

        # Only shuffle and drop last when training
        loader_kwargs.setdefault('drop_last', training)

        if training and loader_kwargs.get('sampler', None) is None and loader_kwargs.get('balance_method', None) \
                is not None:
            method = loader_kwargs.pop('balance_method')
            assert method.lower() in ['undersample', 'oversample']
            if not hasattr(dataset, 'get_targets'):
                print("Failed to create dataloader with {} balancing. {} does not support `get_targets()`.".format(
                    method, dataset.__class__.__name__
                ))
            else:
                sampler = balanced_undersampling(dataset) if method.lower() == 'undersample' \
                    else balanced_oversampling(dataset)
                return DataLoader(dataset, sampler=sampler, **loader_kwargs)

        # Make sure balance method doesn't not passed to DataLoader at this point.
        loader_kwargs.pop('balance_method', None)

        # Shuffle if not already specifying weighted sampler
        loader_kwargs.setdefault('shuffle', training)

        return DataLoader(dataset, **loader_kwargs)


def _get_label_balance(dataset):
    labels = dataset.get_targets()
    counts = np.bincount(labels)
    train_weights = 1. / torch.tensor(counts, dtype=torch.float)
    sample_weights = train_weights[labels]
    tqdm.tqdm.write('Class frequency: {}'.format(counts/counts.sum()))
    return sample_weights, counts


def balanced_undersampling(dataset, replacement=False):
    sample_weights, counts = _get_label_balance(dataset)
    return WeightedRandomSampler(sample_weights, len(counts) * int(counts.min()), replacement=replacement)


def balanced_oversampling(dataset, replacement=True):
    sample_weights, counts = _get_label_balance(dataset)
    return WeightedRandomSampler(sample_weights, len(counts) * int(counts.max()), replacement=replacement)
