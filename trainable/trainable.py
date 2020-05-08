import torch
import tqdm

from pandas import DataFrame
from collections import OrderedDict
from data.dataset import DNPTDataset


class BaseTrainable(object):

    def __init__(self, optimizer=None, scheduler=None, cuda=False):
        self.cuda = cuda
        self.optimizer = torch.optim.Adam(self.all_trainable_params()) if optimizer is None else optimizer
        self.scheduler = scheduler

    def all_trainable_params(self):
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        metrics : OrderedDict
                  Dictionary of metric quantities.
        """

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

    def evaluate(self, dataset: DNPTDataset):
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
                outputs = self.forward(inputs)
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


class SimpleClassifier(BaseTrainable):

    def __init__(self, classifier: torch.nn.Module, loss_fn=None, cuda=False):
        self.classifier = classifier
        self.loss = torch.nn.CrossEntropyLoss() if loss_fn is None else loss_fn
        # TODO elegant cuda setting
        super().__init__(cuda=cuda)

    def all_trainable_params(self):
        return self.classifier.parameters()

    def train_step(self, *inputs):
        self.classifier.train(True)
        return super(SimpleClassifier, self).train_step(*inputs)

    def evaluate(self, dataset: DNPTDataset):
        self.classifier.train(False)
        return super(SimpleClassifier, self).evaluate(dataset)

    def forward(self, *inputs):
        return self.classifier(inputs[0])

    def calculate_loss(self, inputs, outputs):
        return self.loss(outputs, inputs[-1])

    def fit(self, training_dataset: DNPTDataset, epochs=1, validation_dataset=None, step_callback=None,
            epoch_callback=None):
        """
        sklearn/keras-like convenience method to simply proceed with training across multiple epochs of the provided
        dataset
        Parameters
        ----------
        training_dataset : DNPTDataset
        validation_dataset : DNPTDataset
        epochs : int
        step_callback : callable
                        Function to run after every training step that has signature: fn(train_metrics) -> None
        epoch_callback : callable
                        Function to run after every epoch that has signature: fn(validation_metrics) -> None
        Returns
        -------
        train_log : Dataframe
                    Metrics after each iteration of training as a pandas dataframe
        validation_log : Dataframe
                         Validation metrics after each epoch of training as a pandas dataframe
        """
        def get_batch(iterator):
            if self.cuda:
                return [x.cuda() for x in next(iterator)]
            else:
                return next(iterator)

        validation_log = DataFrame()
        train_log = DataFrame()

        epoch_bar = tqdm.trange(1, epochs+1, desc="Epoch")
        for epoch in epoch_bar:
            pbar = tqdm.trange(1, len(training_dataset)+1, desc="Iteration")
            data_iterator = iter(training_dataset)
            for iteration in pbar:
                inputs = get_batch(data_iterator)
                outputs = self.forward(*inputs)
                train_metrics = self.calculate_metrics(inputs, outputs)
                pbar.set_postfix(train_metrics)
                train_metrics['epoch'] = epoch
                train_metrics['iteration'] = iteration
                train_metrics['lr'] = self.optimizer.param_groups[0]['lr']
                train_log.append(train_metrics)
                if callable(step_callback):
                    step_callback(train_metrics)

            val_metrics = self.evaluate(validation_dataset)

            self.standard_logging(val_metrics, "End of Epoch {}".format(epoch))

            val_metrics['epoch'] = epoch
            validation_log.append(val_metrics)
            if callable(epoch_callback):
                epoch_callback(val_metrics)

        return train_log, validation_log


class DonchinSpeller(BaseTrainable):

    def __init__(self, p300_detector: torch.nn.Module, detector_len: int, aggregator: torch.nn.Module, end_to_end=False,
                 loss_fn=None, cuda=False):
        self.detector = p300_detector
        self.detector_len = detector_len
        self.aggregator = aggregator
        self.loss = torch.nn.CrossEntropyLoss() if loss_fn is None else loss_fn
        # TODO elegant cuda setting
        super().__init__(cuda=cuda)

    def all_trainable_params(self):
        return *self.detector.parameters(), *self.aggregator.parameters()

    def train_step(self, *inputs):
        self.classifier.train(True)
        return super(SimpleClassifier, self).train_step(*inputs)

    def evaluate(self, dataset: DNPTDataset):
        self.classifier.train(False)
        return super(SimpleClassifier, self).evaluate(dataset)

    def forward(self, *inputs):
        return self.classifier(inputs[0])

    def calculate_loss(self, inputs, outputs):
        return self.loss(outputs, inputs[-1])
