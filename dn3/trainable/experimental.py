import torch
from dn3.data.dataset import DN3ataset
from .trainable import BaseTrainable


class DonchinSpeller(BaseTrainable):

    def __init__(self, p300_detector: torch.nn.Module, detector_len: int, aggregator: torch.nn.Module, end_to_end=False,
                 loss_fn=None, cuda=False):
        self.detector = p300_detector
        self.detector_len = detector_len
        self.aggregator = aggregator
        self.loss = torch.nn.CrossEntropyLoss() if loss_fn is None else loss_fn
        super().__init__(cuda=cuda)

    def parameters(self):
        return *self.detector.parameters(), *self.aggregator.parameters()

    def train_step(self, *inputs):
        self.classifier.train(True)
        return super(BaseTrainable, self).train_step(*inputs)

    def evaluate(self, dataset: DN3ataset):
        self.classifier.train(False)
        return super(BaseTrainable, self).evaluate(dataset)

    def forward(self, *inputs):
        return self.classifier(inputs[0])

    def calculate_loss(self, inputs, outputs):
        return self.loss(outputs, inputs[-1])