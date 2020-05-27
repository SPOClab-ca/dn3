from ..data.dataset import DN3ataset
from .layers import *


class LogRegNetwork(nn.Module):
    def __init__(self, targets, samples, channels):
        super().__init__()
        self.targets = targets
        self.samples = samples
        self.channels = channels
        self.weights = nn.Linear(samples*channels, targets)

    def forward(self, x):
        return self.weights(x.view(x.shape[0], -1))

    @staticmethod
    def from_dataset(dataset: DN3ataset, targets=2):
        """Convenience method creates new TIDNet from a DN3ataset instance"""
        assert isinstance(dataset, DN3ataset)
        return LogRegNetwork(targets, dataset.sequence_length, len(dataset.channels))


class TIDNet(nn.Module):

    def __init__(self, targets, samples, channels, dropout=0.4, activation=nn.ReLU):
        super().__init__()
        self.targets = targets
        self.samples = samples
        self.channels = channels
        self.do = dropout
        self.activation = activation

    @staticmethod
    def from_dataset(dataset: DN3ataset, targets=2):
        """Convenience method creates new TIDNet from a DN3ataset instance"""
        assert isinstance(dataset, DN3ataset)
        return TIDNet(targets, dataset.sequence_length, len(dataset.channels))
