import math

from copy import deepcopy
from ..data.dataset import DN3ataset
from .layers import *


class DN3BaseModel(nn.Module):
    """
    This is a base model used by the provided models in the library more out of convenience than anything.
    It is not strictly necessary to have new modules inherit from this, any nn.Module should suffice, but it provides
    some integrated conveniences...
    """
    def __init__(self, targets, samples, channels):
        super().__init__()
        self.targets = targets
        self.samples = samples
        self.channels = channels
        self.make_new_classification_layer()

    @property
    def num_features_for_classification(self):
        raise NotImplementedError

    def clone(self):
        """
        This provides a standard way to copy models, weights and all.
        """
        return deepcopy(self)

    def freeze_features(self, unfreeze=False):
        """
        In many cases, the features learned by a model in one domain can be applied to another case.

        This method freezes (or un-freezes) all but the `classifier` layer. So that any further training does not (or
        does if unfreeze=True) affect these weights.

        Parameters
        ----------
        unfreeze : bool
                   To unfreeze weights after a previous call to this.
        """
        for param in self.parameters():
            param.requires_grad = unfreeze
        for param in self.classifier.parameters():
            param.requires_grad = True

    def make_new_classification_layer(self):
        """
        This allows for a distinction between the classification layer(s) and the rest of the network. Using a basic
        formulation of a network being composed of two parts feature_extractor & classifier.

        This method is for implementing the classification side, so that methods like :py:meth:`freeze_features` works
        as intended.

        Anything besides a layer that just flattens anything incoming to a vector and Linearly weights this to the 
        target should override this method, and there should be a variable called `self.classifier`

        """
        classifier = nn.Linear(self.num_features_for_classification, self.targets)
        classifier.weight.data.normal_(std=0.02)
        classifier.bias.data.zero_()
        self.classifier = nn.Sequential(Flatten(), classifier)

    def forward(self, x):
        raise NotImplementedError

    @classmethod
    def from_dataset(cls, dataset: DN3ataset, targets):
        assert isinstance(dataset, DN3ataset)
        return cls(targets, dataset.sequence_length, len(dataset.channels))


class LogRegNetwork(DN3BaseModel):
    """
    In effect, simply an implementation of linear kernel (multi)logistic regression
    """
    def __init__(self, targets, samples, channels):
        super().__init__(targets, samples, channels)

    @property
    def num_features_for_classification(self):
        return self.samples * self.channels

    def forward(self, x):
        return self.classifier(x)


class TIDNet(DN3BaseModel):
    """
    The Thinker Invariant Densenet from Kostas & Rudzicz 2020 (under review).

    This alone is not strictly "thinker invariant", but it for the most part performs as well as EEGNet, or better
    when there is abundant data.
    """

    def __init__(self, targets, samples, channels, s_growth=24, t_filters=32, do=0.4, pooling=20,
                 temp_layers=2, spat_layers=2, temp_span=0.05, bottleneck=3, summary=-1):
        self.temp_len = math.ceil(temp_span * samples)
        summary = samples // pooling if summary == -1 else summary
        self._num_features = (t_filters + s_growth * spat_layers) * summary
        super().__init__(targets, samples, channels)

        self.temporal = nn.Sequential(
            Expand(axis=1),
            TemporalFilter(1, t_filters, depth=temp_layers, temp_len=self.temp_len),
            nn.MaxPool2d((1, pooling)),
            nn.Dropout2d(do),
        )

        self.spatial = DenseSpatialFilter(self.channels, s_growth, spat_layers, in_ch=t_filters, dropout_rate=do,
                                          bottleneck=bottleneck)
        self.extract_features = nn.Sequential(
            nn.AdaptiveAvgPool1d(int(summary)),
        )

    @property
    def num_features_for_classification(self):
        return self._num_features

    def forward(self, x, **kwargs):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.extract_features(x)

        return self.classifier(x)


class EEGNet(DN3BaseModel):
    """
    This is the DN3 re-implementation of Lawhern et. al.'s EEGNet from:
    https://iopscience.iop.org/article/10.1088/1741-2552/aace8c

    Notes
    -----
    The implementation below is in no way officially sanctioned by the original authors, and in fact is  missing the
    constraints the original authors have on the convolution kernels, and may or may not be missing more...

    That being said, in *our own personal experience*, this implementation has fared no worse when compared to
    implementations that include this constraint (albeit, those were *also not written* by the original authors).
    """

    def __init__(self, targets, channels, samples, do=0.25, pooling=8, F1=8, D=2, t_len=65, F2=16):
        super().__init__(targets, channels, samples)

        self.init_conv = nn.Sequential(
            Expand(1),
            nn.Conv2d(1, F1, (1, t_len), padding=(0, t_len // 2), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(F1, D * F1, (channels, 1), bias=False, groups=F1),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, pooling // 2)),
            nn.Dropout(do)
        )
        samples = samples // (pooling // 2)

        self.sep_conv = nn.Sequential(
            # Separate into two convs, one that doesnt operate across filters, one isolated to filters
            nn.Conv2d(D*F1, D*F1, (1, 17), bias=False, padding=(0, 8), groups=D*F1),
            nn.Conv2d(D*F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling)),
            nn.Dropout(do)
        )
        samples = samples // pooling

        self._num_features = F2 * samples

    @property
    def num_features_for_classification(self):
        return self._num_features

    def forward(self, x):
        x = self.init_conv(x)
        x = self.depth_conv(x)
        x = self.sep_conv(x)

        return self.classifier(x)
