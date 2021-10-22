import nntplib
from abc import ABCMeta

import math

from copy import deepcopy

import numpy as np

from ..data.dataset import DN3ataset
from .layers import *


class DN3BaseModel(nn.Module):
    """
    This is a base model used by the provided models in the library that is meant to make those included in this
    library as powerful and multi-purpose as is reasonable.

    It is not strictly necessary to have new modules inherit from this, any nn.Module should suffice, but it provides
    some integrated conveniences...

    The premise of this model is that deep learning models can be understood as *learned pipelines*. These
    :any:`DN3BaseModel` objects, are re-interpreted as a two-stage pipeline, the two stages being *feature extraction*
    and *classification*.
    """
    def __init__(self, samples, channels, return_features=True):
        super().__init__()
        self.samples = samples
        self.channels = channels
        self.return_features = return_features

    def forward(self, x):
        raise NotImplementedError

    def internal_loss(self, forward_pass_tensors):

        return None

    def clone(self):
        """
        This provides a standard way to copy models, weights and all.
        """
        return deepcopy(self)

    def load(self, filename, strict=True):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze

    @classmethod
    def from_dataset(cls, dataset: DN3ataset, **modelargs):
        print("Creating {} using: {} channels with trials of {} samples at {}Hz".format(cls.__name__,
                                                                                        len(dataset.channels),
                                                                                        dataset.sequence_length,
                                                                                        dataset.sfreq))
        assert isinstance(dataset, DN3ataset)
        return cls(samples=dataset.sequence_length, channels=len(dataset.channels), **modelargs)


class Classifier(DN3BaseModel):
    """
    A generic Classifer container. This container breaks operations up into feature extraction and feature
    classification to enable convenience in transfer learning and more.
    """

    @classmethod
    def from_dataset(cls, dataset: DN3ataset, **modelargs):
        """
        Create a classifier from a dataset.

        Parameters
        ----------
        dataset
        modelargs: dict
                   Options to construct the dataset, if dataset does not have listed targets, targets must be specified
                   in the keyword arguments or will fall back to 2.

        Returns
        -------
        model: Classifier
               A new `Classifier` ready to classifiy data from `dataset`
        """
        if hasattr(dataset, 'get_targets'):
            targets = len(np.unique(dataset.get_targets()))
        elif dataset.info is not None and isinstance(dataset.info.targets, int):
            targets = dataset.info.targets
        else:
            targets = 2
        modelargs.setdefault('targets', targets)
        print("Creating {} using: {} channels x {} samples at {}Hz | {} targets".format(cls.__name__,
                                                                                        len(dataset.channels),
                                                                                        dataset.sequence_length,
                                                                                        dataset.sfreq,
                                                                                        modelargs['targets']))
        assert isinstance(dataset, DN3ataset)
        return cls(samples=dataset.sequence_length, channels=len(dataset.channels), **modelargs)

    def __init__(self, targets, samples, channels, return_features=True):
        super(Classifier, self).__init__(samples, channels, return_features=return_features)
        self.targets = targets
        self.make_new_classification_layer()
        self._init_state = self.state_dict()

    def reset(self):
        self.load_state_dict(self._init_state)

    def forward(self, *x):
        features = self.features_forward(*x)
        if self.return_features:
            return self.classifier_forward(features), features
        else:
            return self.classifier_forward(features)

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
        nn.init.xavier_normal_(classifier.weight)
        classifier.bias.data.zero_()
        self.classifier = nn.Sequential(Flatten(), classifier)

    def freeze_features(self, unfreeze=False, freeze_classifier=False):
        """
        In many cases, the features learned by a model in one domain can be applied to another case.

        This method freezes (or un-freezes) all but the `classifier` layer. So that any further training does not (or
        does if unfreeze=True) affect these weights.

        Parameters
        ----------
        unfreeze : bool
                   To unfreeze weights after a previous call to this.
        freeze_classifier: bool
                   Commonly, the classifier layer will not be frozen (default). Setting this to `True` will freeze this
                   layer too.
        """
        super(Classifier, self).freeze_features(unfreeze=unfreeze)

        if isinstance(self.classifier, nn.Module) and not freeze_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = True

    @property
    def num_features_for_classification(self):
        raise NotImplementedError

    def classifier_forward(self, features):
        return self.classifier(features)

    def features_forward(self, x):
        raise NotImplementedError

    def load(self, filename, include_classifier=False, freeze_features=True):
        state_dict = torch.load(filename)
        if not include_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=False)
        if freeze_features:
            self.freeze_features()

    def save(self, filename, ignore_classifier=False):
        state_dict = self.state_dict()
        if ignore_classifier:
            for key in [k for k in state_dict.keys() if 'classifier' in k]:
                state_dict.pop(key)
        print("Saving to {} ...".format(filename))
        torch.save(state_dict, filename)


class StrideClassifier(Classifier, metaclass=ABCMeta):

    def __init__(self, targets, samples, channels, stride_width=2, return_features=False):
        """
        Instead of summarizing the entire temporal dimension into a single prediction, a prediction kernel is swept over
        the final sequence representation and generates predictions at each step.

        Parameters
        ----------
        targets
        samples
        channels
        stride_width
        return_features
        """
        self.stride_width = stride_width
        super(StrideClassifier, self).__init__(targets, samples, channels, return_features=return_features)

    def make_new_classification_layer(self):
        self.classifier = torch.nn.Conv1d(self.num_features_for_classification, self.targets,
                                          kernel_size=self.stride_width)
        torch.nn.init.xavier_normal_(self.classifier.weight)
        self.classifier.bias.data.zero_()


class LogRegNetwork(Classifier):
    """
    In effect, simply an implementation of linear kernel (multi)logistic regression
    """
    def features_forward(self, x):
        return x

    @property
    def num_features_for_classification(self):
        return self.samples * self.channels


class TIDNet(Classifier):
    """
    The Thinker Invariant Densenet from Kostas & Rudzicz 2020, https://doi.org/10.1088/1741-2552/abb7a7

    This alone is not strictly "thinker invariant", but on average outperforms shallower models at inter-subject
    prediction capability.
    """

    def __init__(self, targets, samples, channels, s_growth=24, t_filters=32, do=0.4, pooling=20,
                 activation=nn.LeakyReLU, temp_layers=2, spat_layers=2, temp_span=0.05, bottleneck=3,
                 summary=-1, return_features=False):
        self.temp_len = math.ceil(temp_span * samples)
        summary = samples // pooling if summary == -1 else summary
        self._num_features = (t_filters + s_growth * spat_layers) * summary
        super().__init__(targets, samples, channels, return_features=return_features)

        self.temporal = nn.Sequential(
            Expand(axis=1),
            TemporalFilter(1, t_filters, depth=temp_layers, temp_len=self.temp_len, activation=activation),
            nn.MaxPool2d((1, pooling)),
            nn.Dropout2d(do),
        )

        self.spatial = DenseSpatialFilter(self.channels, s_growth, spat_layers, in_ch=t_filters, dropout_rate=do,
                                          bottleneck=bottleneck, activation=activation)
        self.extract_features = nn.Sequential(
            nn.AdaptiveAvgPool1d(int(summary)),
        )

    @property
    def num_features_for_classification(self):
        return self._num_features

    def features_forward(self, x, **kwargs):
        x = self.temporal(x)
        x = self.spatial(x)
        return self.extract_features(x)


class EEGNet(Classifier):
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

    def __init__(self, targets, samples, channels, do=0.25, pooling=8, F1=8, D=2, t_len=65, F2=16,
                 return_features=False):
        if t_len >= samples:
            print("Warning: EEGNet `t_len` too long for sample length, reverting to 0.25 sample length")
            t_len = samples // 4
            t_len = t_len if t_len % 2 else t_len+1

        samples = samples // (pooling // 2)
        samples = samples // pooling
        self._num_features = F2 * samples
        super().__init__(targets, samples, channels, return_features=return_features)

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

        self.sep_conv = nn.Sequential(
            # Separate into two convs, one that doesnt operate across filters, one isolated to filters
            nn.Conv2d(D*F1, D*F1, (1, 17), bias=False, padding=(0, 8), groups=D*F1),
            nn.Conv2d(D*F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling)),
            nn.Dropout(do)
        )

    @property
    def num_features_for_classification(self):
        return self._num_features

    def features_forward(self, x):
        x = self.init_conv(x)
        x = self.depth_conv(x)
        return self.sep_conv(x)


class EEGNetStrided(StrideClassifier):
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

    def __init__(self, targets, samples, channels, do=0.25, pooling=8, F1=8, D=2, t_len=65, F2=16,
                 return_features=False, stride_width=2):
        if t_len >= samples:
            print("Warning: EEGNet `t_len` too long for sample length, reverting to 0.25 sample length")
            t_len = samples // 4
            t_len = t_len if t_len % 2 else t_len+1

        samples = samples // (pooling // 2)
        samples = samples // pooling
        self._num_features = F2
        super().__init__(targets, samples, channels, return_features=return_features, stride_width=stride_width)

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

        self.sep_conv = nn.Sequential(
            # Separate into two convs, one that doesnt operate across filters, one isolated to filters
            nn.Conv2d(D*F1, D*F1, (1, 17), bias=False, padding=(0, 8), groups=D*F1),
            nn.Conv2d(D*F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pooling)),
            nn.Dropout(do)
        )

    @property
    def num_features_for_classification(self):
        return self._num_features

    def features_forward(self, x):
        x = self.init_conv(x)
        x = self.depth_conv(x)
        return self.sep_conv(x).squeeze(-2)


class BENDRClassifier(Classifier):

    def __init__(self, targets, samples, channels,
                 return_features=True,
                 encoder_h=256,
                 encoder_w=(3, 2, 2, 2, 2, 2),
                 encoder_do=0.,
                 projection_head=False,
                 encoder_stride=(3, 2, 2, 2, 2, 2),
                 hidden_feedforward=3076,
                 heads=8,
                 context_layers=8,
                 context_do=0.15,
                 activation='gelu',
                 position_encoder=25,
                 layer_drop=0.0,
                 mask_p_t=0.1,
                 mask_p_c=0.004,
                 mask_t_span=6,
                 mask_c_span=64,
                 start_token=-5,
                 **kwargs):
        self._context_features = encoder_h
        super(BENDRClassifier, self).__init__(targets, samples, channels, return_features=return_features)
        self.encoder = ConvEncoderBENDR(in_features=channels, encoder_h=encoder_h, enc_width=encoder_w,
                                        dropout=encoder_do, projection_head=projection_head,
                                        enc_downsample=encoder_stride)
        self.contextualizer = BENDRContextualizer(encoder_h, hidden_feedforward=hidden_feedforward, heads=heads,
                                                  layers=context_layers, dropout=context_do, activation=activation,
                                                  position_encoder=position_encoder, layer_drop=layer_drop,
                                                  mask_p_t=mask_p_t, mask_p_c=mask_p_c, mask_t_span=mask_t_span,
                                                  finetuning=True)


    @property
    def num_features_for_classification(self):
        return self._context_features

    def easy_parallel(self):
        self.encoder = nn.DataParallel(self.encoder)
        self.contextualizer = nn.DataParallel(self.contextualizer)
        self.classifier = nn.DataParallel(self.classifier)

    def features_forward(self, x):
        x = self.encoder(x)
        x = self.contextualizer(x)
        return x[0]


