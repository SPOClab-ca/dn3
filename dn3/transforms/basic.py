import torch
from numpy import ndarray

from .channels import map_channels_1010


class BaseTransform(object):
    """
    Transforms are, for the most part, simply operations that are performed on the loaded tensors when they are fetched
    via the :meth:`transform` method. Ideally this is implemented with pytorch operations for ease of execution graph
     integration.
    """
    def __init__(self, only_trial_data=True):
        self.only_trial_data = only_trial_data

    def __call__(self, *inputs):
        if self.only_trial_data:
            x = self.transform(inputs[0])
            inputs = (x, *inputs[1:])
        else:
            inputs = self.transform(*inputs)
        return inputs

    def transform(self, x):
        """
        Modifies a batch of tensors.
        Parameters
        ----------
        x : torch.Tensor, tuple
            The trial tensor, not including a batch-dimension. If initialized with `only_trial_data=False`, then this
            is a tuple of all ids, labels, etc. being propagated.
        Returns
        -------
        x : torch.Tensor, tuple
            The modified trial tensor, or tensors if not `only_trial_data`
        """
        raise NotImplementedError()

    def new_channels(self, old_channels):
        """
        This is an optional method that indicates the transformation modifies the representation and/or presence of
        channels.

        Parameters
        ----------
        old_channels : ndarray
                       An array with the channel names as they are up until this transformation

        Returns
        -------
        new_channels : ndarray
                      An array with the channel names as they are after this transformation. Supports conversion of 1D
                      channel set into more dimensions, e.g. a list of channels into a rectangular grid.
        """
        return old_channels

    def new_sfreq(self, old_sfreq):
        """
        This is an optional method that indicates the transformation modifies the sampling frequency of the underlying
        time-series.

        Parameters
        ----------
        old_sfreq : float

        Returns
        -------
        new_sfreq : float
        """
        return old_sfreq

    def new_sequence_length(self, old_sequence_length):
        """
        This is an optional method that indicates the transformation modifies the length of the acquired extracts,
        specified in number of samples.

        Parameters
        ----------
        old_sequence_length : int

        Returns
        -------
        new_sequence_length : int
        """
        return old_sequence_length


class ZScore(BaseTransform):
    """
    Z-score normalization of trials
    """
    def transform(self, x):
        return (x - x.mean()) / x.std()


class TemporalPadding(BaseTransform):

    def __init__(self, start_padding, end_padding, mode='constant', constant_value=0):
        """
        Pad the number of samples.

        Parameters
        ----------
        start_padding : int
                        The number of padded samples to add to the beginning of a trial
        end_padding : int
                      The number of padded samples to add to the end of a trial
        mode : str
               See `pytorch documentation <https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad>`_
        constant_value : float
               If mode is 'constant' (the default), the value to compose the samples of.
        """
        super().__init__()
        self.start_padding = start_padding
        self.end_padding = end_padding
        self.mode = mode
        self.constant_value = constant_value

    def transform(self, x):
        pad = [self.start_padding, self.end_padding] + [0 for _ in range(2, x.shape[-1])]
        return torch.nn.functional.pad(x, pad, mode=self.mode, value=self.constant_value)

    def new_sequence_length(self, old_sequence_length):
        return old_sequence_length + self.start_padding + self.end_padding


class MappingDeep1010(BaseTransform):
    """
    Maps various channel sets into the Deep10-10 scheme.
    TODO - refer to eventual literature on this
    """
    def __init__(self, ch_names, EOG=None, reference=None, add_scale_ind=True, return_mask=True):
        super().__init__()
        self.mapping = map_channels_1010(ch_names, EOG, reference)
        self.add_scale_ind = add_scale_ind
        self.return_mask = return_mask

    def transform(self, x):
        x = (x.transpose(1, 0) @ self.mapping).transpose(1, 0)
        return (x, self.mapping.sum(dim=0))

    def new_channels(self, old_channels: ndarray):
        channels = old_channels @ self.mapping.gt(0)
        channels[channels == 0] = None
        return channels
