import torch
from numpy import ndarray

from .channels import map_channels_1010


class BaseTransform(object):
    """
    Transforms are, for the most part, simply callable objects.

    __call__ should ideally be implemented with pytorch operations for ease of execution graph integration
    """

    def __call__(self, *inputs, **kwargs):
        """
        Modifies a batch of tensors.
        Parameters
        ----------
        *inputs : torch.Tensor
                  inputs will always have a length of at least one, storing the canonical data used for training.
                  If labels are provided by EpochRecording(s), they will be the last `input` provided.
                  Each tensor will start with a batch dim.
        **kwargs :
                 Currently unused. Subject id? Other ids?

        Returns
        -------
        The transformed inputs.
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
        channels : ndarray
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
        sfreq : float
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
        sequence_length : int
        """
        return old_sequence_length


class MappingDeep1010(BaseTransform):
    """
    Maps various channel sets into the Deep10-10 scheme.
    TODO - refer to eventual literature on this
    """
    def __init__(self, ch_names, EOG=None, reference=None, add_scale_ind=True, return_mask=True):
        self.mapping = map_channels_1010(ch_names, EOG, reference)
        self.add_scale_ind = add_scale_ind
        self.return_mask = return_mask

    def __call__(self, *inputs, **kwargs):
        x = (inputs[0].transpose(1, 0) @ self.mapping).transpose(1, 0)
        return (x, *inputs[1:], self.mapping.sum(dim=0))

    def new_channels(self, old_channels: ndarray):
        channels = old_channels @ self.mapping.gt(0)
        channels[channels == 0] = None
        return channels
