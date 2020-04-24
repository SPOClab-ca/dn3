import torch

from .channels import map_channels_1010


class BaseTransform(object):
    """
    Transforms are, for the most part, simply callable objects.

    __call__ should ideally be implemented with pytorch operations for ease of graph execution
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


class Mapping1010Plus(BaseTransform):
    """
    Maps various channel sets into the 10-10 plus scheme.
    TODO - refer to eventual literature on this
    """
    def __init__(self, ch_names, EOG=None, reference=None, add_scale_ind=True, return_mask=True):
        self.mapping = map_channels_1010(ch_names, EOG, reference)
        self.add_scale_ind = add_scale_ind
        self.return_mask = return_mask

    def __call__(self, *inputs, **kwargs):
        x = (inputs[0].transpose(1, 0) @ self.mapping).transpose(1, 0)
        return (x, *inputs[1:], self.mapping.sum(dim=0))
