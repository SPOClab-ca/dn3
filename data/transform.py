import torch


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
