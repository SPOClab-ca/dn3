import mne
import torch
import numpy as np

from torch.utils.data import DataLoader


class DNPTLoader(DataLoader):
    """
    DNPT-specific data loader extending dataloader-specific .

    See :py:mod:`torch.utils.data` documentation page for more details.

     Parameters
     ----------
    *thinkers : Thinker
                The thinkers from which to load data.
    batch_transforms : (Iterable)
                 The transforms to be applied to each fetched batch.
    batch_size : int, optional
                 How many samples per batch to load (default: ``1``).
    shuffle : bool, optional
                 Set to ``True`` to have the data reshuffled at every epoch (default: ``False``).
    sampler : (Sampler, optional)
              defines the strategy to draw samples from the dataset. If specified, :attr:`shuffle` must be ``False``.
    batch_sampler : Sampler, optional
                    like :attr:`sampler`, but returns a batch of indices at a time. Mutually exclusive with
                    :attr:`batch_size`,:attr:`shuffle`, :attr:`sampler`, and :attr:`drop_last`.
    num_workers : int, optional
                  how many subprocesses to use for data loading. ``0`` means that the data will be loaded in the
                  main process. (default: ``0``)
    collate_fn : callable, optional
                merges a list of samples to form a mini-batch of Tensor(s).  Used when using batched loading from a
                map-style dataset.
    pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
        into CUDA pinned memory before returning them.  If your data elements
        are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
        see the example below.
    drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
        if the dataset size is not divisible by the batch size. If ``False`` and
        the size of dataset is not divisible by the batch size, then the last batch
        will be smaller. (default: ``False``)
    timeout (numeric, optional): if positive, the timeout value for collecting a batch
        from workers. Should always be non-negative. (default: ``0``)
    worker_init_fn (callable, optional): If not ``None``, this will be called on each
        worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
        input, after seeding and before data loading. (default: ``None``)
    """
    def __init__(self):
        super().__init__(da)
