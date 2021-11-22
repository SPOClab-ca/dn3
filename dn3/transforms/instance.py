import warnings

import torch
import numpy as np

# from random import choices
from typing import List

from .channels import map_dataset_channels_deep_1010, DEEP_1010_CH_TYPES, SCALE_IND, \
    EEG_INDS, EOG_INDS, REF_INDS, EXTRA_INDS, DEEP_1010_CHS_LISTING
from dn3.utils import min_max_normalize

from torch.nn.functional import interpolate


def same_channel_sets(channel_sets: list):
    """Validate that all the channel sets are consistent, return false if not"""
    for chs in channel_sets[1:]:
        if chs.shape[0] != channel_sets[0].shape[0] or chs.shape[1] != channel_sets[0].shape[1]:
            return False
        # if not np.all(channel_sets[0] == chs):
        #     return False
    return True


class InstanceTransform(object):

    def __init__(self, only_trial_data=True):
        """
        Trial transforms are, for the most part, simply operations that are performed on the loaded tensors when they are
        fetched via the :meth:`__call__` method. Ideally this is implemented with pytorch operations for ease of execution
        graph integration.
        """
        self.only_trial_data = only_trial_data

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, *x):
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
                       An array whose last two dimensions are channel names and channel types.

        Returns
        -------
        new_channels : ndarray
                      An array with the channel names and types after this transformation. Supports the addition of
                      dimensions e.g. a list of channels into a rectangular grid, but the *final two dimensions* must
                      remain the channel names, and types respectively.
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


class _PassThroughTransform(InstanceTransform):

    def __init__(self):
        super().__init__(only_trial_data=False)

    def __call__(self, *x):
        return x


class ZScore(InstanceTransform):
    """
    Z-score normalization of trials
    """
    def __init__(self, mean=None, std=None):
        super(ZScore, self).__init__()
        self.mean = mean
        self.std = std
    
    def __call__(self, x):
        mean = x.mean() if self.mean is None else self.mean
        std = x.std() if self.std is None else self.std
        return (x - mean) / std


class FixedScale(InstanceTransform):
    """
    Scale the input to range from low to high
    """
    def __init__(self, low_bound=-1, high_bound=1):
        super().__init__()
        self.low_bound = low_bound
        self.high_bound = high_bound

    def __call__(self, x):
        return min_max_normalize(x, self.low_bound, self.high_bound)


class TemporalPadding(InstanceTransform):

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

    def __call__(self, x):
        pad = [self.start_padding, self.end_padding] + [0 for _ in range(2, x.shape[-1])]
        return torch.nn.functional.pad(x, pad, mode=self.mode, value=self.constant_value)

    def new_sequence_length(self, old_sequence_length):
        return old_sequence_length + self.start_padding + self.end_padding


class TemporalInterpolation(InstanceTransform):

    def __init__(self, desired_sequence_length, mode='nearest', new_sfreq=None):
        """
        This is in essence a DN3 wrapper for the pytorch function
        `interpolate() <https://pytorch.org/docs/stable/nn.functional.html>`_

        Currently only supports single dimensional samples (i.e. channels have not been projected into more dimensions)

        Warnings
        --------
        Using this function to downsample data below a suitable nyquist frequency given the low-pass filtering of the
        original data will cause dangerous aliasing artifacts that will heavily affect data quality to the point of it
        being mostly garbage.

        Parameters
        ----------
        desired_sequence_length: int
                                 The desired new sequence length of incoming data.
        mode: str
              The technique that will be used for upsampling data, by default 'nearest' interpolation. Other options
              are listed under pytorch's interpolate function.
        new_sfreq: float, None
                   If specified, registers the change in sampling frequency
        """
        super().__init__()
        self._new_sequence_length = desired_sequence_length
        self.mode = mode
        self._new_sfreq = new_sfreq

    def __call__(self, x):
        # squeeze and unsqueeze because these are done before batching
        if len(x.shape) == 2:
            return interpolate(x.unsqueeze(0), self._new_sequence_length, mode=self.mode).squeeze(0)
        # Supports batch dimension
        elif len(x.shape) == 3:
            return interpolate(x, self._new_sequence_length, mode=self.mode)
        else:
            raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")

    def new_sequence_length(self, old_sequence_length):
        return self._new_sequence_length

    def new_sfreq(self, old_sfreq):
        if self._new_sfreq is not None:
            return self._new_sfreq
        else:
            return old_sfreq


class CropAndUpSample(TemporalInterpolation):

    def __init__(self, original_sequence_length, crop_sequence_min):
        super().__init__(original_sequence_length)
        self.crop_sequence_min = crop_sequence_min

    def __call__(self, x):
        crop_len = np.random.randint(low=self.crop_sequence_min, high=x.shape[-1])
        return super(CropAndUpSample, self).__call__(x[:, :crop_len])

    def new_sequence_length(self, old_sequence_length):
        return old_sequence_length - self.crop_sequence_min


class TemporalCrop(InstanceTransform):

    def __init__(self, cropped_length, start_offset=None):
        """
        Crop to a new length of `cropped_length` from the specified `start_offset`, or randomly select an offset.

        Parameters
        ----------
        cropped_length : int
                         The cropped sequence length (in samples).
        start_offset : int, None, List[int]
                       If a single int, the sample to start the crop from. If `None`, will uniformly select from
                       within the possible start offsets (such that `cropped_length` is not violated). If
                       a list of ints, these specify the *un-normalized* probabilities for different start offsets.
        """
        super().__init__()
        if isinstance(start_offset, int):
            start_offset = [0 for _ in range(start_offset-1)] + [1]
        elif isinstance(start_offset, list):
            summed = sum(start_offset)
            start_offset = [w / summed for w in start_offset]
        self.start_offset = start_offset
        self._new_length = cropped_length

    def _get_start_offset(self, full_length):
        possible_starts = full_length - self._new_length
        assert possible_starts >= 0
        if self.start_offset is None:
            start_offset = np.random.choice(possible_starts) if possible_starts > 0 else 0
        elif isinstance(self.start_offset, list):
            start_offset = self.start_offset + [0 for _ in range(len(self.start_offset), possible_starts)]
            start_offset = np.random.choice(possible_starts, p=start_offset)
        else:
            start_offset = self.start_offset
        return int(start_offset)

    def __call__(self, x):
        start_offset = self._get_start_offset(x.shape[-1])
        return x[..., start_offset:start_offset + self._new_length]

    def new_sequence_length(self, old_sequence_length):
        return self._new_length


class CropAndResample(TemporalInterpolation):

    def __init__(self, desired_sequence_length, stdev, truncate=None, mode='nearest', new_sfreq=None,
                 crop_side='both', allow_uncroppable=False):
        super().__init__(desired_sequence_length, mode=mode, new_sfreq=new_sfreq)
        self.stdev = stdev
        self.allow_uncrop = allow_uncroppable
        self.truncate = float("inf") if truncate is None else truncate
        if crop_side not in ['right', 'left', 'both']:
            raise ValueError("The crop-side should either be 'left', 'right', or 'both'")
        self.crop_side = crop_side

    @staticmethod
    def trunc_norm(mean, std, max_diff):
        val = None
        while val is None or abs(val - mean) > max_diff:
            val = int(np.random.normal(mean, std))
        return val

    def new_sequence_length(self, old_sequence_length):
        return self._new_sequence_length

    def __call__(self, x):
        max_diff = min(x.shape[-1] - self._new_sequence_length, self.truncate)
        if self.allow_uncrop and max_diff == 0:
            return x
        assert max_diff > 0
        crop = np.random.choice(['right', 'left']) if self.crop_side == 'both' else self.crop_side
        if self.crop_side == 'right':
            offset = self.trunc_norm(self._new_sequence_length, self.stdev, max_diff)
            return super(CropAndResample, self).__call__(x[:, :offset])
        else:
            offset = self.trunc_norm(x.shape[-1] - self._new_sequence_length, self.stdev, max_diff)
            return super(CropAndResample, self).__call__(x[:, offset:])


class MappingDeep1010(InstanceTransform):
    """
    Maps various channel sets into the Deep10-10 scheme, and normalizes data between [-1, 1] with an additional scaling
    parameter to describe the relative scale of a trial with respect to the entire dataset.

    See https://doi.org/10.1101/2020.12.17.423197  for description.
    """
    def __init__(self, dataset, max_scale=None, return_mask=False):
        """
        Creates a Deep10-10 mapping for the provided dataset.

        Parameters
        ----------
        dataset : Dataset

        max_scale : float
                    If specified, the scale ind is filled with the relative scale of the trial with respect
                    to this, otherwise uses dataset.info.data_max - dataset.info.data_min.
        return_mask : bool
                      If `True` (`False` by default), an additional tensor is returned after this transform that
                      says which channels of the mapping are in fact in use.
        """
        super().__init__()
        self.mapping = map_dataset_channels_deep_1010(dataset.channels)
        if max_scale is not None:
            self.max_scale = max_scale
        elif dataset.info is None or dataset.info.data_max is None or dataset.info.data_min is None:
            print(f"Warning: Did not find data scale information for {dataset}")
            self.max_scale = None
            pass
        else:
            self.max_scale = dataset.info.data_max - dataset.info.data_min
        self.return_mask = return_mask

    @staticmethod
    def channel_listing():
        return DEEP_1010_CHS_LISTING

    def __call__(self, x):
        if self.max_scale is not None:
            scale = 2 * (torch.clamp_max((x.max() - x.min()) / self.max_scale, 1.0) - 0.5)
        else:
            scale = 0

        x = (x.transpose(1, 0) @ self.mapping).transpose(1, 0)

        for ch_type_inds in (EEG_INDS, EOG_INDS, REF_INDS, EXTRA_INDS):
            x[ch_type_inds, :] = min_max_normalize(x[ch_type_inds, :])

        used_channel_mask = self.mapping.sum(dim=0).bool()
        x[~used_channel_mask, :] = 0

        x[SCALE_IND, :] = scale

        if self.return_mask:
            return (x, used_channel_mask)
        else:
            return x

    def new_channels(self, old_channels: np.ndarray):
        channels = list()
        for row in range(self.mapping.shape[1]):
            active = self.mapping[:, row].nonzero().numpy()
            if len(active) > 0:
                channels.append("-".join([old_channels[i.item(), 0] for i in active]))
            else:
                channels.append(None)
        return np.array(list(zip(channels, DEEP_1010_CH_TYPES)))


class Deep1010ToEEG(InstanceTransform):

    def __init__(self):
        super().__init__(only_trial_data=False)

    def new_channels(self, old_channels):
        return old_channels[EEG_INDS]

    def __call__(self, *x):
        x = list(x)
        for i in range(len(x)):
            # Assume every tensor that has deep1010 length should be modified
            if len(x[i].shape) > 0 and x[i].shape[0] == len(DEEP_1010_CHS_LISTING):
                x[i] = x[i][EEG_INDS, ...]

        return x


class To1020(InstanceTransform):

    EEG_20_div = [
               'FP1', 'FP2',
        'F7', 'F3', 'FZ', 'F4', 'F8',
        'T7', 'C3', 'CZ', 'C4', 'T8',
        'T5', 'P3', 'PZ', 'P4', 'T6',
                'O1', 'O2'
    ]

    def __init__(self, only_trial_data=True, include_scale_ch=True, include_ref_chs=False):
        """
        Transforms incoming Deep1010 data into exclusively the more limited 1020 channel set.
        """
        super(To1020, self).__init__(only_trial_data=only_trial_data)
        self._inds_20_div = [DEEP_1010_CHS_LISTING.index(ch) for ch in self.EEG_20_div]
        if include_ref_chs:
            self._inds_20_div.append([DEEP_1010_CHS_LISTING.index(ch) for ch in ['A1', 'A2']])
        if include_scale_ch:
            self._inds_20_div.append(SCALE_IND)

    def new_channels(self, old_channels):
        return old_channels[self._inds_20_div]

    def __call__(self, *x):
        x = list(x)
        for i in range(len(x)):
            # Assume every tensor that has deep1010 length should be modified
            if len(x[i].shape) > 0 and x[i].shape[0] == len(DEEP_1010_CHS_LISTING):
                x[i] = x[i][self._inds_20_div, ...]
        return x


class MaskAuxiliariesDeep1010(InstanceTransform):

    MASK_THESE = REF_INDS + EOG_INDS + EXTRA_INDS

    def __init__(self, randomize=False):
        super(MaskAuxiliariesDeep1010, self).__init__()
        self.randomize = randomize

    def __call__(self, x):
        if self.randomize:
            x[self.MASK_THESE, :] = 2 * torch.rand_like(x[self.MASK_THESE, :]) - 1
        else:
            x[self.MASK_THESE] = 0
        return x


class NoisyBlankDeep1010(InstanceTransform):
    """

    """

    def __init__(self, mask_index=1, purge_mask=False):
        super().__init__(only_trial_data=False)
        self.mask_index = mask_index
        self.purge_mask = purge_mask

    def __call__(self, *x):
        assert isinstance(x, (list, tuple)) and len(x) > 1
        x = list(x)
        blanks = x[0][~x[self.mask_index], :]
        x[0][~x[self.mask_index], :] = 2 * torch.rand_like(blanks) - 1
        if self.purge_mask:
            x = x.pop(self.mask_index)
        return x


class AdditiveEogDeep1010(InstanceTransform):

    EEG_IND_END = EOG_INDS[0] - 1

    def __init__(self, p=0.1, max_intensity=0.3, blank_eog_p=1.0):
        super().__init__()
        self.max_intensity = max_intensity
        self.p = p
        self.blanking_p = blank_eog_p

    def __call__(self, x):
        affected_inds = (torch.rand(EOG_INDS[0] - 1).lt(self.p)).nonzero()
        for which_eog in EOG_INDS:
            this_affected_ids = affected_inds[torch.rand_like(affected_inds.float()).lt(0.25)]
            x[this_affected_ids, :] = self.max_intensity * torch.rand_like(this_affected_ids.float()).unsqueeze(-1) *\
                                      x[which_eog]
            # Short circuit the random call as these are more frequent
            if self.blanking_p != 0 and (self.blanking_p == 1 or torch.rand(1) < self.blanking_p):
                x[which_eog, :] = 0
        return x


class UniformTransformSelection(InstanceTransform):

    def __init__(self, transform_list, weights=None, suppress_warnings=False):
        """
        Uniformly selects a transform from the `transform_list` with probabilities according to `weights`.

        Parameters
        ----------
        transform_list: List[InstanceTransform]
                        List of transforms to select from.
        weights: None, List[float]
           This is either `None`, in which case the transforms are selected with equal probability. Or relative
           probabilities to select from `transform_list`. This means, it does not have to be normalized probabilities
           *(this doesn't have to sum to one)*. If `len(transform_list) == len(weights) - 1`, it will be assumed that
           the final weight expresses the likelihood of *no transform*.
        """
        super().__init__(only_trial_data=False)
        self.suppress_warnings = suppress_warnings
        for x in transform_list:
            assert isinstance(x, InstanceTransform)
        self.transforms = transform_list
        if weights is None:
            self._choice_weights = [1 / len(transform_list) for _ in range(len(transform_list))]
        else:
            if len(weights) == len(transform_list) + 1:
                self.transforms.append(_PassThroughTransform())
            total_weight = sum(weights)
            self._choice_weights = [p / total_weight for p in weights]
            assert len(weights) == len(transform_list)

    def __call__(self, *x):
        transform = np.random.choice(self.transforms, p=self._choice_weights)
        # if don't need to support <3.6, this is faster
        # which = choices(self.transforms, self._choice_weights)
        if hasattr(transform, 'only_trial_data') and transform.only_trial_data:
            return [transform(x[0]), *x[1:]]
        else:
            return transform(*x)

    def new_channels(self, old_channels):
        all_new = [transform.new_channels(old_channels) for transform in self.transforms]
        if not self.suppress_warnings and not same_channel_sets(all_new):
            warnings.warn('Multiple channel representations!')
        return all_new[0]

    def new_sfreq(self, old_sfreq):
        all_new = set(transform.new_sfreq(old_sfreq) for transform in self.transforms)
        if not self.suppress_warnings and len(all_new) > 1:
            warnings.warn('Multiple new sampling frequencies!')
        return all_new.pop()

    def new_sequence_length(self, old_sequence_length):
        all_new = set(transform.new_sequence_length(old_sequence_length) for transform in self.transforms)
        if not self.suppress_warnings and len(all_new) > 1:
            warnings.warn('Multiple new sequence lengths!')
        return all_new.pop()


class EuclideanAlignmentTransform(InstanceTransform):

    def __init__(self, reference_matrices, inds):
        super(EuclideanAlignmentTransform, self).__init__(only_trial_data=False)
        self.reference_matrices = reference_matrices
        self.inds = inds

    def __call__(self, *x):
        # Check that we have at least x, thinker_id, session_id, label if more than one referecing matrix
        # TODO expand so that this works for a single subject and multiple sessions
        if not isinstance(self.reference_matrices, dict):
            xform = torch.transpose(self.reference_matrices, 0, 1)
            inds = self.inds
        else:
            # Missing thinker and session index
            if len(x) == 3:
                thid = 0
                sid = 0
            elif len(x) == 4:
                thid = 0
                sid = int(x[-2])
            else:
                thid = int(x[-3])
                sid = int(x[-2])

            xform = torch.transpose(self.reference_matrices[thid][sid], 0, 1)
            inds = self.inds[thid][sid]

        x[0][inds, :] = torch.matmul(xform, x[0][inds, :])
        return x
