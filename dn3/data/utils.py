from .dataset import *


class MultiDatasetContainer(TorchDataset):

    def __init__(self, *datasets, oversample=False, return_dataset_ids=False, max_artificial_size=None):
        """
        This integrates the loading of multiple datasets, and facilitates over-sampling smaller ones to prevent larger
        datasets from crowding smaller ones.

        Parameters
        ----------
        datasets :
                 A number of datasets to select from.
        oversample : bool
                    Whether to oversample minority datasets or not.
        return_dataset_ids : bool
                             Whether to return the `dataset_ids` as the last returned Tensor in addition to all other
                             values returned.
        max_artificial_size : int, None
                            An upper limit on how many samples to maximally inflate smaller datasets to. This only
                            applies if oversample is `True`. If so, the smallest dataset :math: `d_s` with a
                            max_artificial size :math: `K` and largest dataset length :math: `d_l`, has effective
                            dataset size:

                            .. math:: \bar{d_s} = min(K, d_l)

        Notes
        -----
        This container never undersamples datasets, mainly to preserve to the deterministic nature of the balancing.
        The `cumulative_sizes` attribute of this class could be used to develop an undersampling strategy.
        """
        self.datasets = datasets
        self.return_dataset_ids = return_dataset_ids
        dataset_sizes = [len(d) for d in datasets]
        if oversample:
            max_artificial_size = max(dataset_sizes) if max_artificial_size is None else max_artificial_size
            largest_oversample = min(max(dataset_sizes), max_artificial_size)
            sample_sizes = [largest_oversample if s < largest_oversample else s for s in dataset_sizes]
        else:
            sample_sizes = dataset_sizes

        self.cumulative_sizes = np.cumsum(sample_sizes)
        self.index_map = list()

        for ds_len, inflated_size in zip(dataset_sizes, sample_sizes):
            ind_to_ind = np.tile(np.arange(ds_len), int(np.ceil(inflated_size / ds_len)))
            self.index_map.append(ind_to_ind[:inflated_size])

    def __getitem__(self, item):
        if item < 0:
            if -item > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            item = len(self) + item
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, item)
        if dataset_idx == 0:
            sample_idx = item
        else:
            sample_idx = item - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = self.index_map[dataset_idx][sample_idx]

        if self.return_dataset_ids:
            return (*self.datasets[dataset_idx][sample_idx],
                    self.datasets[dataset_idx].dataset_id)
        else:
            return self.datasets[dataset_idx][sample_idx]

    def __len__(self):
        return sum(len(inds) for inds in self.index_map)


def get_dataset_max_and_min(dataset: Dataset):
    """
    This utility function is used early on to determine the *data_max* and *data_min* parameters that are added to the
    configuratron to properly create the Deep1010 mapping.

    Parameters
    ----------
    dataset: Dataset

    Returns
    -------
    max, min
    """
    dmax = None
    dmin = None

    pbar = tqdm.tqdm(dataset)
    for data in pbar:
        data = data[0]
        _max = data.max()
        _min = data.min()
        if dmax is None:
            dmax = _max
        if dmin is None:
            dmin = _min
        dmax = max(dmax, _max)
        dmin = min(dmin, _min)
        pbar.set_postfix(dict(dmax=dmax, dmin=dmin))

    return dmax, dmin


def get_largest_trial_id(dataset: Dataset, trial_id_offset=-2):
    """
    This utility is for determining the largest trial id from a dataset.

    Parameters
    ----------
    dataset: Dataset
    trial_id_offset: int
                    The offset from the returned data that the trial_id value is expected (likely -1 for Raw recordings
                    and -2 for epoched).

    Returns
    -------
    max_trial_id: int
                 The largest trial_id value to expect from this dataset.
    """
    old_return_trial_id = dataset.return_trial_id
    dataset.update_id_returns(trial=True)
    max_trial_id = 0

    pbar = tqdm.tqdm(dataset)
    for data in pbar:
        data = data[trial_id_offset]
        max_trial_id = max(data, max_trial_id)
        pbar.set_postfix(dict(max_trial_id=max_trial_id))

    dataset.update_id_returns(trial=old_return_trial_id)
    return max_trial_id
