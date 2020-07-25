import tqdm
from .dataset import *


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
