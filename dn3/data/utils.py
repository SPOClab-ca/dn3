from .dataset import *
from dn3.transforms.channels import EEG_INDS
from yaml import safe_dump


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


def deviation_based_span_rejection(dataset: Dataset, num_deviations=4, max_passes=10, save_to_yaml=None,
                                   **dataloader_kwargs):
    """
    Rejects time-spans from sessions based on dataset-wide statistics.

    Warnings
    ----------
    This only works with :any:`Dataset` that have raw data underlying, in other words, constructed using
    :any:`RawTorchRecording`, with Deep1010 mapping that includes a mask (which allows focus on only used EEG channels
    but allows for varying channel configurations).

    Additionally, with a very small stride (much smaller than sequence_length), corrupt portions (by the deviation
    definition here) are likely to still propagate through at the tails of some points.

    Parameters
    ----------
    dataset : Dataset
    num_deviations : float
                    The number of standard deviations that a trial's deviation needs to differ, from the average
                    standard deviation of all (not bad) trials.
    max_passes : int
                 The number of times to re-calculate the average standard deviation of not bad trials to reject those
                 missed on the first pass.
    save_to_yaml : str, None
                   If not None, should be a filepath to where to store the exlcusions. This can then be used by the
                   configuratron by specifying *exclude: !include <save_to_yaml filepath>
    dataloader_kwargs : dict
                        A pytorch dataloader is used to load data in batches. Keyword arguments propagated here.

    Returns
    -------
    exclude,

    """
    old_trial, old_session, old_person = dataset.return_trial_id, dataset.return_session_id, dataset.return_person_id
    dataset.update_id_returns(trial=True, session=True, person=True)
    sfreq, sequence_length = dataset.sfreq, dataset.sequence_length
    deviations = {thid: {sid: [] for sid in dataset.thinkers[thid].sessions.keys()} for thid in dataset.get_thinkers()}
    exclude = dict()

    def add_bad_span(tid, sid, start, end):
        if tid in exclude and sid in exclude[tid]:
            exclude[tid][sid].append([start / sfreq, end / sfreq])
        elif tid in exclude:
            exclude[tid][sid] = [[start / sfreq, end / sfreq]]
        else:
            exclude[tid] = {sid: [[start / sfreq, end / sfreq]]}

    def masked_dev_stats():
        total = 0
        results = list()
        # First pass mean, second pass stdev. Total calculated once
        for i in range(2):
            accumulator = 0
            for tid in deviations:
                for sid in deviations[tid]:
                    if i == 0:
                        accumulator += deviations[tid][sid].sum()
                        total += np.ma.count(deviations[tid][sid])
                    else:
                        accumulator += np.ma.power((deviations[tid][sid] - results[0]), 2).sum()
            results.append(accumulator / total)
        return results[0], np.sqrt(results[1])

    dataloader_kwargs.setdefault('shuffle', False)
    dataloader_kwargs.setdefault('drop_last', False)
    dataloader_kwargs.setdefault('batch_size', 128)
    batch_loader = DataLoader(dataset, **dataloader_kwargs)

    for data in tqdm.tqdm(batch_loader, desc="Loading Dataset"):
        think_id, sess_id = data[-3:-1]
        x = data[0][:, EEG_INDS, :].numpy().std(axis=-1).view(np.ma.MaskedArray)
        x[np.invert(data[1][:, EEG_INDS].numpy())] = np.ma.masked
        for i, (t, s) in enumerate(zip(think_id, sess_id)):
            who_dis = dataset.get_thinkers()[int(t)]
            which_sess = list(dataset.thinkers[who_dis].sessions.keys())[s]
            deviations[who_dis][which_sess].append(x[i])

    return_deviations = deviations.copy()

    for tid in deviations:
        for sid in deviations[tid]:
            deviations[tid][sid] = np.ma.stack(deviations[tid][sid])

    for i in tqdm.trange(max_passes, desc='Rejecting:'):
        total_mean_dev, total_std_from_mean_dev = masked_dev_stats()
        reject_count = 0

        for tid in deviations:
            for sid in deviations[tid]:
                normed_diffs = np.abs(deviations[tid][sid] - total_mean_dev) / total_std_from_mean_dev

                new_rejections = np.unique(np.ma.nonzero(normed_diffs >= num_deviations)[0])
                deviations[tid][sid][new_rejections] = np.ma.masked
                reject_count += len(new_rejections)

        if reject_count == 0:
            print("Stabilized early, stopping...")
            break

    # print("Rejected {} dataset sequences.".format(reject_count))

    def get_start_end(tid, sid, trial_id):
        stride = dataset.thinkers[tid].sessions[sid].stride
        return trial_id * stride, trial_id * stride + sequence_length

    print("Cleaning up...")
    for tid in deviations:
        for sid in deviations[tid]:
            rejects = np.unique(np.ma.nonzero(deviations[tid][sid].count(axis=1) == 0)[0])
            if len(rejects) == 0:
                continue

            start, end = get_start_end(tid, sid, rejects[0])
            if len(rejects) == 1:
                add_bad_span(tid, sid, *get_start_end(tid, sid, rejects[0]))
                continue

            for i, trial_id in enumerate(rejects[1:]):
                if rejects[i] == trial_id-1:
                    _, end = get_start_end(tid, sid, trial_id)
                else:
                    add_bad_span(tid, sid, start, end)
                    start, end = get_start_end(tid, sid, trial_id)

            add_bad_span(tid, sid, start, end)

    dataset.update_id_returns(trial=old_trial, person=old_person, session=old_session)
    if save_to_yaml is not None:
        print("Saving to", save_to_yaml)
        with open(save_to_yaml, 'w') as f:
            safe_dump(exclude, f)

    print('Done.')
    return exclude, return_deviations
