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


def get_dataset_max_and_min(dataset: Dataset, rd=0.9):
    """
    This utility function is used early on to determine the *data_max* and *data_min* parameters that are added to the
    configuratron to properly create the Deep1010 mapping. Running factor tracks how much the individual max and mins
    deviate from the max and min while searching, useful to track how large the peaks in data appear.

    Parameters
    ----------
    dataset: Dataset
    rd: float

    Returns
    -------
    max, min
    """
    dmax = None
    dmin = None
    running_dev_max = 0
    running_dev_min = 0

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
        running_dev_max = rd * running_dev_max + (1 - rd) * (dmax - _max) ** 2
        running_dev_min = rd * running_dev_min + (1 - rd) * (dmin - _min) ** 2
        pbar.set_postfix(dict(dmax=dmax, dmin=dmin, dev_max=running_dev_max, dev_min=running_dev_min))

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


def _trial_stdev(x):
    return x.numpy().std(axis=-1)


class SingleStatisticSpanRejection:

    def __init__(self, dataset, mask_ind=-1, stat_fn=_trial_stdev, **dataloader_kwargs):
        """
        With larger datasets, it may be prudent to triage the data based on global statistics. Here spans of time are
        rejected based on a provided statistic (default stdev) by creating a report readable by the :any:`Configuratron`

        Parameters
        ----------
        dataset
        mask_ind
        stat_fn
        dataloader_kwargs
        """
        assert callable(stat_fn)
        self.stat_fn = stat_fn

        self.dataset = dataset
        self.mask_ind = mask_ind
        dataloader_kwargs.setdefault('shuffle', False)
        dataloader_kwargs.setdefault('drop_last', False)
        dataloader_kwargs.setdefault('batch_size', 128)
        self.dataloader_kwargs = dataloader_kwargs
        self.reset()

    @staticmethod
    def from_precollected_statistics(dataset, precollected):
        new = SingleStatisticSpanRejection(dataset)
        new.reset()
        new.statistic_lookup = precollected
        return new

    def reset(self, rejections_only=False):
        self.rejections = {thid: {sid: [] for sid in self.dataset.thinkers[thid].sessions.keys()} for thid in
                           self.dataset.get_thinkers()}
        self.exclude = dict()

        if rejections_only:
            return

        self._sfreq, self._sequence_length = self.dataset.sfreq, self.dataset.sequence_length
        self.old_trial, self.old_session, self.old_person = self.dataset.return_trial_id, \
                                                            self.dataset.return_session_id, \
                                                            self.dataset.return_person_id

        self.statistic_lookup = {thid: {sid: [] for sid in self.dataset.thinkers[thid].sessions.keys()} for thid in
                                 self.dataset.get_thinkers()}

    @property
    def valid_stats(self):
        valid_stats = list()
        for tid in tqdm.tqdm(self.statistic_lookup, desc="Collecting...", unit="Person"):
            for sid in self.statistic_lookup[tid]:
                for i, stat in enumerate(self.statistic_lookup[tid][sid]):
                    if i not in self.rejections[tid][sid]:
                        x = self.statistic_lookup[tid][sid][i]
                        valid_stats.append(x[x.mask == False].data.ravel())
        return np.concatenate(valid_stats)

    @property
    def rejected_stats(self):
        rejected_stats = list()
        for tid in self.statistic_lookup:
            for sid in self.statistic_lookup[tid]:
                for i, stat in enumerate(self.statistic_lookup[tid][sid]):
                    if i in self.rejections[tid][sid]:
                        x = self.statistic_lookup[tid][sid][i]
                        rejected_stats.append(x[x.mask == False].data.ravel())
        if len(rejected_stats) > 0:
            return np.concatenate(rejected_stats)
        return np.array([])

    def collect_statistic(self):
        self.dataset.update_id_returns(trial=True, session=True, person=True)
        thinker_list = self.dataset.get_thinkers()
        session_lists = {who_dis: list(self.dataset.thinkers[who_dis].sessions.keys()) for who_dis in thinker_list}

        batch_loader = DataLoader(self.dataset, **self.dataloader_kwargs)

        for data in tqdm.tqdm(batch_loader, desc="Loading Dataset"):
            think_id, sess_id = data[1:3] if self.mask_ind == -1 else data[-3:-1]

            x = self.stat_fn(data[0][:, EEG_INDS, :]).view(np.ma.MaskedArray)

            x[np.invert(data[self.mask_ind][:, EEG_INDS].numpy())] = np.ma.masked

            for i, (t, s) in enumerate(zip(think_id, sess_id)):
                who_dis = thinker_list[int(t)]
                which_sess = session_lists[who_dis][s]
                self.statistic_lookup[who_dis][which_sess].append(x[i])

        self.dataset.update_id_returns(trial=self.old_trial, person=self.old_person, session=self.old_session)

    def _make_numpy_convenience(self):
        stats = {thid: {sid: [] for sid in self.statistic_lookup[thid].keys()} for thid in self.statistic_lookup}
        for tid in stats:
            for sid in stats[tid]:
                stats[tid][sid] = np.ma.stack(self.statistic_lookup[tid][sid])
        return stats

    def deviation_threshold_rejection(self, reject_iterations=10, num_deviations=4):
        stats = self._make_numpy_convenience()
        num_rejections = 0

        for i in tqdm.trange(reject_iterations, desc='Rejecting:'):
            valid_stats = self.valid_stats
            total_mean_dev = valid_stats.mean()
            total_stdev_of_dev = valid_stats.std()
            reject_count = 0

            for tid in stats:
                for sid in stats[tid]:
                    normed_diffs = np.abs(stats[tid][sid] - total_mean_dev) / total_stdev_of_dev

                    new_rejections = np.unique(np.ma.nonzero(normed_diffs >= num_deviations)[0])
                    stats[tid][sid][new_rejections] = np.ma.masked
                    reject_count += len(new_rejections)
                    num_rejections += len(new_rejections)
                    self.rejections[tid][sid] = np.union1d(new_rejections, self.rejections[tid][sid]).tolist()

            if reject_count == 0:
                print("Stabilized early, stopping...")
                break

        return num_rejections

    def keep_window(self, low=None, high=None):
        """
        Rejects any statistics that lie outside the specified windown limits of low and high. If one is not specified,
        one side of the window is ignored.

        Parameters
        ----------
        low : float, None
              The lowest acceptable stat value. If None, no low threshold.
        high: float, None
              The highest acceptable stat value. If None, not high threshold.
        """
        if low is None and high is None:
            raise ValueError("One of 'low' or 'high' needs to be specified.")
        stats = self._make_numpy_convenience()

        num_rejections = 0
        for tid in stats:
            for sid in stats[tid]:
                low_rejections = np.ma.nonzero(stats[tid][sid] < low)[0]
                high_rejections = np.ma.nonzero(stats[tid][sid] > high)[0]

                new_rejections = np.concatenate([low_rejections, high_rejections])
                num_rejections += len(new_rejections)
                stats[tid][sid][new_rejections] = np.ma.masked
                self.rejections[tid][sid] = np.setdiff1d(new_rejections, self.rejections[tid][sid]).tolist()

        return num_rejections

    def reject(self, thinker=None, session=None, trial=None, bulk=None):
        pass

    def _add_bad_span(self, tid, sid, start, end):
        span = [float(start / self._sfreq), float(end / self._sfreq)]

        if tid in self.exclude and sid in self.exclude[tid]:
            self.exclude[tid][sid].append(span)
        elif tid in self.exclude:
            self.exclude[tid][sid] = [span]
        else:
            self.exclude[tid] = {sid: [span]}

    def get_configuratron_exclusions(self, save_to_file=None):
        """
        Creates exclusions to be used by configuratron on next dataset loading.

        Parameters
        ----------
        save_to_file : str (path)
                      If not None, will save the exlcusions as a yaml file for use by the configuratron.

        Returns
        -------
        exclusions
        """

        def get_start_end(tid, sid, trial_id):
            stride = self.dataset.thinkers[tid].sessions[sid].stride
            return trial_id * stride, trial_id * stride + self._sequence_length

        for tid in tqdm.tqdm(self.statistic_lookup, desc="Determining bad spans...", unit='person'):
            for sid in self.statistic_lookup[tid]:
                rejects = self.rejections[tid][sid]
                if len(rejects) == 0:
                    continue

                start, end = get_start_end(tid, sid, rejects[0])
                if len(rejects) == 1:
                    self._add_bad_span(tid, sid, *get_start_end(tid, sid, rejects[0]))
                    continue

                for i, trial_id in enumerate(rejects[1:]):
                    if rejects[i] == trial_id - 1:
                        _, end = get_start_end(tid, sid, trial_id)
                    else:
                        self._add_bad_span(tid, sid, start, end)
                        start, end = get_start_end(tid, sid, trial_id)

                self._add_bad_span(tid, sid, start, end)

        if save_to_file is not None:
            print("Saving to", save_to_file)
            with open(save_to_file, 'w') as f:
                safe_dump(self.exclude, f)

        return self.exclude
