import bisect
from .dataset import Dataset, DN3ataset, ConcatDataset, np, same_channel_sets, torch


class MultiDataset(ConcatDataset):
    """
    A wrapper for DN3 datasets so that they can be concatenated, and the `get_targets()` method is not broken.

    This also supports :any:`DomainAwareProcess`, where each dataset used to construct this object is considered
    a seperate domain.

    This allows for cross-dataset training that supports the normal DN3 :any:`BaseProcess` interface.
    """
    def get_targets(self):
        targets = list()
        for ds in self.datasets:
            if hasattr(ds, 'get_targets'):
                targets.append(ds.get_targets())
            else:
                targets.append([d[-1] for d in ds])
        return np.concatenate(targets)

    def get_domains(self):
        return self.datasets

    @property
    def sfreq(self):
        sfreq = set(d.sfreq for d in self.datasets)
        if len(sfreq) > 1:
            print("Warning: Multiple sampling frequency values found. Over/re-sampling may be necessary.")
            return unfurl(sfreq)
        sfreq = sfreq.pop()
        return sfreq

    @property
    def channels(self):
        channels = [d.channels for d in self.datasets]
        if not same_channel_sets(channels):
            raise ValueError("Multiple channel sets found. A consistent mapping like Deep1010 is necessary to proceed.")
        channels = channels.pop()
        return channels

    @property
    def sequence_length(self):
        sequence_length = set(d.sequence_length for d in self.datasets)
        if len(sequence_length) > 1:
            print("Warning: Multiple sequence lengths found. A cropping transformation may be in order.")
            return unfurl(sequence_length)
        sequence_length = sequence_length.pop()
        return sequence_length


class PersonIDAggregator(MultiDataset):
    """
    A wrapper to concatenate DN3 datasets so that they retain a consistent person index across multiple datasets.

    Automatically turns off return_task_id and dataset_ids are modified to be the 0 indexed id of the order that
    datasets are provided in.
    """
    def __init__(self, datasets, return_dataset_idx=False, oversample_by_dataset=False):
        super(PersonIDAggregator, self).__init__(datasets)
        self.people_offset = self.cumsum([d.get_thinkers() if hasattr(d, 'get_thinkers') else [1] for d in datasets])
        self._return_dataset_idx = return_dataset_idx
        self.oversample_by_dataset = oversample_by_dataset

    def __str__(self):
        pass

    def num_people(self):
        return self.people_offset[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)

        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        fetched = self.datasets[dataset_idx][sample_idx]



        if len(fetched) > 1 and fetched[1].dtype == torch.bool:
            fetched = [fetched[0], fetched[1], torch.tensor(0).long(), *fetched[2:]]
        else:
            fetched = [fetched[0], torch.zeros(1), *fetched[1:]]

        # Skip deep1010 mask
        if fetched[1].dtype == torch.bool:
            x, ds_p_id = fetched[0], fetched[2]
        else:
            x, ds_p_id = fetched[:2]

        to_return = [x, ds_p_id + bisect.bisect_right(self.people_offset, ds_p_id), *fetched[2:]]
        if self._return_dataset_idx:
            to_return.insert(1, torch.tensor(dataset_idx, dtype=torch.long))
        return to_return


def fracture_dataset_into_thinker_domains(dataset: Dataset):
    """
    Simple helper function to break apart a dataset of many thinkers into a thinker-per-domain configuration.

    Parameters
    ----------
    dataset: Dataset

    Returns
    -------
    container: MultiDataset
    """
    return MultiDataset(list(dataset.thinkers.values()))
