import bisect
from .dataset import Dataset, DN3ataset, ConcatDataset, np, same_channel_sets, torch


class MultiDataset(ConcatDataset, DN3ataset):
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

    Parameters
    ----------
    datasets: list
              The datasets (:any:`Thinker` and :any:`Dataset`) to combine into a single aggregated dataset.
    """
    def __init__(self, datasets, return_dataset_idx=False):
        super(PersonIDAggregator, self).__init__(datasets)
        self.missing_return_ids = list()
        for d in datasets:
            if hasattr(d, 'return_task_id'):
                d.return_task_id = False
            return_ids = [False] * 4 # dataset, person, sess, trial
            if hasattr(d, 'return_dataset_id') and d.return_dataset_id:
                return_ids[0] = True
            if hasattr(d, 'return_person_id') and d.return_person_id:
                return_ids[1] = True
            if hasattr(d, 'return_session_id') and d.return_session_id:
                return_ids[2] = True
            if hasattr(d, 'return_trial_id') and d.return_trial_id:
                return_ids[3] = True
            self.missing_return_ids.append(return_ids)
        self.missing_return_ids = np.array(self.missing_return_ids)
        concerns = self.missing_return_ids[:, 2:].sum(axis=0)
        if np.any(np.logical_and(concerns < len(datasets), concerns > 0)):
            raise ValueError("All datasets do not have the same return status for session and trials.")
        self.people_offset = self.cumsum([d.get_thinkers() if hasattr(d, 'get_thinkers') else [1] for d in datasets])
        self._return_dataset_idx = return_dataset_idx

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

        fetched = list(self.datasets[dataset_idx][sample_idx])

        x = fetched[0]
        p_id = 0
        p_id_idx = 2 if fetched[1].dtype == torch.bool else 1
        # Check if person Id is returned
        if self.missing_return_ids[dataset_idx, 1]:
            p_id = fetched[p_id_idx]
            p_id_idx += 1

        p_id = torch.tensor(p_id + bisect.bisect_right(self.people_offset, p_id))
        to_return = [x, p_id, *fetched[p_id_idx:]]
        if self._return_dataset_idx:
            to_return.insert(1, torch.tensor(dataset_idx, dtype=torch.long))
        return self._execute_transforms(*to_return)


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
