from .dataset import Dataset, DN3ataset


class MultiDomainContainer:

    def __init__(self, *domains):
        """
        A container class that facilitates training with multiple disparate domains at the same time.

        Parameters
        ----------
        domains: DN3ataset
                 These should be various different :any:`DN3atatset`s (e.g. Thinkers, Sessions, Datasets) to be used as
                 separate domains for domain-sensitive training.
        """
        self.datasets = domains

    def get_domains(self):
        return self.datasets

    def __len__(self):
        return max([len(ds) for ds in self.datasets])


def fracture_dataset_into_thinker_domains(dataset: Dataset):
    """
    Simple helper function to break apart a dataset of many thinkers into a thinker-per-domain configuration.

    Parameters
    ----------
    dataset: Dataset

    Returns
    -------
    container: MultiDomainContainer
    """
    return MultiDomainContainer(*list(dataset.thinkers.values()))
