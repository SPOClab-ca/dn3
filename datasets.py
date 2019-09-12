import copy
import mne
import dataloaders
from moabb import datasets


class DatasetInterface:
    """
    This should be subclassed to create a new dataset.
    """

    def subjects(self):
        """
        :return: List of subjects present in the dataset
        """
        raise NotImplementedError()

    def sessions(self):
        """
        :return: List of named sessions in the dataset
        """
        raise NotImplementedError()

    def runs(self):
        """
        :return: List of runs in the dataset
        """
        raise NotImplementedError()

    def load(self, *subjects, session=-1, run=-1, fully_label=True):
        raise NotImplementedError()


class BNCI2014001(DatasetInterface, datasets.BNCI2014001):
    """
    Also known as the BCI competitions IV dataset 2a dataset.
    Consists of 9 subjects performing a 4-way motor-imagery task

    While dataset is downloaded and stored with the moabb package, specifying lazy loading will save the data in mne
    native formats (if not already done), and allow for disk-based loading. Use the cacheing mechanisms from the
    dataloaders to prevent thrashing the disk.
    """
    def __init__(self, location=None, lazy_load=True):
        # TODO load raws without preloading
        super().__init__()
        # ds.download(path=location)
        self.raw_data = self.get_data()
        self._sessions = [list(self.raw_data[x].keys()) for x in self.raw_data]
        self._runs = [list(self.raw_data[x][y].keys()) for x in self.raw_data for y in self.raw_data[x]]

    def subjects(self):
        return self.subject_list

    def sessions(self):
        return self._sessions

    def runs(self):
        return self._runs

    # def epoch_all(self, *subjects, session=-1, run=-1, **epochs_dataloader_kwargs):
    #     def epoch_me(raw):
    #         events = mne.find_events(raw)
    #         return dataloaders.EpochsDataLoader(raw, events, **epochs_dataloader_kwargs)
    #
    #     epoched = copy.copy(self.data)
    #     combined = None
    #     for i in self.subject_list:
    #         for j in self._sessions:
    #             for k in self._runs:
    #                 epoched[i][j][k] = epoch_me(self.data[i][j][k])
    #             if combine:
    #                 combined = dataloaders.labelled_concat(*epoched[i][j].values())
    #         if combine:
    #             combined =
    #
    #     return epoched if combined is None else combined

