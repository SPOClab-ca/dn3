import moabb.datasets as mbd
from dn3.utils import make_epochs_from_raw, DN3ConfigException


# These are hard-coded in MOABB, if you are having trouble with this, check if the "sign" has changed
SUPPORTED_DATASETS = {
    'BNCI2014001': mbd.BNCI2014001,
    'PhysionetMI': mbd.PhysionetMI,
    'Cho2017': mbd.Cho2017
}


class MoabbDataset:

    def __init__(self, ds_name, data_location, **kwargs):
        try:
            self.ds = SUPPORTED_DATASETS[ds_name](**kwargs)
        except KeyError:
            raise DN3ConfigException("No support for MOABB dataset called {}".format(ds_name))
        self.path = data_location
        self.data_dict = None
        self.run_map = dict()

    def _get_ds_data(self):
        if self.data_dict is None:
            self.ds.download(path=str(self.path), update_path=True)
            self.data_dict = self.ds.get_data()

    def get_pseudo_mapping(self, exclusion_cb):
        self._get_ds_data()
        # self.run_map = {th: dict() for th in self.data_dict.keys()}
        # DN3 collapses sessions and runs
        mapping = dict()

        for th in self.data_dict.keys():
            for sess in self.data_dict[th].keys():
                for run in self.data_dict[th][sess].keys():
                    id = '-'.join((str(th), str(sess), str(run)))
                    self.run_map[id] = self.data_dict[th][sess][run]
                    if exclusion_cb(self.data_dict[th][sess][run].filenames[0], str(th), id):
                        continue
                    if th in mapping:
                        mapping[th].append(id)
                    else:
                        mapping[th] = [id]
        return mapping

    def get_raw(self, pseudo_path):
        return self.run_map[str(pseudo_path)]
