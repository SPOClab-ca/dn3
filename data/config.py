import yaml
import mne.io as loader

from pathlib import Path
from .dataset import Dataset, RawTorchRecording, EpochTorchRecording, Thinker


_SUPPORTED_EXTENSIONS = {
    '.edf': loader.read_raw_edf,
    '.fif': loader.read_raw_fif,
    # TODO: add in the rest of this support
}


class DNPTConfigException(BaseException):
    """
    Exception to be triggered when DNPT-configuration parsing fails.
    """
    pass


class ExperimentConfig:
    """
    Parses DNPT configuration files. Checking the DNPT token for listed datasets.
    """
    def __init__(self, config_filename: str, auto_create_datasets=True, adopt_extra_config=True):
        """
        Parses DNPT configuration files. Checking the DNPT token for listed datasets.
        Parameters
        ----------
        config_filename : str
                          String for path to yaml formatted configuration file
        adopt_extra_config : bool
                             For any additional tokens aside from DNPT and specified datasets, integrate them into this
                             object for later use. Defaults to True.
        """
        self._original_config = yaml.load(open(config_filename, 'r'))
        working_config = self._original_config.copy()

        if 'DNPT' not in working_config.keys():
            raise DNPTConfigException("Toplevel `DNPT` not found in: {}".format(config_filename))
        if 'datasets' not in working_config['DNPT'].keys():
            raise DNPTConfigException("`datasets` not found in {}".format([k.lower() for k in
                                                                           working_config["DNPT"].keys()]))
        if not isinstance(working_config['DNPT']['datasets'], list):
            raise DNPTConfigException("`datasets` must be a list")
        self._dataset_names = working_config['DNPT']['datasets']

        self.datasets = dict()
        for ds in self._dataset_names:
            if ds not in working_config.keys():
                raise DNPTConfigException("Dataset: {} not found in {}".format(
                    ds, [k for k in working_config.keys() if k != 'DNPT']))
            self.datasets[ds] = DatasetConfig(working_config.pop(ds))
        print("Found {} dataset(s).".format(len(self.datasets)))

        self.experiment = working_config.pop('DNPT')
        if adopt_extra_config:
            self.__dict__.update(working_config)


class DatasetConfig:
    def __init__(self, config: dict):
        self._original_config = config.copy()
        # Required args
        try:
            self.toplevel = config.pop('toplevel')
        except KeyError as e:
            raise DNPTConfigException("Could not find required value: {}".format(e.args[0]))

        # Known optional args
        def get_pop(key, default=None):
            config.setdefault(key, __default=default)
            return config.pop(key)
        self.data_max = get_pop('max')
        self.data_min = get_pop('min')

        # The rest
        self.__dict__.update(config)

    def scan_toplevel(self):
        print("Scanning {}. If there are a lot of files, this may take a while...".format(self.toplevel))
        return list()

    def auto_mapping(self):
        """
        Generates a mapping of sessions and people assuming files are stored in the structure:
         `toplevel`/(*optional - <version>)/<person-id>/<session-id>.{ext}
        Returns
        -------

        """
        files = self.scan_toplevel()
        mapping = dict()
        for sess_file in files:
            sess_file = Path(sess_file)
            person = sess_file.parent.name
            if person in mapping:
                mapping[person].append(str(sess_file))
            else:
                mapping[person] = [str(sess_file)]
        return mapping

    def auto_construct_dataset(self, mapping=None):
        """
        This creates a dataset using the config values. If tmax and tmin are specified in the config, creates epoched
        dataset, otherwise Raw.
        Parameters
        ----------
        mapping : dict, optional
                A dict specifying a list of sessions (as paths to files) for each person_id in the dataset. e.g.
                {
                  person_1: [sess_1.edf, ...],
                  person_2: [sess_1.edf],
                  ...
                }
                If not specified, will use `auto_mapping()` to generate.
        Returns
        -------
        An instance of `Dataset`, constructed according to mapping.
        """
        if mapping is None:
            return self.auto_construct_dataset(self.auto_mapping())
