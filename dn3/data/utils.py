import tqdm
from .dataset import *


def get_dataset_max_and_min(dataset: DN3ataset):
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


def get_session_lengths(dataste: DN3ataset):
    pass
