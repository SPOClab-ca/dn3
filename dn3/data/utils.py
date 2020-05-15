import mne
from numpy import unique
from collections import Iterable

import torch
from torch.utils.data.dataset import random_split


def rand_split(dataset, frac=0.75):
    if frac >= 1:
        return dataset
    samples = len(dataset)
    return random_split(dataset, lengths=[round(x) for x in [samples*frac, samples*(1-frac)]])


def unfurl(_set: set):
    _list = list(_set)
    for i in range(len(_list)):
        if not isinstance(_list[i], Iterable):
            _list[i] = [_list[i]]
    return tuple(x for z in _list for x in z)


def min_max_normalize(x: torch.Tensor):
    if len(x.shape) == 2:
        return (x - x.min()) / (x.max() - x.min())
    elif len(x.shape) == 3:
        return (x - torch.min(torch.min(x, keepdim=True, dim=-1)[0], keepdim=True, dim=-1)[0]) / \
               (torch.max(torch.max(x, keepdim=True, dim=-1)[0], keepdim=True, dim=-1)[0] -
                torch.min(torch.min(x, keepdim=True, dim=-1)[0], keepdim=True, dim=-1)[0])


def map_events_to_class_labels(events, ):
    pass


def make_epochs_from_raw(raw: mne.io.Raw, tmin, tlen, event_ids=None, baseline=None, decim=1, filter_bp=None,
                         drop_bad=False):
    sfreq = raw.info['sfreq']
    if filter_bp is not None:
        if isinstance(filter_bp, (list, tuple)) and len(filter_bp) == 2:
            raw.load_data()
            raw.filter(filter_bp[0], filter_bp[1])
        else:
            print('Filter must be provided as a two-element list [low, high]')

    if isinstance(event_ids, dict):
        events = mne.events_from_annotations(raw, event_id=event_ids)[0]
    else:
        try:
            events = mne.find_events(raw)
        except ValueError():
            print("Falling back to annotations")

    # Map various event codes to a incremental label system
    _, events[:, -1] = unique(events[:, -1], return_inverse=True)

    return mne.Epochs(raw, events, tmin=tmin, tmax=tmin + tlen - 1 / sfreq, preload=True, decim=decim,
                      baseline=baseline, reject_by_annotation=drop_bad)
