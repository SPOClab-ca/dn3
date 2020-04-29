import torch
from torch.utils.data.dataset import random_split


def rand_split(dataset, frac=0.75):
    if frac >= 1:
        return dataset
    samples = len(dataset)
    return random_split(dataset, lengths=[round(x) for x in [samples*frac, samples*(1-frac)]])


def min_max_normalize(x: torch.Tensor):
    if len(x.shape) == 2:
        return (x - x.min()) / (x.max() - x.min())
    elif len(x.shape) == 3:
        return (x - torch.min(torch.min(x, keepdim=True, dim=-1)[0], keepdim=True, dim=-1)[0]) / \
               (torch.max(torch.max(x, keepdim=True, dim=-1)[0], keepdim=True, dim=-1)[0] -
                torch.min(torch.min(x, keepdim=True, dim=-1)[0], keepdim=True, dim=-1)[0])