import torch
import numpy as np

_LEFT_NUMBERS = list(reversed(range(1, 9, 2)))
_RIGHT_NUMBERS = list(range(2, 10, 2))

STANDARD_10_10_CHS_PLUS = [
    # EEG
    "NZ",
    "FP1", "FPZ", "FP2",
    "AF7", "AF3", "AFZ", "AF4", "AF8",
    "F9", *["F{}".format(n) for n in _LEFT_NUMBERS], "FZ", *["F{}".format(n) for n in _RIGHT_NUMBERS], "F10",

    "FT9", "FT7", *["FC{}".format(n) for n in _LEFT_NUMBERS[1:]], "FCZ",
    *["FC{}".format(n) for n in _RIGHT_NUMBERS[:-1]], "FT8", "FT10",
                                                                                                                                  
    "T9", "T7", "T3",  *["C{}".format(n) for n in _LEFT_NUMBERS[1:]], "CZ",
    *["C{}".format(n) for n in _RIGHT_NUMBERS[:-1]], "T4", "T8", "T10",

    "TP9", "TP7", *["CP{}".format(n) for n in _LEFT_NUMBERS[1:]], "CPZ",
    *["CP{}".format(n) for n in _RIGHT_NUMBERS[:-1]], "TP8", "TP10",

    "P9", "P7", "T5",  *["P{}".format(n) for n in _LEFT_NUMBERS[1:]], "PZ",
    *["P{}".format(n) for n in _RIGHT_NUMBERS[:-1]],  "T6", "P8", "P10",

    "PO7", "PO3", "POZ", "PO4", "PO8",
    "O1",  "OZ", "O2",
    "IZ",
    # EOG
    "VEOGL", "VEOGR", "HEOGL", "HEOGR",
    # References
    "A1", "A2",
    # SCALING
    "SCALE",
    # Extra
    ["EX{}".format(n) for n in range(1, 4)]
]
EOG_INDS = [STANDARD_10_10_CHS_PLUS.index(ch) for ch in ["VEOGL", "VEOGR", "HEOGL", "HEOGR"]]
REF_INDS = [STANDARD_10_10_CHS_PLUS.index(ch) for ch in ["A1", "A2"]]
SCALE_IND = -5 + len(STANDARD_10_10_CHS_PLUS)
EXTRA_INDS = list(range(len(STANDARD_10_10_CHS_PLUS) - 4, len(STANDARD_10_10_CHS_PLUS)))


def map_channels_1010(channel_names: list, EOG=None, reference=None, extra_channels=None):
    """
    Maps provided channel names to the standard format, will automatically map EOG and extra channels if they have been
    named according to standard convention. Otherwise provide as keyword arguments.
    :param channel_names: List of channel names from dataset
    :param EOG: Must be a single channel name, or left and right EOG channels, optionally vertical L/R then horizontal
                L/R for four channels.
    :param reference: None, one or two channels to be used as references. If two, should be left and right in that
                      order.
    :param extra_channels: List of up to 3 extra channels to include
    :return: torch.Tensor -> Mapping matrix from dataset channel space to standard space.
    """
    map = np.zeros((len(channel_names), len(STANDARD_10_10_CHS_PLUS)))

    if isinstance(EOG, str):
        EOG = [EOG] * 4
    elif EOG is None:
        EOG = []
    elif len(EOG) == 2:
        EOG = EOG * 2
    else:
        assert len(EOG) == 4
    for eog_map, eog_std in zip(EOG, EOG_INDS):
        try:
            map[channel_names.index(eog_map), eog_std] = 1.0
        except ValueError:
            raise ValueError("EOG channel {} not found in provided channels.".format(eog_map))

    if isinstance(reference, str):
        reference = [reference] * 2
    elif reference is None:
        reference = []
    else:
        assert len(reference) <= 2
    for ref_map, ref_std in zip(reference, REF_INDS):
        try:
            map[channel_names.index(ref_map), ref_std] = 1.0
        except ValueError:
            raise ValueError("Reference channel {} not found in provided channels.".format(ref_map))

    if isinstance(extra_channels, str):
        extra_channels = [extra_channels]
    elif extra_channels is None:
        extra_channels = []
    assert len(extra_channels) <= 3
    for ch, place in zip(extra_channels, EXTRA_INDS):
        map[channel_names.index(ch), place] = 1.0

    for i, ch in enumerate(channel_names):
        if ch not in EOG and ch not in reference and ch not in extra_channels:
            try:
                map[i, STANDARD_10_10_CHS_PLUS.index(str(ch).upper())] = 1.0
            except ValueError:
                print("Warning: channel {} not found in standard layout. Skipping...".format(ch))
                continue

    # Normalize for if multiple values mapped to single location
    summed = map.sum(axis=0)[np.newaxis, :]
    return torch.from_numpy(np.divide(map, summed, out=np.zeros_like(map), where=summed != 0)).float()
