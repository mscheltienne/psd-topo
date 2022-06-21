from typing import List, Tuple

import mne
import numpy as np
from mne.io import BaseRaw
from mne.io.pick import _DATA_CH_TYPES_ORDER_DEFAULT
from pyxdf import load_xdf

from ..config import load_config


def read_raw_xdf(fname) -> List[BaseRaw]:
    """Read raw XDF files saved with the LabRecorder.

    All streams that have "WS-" in their name will be loaded in a separate
    Raw instance. The TRIGGER channel from the "WS-" streams will be replaced
    with the data on the marker stream.

    Parameters
    ----------
    fname : file-like
        Path to the -raw.fif file to load.

    Returns
    -------
    raws : list of Raw
        List of loaded MNE raw instances.
    """
    streams, _ = load_xdf(fname)
    amp_prefix, trigger_stream_name = load_config()
    eeg_streams = _find_streams(streams, stream_name=amp_prefix)
    assert len(eeg_streams) != 0  # sanity-check
    marker_stream = _find_streams(streams, stream_name=trigger_stream_name)
    assert len(marker_stream) in (0, 1)  # sanity-check

    # retrieve the marker stream
    if len(marker_stream) == 1:
        stream = marker_stream[0][1]
        marker = (stream["time_stamps"], stream["time_series"][:, 0])
        del stream
    else:
        marker = None

    # create the raw instances
    raws = list()
    for _, stream in eeg_streams:
        # retrieve information
        ch_names, ch_types, units = _get_eeg_ch_info(stream)
        sfreq = int(eval(stream["info"]["nominal_srate"][0]))
        data = stream["time_series"].T

        # create MNE raw
        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(data, info, first_samp=0)

        # drop AUX and reference channels
        raw.drop_channels(["X1", "X2", "X3", "A2"])

        # rename trigger channels and set channel type
        raw.set_channel_types(mapping=dict(TRG="stim"))
        raw.rename_channels(mapping=dict(TRG="TRIGGER"))
        assert raw.ch_names[-1] == "TRIGGER"  # sanity-check
        raw.reorder_channels([raw.ch_names[-1]] + raw.ch_names[:-1])

        # scaling
        def uVolt2Volt(timearr):
            """Convert from uV to Volts."""
            return timearr * 1e-6

        raw.apply_function(uVolt2Volt, picks="eeg", channel_wise=True)

        # add marker on trigger channel
        if marker is not None:
            events = list()
            for ts, value in zip(*marker):
                next_index = np.searchsorted(stream["time_stamps"], ts)
                events.append([next_index, 0, value])
            events = np.array(events)
            raw.add_events(events, stim_channel="TRIGGER", replace=True)

        # save and clean-up
        del stream
        raws.append(raw)

    return raws


def _find_streams(
    streams: List[dict], stream_name: str
) -> List[Tuple[int, dict]]:
    """Find the stream including 'stream_name' in the name attribute.

    Parameters
    ----------
    streams : list of dict
        List of stream dictionary loaded by pyxdf.
    stream_name : str
        Substring that has to be present in the name attribute.

    Returns
    -------
    list of tuples : (k: int, stream: dict)
        k is the idx of stream in streams.
        stream is the stream that contains stream_name in its name.
    """
    return [
        (k, stream)
        for k, stream in enumerate(streams)
        if stream_name in stream["info"]["name"][0]
    ]


def _get_eeg_ch_info(stream: dict) -> Tuple[List, List, List]:
    """Extract the info for each eeg channels (label, type and unit)."""
    ch_names, ch_types, units = [], [], []

    # get channels labels, types and units
    for ch in stream["info"]["desc"][0]["channels"][0]["channel"]:
        ch_type = ch["type"][0].lower()
        if ch_type not in _DATA_CH_TYPES_ORDER_DEFAULT:
            # to be changed to a dict if to many entries exist.
            ch_type = "stim" if ch_type == "markers" else ch_type
            ch_type = "misc" if ch_type == "aux" else ch_type

        ch_names.append(ch["label"][0])
        ch_types.append(ch_type)
        units.append(ch["unit"][0])

    return ch_names, ch_types, units
