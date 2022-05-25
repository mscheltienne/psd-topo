import mne
from mne.io.pick import _DATA_CH_TYPES_ORDER_DEFAULT
from pyxdf import load_xdf


def read_raw_xdf(fname):
    """
    Read raw XDF files saved with the LabRecorder.

    Parameters
    ----------
    fname : file-like
        Path to the -raw.fif file to load.

    Returns
    -------
    raw : Raw
        MNE raw instance.
    """
    streams, _ = load_xdf(fname)
    assert len(streams) == 1
    stream = streams[0]
    del streams

    # retrieve information
    ch_names, ch_types, units = _get_eeg_ch_info(stream)
    sfreq = int(stream["info"]["nominal_srate"][0])
    data = stream["time_series"].T

    # create MNE raw
    info = mne.create_info(ch_names, sfreq, ch_types)
    raw = mne.io.RawArray(data, info, first_samp=0)

    # drop AUX and reference channels
    raw.drop_channels(["X1", "X2", "X3", "A2"])

    # scaling
    def uVolt2Volt(timearr):
        """Converts from uV to Volts."""
        return timearr * 1e-6

    raw.apply_function(
        uVolt2Volt, picks=["eeg", "eog", "ecg", "misc"], channel_wise=True
    )

    # rename trigger channels and set channel type
    raw.set_channel_types(mapping=dict(TRG='stim'))
    raw.rename_channels(mapping=dict(TRG='TRIGGER'))
    assert raw.ch_names[-1] == 'TRIGGER'  # sanity-check
    raw.reorder_channels([raw.ch_names[-1]] + raw.ch_names[:-1])

    return raw


def _get_eeg_ch_info(stream):
    """
    Extract the info for each eeg channels (label, type and unit)
    """
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
