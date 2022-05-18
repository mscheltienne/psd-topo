import time

from bsl import StreamReceiver
from mne import create_info

from . import TopoMapMPL
from .fft import _fft
from .utils._checks import _check_type


def nfb(stream_name: str):
    """Neurofeedback loop.

    Parameters
    ----------
    stream_name : str
        The name of the LSL stream to connect to.
    """
    _check_type(stream_name, (str,), "stream_name")

    # create receiver and feedback
    sr = StreamReceiver(bufsize=1, winsize=1, stream_name=stream_name)

    # retrieve sampling rate and channels
    fs = sr.streams[stream_name].sample_rate
    ch_names = sr.streams[stream_name].ch_list
    # remove unwanted channels: TRIGGER
    ch_names = ch_names[1:]

    # create feedback
    info = create_info(ch_names=ch_names, sfreq=fs, ch_types='eeg')
    info.set_montage('standard_1020')
    feedback = TopoMapMPL(info)

    # wait to fill one buffer
    time.sleep(1)

    # loop for 30 seconds
    while True:
        # retrieve data
        sr.acquire()
        data, _ = sr.get_window()
        # remove trigger channel
        data = data[:, 1:]
        # compute metric
        metric = _fft(data.T, fs=fs, band=(8, 13), dB=True)  # (n_channels, )
        # update feedback
        feedback.update(metric)
