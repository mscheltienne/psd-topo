import time
from typing import Tuple

import numpy as np
from bsl import StreamReceiver
from mne import create_info

from .feedback import FeedbackMPL
from .fft import _fft
from .utils._checks import _check_band, _check_type


def nfb(stream_name: str, band: Tuple[float, float] = (8, 13)):
    """Neurofeedback loop.

    Parameters
    ----------
    stream_name : str
        The name of the LSL stream to connect to.
    %(band)s
    """
    _check_type(stream_name, (str,), "stream_name")
    _check_band(band)

    # create receiver and feedback
    sr = StreamReceiver(bufsize=1, winsize=1, stream_name=stream_name)

    # retrieve sampling rate and channels
    fs = sr.streams[stream_name].sample_rate
    ch_names = sr.streams[stream_name].ch_list
    # remove unwanted channels
    ch2remove = ("TRIGGER", "TRG", "X1", "X2", "X3", "A1", "A2")
    ch_idx = np.array(
        [k for k, ch in enumerate(ch_names) if ch not in ch2remove]
    )
    # retrieve indices of channels to average for line-plot
    ch2average = ("O1", "O2")
    ch2average_idx = np.array(
        [k for k, ch in enumerate(ch_names) if ch in ch2average]
    )
    # filter channel name list
    ch_names = [ch for ch in ch_names if ch not in ch2remove]

    # create feedback
    info = create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    info.set_montage("standard_1020")
    feedback = FeedbackMPL(info)

    # wait to fill one buffer
    time.sleep(1)

    # loop for 10 seconds
    start = time.time()
    while time.time() - start <= 10:
        # retrieve data
        sr.acquire()
        data, tslist = sr.get_window()
        # remove trigger channel
        data = data[:, ch_idx]
        # compute metric
        fftval = _fft(data.T, fs=fs, band=band, dB=True)  # (n_channels, )
        avg = np.average(fftval[ch2average_idx])  # float
        # update feedback
        feedback.update(fftval, tslist[0], avg)
