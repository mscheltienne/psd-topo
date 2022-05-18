import time
from typing import Tuple

import numpy as np
from bsl import StreamReceiver
from mne import create_info

from .fft import _fft
from .topomap import TopoMapMPL
from .utils._checks import _check_band, _check_type


def nfb(stream_name: str, band: Tuple[float, float] = (8, 13), figsize=(6, 3)):
    """Neurofeedback loop.

    Parameters
    ----------
    stream_name : str
        The name of the LSL stream to connect to.
    %(band)s
    %(figsize)s
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
    ch_names = [ch for ch in ch_names if ch not in ch2remove]

    # create feedback
    info = create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    info.set_montage("standard_1020")
    feedback = TopoMapMPL(info, figsize)

    # wait to fill one buffer
    time.sleep(1)

    # loop for 30 seconds
    start = time.time()
    while time.time() - start <= 30:
        # retrieve data
        sr.acquire()
        data, _ = sr.get_window()
        # remove trigger channel
        data = data[:, ch_idx]
        # compute metric
        metric = _fft(data.T, fs=fs, band=band, dB=True)  # (n_channels, )
        # update feedback
        feedback.update(metric)
