import time
from typing import Tuple

import numpy as np
from bsl import StreamReceiver
from mne import create_info

from .fft import _fft
from .topomap import TopomapMPL
from .utils._checks import _check_band, _check_type
from .utils._logs import logger


def nfb(
    stream_name: str,
    band: Tuple[float, float] = (8, 13),
    winsize: float = 5.0,
    figsize: Tuple[float, float] = (3, 3),
):
    """Neurofeedback loop.

    Parameters
    ----------
    stream_name : str
        The name of the LSL stream to connect to.
    %(band)s
    winsize : float
        Duration of the acquisition window in seconds.
    %(figsize)s
    """
    _check_type(stream_name, (str,), "stream_name")
    _check_band(band)
    _check_type(winsize, ("numeric",), "winsize")
    if winsize <= 0:
        raise ValueError("The window size must be a strictly positive number.")

    # create receiver and feedback
    sr = StreamReceiver(
        bufsize=winsize, winsize=winsize, stream_name=stream_name
    )

    # retrieve sampling rate and channels
    fs = sr.streams[stream_name].sample_rate
    ch_names = sr.streams[stream_name].ch_list
    # remove unwanted channels
    ch2remove = ("TRIGGER", "TRG", "X1", "X2", "X3", "A1", "A2")
    ch_idx = np.array(
        [k for k, ch in enumerate(ch_names) if ch not in ch2remove]
    )
    # filter channel name list
    ch_names = [ch for ch in ch_names if ch not in ch2remove]

    # wait to fill one buffer
    logger.info(
        "Buffer: waiting for an entire %.2f seconds buffer to be fill..",
        winsize,
    )
    time.sleep(winsize)
    logger.info("Buffer: ready!")

    # create feedback
    info = create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    info.set_montage("standard_1020")
    logger.info("Topomap: creating display window..")
    feedback = TopomapMPL(info, figsize)
    logger.info("Topomap: ready!")

    # main loop
    while True:
        # retrieve data
        sr.acquire()
        data, _ = sr.get_window()
        # remove trigger channel
        data = data[:, ch_idx]
        # compute metric
        fftval = _fft(data.T, fs=fs, band=band, dB=True)  # (n_channels, )
        # update feedback
        feedback.update(fftval)
