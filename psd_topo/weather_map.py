import time
from typing import Optional, Tuple, Union

import numpy as np
from bsl import StreamReceiver
from mne import create_info

from .fft import _fft
from .topomap import TopomapMPL
from .utils._checks import _check_band, _check_type
from .utils._docs import fill_doc
from .utils._logs import logger, set_log_level


@fill_doc
def weather_map(
    stream_name: str,
    band: Tuple[float, float],
    winsize: float,
    figsize: Optional[Tuple[float, float]] = None,
    verbose: Optional[Union[str, int]] = None,
) -> None:
    """Online loop to create a "weather map" from an EGI recording.

    Parameters
    ----------
    %(stream_name)s
    %(band)s
    %(winsize)s
    %(figsize)s
    %(verbose)s
    """
    set_log_level(verbose)
    _check_type(stream_name, (str,), "stream_name")
    _check_band(band)
    _check_type(winsize, ("numeric",), "winsize")
    if winsize <= 0:
        raise ValueError("The window size must be a strictly positive number.")
    figsize = TopomapMPL._check_figsize(figsize)

    # create receiver and feedback
    sr = StreamReceiver(
        bufsize=winsize, winsize=winsize, stream_name=stream_name
    )

    # retrieve sampling rate and channels
    fs = sr.streams[stream_name].sample_rate
    ch_names = sr.streams[stream_name].ch_list
    # remove trigger channel
    trigger_idx = ch_names.index("TRIGGER")
    ch_idx = np.array(
        [k for k, ch in enumerate(ch_names) if ch != "TRIGGER"]
    )
    ch_names = [ch for ch in ch_names if ch != "TRIGGER"]
    # replace E257 with Cz
    ch_names[ch_names.index("E257")] = "Cz"

    # wait to fill one buffer
    logger.info(
        "Buffer: waiting for an entire %.2f seconds buffer to be fill..",
        winsize,
    )
    time.sleep(winsize)
    logger.info("Buffer: ready!")

    # create feedback
    info = create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
    info.set_montage("GSN-HydroCel-257")
    logger.info("Topomap: creating display window..")
    feedback = TopomapMPL(info, "hsv", figsize)
    logger.info("Topomap: ready!")

    # main loop
    while True:
        # retrieve data
        sr.acquire()
        data, _ = sr.get_window()
        # remove unwanted channels
        trigger = data[:, trigger_idx]  # retrieve trigger channel
        data = data[:, ch_idx]  # retrieve EEG channels
        # apply CAR
        data = (data.T - np.average(data, axis=1)).T
        # compute metric
        fftval = _fft(data.T, fs=fs, band=band, dB=True)  # (n_channels, )
        # update feedback
        feedback.update(fftval)
        if np.any(trigger):
            idx = np.nonzero(trigger)[0][-1]
            feedback.axes.set_title(str(trigger[idx]))
        feedback.redraw()
