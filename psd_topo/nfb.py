import time

from bsl import StreamReceiver

from . import fft
from .utils._checks import _check_type


def nfb(stream_name: str):
    """A basic NFB loop that runs 30 seconds.

    Parameters
    ----------
    stream_name : str
        The name of the LSL stream to connect to.
    """
    _check_type(stream_name, (str,), 'stream_name')

    # create receiver and feedback
    sr = StreamReceiver(bufsize=1, winsize=1, stream_name=stream_name)

    # retrieve sampling rate
    fs = sr.streams[stream_name].sample_rate

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
        metric = fft(data.T, fs=fs, band=(8, 13))  # (n_channels, )
        # update feedback
