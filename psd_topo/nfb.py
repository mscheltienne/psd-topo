import time

from bsl import StreamReceiver

from . import fft


def basic(stream_name: str):
    """A basic NFB loop that runs 30 seconds.

    Parameters
    ----------
    stream_name : str
        The name of the LSL stream to connect to.
    """
    # create receiver and feedback
    sr = StreamReceiver(bufsize=1, winsize=1, stream_name=stream_name)

    # retrieve sampling rate
    fs = sr.streams[stream_name].sample_rate

    # wait to fill one buffer
    time.sleep(1)

    # loop for 30 seconds
    start = time.time()
    while time.time() - start <= 30:
        # retrieve data
        sr.acquire()
        data, _ = sr.get_window()
        # compute metric
        metric = fft(data.T, fs=fs, band=(8, 13))
