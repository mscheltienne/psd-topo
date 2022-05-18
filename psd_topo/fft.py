from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .utils._checks import _check_band, _check_type
from .utils._docs import copy_doc, fill_doc


@fill_doc
def fft(data: NDArray[float], fs: float, band: Tuple[float, float], dB: bool):
    """Apply FFT to the data after applying a hamming window.

    Parameters
    ----------
    data : array
        2D array of shape (n_channels, n_times) containing the received data.
    fs : float
        Sampling frequency in Hz.
    %(band)s
    dB : bool
        If True, the fftval are converted to dB with 10 * np.log10(fftval).

    Returns
    -------
    fftval : array
        1D array of shape (n_channels, ) containing the absolute value of the
        FFT for all channels averaged across the frequency band.
    """
    _check_type(data, (np.ndarray,), "data")
    if data.ndim != 2:
        raise ValueError(
            "The data array 'data' must be a 2D array of shape "
            "(n_channels, n_times)."
        )
    _check_type(fs, ("numeric",), "fs")
    if 0 <= fs:
        raise ValueError(
            "The sampling frequency 'fs' must be strictly positive."
        )
    _check_band(band)
    _check_type(dB, (bool,), "dB")
    return _fft(data, fs, band, dB)


@copy_doc(fft)
def _fft(data: NDArray[float], fs: float, band: Tuple[float, float], dB: bool):
    winsize = data.shape[-1]
    # mutliply the data with a window
    window = np.hamming(winsize)
    data = data * window
    # retrieve fft
    frequencies = np.fft.rfftfreq(winsize, 1 / fs)
    band_idx = np.where((band[0] <= frequencies) & (frequencies <= band[1]))[0]
    fftval = np.abs(np.fft.rfft(data, axis=-1)[:, band_idx])
    fftval = np.average(fftval, axis=1)
    fftval = 10 * np.log10(fftval) if dB else fftval
    return fftval
