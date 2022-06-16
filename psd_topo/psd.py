import numpy as np
from matplotlib import pyplot as plt
from mne.io import BaseRaw
from mne.time_frequency import psd_welch

from ._typing import FigSize, Picks
from .utils._checks import _check_type
from .utils._docs import fill_doc


@fill_doc
def plot_psd(
    raw: BaseRaw,
    winsize: float,
    overlap: float,
    fmin: float = 8.0,
    fmax: float = 13.0,
    picks: Picks = "eeg",
    figsize: FigSize = (5, 5),
):
    """Plot the power spectral density using welch windows.

    Parameters
    ----------
    raw : Raw
        MNE raw instance.
    winsize : float
        Duration of a welch windows in seconds
    overlap : float
        Overlap between welch windows in seconds.
    fmin : float
        Minimum frequency of interest in Hz.
    fmax : float
        Maximum frequency of interest in Hz.
    %(picks_all)s
    %(figsize)s

    Returns
    -------
    f : Figure
    ax : Axes
    """
    _check_type(raw, (BaseRaw,), "raw")
    _check_type(winsize, ("numeric",), "winsize")
    _check_type(overlap, ("numeric",), "winsize")
    _check_type(fmin, ("numeric",), "winsize")
    _check_type(fmax, ("numeric",), "winsize")
    assert 0 < winsize
    assert 0 < overlap
    assert overlap < winsize
    assert 0 < fmin
    assert 0 < fmax
    kwargs = dict(
        fmin=fmin,
        fmax=fmax,
        n_fft=int(winsize * raw.info["sfreq"]),
        n_per_seg=int(winsize * raw.info["sfreq"]),
        n_overlap=int(overlap * raw.info["sfreq"]),
        picks=picks,
    )
    psds, _ = psd_welch(
        raw,
        average=None,
        window="hamming",
        **kwargs,
    )
    assert psds.ndim == 3  # sanity-check, shape is (n_ch, n_freqs, n_seg)
    psds = np.average(psds, axis=(0, 1))  # average across the channels / freqs
    assert psds.ndim == 1  # sanity-check
    # recreate time-axis
    step = winsize - overlap
    times = np.arange(0, step * psds.size, step) + winsize / 2
    # scaling to dB
    scaling = 1e6  # default scaling
    psds *= scaling * scaling
    np.log10(np.maximum(psds, np.finfo(float).tiny), out=psds)
    psds *= 10
    # create figure
    f, ax = plt.subplots(1, 1)
    ax.plot(times, psds)
    # format figure
    ax.set_xlabel("Time (s)")
    ax.set_xticks(np.arange(0, times[-1], 60))
    ax.set_xticks(np.arange(0, times[-1], 20), minor=True)
    ax.set_ylabel("PSD $\\mathrm{µV²/Hz}$$\ \mathrm{(dB)}$")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return f, ax
