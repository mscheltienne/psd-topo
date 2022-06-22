from typing import List, Optional, Tuple, Union

import mne
import numpy as np
from matplotlib import colors as mpl_colors
from matplotlib import pyplot as plt
from mne.io import BaseRaw
from mne.time_frequency import psd_welch

from ._typing import Color, FigSize, Picks
from .config import load_triggers
from .utils._checks import _check_type
from .utils._docs import fill_doc


@fill_doc
def plot_psd(
    raws: Union[List[BaseRaw], Tuple[BaseRaw]],
    winsize: float,
    overlap: float,
    fmin: float = 8.0,
    fmax: float = 13.0,
    picks: Picks = "eeg",
    labels: Optional[Union[List[str], Tuple[str, ...]]] = None,
    default_color: Color = "crimson",
    colors: Optional[Union[List[Color], Tuple[Color, ...]]] = None,
    figsize: FigSize = (10, 5),
):
    """Plot the power spectral density using welch windows.

    The raws instance must be loaded with the io.read_raw_xdf function and must
    have been recorded with the 'task' entry-point. Each raw instance must
    include all the triggers define in the triggers.ini.

    Parameters
    ----------
    raws : list of raws | tuple of raws
        List of MNE raw instance.
    winsize : float
        Duration of a welch windows in seconds
    overlap : float
        Overlap between welch windows in seconds.
    fmin : float
        Minimum frequency of interest in Hz.
    fmax : float
        Maximum frequency of interest in Hz.
    %(picks_all)s
    labels : list of str | tuple of str
    default_color : color
        The color to use outside the event time-ranges.A color can be defined
        as a string or a RGB / RGBA tuple.
    colors : list of colors | tuple of colors
        The colors to use for the different events. A color can be defined as
        a string or a RGB / RGBA tuple.
    %(figsize)s

    Returns
    -------
    fig : Figure
    axis : Axes
    """
    _check_arguments(raws, winsize, overlap, fmin, fmax, picks, labels)
    colors = _check_colors(default_color, colors)
    # prepare kwargs
    kwargs = dict(
        fmin=fmin,
        fmax=fmax,
        picks=picks,
    )
    # compute psds, times and retrieve events
    psds = list()
    for raw in raws:
        psd, _ = psd_welch(
            raw,
            average=None,
            window="hamming",
            n_fft=int(winsize * raw.info["sfreq"]),
            n_per_seg=int(winsize * raw.info["sfreq"]),
            n_overlap=int(overlap * raw.info["sfreq"]),
            **kwargs,
        )
        assert psd.ndim == 3  # sanity-check, shape is (n_ch, n_freqs, n_seg)
        psd = np.average(psd, axis=(0, 1))  # avg across the channels / freqs
        assert psd.ndim == 1  # sanity-check
        # scaling to dB
        scaling = 1e6  # default scaling
        psd *= scaling * scaling
        np.log10(np.maximum(psd, np.finfo(float).tiny), out=psd)
        psd *= 10
        # recreate time-axis
        step = winsize - overlap
        times = np.arange(0, step * psd.size, step) + winsize / 2
        # retrieve events
        events = mne.find_events(raw, "TRIGGER")
        # assume events are in the correct order
        events = (events[:, 0] / raw.info["sfreq"]) + winsize / 2
        # store
        psds.append((times, psd, events))

    # create figure
    fig, axis = plt.subplots(len(raws), 1, figsize=figsize, sharex=True)
    axis = [axis] if len(raws) == 1 else list(axis)
    for k, (times, psd, events) in enumerate(psds):  # sane number as raws
        previous_ev = 0
        for i, event in enumerate(events):
            idx = np.searchsorted(times, event)
            axis[k].plot(
                times[previous_ev:idx], psd[previous_ev:idx], color=colors[i]
            )
            previous_ev = idx
        # plot the last event
        axis[k].plot(times[previous_ev:], psd[previous_ev:], color=colors[-1])

    # retrieve min/max times
    tmin = np.min([time[0] for time, _, _ in psds])
    tmax = np.min([time[-1] for time, _, _ in psds])
    # format figure
    _format_figure(fig, axis, tmin, tmax, labels)
    return fig, axis


def _check_arguments(
    raws: Union[List[BaseRaw], Tuple[BaseRaw]],
    winsize: float,
    overlap: float,
    fmin: float,
    fmax: float,
    picks: Picks,
    labels: Union[List[str], Tuple[str, ...], None],
):
    """Check the arguments of plot_psd."""
    triggers = load_triggers()
    # check raw instances
    _check_type(raws, (list, tuple), "raws")
    assert len(raws) != 0
    for raw in raws:
        _check_type(raw, (BaseRaw,), "raw")
    events = [mne.find_events(raw, "TRIGGER") for raw in raws]
    assert all(event.shape == events[0].shape for event in events)
    assert events[0].size != 0
    assert all(event.shape[0] == len(triggers.by_value) for event in events)
    assert all(set(event[:, 2]) == set(triggers.by_value) for event in events)
    # check PSD settings
    _check_type(winsize, ("numeric",), "winsize")
    _check_type(overlap, ("numeric",), "winsize")
    _check_type(fmin, ("numeric",), "winsize")
    _check_type(fmax, ("numeric",), "winsize")
    assert 0 < winsize
    assert 0 < overlap
    assert overlap < winsize
    assert 0 < fmin
    assert 0 < fmax
    # check labels
    if labels is not None:
        _check_type(labels, (list, tuple), "labels")
        for label in labels:
            _check_type(label, (str,), "label")
        assert len(labels) == len(raws)


def _check_colors(
    default_color: Color,
    colors: Union[List[Color], Tuple[Color, ...], None],
) -> Tuple[Tuple[float, float, float, float], ...]:
    """Check and format colors as RGBA tuples."""
    triggers = load_triggers()
    if colors is not None:
        _check_type(colors, (list, tuple), "colors")
        assert len(triggers.by_value) == len(colors)
        colors = list(colors)  # cast to mutable list
        for k, color in enumerate(colors):
            _check_type(color, (tuple, str), "color")
            if isinstance(color, str):
                color = mpl_colors.to_rgba(color)  # returns a tuple
            if len(color) == 3:
                color = tuple(list(color) + [1.0])
            assert len(color) == 4
            assert all(0 <= c <= 1 for c in color)
            colors[k] = color
    else:
        cmap = plt.cm.get_cmap("viridis", len(triggers.by_value))
        colors = [cmap(k) for k in range(len(triggers.by_value))]

    _check_type(default_color, (tuple, str), "default_color")
    if isinstance(default_color, str):
        default_color = mpl_colors.to_rgba(default_color)  # returns a tuple
        if len(default_color) == 3:
            default_color = tuple(list(default_color) + [1.0])
        assert len(default_color) == 4
        assert all(0 <= c <= 1 for c in default_color)

    colors = tuple([default_color] + colors)  # cast to immutable

    return colors


def _format_figure(
    fig: plt.Figure,
    axis: List[plt.Axes],
    tmin: float,
    tmax: float,
    labels: Union[List[str], Tuple[str, ...], None],
):
    """Format the PSD figure."""
    for k, ax in enumerate(axis):
        ax.set_xlim(tmin, tmax)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xticks([])
        if labels is not None:
            ax.set_title(labels[k])
    axis[-1].set_xlabel("Time (s)")
    axis[-1].set_xticks(np.arange(0, tmax, 60))
    axis[-1].set_xticks(np.arange(0, tmax, 20), minor=True)
    fig.text(
        0.03,
        0.5,
        "PSD $\\mathrm{µV²/Hz}$$\ \mathrm{(dB)}$",
        va="center",
        rotation="vertical",
    )
    fig.tight_layout()
    fig.subplots_adjust(left=0.1)
