from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from mne import Info
from mne.viz import plot_topomap
from numpy.typing import NDArray

from .utils._checks import _check_type
from .utils._docs import copy_doc, fill_doc


@fill_doc
class _Feedback(ABC):
    """Abstract class defining a feedback window.

    Parameters
    ----------
    %(info)s
    """

    @abstractmethod
    def __init__(self, info):
        self._info = _Feedback._check_info(info)
        # define colorbar range
        self._vmin = None
        self._vmax = None

    @abstractmethod
    def update(
        self, topodata: NDArray[float], timestamp: float, lineplotdata: float
    ):
        """
        Update the topographic map with the new data array (n_channels, ).

        Parameters
        ----------
        topodata : array
            1D array of shape (n_channels, ) containing the new data samples to
            plot.
        timestamp : float
            Timestamp of the new data-point.
        lineplotdata : float
            Y-value plotted on a lineplot at X=timestamp.
        """
        topodata -= np.mean(topodata)
        return topodata

    # ------------------------------------------------------------------------
    @property
    def info(self):
        """MNE Info instance with a montage.

        :type: `mne.Info`
        """
        return self._info

    @property
    def vmin(self):
        """Minimum value of the colormap range."""
        return self._vmin

    @property
    def vmax(self):
        """Maximum value of the colormap range."""
        return self._vmax

    # ------------------------------------------------------------------------
    @staticmethod
    def _check_info(info):
        """Static checker for the provided info instance."""
        _check_type(info, (Info,), "info")
        if info.get_montage() is None:
            raise ValueError(
                "The provided info instance 'info' does not have "
                "a DigMontage attached."
            )
        return info


class FeedbackMPL(_Feedback):
    """
    Feedback window using matplotlib.

    Parameters
    ----------
    %(info)s
    """

    def __init__(
        self,
        info: Info,
    ):
        if plt.get_backend() != "Qt5Agg":
            plt.switch_backend("Qt5Agg")
        if not plt.isinteractive():
            plt.ion()  # enable interactive mode
        super().__init__(info)
        self._fig, self._axes = plt.subplots(1, 2, figsize=(6, 3))
        # define kwargs for plot_topomap
        self._kwargs = dict(
            vmin=self._vmin,
            vmax=self._vmax,
            cmap="hsv",
            sensors=True,
            res=64,
            axes=self._axes[0],
            names=None,
            outlines="head",
            contours=6,
            onselect=None,
            extrapolate="auto",
            show=False,
        )
        # create initial topographic plot
        plot_topomap(
            np.zeros(len(self._info["ch_names"])),
            self._info,
            **self._kwargs,
        )
        # prepare axes for line plot
        self._axes[1].axis("off")
        self._points = []

    @copy_doc(_Feedback.update)
    def update(
        self, topodata: NDArray[float], timestamp: float, lineplotdata: float
    ):
        topodata = super().update(topodata, timestamp, lineplotdata)
        self._update_topoplot(topodata)
        self._update_lineplot(timestamp, lineplotdata)
        # redraw figure
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def _update_topoplot(self, topodata: NDArray[float]):
        """Update topographic plot."""
        self._axes[0].clear()
        plot_topomap(topodata, self._info, **self._kwargs)

    def _update_lineplot(self, timestamp: float, lineplotdata: float):
        """Update the line plot."""
        if 20 < len(self._points):
            self._points[0].remove()
            del self._points[0]
        self._points.append(
            self._axes[1].scatter(timestamp, lineplotdata, c="black")
        )
        if 1 < len(self._points):  # update x-range
            x0 = self._points[0].get_offsets().data[0][0]
            xf = self._points[-1].get_offsets().data[0][0]
            self._axes[1].set_xlim(x0, xf)

    # ------------------------------------------------------------------------
    @property
    def fig(self) -> plt.Figure:
        """Matplotlib figure."""
        return self._fig

    @property
    def axes(self) -> plt.Axes:
        """Matplotlib axes."""
        return self._axes
