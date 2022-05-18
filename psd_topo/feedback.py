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
    def update(self, data: NDArray[float]):
        """
        Update the topographic map with the new data array (n_channels, ).

        Parameters
        ----------
        data : array
            1D array of shape (n_channels, ) containing the new data samples to
            plot.
        """
        pass

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
        if not plt.isinteractive():
            plt.ion()  # enable interactive mode
        super().__init__(info)
        self._fig, self._axes = plt.subplots(1, 1, figsize=(4, 4))
        # define kwargs for plot_topomap
        self._kwargs = dict(
            vmin=self._vmin,
            vmax=self._vmax,
            cmap="RdBu_r",
            sensors=True,
            res=64,
            axes=self._axes,
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

    @copy_doc(_Feedback.update)
    def update(self, data: NDArray[float]):
        self._axes.clear()
        plot_topomap(data, self._info, **self._kwargs)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    # ------------------------------------------------------------------------
    @property
    def fig(self) -> plt.Figure:
        """Matplotlib figure."""
        return self._fig

    @property
    def axes(self) -> plt.Axes:
        """Matplotlib axes."""
        return self._axes
