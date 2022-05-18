from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from mne import Info
from mne.viz import plot_topomap
from numpy.typing import NDArray

from .utils._checks import _check_type
from .utils._docs import copy_doc, fill_doc


@fill_doc
class TopoMap(ABC):
    """
    Abstract class representing a base topographic map feedback.

    Parameters
    ----------
    %(info)s
    """

    @abstractmethod
    def __init__(self, info: Info):
        self._info = TopoMap._check_info(info)

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

    @property
    def info(self):
        """MNE Info instance with a montage.

        :type: `mne.Info`
        """
        return self._info

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


@fill_doc
class TopoMapMPL(TopoMap):
    """
    Topographic map feedback using matplotlib.

    Parameters
    ----------
    %(info)s
    figsize : tuple
        2-sequence tuple defining the matplotlib figure size: (width, height)
        in inches.
    """

    def __init__(self, info: Info, figsize: Tuple[float, float] = (5, 5)):
        super().__init__(info)
        self._f, self._ax = plt.subplots(1, 1, figsize=figsize)
        # define kwargs for plot_topomap
        self._kwargs = dict(
            cmap="RdBu_r",
            sensors=True,
            res=64,
            names=None,
            outlines="head",
            contours=6,
            onselect=None,
            extrapolate="auto",
        )
        # create initial plot
        plot_topomap(
            np.zeros(len(self._info["ch_names"])),
            self._info,
            axes=self._ax,
            show=True,
            **self._kwargs,
        )

    @copy_doc(TopoMap.update)
    def update(self, data: NDArray[float]):
        self._ax.clear()
        plot_topomap(
            data, self._info, axes=self._ax, show=True, **self._kwargs
        )
        self._f.canvas.draw()
        self._f.canvas.flush_events()

    @property
    def f(self):
        """Matplotlib figure."""
        return self._f

    @property
    def ax(self):
        """Matplotlib axes."""
        return self._ax
