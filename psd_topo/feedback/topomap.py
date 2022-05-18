import numpy as np
from matplotlib import pyplot as plt
from mne import Info
from mne.viz import plot_topomap
from numpy.typing import NDArray

from ..utils._checks import _check_type


class TopoMap:
    """
    Topographic map feedback using matplotlib.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure.
    axes : Axes
        Matplotlib axes on which the topographic map is drawn.
    info : Info
        MNE Info instance with a montage.
    """

    def __init__(
        self,
        fig: plt.Figure,
        axes: plt.Axes,
        info: Info,
    ):
        self._fig = _check_type(fig, (plt.Figure,), "fig")
        self._axes = _check_type(axes, (plt.Axes,), "axes")
        self._info = TopoMap._check_info(info)
        # define kwargs for plot_topomap
        self._kwargs = dict(
            vmin=None,
            vmax=None,
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

    def update(self, data: NDArray[float]):
        """
        Update the topographic map with the new data array (n_channels, ).

        Parameters
        ----------
        data : array
            1D array of shape (n_channels, ) containing the new data samples to
            plot.
        """
        self._axes.clear()
        plot_topomap(data, self._info, **self._kwargs)
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    # ------------------------------------------------------------------------
    @property
    def fig(self):
        """Matplotlib figure."""
        return self._fig

    @property
    def axes(self):
        """Matplotlib axes."""
        return self._axes

    @property
    def info(self):
        """MNE Info instance with a montage.

        :type: `mne.Info`
        """
        return self._info

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
