from abc import ABC, abstractmethod

from mne import Info
from numpy.typing import NDArray

from .utils._checks import _check_type
from .utils._docs import fill_doc, copy_doc


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
        self._info = TopoMap.check_info(info)

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
        _check_type(info, (Info,), 'info')
        if info.get_montage() is None:
            raise ValueError("The provided info instance 'info' does not have "
                             "a DigMontage attached.")


@fill_doc
class TopoMapMPL(TopoMap):
    """
    Topographic map feedback using matplotlib.

    Parameters
    ----------
    %(info)s
    """

    def __init__(self, info: Info):
        super().__init__(info)

    @copy_doc(TopoMap.update)
    def update(self, data: NDArray[float]):
        pass
