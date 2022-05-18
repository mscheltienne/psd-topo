from abc import ABC, abstractmethod

from numpy.typing import NDArray


class TopoMap(ABC):
    """
    Abstract class representing a base topographic map feedback.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def update(self, data: NDArray):
        """
        Update the topographic map with the new data array (n_channels, ).

        Parameters
        ----------
        data : array
            1D array of shape (n_channels, ) containing the new data samples to
            plot.

        """
        pass


class TopoMapMPL(TopoMap):
    """
    Topographic map feedback using matplotlib.
    """

    pass
