"""Type hints."""

from typing import List, Tuple, Union

from numpy.typing import NDArray

FigSize = Tuple[float, float]
Picks = Union[str, List[str], List[int], NDArray[int]]
Color = Union[
    str, Tuple[float, float, float], Tuple[float, float, float, float]
]
