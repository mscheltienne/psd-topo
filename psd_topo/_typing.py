"""Type hints."""

from typing import List, Tuple, Union

from numpy.typing import NDArray

FigSize = Tuple[float, float]
Picks = Union[str, List[str], List[int], NDArray[int]]
