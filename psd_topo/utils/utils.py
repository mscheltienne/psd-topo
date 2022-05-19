from typing import List

from bsl.utils.lsl import list_lsl_streams

from ._checks import _check_type


def search_amplifiers(n: int) -> List[str]:
    """Search the available DSI-24 amplifiers on the network.

    Parameters
    ----------
    n : int
        Number of amplifiers to search.

    Returns
    -------
    stream_names : list of str
        List of amplifiers' names.
    """
    _check_type(n, ("int",), "n")
    if n <= 0:
        raise ValueError("The number of amplifiers to look for 'n' must be a "
                         "strictly positive integer.")

    stream_names, _ = list_lsl_streams(ignore_markers=True)
    return stream_names
