import argparse

from bsl.utils.lsl import search_lsl

from psd_topo import nfb


def run():
    """Entrypoint for main <command> usage."""
    parser = argparse.ArgumentParser(
        prog="PSD-Topo", description="Real-time PSD topography"
    )
    parser.add_argument(
        "-fmin",
        type=float,
        metavar="float",
        help="minimum frequency of interest (Hz)",
        default=8.,
    )
    parser.add_argument(
        "-fmax",
        type=float,
        metavar="float",
        help="maximum frequency of interest (Hz)",
        default=13.,
    )
    args = parser.parse_args()

    # look for EEG amplifier
    stream_name = search_lsl(ignore_markers=True, timeout=5)

    # start plotting
    nfb(stream_name, (args.fmin, args.fmax))
