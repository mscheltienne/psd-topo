import argparse

from psd_topo import set_log_level
from psd_topo.io import read_raw_xdf
from psd_topo.psd import plot_psd


def run():
    """Entrypoint for plot_psd <command> usage."""
    parser = argparse.ArgumentParser(
        prog="PSD-Topo plot", description="PSD plot for Magda"
    )
    parser.add_argument(
        "--fname",
        type=str,
        metavar="str",
        help=".xdf file to load",
    )
    parser.add_argument(
        "--verbose", help="enable debug logs", action="store_true"
    )
    args = parser.parse_args()

    verbose = "DEBUG" if args.verbose else "INFO"
    set_log_level(verbose)

    raws, streams = read_raw_xdf(args.fname)
    fig, ax = plot_psd(
        raws,
        winsize=5,
        overlap=4.8,
        fmin=8.0,
        fmax=13.0,
        picks=["O1", "O2"],
        labels=streams,
        default_color="lightblue",
        colors=[(0, 153, 153), (255, 102, 102), "black"],
    )
    fig.savefig("psd-plot.png", dpi=300)
