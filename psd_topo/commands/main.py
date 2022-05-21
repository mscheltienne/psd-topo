import argparse
import multiprocessing as mp

from psd_topo import nfb, set_log_level
from psd_topo.utils import search_amplifiers


def run():
    """Entrypoint for main <command> usage."""
    parser = argparse.ArgumentParser(
        prog="PSD-Topo", description="Real-time PSD topography"
    )
    parser.add_argument(
        "-n",
        type=int,
        metavar="int",
        help="number of amplifiers to connect to",
        default=1,
    )
    parser.add_argument(
        "--fmin",
        type=float,
        metavar="float",
        help="minimum frequency of interest (Hz)",
        default=8.0,
    )
    parser.add_argument(
        "--fmax",
        type=float,
        metavar="float",
        help="maximum frequency of interest (Hz)",
        default=13.0,
    )
    parser.add_argument(
        "--winsize",
        type=float,
        metavar="float",
        help="acquisition window duration (seconds)",
        default=5.0,
    )
    parser.add_argument(
        "--figsize",
        type=float,
        metavar="float",
        nargs=2,
        help="figure size for the matplotlib backend",
    )
    parser.add_argument(
        "--verbosity", help="enable debug logs", action="store_true"
    )
    args = parser.parse_args()

    # set verbosity
    if args.verbosity:
        set_log_level("DEBUG")
        verbose = "DEBUG"
    else:
        set_log_level("INFO")
        verbose = "INFO"

    # start individual processes
    print("\n>> Press ENTER to stop.\n")
    stream_names = search_amplifiers(args.n)
    processes = list()
    for stream_name in stream_names:
        process = mp.Process(
            target=nfb,
            args=(
                stream_name,
                (args.fmin, args.fmax),
                args.winsize,
                args.figsize,
                verbose,
            ),
        )
        process.start()
        processes.append(process)

    # stop
    input()
    for process in processes:
        process.kill()
