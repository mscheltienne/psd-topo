import argparse
import multiprocessing as mp

from psd_topo import set_log_level, weather_map


def run():
    """Entrypoint for weather_map <command> usage."""
    parser = argparse.ArgumentParser(
        prog="PSD-Topo", description="Weather-map topography"
    )
    parser.add_argument(
        "--stream",
        type=str,
        metavar="str",
        help="LSL stream to connect to",
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
        "--verbose", help="enable debug logs", action="store_true"
    )
    args = parser.parse_args()

    # set verbosity
    verbose = "DEBUG" if args.verbose else "INFO"
    set_log_level(verbose)

    # start individual processes
    print("\n>> Press ENTER to stop.\n")
    process = mp.Process(
        target=weather_map,
        args=(
            args.stream,
            (args.fmin, args.fmax),
            args.winsize,
            args.figsize,
            verbose,
        ),
    )
    process.start()
    # stop
    input()
    process.kill()
