import argparse
import time
import multiprocessing as mp

from psd_topo import nfb
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
    stream_names = search_amplifiers(args.n)

    # start individual processes
    processes = list()
    for stream_name in stream_names:
        process = mp.Process(target=nfb, args=(stream_name, (args.fmin, args.fmax)))
        process.start()
        processes.append(process)

    time.sleep(5)  # give time to the process to start
    # stop
    input('>> Press ENTER to stop.\n')
    for process in processes:
        process.kill()
