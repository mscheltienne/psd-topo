import argparse

from bsl.triggers import LSLTrigger

from psd_topo import set_log_level


def run():
    """Entrypoint for task <command> usage."""
    parser = argparse.ArgumentParser(
        prog="PSD-Topo", description="Real-time PSD topography"
    )
    parser.add_argument(
        "--verbose", help="enable debug logs", action="store_true"
    )
    args = parser.parse_args()

    # set verbosity
    verbose = "DEBUG" if args.verbose else "INFO"
    set_log_level(verbose)

    # set trigger
    trigger = LSLTrigger("PSD-markers", verbose=True)

    # wait for LabRecorder to be started
    input(">>> Press ENTER after starting the recording on the LabRecorder.")

    # tasks with manual key-input triggers
    input(">>> Press ENTER to send 'eye-open' trigger.")
    trigger.signal(1)
    input(">>> Press ENTER to send 'eye-close' trigger.")
    trigger.signal(2)
    input(">>> Press ENTER to send 'lecture' trigger.")
    trigger.signal(3)
    input(">>> Press ENTER to close.")
