from configparser import ConfigParser
from pathlib import Path
from typing import Tuple

from bsl.triggers import TriggerDef


def load_triggers():
    """Load triggers from triggers.ini into a TriggerDef instance.

    Returns
    -------
    tdef : TriggerDef
        Trigger definitiopn containing: eye_open, eye_close and lecture.
    """
    directory = Path(__file__).parent
    tdef = TriggerDef(directory / "triggers.ini")

    keys = (
        "eye_open",
        "eye_close",
        "lecture",
    )
    for key in keys:
        if not hasattr(tdef, key):
            raise ValueError(
                f"Key '{key}' is missing from trigger definition."
            )

    return tdef


def load_config() -> Tuple[str, str]:
    """Load config from config.ini.

    Returns
    -------
    amplifier_prefix : str
        Prefix of the amplifier name on the LSL network.
    trigger_stream_name : str
        Name of the LSL outlet of the software trigger.
    """
    directory = Path(__file__).parent
    config = ConfigParser(inline_comment_prefixes=("#", ";"))
    config.optionxform = str
    config.read(str(directory / "config.ini"))

    keys = ("amplifier", "trigger")
    for key in keys:
        if not config.has_section(key):
            raise ValueError(f"Key '{key}' is missing from configuration.")

    amplifier_prefix = config["amplifier"]["prefix"]
    trigger_stream_name = config["trigger"]["stream_name"]

    return amplifier_prefix, trigger_stream_name
