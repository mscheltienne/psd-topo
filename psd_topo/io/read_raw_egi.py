import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mne
import numpy as np
from mne import create_info
from mne.channels import make_dig_montage
from mne.io import BaseRaw, RawArray


def read_raw_egi(fname) -> BaseRaw:
    """Read raw MFF file saved with the EGI software.

    Parameters
    ----------
    fname : file-like
        Path to the .mff file to load.

    Returns
    -------
    raw : Raw
        MNE raw instance.
    """
    raw = mne.io.read_raw_egi(fname, preload=True)

    # drop bad channels
    bads = (
        "31 67 73 82 91 92 102 111 120 133 145 165 174 187 199 208 209 "
        + "216 217 218 219 225 226 227 228 229 230 231 232 233 234 235 "
        + "236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 "
        + "251 252 253 254 255 256"
    )
    bads = bads.split(" ")
    bads = [f"E{k}" for k in bads]
    raw.drop_channels(bads + ["diode"])

    # drop synthetic trigger channel and reconstruct
    raw.drop_channels(["STI 014"])
    ch_pattern = re.compile(r"(E\d{1,3})")
    trigger_chs = [ch for ch in raw.ch_names if not re.match(ch_pattern, ch)]
    stim = raw.get_data(picks=trigger_chs)
    stim = np.sum(stim, axis=0).reshape(1, -1)
    info = create_info(["TRIGGER"], sfreq=raw.info["sfreq"], ch_types="stim")
    stim = RawArray(stim, info)
    raw.add_channels([stim], force_update_info=True)
    raw.drop_channels(trigger_chs)

    # rename ref
    raw.rename_channels(dict(E257="VREF"))

    # load montage
    montage_fname = Path(fname) / "coordinates.xml"
    # the montage contains 257 channels with the field <number> from 1 to  257
    # and 3 fiducials: Nasion, LPA, RPA with the number and names:
    # - Nasion - 258
    # - Left periauricular point - 259
    # - Right periauricular point - 260
    namespaces = {"": "http://www.egi.com/coordinates_mff"}
    tree = ET.parse(montage_fname)
    root = tree.getroot()
    sensors = root.findall(".//sensorLayout/sensors/sensor", namespaces)
    ch_pos = dict()
    for sensor in sensors:
        name = sensor.find("name", namespaces).text
        number = sensor.find("number", namespaces).text
        name = f"E{number}" if name is None else name
        x = sensor.find("x", namespaces).text
        y = sensor.find("y", namespaces).text
        z = sensor.find("z", namespaces).text
        ch_pos[name] = np.array([float(x), float(y), float(z)])
    nasion = ch_pos["Nasion"]
    lpa = ch_pos["Left periauricular point"]
    rpa = ch_pos["Right periauricular point"]
    del ch_pos["Nasion"]
    del ch_pos["Left periauricular point"]
    del ch_pos["Right periauricular point"]
    montage = make_dig_montage(ch_pos, nasion, lpa, rpa, coord_frame="unknown")
    # set montage
    raw.set_montage(montage)

    return raw
