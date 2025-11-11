
import sys
import logging
import os
import matplotlib.pyplot as plt

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import yaml
import json, ast



def _to_dict_if_str(obj):
    """
    Return *obj* unchanged unless it is a **str** that represents a mapping.
    Parsing order: JSON ➜ YAML ➜ Python-literal.  Any failure leaves the
    original string in place.
    """
    if not isinstance(obj, str):
        return obj

    # 1 YAML
    try:
        parsed = yaml.safe_load(obj)
        if isinstance(parsed, dict):
            return parsed
    except yaml.YAMLError:
        pass

    # 2 JSON
    try:
        parsed = json.loads(obj)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 3️⃣ Python literal
    try:
        parsed = ast.literal_eval(obj)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        pass

    return {} # Give up, return empty dict


def load_dataframe(filename):
    # Feather V2 files are Arrow IPC “file” streams under the hood
    with ipc.open_file(filename) as reader:
        schema = reader.schema                   # pyarrow.Schema
        meta   = schema.metadata or {}           # dict-like, keys/values are bytes

        # Decode bytes → str for convenience
        decoded_meta = {k.decode(): v.decode() for k, v in meta.items()}

    if "configuration" in decoded_meta:
        decoded_meta["configuration"] = _to_dict_if_str(
            decoded_meta["configuration"]
        )

    # Load dataframe
    df = pd.read_feather(filename)

    return df, decoded_meta

# Custom logging formatter (cf https://stackoverflow.com/questions/1343227/can-pythons-logging-format-be-modified-depending-on-the-message-log-level)
class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    fmt = "%(asctime)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + fmt + reset,
        logging.INFO: "%(message)s",
        logging.WARNING: yellow + fmt + reset,
        logging.ERROR: red + fmt + reset,
        logging.CRITICAL: bold_red + fmt + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def init_logging(verbose=False):
    """
    Initialize logging using the CustomFormatter and declare the global logger variable.
    """
    root = logging.getLogger()
    if not root.handlers:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(CustomFormatter())
        root.addHandler(ch)
    # Levels
    root.setLevel(logging.DEBUG if verbose else logging.INFO)


def save_figure(filename: str, dpi: int = 300) -> None:
    """Save current matplotlib figure *only* if SAVE_FIGURES is True."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=dpi)
    plt.close()


# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
