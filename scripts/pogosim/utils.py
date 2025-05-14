
import sys
import logging

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc

def load_dataframe(filename):
    # Feather V2 files are Arrow IPC “file” streams under the hood
    with ipc.open_file(filename) as reader:
        schema = reader.schema                   # pyarrow.Schema
        meta   = schema.metadata or {}           # dict-like, keys/values are bytes

        # Decode bytes → str for convenience
        decoded_meta = {k.decode(): v.decode() for k, v in meta.items()}

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
    format = "%(asctime)s - %(levelname)s - (%(filename)s:%(lineno)d) - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: "%(message)s",
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def init_logging(verbose=False):
    """
    Initialize logging using the CustomFormatter and declare the global logger variable.
    """
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(CustomFormatter())
    logging.root.addHandler(ch)
    logging.root.setLevel(logging.INFO if verbose else logging.INFO)


# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
