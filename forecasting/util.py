import datetime
import hashlib
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Union
from urllib.request import urlretrieve

import numpy as np
from joblib import Parallel
from tqdm import tqdm

FREQ_ENUM = {"D": "daily", "W-MON": "weekly", "M": "monthly"}


def get_latest_date(directory: Path):
    subfolders = sorted(
        [
            sub
            for sub in directory.iterdir()
            if sub.is_dir()
            and (
                bool(re.match("[0-9]{8}", sub.name[-8:]))
                or bool(re.match("[0-9]{6}", sub.name[-6:]))
            )
        ],
    )
    if not subfolders:
        raise OSError(f"Could not find any subfolder with dates in {directory}")
    return subfolders[-1].stem


def mkdir_if_needed(path: Path):
    if not path.exists():
        path.mkdir(parents=True)


def return_or_load(object_or_path, object_type, load_func):
    """Returns the input directly or load the object from file.
    Returns the input if its type is object_type, otherwise load the object using the
    load_func
    """
    if isinstance(object_or_path, object_type):
        return object_or_path
    return load_func(object_or_path)


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def get_file_modified_date(file_path: str):
    stat = os.stat(file_path)
    return stat.st_mtime


def get_formatted_duration(seconds: float, fmt: str = "hms") -> str:
    """Format a time in seconds.
    :param format: "hms" for hours mins secs or "ms" for min secs.
    """
    mins, secs = divmod(seconds, 60)
    if fmt == "ms":
        t = f"{int(mins):d}m:{int(secs):02d}s"
    elif fmt == "hms":
        h, mins = divmod(mins, 60)
        t = f"{int(h):d}h:{int(mins):02d}m:{int(secs):02d}s"
    else:
        raise Exception(f"Format {fmt} not available, only 'hms' or 'ms'")
    return t


def get_current_time(format="%Y-%m-%d_%H-%m-%S"):
    return datetime.datetime.now().strftime(format)


def get_current_date(format="%Y-%m-%d"):
    return datetime.datetime.now().strftime(format)


# Convert the Python objects into serializable objects for dumping JSON file
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def compute_sha256(filename: Union[Path, str]):
    """Return SHA256 checksum of a file."""
    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        Parameters
        ----------
        blocks: int, optional
            Number of blocks transferred so far [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize  # pylint: disable=attribute-defined-outside-init
        self.update(blocks * bsize - self.n)  # will also set self.n = b * bsize


def download_url(url, filename):
    """Download a file from url to filename, with a progress bar."""
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)  # nosec


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


class TqdmToLoguru:
    """
    Output stream for TQDM which will output to loguru logger instead of the standard output.
    """

    def __init__(self, logger, level="INFO"):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message):
        # Remove any trailing newlines and avoid empty messages
        message = message.strip("\n")
        if message:  # Avoid logging empty messages
            self.logger.log(self.level, message)

    def flush(self):
        pass  # No need to implement flush for loguru
