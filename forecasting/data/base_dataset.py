import argparse
import shutil
from abc import ABC
from copy import deepcopy
from pathlib import Path
from typing import Dict, Generic, Type, TypeVar

import numpy as np
import pandas as pd
import yaml
from darts.dataprocessing.pipeline import Pipeline

from forecasting.configs.logging_config import logger
from forecasting.metadata.shared import Metadata
from forecasting.util import import_class


def load_and_print_info(data_module_class) -> "BaseDataset":
    """Load data and print info."""
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    params = data_module_class.process_args(args)
    dataset = data_module_class(params)
    dataset.prepare_data()
    dataset.setup()
    logger.info(dataset)
    return dataset


def _copy_raw_dataset(source: Path, desc: Path) -> Path:
    if desc.is_file():
        desc.unlink()
    else:
        desc.parent.mkdir(parents=True, exist_ok=True)
    return shutil.copy(source, desc)


T = TypeVar("T", bound=Type[Metadata])


class BaseDataset(ABC, Generic[T]):
    """Class that load a custom dataset."""

    def __init__(self, metadata: T, root_dir: Path = None):
        self._metadata: T = deepcopy(metadata)
        if root_dir is None:
            self._root_dir = self._metadata.source_data_path
        else:
            self._root_dir = root_dir

        transform_list = []
        if self._metadata.transforms:
            for tran in self._metadata.transforms:
                tran_class = tran["class"]
                tran_args = _process_tran_agrs(tran["args"])
                tran_class = import_class(
                    f"darts.dataprocessing.transformers.{tran_class}"
                )
                transform = tran_class(**tran_args)
                transform_list.append(transform)

        self.pipeline = Pipeline(transform_list) if len(transform_list) > 0 else None

    @classmethod
    def make_metadata(config: Dict) -> T:
        metadata = T(**config)
        return metadata

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--config_file_path", type=str, require=True)
        return parser

    @staticmethod
    def process_args(args):
        with open(args.config_file_path, encoding="utf-8") as f:
            configs = yaml.safe_load(f)
        return configs

    def _download_dataset(self):
        pass


_modules = {"numpy": np, "pandas": pd}


def _process_tran_agrs(args: Dict):
    result = {}
    for key, value in args.items():
        if isinstance(value, str) and value.endswith("()"):
            components = value.split(".")
            if len(components) > 1:
                module = _modules[components[0]]
                assert module is not None, f"Don't support module {module}"
                func_str = ".".join(components[1:])[:-2]
                result[key] = getattr(module, func_str)
            else:
                result[key] = globals()[value[:-2]]
        else:
            result[key] = value
    return result
