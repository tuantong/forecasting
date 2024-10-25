from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

from pydantic.dataclasses import dataclass

from forecasting import PROJECT_DATA_PATH


@dataclass
class Metadata:
    """
    This class holds information about the configuration of the dataset
    """

    # version of dataset
    version: str
    # name of dataset
    name: str
    # prediction length
    prediction_length: int
    # used to indicate the freq
    freq: str
    # date time column
    time_col: str = "date"
    # target column
    target_col: str = "quantity_order"
    # id column
    id_col: str = "id"
    # transformations to be applied on dataset
    transforms: List[Dict] = None

    _source_data_path = None
    _source_meta_data_path = None

    @property
    def source_data_path(self):
        return (
            Path(PROJECT_DATA_PATH) / self.version
            if self._source_data_path is None
            else self._source_data_path
        )

    @source_data_path.setter
    def source_data_path(self, new_path: Union[str, Path]):
        self._source_data_path = new_path

    @property
    def source_meta_data_path(self):
        return (
            Path(PROJECT_DATA_PATH) / self.version
            if self._source_meta_data_path is None
            else self._source_meta_data_path
        )

    @source_meta_data_path.setter
    def source_meta_data_path(self, new_path: Union[str, Path]):
        self._source_meta_data_path = new_path

    def __post_init__(self):
        # Check version string
        # Get latest date string if version is 'latest'
        if self.version == "latest":
            self.version = datetime.now().strftime("%Y%m%d")
