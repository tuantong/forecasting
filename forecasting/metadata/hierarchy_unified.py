from pathlib import Path

from pydantic.dataclasses import dataclass

from forecasting.metadata.shared import DATA_DIRNAME, Metadata

RAW_DATA_DIRNAME = DATA_DIRNAME / "raw"
METADATA_PATH = RAW_DATA_DIRNAME / "meta_data.yaml"
TRAIN_DATA_PATH = RAW_DATA_DIRNAME / "weekly_dataset" / "train.parquet"
TEST_DATA_PATH = RAW_DATA_DIRNAME / "weekly_dataset" / "test.parquet"
PROCESSED_DATA_DIRNAME = DATA_DIRNAME / "processed" / "weekly_dataset"


@dataclass
class HierarchyUnifiedMetadata(Metadata):
    """
    This class holds information about the configuration of the dataset
    """

    @property
    def source_train_path(self) -> Path:
        return self.source_data_path / "weekly_dataset" / "train.parquet"

    @property
    def source_test_path(self) -> Path:
        return self.source_data_path / "weekly_dataset" / "test.parquet"

    @property
    def source_metadata_path(self) -> Path:
        return self.source_data_path / "meta_data.parquet"

    @property
    def raw_train_data_path(self) -> Path:
        return TRAIN_DATA_PATH

    @property
    def raw_test_data_path(self) -> Path:
        return TEST_DATA_PATH

    @property
    def processed_train_data_path(self) -> Path:
        return PROCESSED_DATA_DIRNAME / self.name / "train_processed.pkl"

    @property
    def processed_test_data_path(self) -> Path:
        return PROCESSED_DATA_DIRNAME / self.name / "test_processed.pkl"

    @property
    def meta_data_path(self) -> Path:
        return METADATA_PATH
