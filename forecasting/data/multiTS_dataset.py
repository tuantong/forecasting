import copy
import os
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from cloudpathlib import CloudPath

from forecasting import PROCESSED_DATA_BUCKET
from forecasting.configs.logging_config import logger
from forecasting.data.base_dataset import BaseDataset
from forecasting.data.s3_utils.aws_service import AWSStorage, create_client
from forecasting.data.utils.pandas_utils import clean_static_feat
from forecasting.metadata.unified import UnifiedMetadata


class DatasetLoadingException(BaseException):
    """Class for handling dataset loading exception"""


class MultiTSDataset(BaseDataset):
    """
    Dataset loader multiple time series dataset.

    Args:
    MultiTSDataset(metadata, root_path, processed_data_path):
        metadata (DatasetMetadata): Contains information about the dataset
        ts_data_uri (str): URI path to the time series data
        ts_meta_uri (str): URI path to the metadata data
        root_dir (Path): Local directory to store the dataset
    """

    def __init__(
        self,
        metadata: Union[UnifiedMetadata, Dict],
        ts_data_uri: str,
        ts_meta_uri: str,
        ts_stock_uri: str,
        root_dir: Path = None,
    ):
        if isinstance(metadata, Dict):
            metadata = UnifiedMetadata(**metadata)
        super().__init__(metadata, root_dir)

        self.ts_data_uri = ts_data_uri
        self.ts_meta_uri = ts_meta_uri
        self.ts_stock_uri = ts_stock_uri

        self._metadata.source_data_path = self._root_dir / os.path.basename(
            self.ts_data_uri
        )
        self._metadata.source_meta_data_path = self._root_dir / os.path.basename(
            self.ts_meta_uri
        )
        self._metadata.source_stock_data_path = self._root_dir / os.path.basename(
            self.ts_stock_uri
        )

        # Run sanity checks
        self._prepare_data()

        if self.pipeline:
            self.past_cov_pipeline = copy.deepcopy(self.pipeline)

    def __repr__(self) -> str:
        head = f"{self.__class__.__name__}"
        body = [f"Config: {self._metadata}"]
        body += self._extra_repr().splitlines()
        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)

    def _extra_repr(self) -> str:
        extra_body = f"Past covariates column(s): {self._metadata.past_cov_cols}"
        return extra_body

    def _prepare_data(self) -> None:
        # Sanity checks the dataset and downloads the data to the local directory if it is not already downloaded.

        assert self._metadata.freq in [
            "D",
            "W-MON",
            "M",
        ], f"Only frequency of type 'D', 'W-MON', 'M' is supported!, found type: {self._metadata.freq}"

        if not self._is_already_downloaded():
            logger.info("Data not found in local directory. Downloading from URIs.")
            self._download_dataset()
        else:
            logger.info("Data already exist in local directory. Skipping download.")

    def _download_dataset(self):
        """Download dataset from S3"""
        logger.info(f"Creating directory {self._root_dir.absolute()}")
        os.makedirs(self._root_dir, exist_ok=True)

        aws_client = create_client()
        s3_service = AWSStorage(aws_client, PROCESSED_DATA_BUCKET)
        ts_data_uri = CloudPath(self.ts_data_uri)
        try:
            if ts_data_uri.is_dir():
                s3_service.download_folder(
                    ts_data_uri.key, self._metadata.source_data_path
                )
            else:
                s3_service.download_file(
                    ts_data_uri.key, self._metadata.source_data_path
                )
        except Exception as e:
            raise DatasetLoadingException(
                f"Could not download time series file. Reason: {repr(e)}"
            ) from e
        logger.info("Time series data download successful")

        ts_meta_uri = CloudPath(self.ts_meta_uri)
        try:
            if ts_meta_uri.is_dir():
                s3_service.download_folder(
                    ts_meta_uri.key, self._metadata.source_meta_data_path
                )
            else:
                s3_service.download_file(
                    ts_meta_uri.key, self._metadata.source_meta_data_path
                )
        except Exception as e:
            raise DatasetLoadingException(
                f"Could not download metadata file. Reason: {repr(e)}"
            ) from e
        logger.info("Metadata download successful")

        ts_stock_uri = CloudPath(self.ts_stock_uri)
        try:
            if ts_stock_uri.is_dir():
                s3_service.download_folder(
                    ts_stock_uri.key, self._metadata.source_stock_data_path
                )
            else:
                s3_service.download_file(
                    ts_stock_uri.key, self._metadata.source_stock_data_path
                )
        except Exception as e:
            raise DatasetLoadingException(
                f"Could not download stock file. Reason: {repr(e)}"
            ) from e
        logger.info("Stock data download successful")

    def _get_path_ts_data(self) -> Path:
        return Path(self._metadata.source_data_path)

    def _get_path_metadata(self) -> Path:
        return Path(self._metadata.source_meta_data_path)

    def _get_path_stock_data(self) -> Path:
        return Path(self._metadata.source_stock_data_path)

    def _is_already_downloaded(self) -> bool:
        has_ts_series = os.path.isfile(self._get_path_ts_data()) or os.path.isdir(
            self._get_path_ts_data()
        )
        has_metadata = os.path.isfile(self._get_path_metadata()) or os.path.isdir(
            self._get_path_metadata()
        )
        has_stock_data = os.path.isfile(self._get_path_stock_data()) or os.path.isdir(
            self._get_path_stock_data()
        )
        return has_ts_series and has_metadata and has_stock_data

    def load_metadata(
        self, drop_price: bool = True, filter_active_only: bool = True
    ) -> pd.DataFrame:
        """Load the meta data file

        Returns
            (pd.DataFrame): Load the meta data
        """
        meta_df = pd.read_parquet(
            self._metadata.source_meta_data_path,
            engine="pyarrow",
            use_nullable_dtypes=True,
        )
        # TODO: Drop the 'price' column temporarily
        # will need to comeback and fix later
        if drop_price and "price" in meta_df.columns:
            meta_df = meta_df.drop(columns=["price"])

        # Clean static features in meta_df
        if self._metadata.static_cov_cols is not None:
            logger.info("Cleaning static features")
            clean_static_feat(meta_df, self._metadata.static_cov_cols)

        if filter_active_only:
            logger.info("Filtering items with 'active' status")
            active_ids = meta_df[meta_df["status"] != "deleted"].id.tolist()
            meta_df = meta_df[meta_df.id.isin(active_ids)]
            # meta_df = meta_df[~meta_df["status"].str.contains("deleted", na=False)]
        return meta_df

    def load_pd(
        self, merge_metadata: bool = True, drop_price: bool = True, **kwargs
    ) -> pd.DataFrame:
        """Load pandas dataframe

        Parameters:
            merge_metadata (bool): Whether or not to merge the meta data with the time series data
            drop_price: (bool): Whether or not to drop the 'price' column from the meta data

        Returns
            (pd.DataFrame): The pandas dataframe of the dataset

        """
        time_series_df = pd.read_parquet(
            self._metadata.source_data_path, engine="pyarrow", use_nullable_dtypes=True
        )
        if not isinstance(time_series_df.index, pd.RangeIndex):
            time_series_df.reset_index(inplace=True)
        # Convert `id_col` to string if it is not of type 'string'
        if time_series_df[self._metadata.id_col].dtype != "string":
            logger.info(f"Converting `{self._metadata.id_col}` to string")
            time_series_df = time_series_df.astype({self._metadata.id_col: "string"})

        if merge_metadata:
            # Include drop_price in the kwargs for load_metadata
            meta_df = self.load_metadata(drop_price, **kwargs)
            # Filter time_series_df `id_col` to match meta_df's
            time_series_df = time_series_df[
                time_series_df[self._metadata.id_col].isin(
                    meta_df[self._metadata.id_col]
                )
            ]
            # Merge time_series with metadata
            full_df = time_series_df.merge(
                meta_df, on=self._metadata.id_col, how="left"
            ).reset_index(drop=True)

            logger.info(
                f"Full DataFrame shape with metadata: {full_df.shape}, Unique IDs: {full_df.id.unique().shape}"
            )

            return full_df
        else:
            logger.info(
                f"Time series DataFrame shape without metadata: {time_series_df.shape}, Unique IDs: {time_series_df.id.unique().shape}"
            )
            return time_series_df

    def load_stock_pd(self) -> pd.DataFrame:
        """Load the stock data file

        Returns
            (pd.DataFrame): Load the stock data
        """
        stock_df = pd.read_parquet(
            self._metadata.source_stock_data_path,
            engine="pyarrow",
            use_nullable_dtypes=True,
        )

        return stock_df


if __name__ == "__main__":
    # from forecasting import PROJECT_ROOT_PATH
    FREQ_ENUM = {"D": "daily", "W-MON": "weekly", "M": "monthly"}
    _S3_PROCESED_URI = os.environ["S3_PROCESSED_DATA_URI"]

    meta = UnifiedMetadata(
        version="latest",
        name="all_brand",
        prediction_length=36,
        freq="W-MON",
        time_col="date",
        target_col="quantity_order",
        id_col="id",
        past_cov_cols=[
            "price",
            "views",
            "gg_click",
            "gg_ad_spends",
            "fb_click",
            "fb_ad_spends",
            "ad_click",
            "ad_spends",
        ],
        static_cov_cols=["id", "brand_name", "is_product", "platform", "product_type"],
    )

    s3_uri = CloudPath(_S3_PROCESED_URI)

    data_uri = (
        s3_uri
        / meta.name
        / (FREQ_ENUM[meta.freq] + "_dataset")
        / (FREQ_ENUM[meta.freq] + "_series.parquet")
    ).as_uri()

    meta_uri = (s3_uri / meta.name / "meta_data.parquet").as_uri()

    stock_uri = (s3_uri / meta.name / "inference_set" / "stock_ts.parquet").as_uri()

    full_dataset = MultiTSDataset(
        metadata=meta,
        ts_data_uri=data_uri,
        ts_meta_uri=meta_uri,
        ts_stock_uri=stock_uri,
    )
