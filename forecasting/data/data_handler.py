import ast
import os
from copy import copy
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from cloudpathlib import CloudPath
from darts import TimeSeries
from statsmodels.tsa.stattools import acf
from tqdm import tqdm

from forecasting import PROJECT_DATA_PATH, S3_PROCESSED_DATA_URI
from forecasting.configs.config_helpers import validate_config
from forecasting.configs.logging_config import logger
from forecasting.data import MultiTSDataset
from forecasting.data.preprocessing.preprocess_data import (
    preprocess_for_inference,
    preprocess_for_training,
)
from forecasting.data.preprocessing.utils import extend_df
from forecasting.data.util import find_seasonality, get_series_type
from forecasting.data.utils.darts_utils import convert_df_to_darts_dataset
from forecasting.metadata import UnifiedMetadata
from forecasting.util import FREQ_ENUM, import_class


class DataHandler:
    """Handles data loading and preprocessing for time series forecasting models."""

    seasonal_list_filename = "seasonal_set.txt"

    def __init__(
        self,
        configs: Dict,
        subset: str,
        local_data_dir: str = None,
        model_type: str = "global",
    ):
        """Initialize the DataHandler.

        Args:
            configs (Dict): Configuration dictionary.
            subset (str): The subset to load the data.
                Valid values: ['full', 'train', 'test', 'inference', 'test_new_item', 'test_best_seller', 'test_top_10']
            local_data_dir (str, optional): Local data directory. Defaults to None.
                If not given will be determined from metadata as follow:
                Path(PROJECT_DATA_PATH) / self.metadata.version / self.metadata.name / self.metadata.freq.
        """
        # Assuming we already validated the outer fields, e.g. ['data', 'model', 'config_name']
        # Validate the `data` config
        validate_config(
            config_data=configs["data"],
            required_fields=["class", "metadata_class", "configs"],
        )

        # Validate subset value
        if subset not in [
            "full",
            "train",
            "test",
            "inference",
            "test_new_item",
            "test_best_seller",
            "test_top_10",
        ]:
            raise ValueError(
                f"Unrecognized `subset`: {subset}. "
                "Valid values are: ['full', 'train', 'test', 'inference', 'test_new_item', 'test_best_seller', 'test_top_10']"
            )

        self.configs = configs
        self.subset = subset
        self.model_type = self._get_model_type(model_type)
        self.metadata = self.init_metadata(
            configs["data"]["metadata_class"], configs["data"]["configs"]
        )
        self.ts_data_uri, self.ts_meta_uri, self.ts_stock_uri = self.get_data_uri(
            self.metadata.name, self.subset, self.metadata.freq
        )
        self.local_data_dir = self._get_local_data_dir(local_data_dir)
        self.dataset = self.init_dataset(
            configs["data"]["class"],
            self.metadata,
            self.ts_data_uri,
            self.ts_meta_uri,
            self.ts_stock_uri,
            self.local_data_dir,
        )

    def _get_model_type(self, default_type):
        model_type = self.configs.get("model", {}).get("type")
        return model_type if model_type else default_type

    def _get_local_data_dir(self, local_data_dir):
        if local_data_dir is None:
            local_data_dir = (
                (
                    Path(PROJECT_DATA_PATH)
                    / self.metadata.version
                    / self.metadata.name
                    / FREQ_ENUM[self.metadata.freq]
                )
                if self.subset != "inference"
                else (
                    Path(PROJECT_DATA_PATH)
                    / self.metadata.version
                    / self.metadata.name
                    / FREQ_ENUM[self.metadata.freq]
                    / "inference_set"
                )
            )
        return local_data_dir

    @staticmethod
    def get_data_uri(brand_name: str, subset: str = "full", freq: str = "W-MON"):
        """Utility function to retrieve the URIs of a given brand

        Example data organization:
        <brand_name>
            |_ inference_set
                |_ daily_series.parquet
                |_ monthly_series.parquet
                |_ weekly_series.parquet
                |_ preprocess_infer_df.parquet
                |_ meta_data.parquet
                |_ stock_ts.parquet
            |_ daily_dataset
                |_ daily_series.parquet
                |_ train.parquet
                |_ test.parquet
                |_ <other_test_files>.parquet
            |_ weekly_dataset
                |_ weekly_series.parquet
                |_ train.parquet
                |_ test.parquet
                |_ <other_test_files>.parquet
            |_ monthly_dataset
                |_ monthly_series.parquet
                |_ train.parquet
                |_ test.parquet
                |_ <other_test_files>.parquet
            |_ meta_data.parquet


        Args:
            brand_name (str): The brand name to retrieve URIs from
                            ('all_brand' to retrieve the full dataset)
            subset (str): The split of dataset
            freq (str): The frequency of dataset
        """
        # if freq == "D" and subset != "full":
        #     raise ValueError(
        #         f"Daily frequency currently only support `full` subset, found subset: {subset}"
        #     )

        s3_uri = CloudPath(S3_PROCESSED_DATA_URI)
        # Convert freq to other format, e.g., 'D' -> 'daily'
        data_format = "parquet"
        data_freq_str = FREQ_ENUM[freq]

        data_folder = (
            "inference_set" if subset == "inference" else f"{data_freq_str}_dataset"
        )
        data_filename = (
            f"{data_freq_str}_series.{data_format}"
            if subset in ["full", "inference"]
            else f"{subset}.{data_format}"
        )
        meta_folder = "inference_set" if subset == "inference" else ""
        meta_filename = "meta_data.parquet"
        stock_filename = "stock_ts.parquet"

        ts_data_uri = (s3_uri / brand_name / data_folder / data_filename).as_uri()
        ts_meta_uri = (s3_uri / brand_name / meta_folder / meta_filename).as_uri()
        ts_stock_uri = (s3_uri / brand_name / "inference_set" / stock_filename).as_uri()
        return ts_data_uri, ts_meta_uri, ts_stock_uri

    @staticmethod
    def init_metadata(meta_class_str: str, meta_configs: Dict) -> UnifiedMetadata:
        metadata_cls = import_class(f"forecasting.metadata.{meta_class_str}")
        return metadata_cls(**meta_configs)

    @staticmethod
    def init_dataset(
        data_class_str: str,
        metadata: UnifiedMetadata,
        ts_data_uri: str,
        ts_meta_uri: str,
        ts_stock_uri: str,
        local_data_dir: str = None,
    ) -> MultiTSDataset:
        logger.info(f"Initializing class {data_class_str} with freq: {metadata.freq}")
        data_class: MultiTSDataset = import_class(f"forecasting.data.{data_class_str}")
        return data_class(
            metadata=metadata,
            ts_data_uri=ts_data_uri,
            ts_meta_uri=ts_meta_uri,
            ts_stock_uri=ts_stock_uri,
            root_dir=local_data_dir,
        )

    def load_data(self, **kwargs) -> pd.DataFrame:
        """Load pandas dataframe

        Parameters:
            **kwargs: Keyword arguments to pass to the `self.dataset.load_pd` method.
            These keyword arguments are specific to the underlying dataset class that
            the DataHandler is using.
                You can pass in the `merge_metadata` and `drop_price` arguments to control
                whether or not to merge the metadata with the time series data and
                whether or not to drop the 'price' column from the meta data.

        Returns:
            (pd.DataFrame): The pandas dataframe of the dataset
        """
        logger.info(f"Loading {FREQ_ENUM[self.metadata.freq]}-{self.subset} dataframe")
        return self.dataset.load_pd(**kwargs)

    def load_metadata(self, drop_price: bool = True, filter_active_only: bool = True):
        return self.dataset.load_metadata(drop_price, filter_active_only)

    def load_stock_data(self):
        return self.dataset.load_stock_pd()

    def preprocess_data(self, df, stage, **kwargs):
        """Preprocess the input dataframe for training or inference.

        Parameters:
            df (pd.DataFrame): The input dataframe.
            stage (str): The preprocessing stage, either 'training' or 'inference'.
            **kwargs: Keyword arguments for preprocessing.

                Refer to the `_preprocess_df_for_training` and `_preprocess_df_for_inference` methods for
                more details on the available keyword arguments and their effects.

        Returns:
            - target_df (pd.DataFrame): The preprocessed target dataframe for inference.
            - past_cov_df (pd.DataFrame): The preprocessed past covariates dataframe for inference.
            - similar_item_dict (dict): The dictionary of similar items for each item id.
        """
        assert stage in [
            "training",
            "inference",
        ], "stage must be either 'training' or 'inference'"
        if stage == "training":
            # Performing filtering here:
            # The number of items after filtering will be smaller after processing
            filtered_df = self._preprocess_df_for_training(df, **kwargs)
            return filtered_df
        target_df, past_cov_df, similar_item_dict = self._preprocess_df_for_inference(
            df, **kwargs
        )
        return target_df, past_cov_df, similar_item_dict

    def convert_df_to_darts_format(self, target_df, past_cov_df=None):
        """
        Convert the input target dataframe and past covariates dataframe to Darts format.

        Args:
            target_df (pandas.DataFrame): Target dataframe.
            past_cov_df (pandas.DataFrame, optional): Past covariates dataframe. Defaults to None.

        Returns:
            tuple: A tuple containing the target series and past covariates series in Darts format.
        """

        def convert_to_darts_series(df, value_cols):
            return convert_df_to_darts_dataset(
                df=df,
                id_col=self.metadata.id_col,
                time_col=self.metadata.time_col,
                value_cols=value_cols,
                static_cols=self.metadata.static_cov_cols,
                freq=self.metadata.freq,
                dtype=np.float32,
            )

        target_series = convert_to_darts_series(target_df, self.metadata.target_col)

        if past_cov_df is not None:
            past_cov_series = convert_to_darts_series(
                past_cov_df, self.metadata.past_cov_cols
            )
        elif self.metadata.past_cov_cols:
            past_cov_series = convert_to_darts_series(
                target_df, self.metadata.past_cov_cols
            )
        else:
            past_cov_series = None

        return target_series, past_cov_series

    def _preprocess_df_for_training(self, df, **kwargs):
        """
        Preprocess the input dataframe for training.

        Args:
            df (pandas.DataFrame): Input dataframe.
            **kwargs: Additional keyword arguments.
                - input_chunk_length (int): Length of input chunk.
                - output_chunk_length (int): Length of output chunk.
                - seasonal_set (list): List of seasonal items.

        Returns:
            pandas.DataFrame: Preprocessed dataframe.
        """
        # TODO: Check self.model_type and process separately
        input_chunk_length = kwargs["input_chunk_length"]
        output_chunk_length = kwargs["output_chunk_length"]
        seasonal_set = kwargs["seasonal_set"]
        logger.info("Running preprocessing step for training")
        processed_df = preprocess_for_training(
            df, input_chunk_length, output_chunk_length, seasonal_set
        )
        logger.info(
            f"df_filter shape: {processed_df.shape}, {len(processed_df.id.unique())}"
        )
        return processed_df

    def _preprocess_df_for_inference(self, df, **kwargs):
        prediction_length = kwargs.get(
            "prediction_length", self.metadata.prediction_length
        )
        preprocess_method = kwargs.get("preprocess_method", "fill_zero")
        input_chunk_length = kwargs["input_chunk_length"]
        output_chunk_length = kwargs["output_chunk_length"]
        # TODO: Check self.model_type and process separately
        logger.info("Running preprocessing step for inference")
        processed_df, similar_item_dict = preprocess_for_inference(
            df,
            preprocess_method=preprocess_method,
            min_length_for_new_item=input_chunk_length,
        )
        logger.info(
            f"preprocessed_df shape: {processed_df.shape}, preprocessed_df unique ids: {processed_df.id.unique().shape}"
        )
        # Extend past covariates into the future if prediction length is bigger than output_chunk_length
        past_cov_df = None
        if self.metadata.past_cov_cols:
            past_cov_df = processed_df.drop(columns=self.metadata.target_col)
            if prediction_length > output_chunk_length:
                logger.info("Extending past covariates into the future")
                past_cov_df = extend_df(
                    df=past_cov_df,
                    value_col=self.metadata.past_cov_cols,
                    avg_sim_df=None,
                    required_len=prediction_length - output_chunk_length,
                    direction="future",
                    freq=self.metadata.freq,
                )
                logger.info(
                    f"past_cov_df shape: {past_cov_df.shape}, past_cov_df unique ids: {past_cov_df.id.unique().shape}"
                )

        return processed_df, past_cov_df, similar_item_dict

    def create_or_load_seasonal_item_list(self, work_dir: str):
        """Load the seasonal item list

        Create a new list and save to `work_dir` if the seasonal_file_path is not found
        """
        seasonal_file_path = os.path.join(work_dir, self.seasonal_list_filename)
        if os.path.exists(seasonal_file_path):
            logger.info(f"Loading the seasonal item set from {seasonal_file_path}")
            seasonal_list = []
            with open(seasonal_file_path, encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    try:
                        # Strip the line and convert it to a tuple using ast.literal_eval
                        item = ast.literal_eval(line)
                    except (SyntaxError, ValueError):
                        # If evaluation fails, log a warning and use the line as is
                        logger.warning(
                            f"Failed to parse line: {line}. Treating it as a string."
                        )
                        item = line
                    seasonal_list.append(item)
        else:
            logger.info("Creating the seasonal item set from monthly dataset")
            seasonal_list = self._create_seasonal_set()
            logger.info(f"Saving the the seasonal item set to {seasonal_file_path}")
            # Save the list
            with open(seasonal_file_path, "w+", encoding="utf-8") as file:
                file.writelines("\n".join(map(str, seasonal_list)))

        return seasonal_file_path, seasonal_list

    def _create_seasonal_set(self):
        # Load the monthly dataset
        monthly_ts_data_uri, monthly_ts_meta_uri, monthly_ts_stock_uri = (
            self.get_data_uri(brand_name=self.metadata.name, subset="full", freq="M")
        )
        # Must create a copy instead of creating a reference
        # to avoid modifying the original metadata
        monthly_metadata = copy(self.metadata)
        monthly_metadata.freq = "M"
        monthly_dataset = self.init_dataset(
            data_class_str=type(self.dataset).__name__,
            metadata=monthly_metadata,
            ts_data_uri=monthly_ts_data_uri,
            ts_meta_uri=monthly_ts_meta_uri,
            ts_stock_uri=monthly_ts_stock_uri,
            local_data_dir=self.local_data_dir.parent / FREQ_ENUM["M"],
        )

        return _find_seasonal_set(monthly_dataset.load_pd())


def _find_seasonal_set(monthly_df: pd.DataFrame):
    seasonal_items = []
    threshold = 3
    required_seasonal_periods = set(range(10, 14))

    monthly_df["cumsum"] = monthly_df.groupby("id")["quantity_order"].cumsum()

    for item_id, _df in tqdm(monthly_df.groupby("id"), desc="Processing Items"):
        _df_qty_values_cut = _df["quantity_order"].values[
            _df["cumsum"].values > threshold
        ]

        len_ts = len(_df_qty_values_cut)
        _, adi, _ = get_series_type(_df_qty_values_cut)

        if adi <= 1.06 and len_ts >= 15:
            item_monthly_ts = TimeSeries.from_values(values=_df_qty_values_cut)

            is_seasonal, detected_periods = find_seasonality(
                item_monthly_ts, max_lag=15
            )
            if is_seasonal:
                # Sort detected periods by their significance (ACF values)
                acf_values = acf(
                    item_monthly_ts.values(), nlags=max(detected_periods), fft=False
                )
                significant_periods = sorted(
                    detected_periods, key=lambda x: acf_values[x], reverse=True
                )

                # Check if the most significant peaks are in the required seasonal periods
                most_significant_peak = significant_periods[0]
                if most_significant_peak in required_seasonal_periods:
                    seasonal_items.append((item_id, significant_periods))

    monthly_df.drop(["cumsum"], axis=1, inplace=True)
    logger.info(f"Number of seasonal items: {len(seasonal_items)}")
    return seasonal_items


def merge_updated_data(current_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge updated data into the current dataframe.

    Parameters:
        current_df (pd.DataFrame): The current dataframe containing the existing data.
        new_df (pd.DataFrame): The new dataframe containing the updated data.

    Returns:
        pd.DataFrame: The merged dataframe with the latest data.

    """
    # Get the last date of the current df
    df_final_date = new_df["date"].min()

    # Filter current_df and new_df based on conditions
    current_df_filtered = current_df[current_df["id"].isin(new_df["id"].unique())]
    current_df_filtered = current_df_filtered[
        current_df_filtered["date"] <= df_final_date
    ]
    new_df_filtered = new_df[new_df["date"] > df_final_date]

    # Concatenate and sort the filtered dataframes
    latest_df = (
        pd.concat([current_df_filtered, new_df_filtered])
        .sort_values(["id", "date"])
        .reset_index(drop=True)
    )

    latest_df = latest_df[latest_df.id.isin(new_df["id"].unique())]

    return latest_df
