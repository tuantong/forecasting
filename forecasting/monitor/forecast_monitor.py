import os
from copy import copy
from datetime import datetime, time, timedelta
from typing import Optional, Union

import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from dateutil.relativedelta import SU, relativedelta

from forecasting import (
    AWS_FORECAST_HISTORY_BUCKET_NAME,
    PROJECT_DATA_PATH,
    PROJECT_ROOT_PATH, PROJECT_CONFIG_PATH, # custom-event
    S3_FORECAST_DATA_DIR,
)
from forecasting.configs import logger
from forecasting.data.data_handler import DataHandler, merge_updated_data
from forecasting.data.s3_utils.aws_service import AWSStorage, aws_client
from forecasting.data.util import calculate_sale_pattern
from forecasting.evaluation.metrics import mape
from forecasting.utils.common_utils import load_dict_from_yaml


def define_confidence_score_by_quantile(error, p50_thresh, p75_thresh):
    if error is None:
        return None
    if error <= p50_thresh:
        return 1
    if error <= p75_thresh:
        return 0.5
    return 0


class ForecastMonitor:
    """Class for monitoring forecast accuracy."""

    _prediction_filename = "predictions.csv"
    _data_config_path = os.path.join(PROJECT_CONFIG_PATH, "sample_config.yaml") # custom-event
    _BRANDS_TO_CREATE_ALL_CHANNELS = ["mizmooz", "as98"]

    def __init__(
        self,
        brand_name: str,
        storage_service: AWSStorage = None,
        local_folder_path: str = None,
        s3_folder_path: str = None,
    ):
        """
        Initialize the ForecastMonitor instance.

        Parameters:
        - brand_name: Brand name to retrieve forecast history and actual values.
        - storage_service: An instance of AWSStorage or any service with similar functionality.
        - local_folder_path: The local path to save the downloaded forecast results.
        - s3_folder_path: The folder path in the S3 bucket.
        """
        self.brand_name = brand_name
        self._storage_service = storage_service or AWSStorage(
            client=aws_client, bucket=AWS_FORECAST_HISTORY_BUCKET_NAME
        )
        self._local_folder_path = local_folder_path or os.path.join(
            PROJECT_DATA_PATH, "forecast_history", brand_name
        )
        os.makedirs(self._local_folder_path, exist_ok=True)
        self._prediction_s3_path = s3_folder_path or os.path.join(
            S3_FORECAST_DATA_DIR, brand_name, self._prediction_filename
        )

        stock_path = os.path.join(
            PROJECT_DATA_PATH,
            "downloaded_multiple_datasource",
            self.brand_name,
            "product-ts.csv",
        )
        stock_df = pd.read_csv(
            stock_path,
            parse_dates=["load_date"],
            dtype={"variant_id": "string", "product_id": "string", "stock": "float32"},
        ).rename(columns={"load_date": "date", "from_source": "platform"})
        self.stock_df = stock_df[~stock_df.variant_id.isna()]

        self.config_dict = load_dict_from_yaml(self._data_config_path)
        self.config_dict["data"]["configs"]["freq"] = "D"
        self.config_dict["data"]["configs"]["name"] = self.brand_name
        self.data_handler = DataHandler(self.config_dict, subset="inference")

        self._cached_actual_data = None
        self._local_prediction_paths = {}
        self._cached_forecast_df = {}
        self._cached_actual_df = {}
        self._forecast_version_info = {}

    def _get_latest_version_before_date(self, s3_path: str, date: datetime):
        """
        Get the latest version of objects in the S3 folder before the specified date.

        Parameters:
        - s3_path: The path in the S3 bucket.
        - date: The date to find the latest version before.

        Returns:
            dict: A dictionary with object keys and their latest version IDs before the specified date.
        """
        logger.info(f"Listing all versions in {s3_path} before {date}")
        versions = self._storage_service.list_object_versions(s3_path)

        if not versions:
            logger.info(f"No versions found for {s3_path}")
            return {}

        latest_versions = {}
        for version in versions:
            last_modified = version["LastModified"].replace(tzinfo=None)
            if last_modified < date:
                key = version["Key"]
                if (
                    key not in latest_versions
                    or latest_versions[key]["LastModified"] < version["LastModified"]
                ):
                    latest_versions[key] = {
                        "VersionId": version["VersionId"],
                        "LastModified": version["LastModified"],
                    }
        self._forecast_version = latest_versions
        return latest_versions

    def _download_historical_forecasts(self, s3_path: str, date: datetime) -> bool:
        """
        Download historical forecast results from S3 before the specified date.

        Parameters:
        - s3_path: The path in the S3 bucket where historical forecasts are stored.
        - date: The date to get the latest version of files before.

        Returns:
            bool: True if download is successful, False otherwise.
        """
        logger.info(
            f"Downloading historical last forecast results from {s3_path} to {self._local_folder_path} before {date}"
        )
        latest_versions = self._get_latest_version_before_date(s3_path, date)
        if len(latest_versions) == 0:
            return False

        os.makedirs(self._local_folder_path, exist_ok=True)
        success = True
        for key, version_info in latest_versions.items():
            last_modified = (
                version_info["LastModified"].replace(tzinfo=None).strftime("%Y-%m-%d")
            )
            self._local_prediction_paths[date] = os.path.join(
                self._local_folder_path,
                os.path.splitext(os.path.basename(key))[0] + f"_{last_modified}.csv",
            )

            if not os.path.exists(self._local_prediction_paths[date]):
                try:
                    self._storage_service.download_file(
                        key,
                        self._local_prediction_paths[date],
                        version_id=version_info["VersionId"],
                    )
                except ClientError as e:
                    logger.error(f"Failed to download {key}: {str(e)}")
                    success = False
        return success

    def _adjust_for_stockout(self, df, stock_df, value_col, start_date, end_date):
        if stock_df.empty:
            return df

        if value_col == "forecast_value":
            stock_df["stockout_id"] = stock_df.apply(
                lambda row: (
                    row["variant_id"] + "_" + row["platform"]
                    if row["variant_id"] != "NA"
                    else row["product_id"] + "_" + row["platform"]
                ),
                axis=1,
            )
        elif value_col == "quantity_order":
            stock_df["stockout_id"] = stock_df.apply(
                lambda row: self.brand_name
                + "_"
                + row["platform"]
                + "_"
                + row["product_id"]
                + "_"
                + row["variant_id"],
                axis=1,
            )
        else:
            raise ValueError(
                "value_col must be either 'forecast_value' or 'quantity_order'"
            )

        item_list = df.stockout_id.unique()
        stock_df = stock_df[stock_df.date.between(start_date, end_date)]
        stock_df = stock_df[stock_df.stockout_id.isin(item_list)]

        date_range = pd.date_range(
            start=df["date"].min(), end=df["date"].max(), freq="D"
        )
        multi_index = pd.MultiIndex.from_product(
            [date_range, item_list], names=["date", "stockout_id"]
        )
        stock_df = (
            stock_df.set_index(["date", "stockout_id"])
            .reindex(multi_index)
            .reset_index()
        )
        stock_df["stock"] = stock_df["stock"].fillna(1)

        df = df.merge(
            stock_df[["date", "stockout_id", "stock"]],
            on=["date", "stockout_id"],
            how="left",
        )
        df[value_col] = df.apply(
            lambda row: row[value_col] if row["stock"] > 0 else 0, axis=1
        )
        return df

    def _process_actual_df(
        self, actual_df: pd.DataFrame, start_date: datetime, end_date: datetime
    ):
        df = actual_df[
            (actual_df.brand_name == self.brand_name)
            & (actual_df.date.between(start_date, end_date))
        ]
        logger.info(f"Unique dates: {df.date.unique()}")

        if df.empty:
            raise ValueError(
                f"No actual data for this brand from {start_date} to {end_date}, check data version in config"
            )

        df["variant_id"] = df.apply(
            lambda row: row["variant_id"] if row["is_product"] == False else "NA",
            axis=1,
        )
        df = df.assign(
            **{
                "id": df.apply(
                    lambda row: f"{row['brand_name']}_{row['platform']}_{row['product_id']}_{row['variant_id']}_{row['channel_id']}",
                    axis=1,
                )
            }
        )

        df = df.assign(
            **{"stockout_id": df.id.apply(lambda x: "_".join(x.split("_")[:-1]))}
        )

        df = self._adjust_for_stockout(
            df, self.stock_df.copy(), "quantity_order", start_date, end_date
        )

        variant_result_df = (
            df[~df["is_product"]]
            .groupby(["id"])
            .agg(
                {
                    "quantity_order": np.sum,
                    "price": np.mean,
                    "variant_id": "first",
                    "product_id": "first",
                    "channel_id": "first",
                    "brand_name": "first",
                    "platform": "first",
                    "name": "first",
                    "is_product": "first",
                }
            )
            .reset_index()
            .rename(columns={"name": "title", "quantity_order": "actual_value"})
        )

        product_result_df = (
            variant_result_df.groupby(["platform", "product_id", "channel_id"])
            .agg(
                {
                    "actual_value": np.sum,
                    "price": np.mean,
                    "brand_name": "first",
                    "title": "first",
                }
            )
            .reset_index()
            .assign(variant_id=pd.NA, is_product=True)
            .astype(dtype={"is_product": bool, "variant_id": pd.NA})
        )
        product_result_df["id"] = product_result_df.apply(
            lambda row: f"{row['brand_name']}_{row['platform']}_{row['product_id']}_NA_{row['channel_id']}",
            axis=1,
        )

        if self.brand_name in self._BRANDS_TO_CREATE_ALL_CHANNELS:
            product_all_channels_df = (
                product_result_df.groupby(["platform", "product_id"])
                .agg(
                    {
                        "actual_value": np.sum,
                        "price": np.mean,
                        "brand_name": "first",
                        "title": "first",
                    }
                )
                .reset_index()
                .assign(channel_id="0", variant_id=pd.NA, is_product=True)
                .astype(
                    dtype={
                        "channel_id": "string",
                        "is_product": bool,
                        "variant_id": pd.NA,
                    }
                )
            )
            product_all_channels_df["id"] = product_all_channels_df.apply(
                lambda row: f"{row['brand_name']}_{row['platform']}_{row['product_id']}_NA_0",
                axis=1,
            )
            product_all_channels_df = product_all_channels_df.astype(
                dtype={"id": "string"}
            )

            variant_all_channels_df = (
                variant_result_df.groupby(["platform", "product_id", "variant_id"])
                .agg(
                    {
                        "actual_value": np.sum,
                        "price": np.mean,
                        "brand_name": "first",
                        "title": "first",
                    }
                )
                .reset_index()
                .assign(channel_id="0", is_product=False)
                .astype(dtype={"channel_id": "string", "is_product": bool})
            )
            variant_all_channels_df["id"] = variant_all_channels_df.apply(
                lambda row: f"{row['brand_name']}_{row['platform']}_{row['product_id']}_{row['variant_id']}_0",
                axis=1,
            )

            all_channels_df = pd.concat(
                [product_all_channels_df, variant_all_channels_df], ignore_index=True
            )
            result_df = pd.concat(
                [variant_result_df, product_result_df, all_channels_df],
                ignore_index=True,
            )
        else:
            result_df = pd.concat(
                [variant_result_df, product_result_df], ignore_index=True
            )

        result_df = result_df.astype(
            dtype={
                "id": "string",
                "actual_value": np.float32,
                "price": np.float32,
                "variant_id": "string",
                "product_id": "string",
                "channel_id": "string",
                "brand_name": "string",
                "platform": "string",
                "title": "string",
                "is_product": bool,
            }
        )
        return result_df

    def _process_forecast_df(
        self,
        forecast_df: pd.DataFrame,
        actual_df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
    ):
        actual_df = actual_df[actual_df.date.between(start_date, end_date)]
        forecast_df = forecast_df[forecast_df.date.between(start_date, end_date)]
        platform_list = actual_df.platform.unique().tolist()
        platform = [x for x in platform_list if x != "amazon"][0]
        if "platform" not in forecast_df.columns:
            forecast_df["platform"] = platform
        print(f"Platform list: {forecast_df.platform.unique()}")
        forecast_df = forecast_df.assign(
            **{"stockout_id": forecast_df["item_id"] + "_" + forecast_df["platform"]}
        )

        forecast_df = self._adjust_for_stockout(
            forecast_df, self.stock_df.copy(), "forecast_value", start_date, end_date
        )

        variant_result_df = (
            forecast_df[~forecast_df["is_product"]]
            .groupby(["platform", "item_id", "channel_id"])
            .agg({"forecast_value": np.sum, "is_product": "first"})
            .reset_index()
            .rename(columns={"item_id": "variant_id"})
        )

        # Merging with the actual_df to get product_id
        variant_result_df = variant_result_df.merge(
            actual_df[["platform", "variant_id", "product_id"]].drop_duplicates(),
            on=["platform", "variant_id"],
            how="left",
        )

        product_result_df = (
            variant_result_df.groupby(["platform", "product_id", "channel_id"])
            .agg({"forecast_value": np.sum})
            .round()
            .reset_index()
            .assign(is_product=True)
            .astype(dtype={"is_product": bool})
            .rename(columns={"product_id": "item_id"})
        )

        variant_result_df = variant_result_df.drop(columns=["product_id"]).rename(
            columns={"variant_id": "item_id"}
        )

        result_df = pd.concat([variant_result_df, product_result_df], ignore_index=True)

        return result_df

    def _cache_forecast_data(self, date_ranges, actual_df):
        for start_date, end_date in date_ranges:
            if (start_date, end_date) not in self._cached_forecast_df:
                if not self._download_historical_forecasts(
                    self._prediction_s3_path, start_date
                ):
                    self._cached_forecast_df[(start_date, end_date)] = None
                else:
                    forecast_df = pd.read_csv(
                        self._local_prediction_paths[start_date],
                        parse_dates=["date"],
                        dtype={"item_id": "string", "channel_id": "string"},
                    )
                    self._cached_forecast_df[(start_date, end_date)] = (
                        self._process_forecast_df(
                            forecast_df, actual_df, start_date, end_date
                        )
                    )
                    self._forecast_version_info[(start_date, end_date)] = (
                        self._forecast_version
                    )

    def _cache_actual_data(self, date_ranges, actual_df):
        for start_date, end_date in date_ranges:
            if (start_date, end_date) not in self._cached_actual_df:
                self._cached_actual_df[(start_date, end_date)] = (
                    self._process_actual_df(actual_df, start_date, end_date)
                )

    def _parse_period(self, period: str):
        num, period_type = period.split()
        num = int(num)
        if period_type in ["days", "day"]:
            return num, "days"
        elif period_type in ["months", "month"]:
            return num, "months"
        else:
            raise ValueError("Unsupported period type. Use 'days' or 'months'.")

    def get_forecast_accuracy(
        self,
        product_id: str = None,
        variant_id: str = None,
        channel_id: str = None,
        platform: str = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: str = "1 month",
        method: str = "mape",
    ) -> dict:
        """
        Get the forecast accuracy for a specific item ID for the specified period before the given date.

        Parameters:
        - product_id: The product ID to calculate forecast accuracy for.
        - variant_id: The variant ID to calculate forecast accuracy for.
        - channel_id: The channel ID to calculate forecast accuracy for.
        - end_date: The date to get the latest version of files before.
        - period: Period to calculate accuracy for, e.g., "7 days" or "3 months".
        - method: Method to use for the error calculation, currently only support ["mape", "avg_abs_error"]

        Returns:
            dict: A dictionary containing forecast accuracy metrics.
        """
        assert method in [
            "mape",
            "avg_abs_error",
        ], "method must be either 'mape' or 'avg_abs_error'"

        if product_id is None and variant_id is None:
            raise ValueError("Only one of product_id or variant_id can be None")

        if channel_id is None:
            logger.info("Channel ID not specified, selecting all channel ID: '0'")
            channel_id = "0"
        # Get is_product status to retrieve in the forecast_df later
        is_product = True if variant_id is None else False

        item_id = product_id if is_product is True else variant_id

        num, period_type = self._parse_period(period)
        if period_type not in ["months", "days"]:
            raise ValueError("Period type must be 'months' or 'days'")
        if (period_type == "days") and num != 7:
            raise ValueError("Number of days currently supported is 7")
        if (period_type == "days") and (method != "mape"):
            raise ValueError(
                f"period_type: {period_type} is not supported for method: {method}"
            )

        if end_date is None:
            if period_type == "months":
                # Find last day of previous month
                end_date = datetime(
                    datetime.now().year, datetime.now().month, 1
                ) - relativedelta(days=1)
            elif period_type == "days":
                # Get previous sunday
                end_date = datetime.combine(
                    datetime.now() + relativedelta(weekday=SU(-1)), time(23, 59, 59)
                )
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        if period_type == "months":
            start_dates = [
                datetime(end_date.year, end_date.month, 1) - relativedelta(months=i)
                for i in range(num)
            ]
        else:
            # TODO: Fix later
            start_dates = [
                datetime.combine(end_date - timedelta(days=num - 1), time(0, 0, 0))
            ]

        start_dates.reverse()
        date_ranges = [
            (
                (date, (date + relativedelta(months=1)) - timedelta(days=1))
                if period_type == "months"
                else (date, end_date)
            )
            for date in start_dates
        ]

        start_date = start_dates[0]
        logger.info(f"Start date: {start_date}, End date: {end_date}")

        if self._cached_actual_data is None:
            self._cached_actual_data = self.data_handler.load_data()

        self._cache_actual_data(date_ranges, self._cached_actual_data)
        actual_df = pd.concat(
            [
                self._cached_actual_df[(s_d, e_d)]
                for s_d, e_d in date_ranges
                if self._cached_actual_df[(s_d, e_d)] is not None
            ]
        )

        self._cache_forecast_data(date_ranges, self._cached_actual_data)
        forecast_dfs = [
            self._cached_forecast_df[(s_d, e_d)]
            for s_d, e_d in date_ranges
            if self._cached_forecast_df[(s_d, e_d)] is not None
        ]
        if len(forecast_dfs) == 0:
            return None

        # Get item actual value in time range
        if is_product:
            item_actual_df = actual_df[
                (actual_df.brand_name == self.brand_name)
                & (actual_df.platform == platform)
                & (actual_df.product_id == item_id)
                & (actual_df.variant_id.isna())
                & (actual_df.channel_id == channel_id)
            ].reset_index(drop=True)
        else:
            item_actual_df = actual_df[
                (actual_df.brand_name == self.brand_name)
                & (actual_df.platform == platform)
                & (actual_df.variant_id == item_id)
                & (actual_df.channel_id == channel_id)
            ].reset_index(drop=True)

        if item_actual_df.empty:
            raise ValueError("No actual data found for this item.")

        if method == "mape":
            total_actuals = item_actual_df.actual_value.sum()
            total_forecasts = sum(
                [
                    forecast_df[
                        (forecast_df.item_id == item_id)
                        & (forecast_df.platform == platform)
                        & (forecast_df.channel_id == channel_id)
                        & (forecast_df.is_product == is_product)
                    ].forecast_value.sum()
                    for forecast_df in forecast_dfs
                ]
            )

            logger.info(
                f"Total Actual value: {total_actuals} - Total Forecast value: {total_forecasts}"
            )

            acc = max(
                0, 1 - mape(np.array([total_actuals]), np.array([total_forecasts]))
            )
            error = 1 - acc

        elif method == "avg_abs_error":
            # Calculate average of absoulte error for each period
            actuals = item_actual_df.actual_value.values
            forecasts = np.array(
                [
                    forecast_df[
                        (forecast_df.item_id == item_id)
                        & (forecast_df.platform == platform)
                        & (forecast_df.channel_id == channel_id)
                        & (forecast_df.is_product == is_product)
                    ].forecast_value.sum()
                    for forecast_df in forecast_dfs
                ]
            )
            logger.info(f"Actual values: {actuals} - Forecast values: {forecasts}")

            error = np.mean(np.abs(actuals - forecasts))

        logger.info(
            f"Forecast error for item over {period}:\n"
            f"Product ID: {product_id} - Variant ID: {variant_id} - Channel ID: {channel_id}\n"
            f"- Error: {error}"
        )
        return {"Error": error}

    def get_forecast_accuracy_all_items(
        self,
        forecast_date: Union[str, datetime],
        end_date: Optional[Union[str, datetime]] = None,
        period: str = "1 month",
        method: str = "mape",
        group_method: str = "sale_pattern",
    ) -> dict:
        """
        Get the forecast accuracy for every item (products and variants per channel) for the specified period before the given date.

        Parameters:
        - end_date: The date to get the latest version of files before.
        - period: Period to calculate accuracy for, e.g., "7 days" or "3 months".
        - method: Method to use for the error calculation, currently only support ["mape", "avg_abs_error"]
        - group_method: Method to group the items for calculating the percentiles of errors. ["sale_pattern", "sale_category"]

        Returns:
            dict: A dictionary containing forecast accuracy metrics for all items and missing items information.
        """
        assert method in [
            "mape",
            "avg_abs_error",
        ], "method must be either 'mape' or 'avg_abs_error'"
        assert group_method in [
            "sale_pattern",
            "sale_category",
        ], "method must be either 'sale_pattern' or 'sale_category'"
        num, period_type = self._parse_period(period)
        if period_type not in ["months", "days"]:
            raise ValueError("Period type must be 'months' or 'days'")
        if (period_type == "days") and (num != 7):
            raise ValueError("Number of days currently supported is 7")
        if (period_type == "days") and (method != "mape"):
            raise ValueError(
                f"period_type: {period_type} is not supported for method: {method}"
            )

        if end_date is None:
            if period_type == "months":
                # Find last day of previous month
                end_date = datetime(
                    datetime.now().year, datetime.now().month, 1
                ) - relativedelta(days=1)
            elif period_type == "days":
                # Get previous sunday
                end_date = datetime.combine(
                    datetime.now() + relativedelta(weekday=SU(-1)), time(23, 59, 59)
                )
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        if period_type == "months":
            start_dates = [
                datetime(end_date.year, end_date.month, 1) - relativedelta(months=i)
                for i in range(num)
            ]
        else:
            # TODO: Fix later
            start_dates = [
                datetime.combine(end_date - timedelta(days=num - 1), time(0, 0, 0))
            ]

        start_dates.reverse()
        date_ranges = [
            (
                (date, (date + relativedelta(months=1)) - timedelta(days=1))
                if period_type == "months"
                else (date, end_date)
            )
            for date in start_dates
        ]

        start_date = start_dates[0]
        logger.info(f"Start date: {start_date}, End date: {end_date}")
        if self._cached_actual_data is None:
            self._cached_actual_data = self.data_handler.load_data()

        self._cache_actual_data(date_ranges, self._cached_actual_data)
        actual_dfs = []
        for s_d, e_d in date_ranges:
            actual_df = self._cached_actual_df.get((s_d, e_d))
            if actual_df is not None:
                actual_df = actual_df.assign(date=s_d)
                actual_dfs.append(actual_df)
        actual_df = pd.concat(actual_dfs)

        self._cache_forecast_data(date_ranges, self._cached_actual_data)
        # Create forecast_dfs with date columns
        forecast_dfs = []
        for s_d, e_d in date_ranges:
            forecast_df = self._cached_forecast_df.get((s_d, e_d))
            if forecast_df is not None:
                forecast_df = forecast_df.assign(date=s_d)
                forecast_dfs.append(forecast_df)

        if len(forecast_dfs) > 0:
            forecast_df = pd.concat(forecast_dfs)

        def generate_key(row, date_ranges):
            for i, (start_date, end_date) in enumerate(date_ranges):
                if start_date <= row["date"] <= end_date:
                    return i
            return -1

        actual_df["variant_id"] = actual_df.apply(
            lambda row: row["variant_id"] if row["is_product"] == False else "NA",
            axis=1,
        )
        product_actual_df = actual_df[actual_df["is_product"]]
        variant_actual_df = actual_df[~actual_df["is_product"]]

        product_actual_df = product_actual_df.assign(
            key=lambda df: df.apply(lambda row: generate_key(row, date_ranges), axis=1)
        )
        variant_actual_df = variant_actual_df.assign(
            key=lambda df: df.apply(lambda row: generate_key(row, date_ranges), axis=1)
        )

        if len(forecast_dfs) > 0:
            product_forecast_df = forecast_df[forecast_df["is_product"]]
            variant_forecast_df = forecast_df[~forecast_df["is_product"]]

            product_forecast_df = product_forecast_df.assign(
                key=lambda df: df.apply(
                    lambda row: generate_key(row, date_ranges), axis=1
                )
            )
            variant_forecast_df = variant_forecast_df.assign(
                key=lambda df: df.apply(
                    lambda row: generate_key(row, date_ranges), axis=1
                )
            )

            product_merged_df = pd.merge(
                product_actual_df,
                product_forecast_df,
                left_on=["platform", "product_id", "channel_id", "key", "is_product"],
                right_on=["platform", "item_id", "channel_id", "key", "is_product"],
                how="outer",
                indicator=True,
            )

            variant_merged_df = pd.merge(
                variant_actual_df,
                variant_forecast_df,
                left_on=["platform", "variant_id", "channel_id", "key", "is_product"],
                right_on=["platform", "item_id", "channel_id", "key", "is_product"],
                how="outer",
                indicator=True,
            )

        else:
            product_merged_df = product_actual_df
            variant_merged_df = variant_actual_df

        merged_df = pd.concat([product_merged_df, variant_merged_df], ignore_index=True)
        if "_merge" not in merged_df.columns:
            merged_df["_merge"] = "left_only"

        merged_df["item_key"] = merged_df.apply(
            lambda row: (
                f"{self.brand_name}_{row['platform']}_{row['product_id']}_{row['variant_id']}_{row['channel_id']}"
            ),
            axis=1,
        )

        # Identify missing items in forecast
        missing_in_forecast_groups = merged_df[
            merged_df["_merge"] == "left_only"
        ].groupby("item_key")
        missing_in_forecast_items = missing_in_forecast_groups.filter(
            lambda x: len(x) == num
        )["item_key"].values.tolist()
        print(f"Missing in forecast_items: {len(missing_in_forecast_items)}")

        # Identify missing items in actual
        missing_in_actual = merged_df[merged_df["_merge"] == "right_only"]
        missing_in_actual_items = missing_in_actual.apply(
            lambda row: (
                f"{row['platform']}_{row['item_id']}_{row['channel_id']}_{'product' if row['is_product'] else 'variant'}"
            ),
            axis=1,
        ).values.tolist()

        merged_df = merged_df[merged_df["_merge"] == "both"]

        # Calculate total sales volume
        merged_df["total_sales_volume"] = merged_df.groupby("item_key")[
            "actual_value"
        ].transform("sum")

        # Load the monthly df and calculate sale pattern
        config_dict = copy(self.config_dict)
        config_dict["data"]["configs"]["freq"] = "M"
        config_dict["data"]["configs"]["name"] = self.brand_name
        full_monthly_data_handler = DataHandler(config_dict, subset="full")
        infer_monthly_data_handler = DataHandler(config_dict, subset="inference")
        full_monthly_df = full_monthly_data_handler.load_data()
        infer_monthly_df = infer_monthly_data_handler.load_data()

        monthly_df = merge_updated_data(full_monthly_df, infer_monthly_df)
        monthly_df["variant_id"] = monthly_df.apply(
            lambda row: row["variant_id"] if row["is_product"] == False else "NA",
            axis=1,
        )
        monthly_df = monthly_df.assign(
            **{
                "id": monthly_df.apply(
                    lambda row: f"{row['brand_name']}_{row['platform']}_{row['product_id']}_{row['variant_id']}_{row['channel_id']}",
                    axis=1,
                )
            }
        )
        monthly_df_pivot = calculate_sale_pattern(
            monthly_df,
            forecast_date,
            self._BRANDS_TO_CREATE_ALL_CHANNELS,
        )
        dict_id_sale_pattern = dict(
            zip(monthly_df_pivot.id, monthly_df_pivot.sale_pattern)
        )
        dict_id_sale_category = dict(
            zip(monthly_df_pivot.id, monthly_df_pivot.sale_category)
        )

        merged_df["sale_pattern"] = merged_df["item_key"].map(dict_id_sale_pattern)
        merged_df["sale_category"] = merged_df["item_key"].map(dict_id_sale_category)

        accuracy_results_dict = {}
        all_errors = []

        for item_key, group in merged_df.groupby("item_key"):
            actuals = group["actual_value"].values
            forecasts = group["forecast_value"].values
            bias = abs(np.sum(actuals) - np.sum(forecasts))
            sale_pattern = dict_id_sale_pattern[item_key]
            sale_category = dict_id_sale_category[item_key]

            if method == "mape":
                actuals = actuals.sum()
                forecasts = forecasts.sum()
                mape_value = mape(actuals, forecasts)
                accuracy = round(max(0, 1 - mape_value), 2)
                error = round(1 - accuracy, 2)
                confidence_score = accuracy if sale_pattern != "zero-sales" else None
            elif method == "avg_abs_error":
                # Calculate average of absolute error for each period
                error = np.round(np.mean(np.abs(actuals - forecasts)), 2)
                if error != 0:
                    all_errors.append((error, sale_pattern))
                confidence_score = None  # Initial placeholder, will be set later

            accuracy_results_dict[item_key] = {
                "error": error,
                "forecast_bias": bias,
                "actual_value": actuals,
                "forecast_value": forecasts,
                "confidence_score": confidence_score,
                "sale_pattern": sale_pattern,
                "sale_category": sale_category,
            }

        percentile_dict = {}
        if merged_df.empty == False:
            # Add error to merged_df
            merged_df["error"] = merged_df.apply(
                lambda row: accuracy_results_dict[row["item_key"]]["error"], axis=1
            )

            # Calculate percentiles within each sales volume group and sale pattern
            if method == "avg_abs_error":
                if group_method == "sale_pattern":
                    accuracy_results_dict = self._process_percentiles_by_sale_pattern(
                        merged_df, accuracy_results_dict, percentile_dict
                    )
                elif group_method == "sale_category":
                    accuracy_results_dict = self._process_percentiles_by_sale_category(
                        merged_df, accuracy_results_dict, percentile_dict
                    )

        for item in missing_in_forecast_items:
            accuracy_results_dict[item] = {
                "error": None,
                "forecast_bias": None,
                "actual_value": None,
                "forecast_value": None,
                "confidence_score": None,
                "sale_pattern": dict_id_sale_pattern[item],
                "sale_category": dict_id_sale_category[item],
            }

        return {
            "error_results": accuracy_results_dict,
            "missing_in_forecast": missing_in_forecast_items,
            "missing_in_actual": missing_in_actual_items,
            "forecast_version_info": self._forecast_version_info,
            "report_range": {"start_date": start_date, "end_date": end_date},
            "percentile_dict": percentile_dict,
            "merged_df": merged_df,  # For debug only,
        }

    def _process_percentiles_by_sale_pattern(
        self, merged_df, accuracy_results_dict, percentile_dict
    ):
        # Group items by sales volume within each sale pattern group
        for sale_pattern, group in merged_df[
            merged_df["sale_pattern"] != "zero-sales"
        ].groupby("sale_pattern"):
            sales_volume_groups = group.drop_duplicates("item_key")[
                "total_sales_volume"
            ].quantile([0.25, 0.75])
            if sales_volume_groups[0.25] != sales_volume_groups[0.75]:
                bins = [
                    -np.inf,
                    sales_volume_groups[0.25],
                    sales_volume_groups[0.75],
                    np.inf,
                ]
                labels = ["low", "medium", "high"]
            else:
                bins = [-np.inf, sales_volume_groups[0.25], np.inf]
                labels = ["low", "high"]
            group["sales_volume_group"] = pd.cut(
                group["total_sales_volume"], bins=bins, labels=labels
            )
            # Update accuracy_results_dict with sales_volume_group
            for item_key in group["item_key"].unique():
                accuracy_results_dict[item_key]["sales_volume_group"] = group[
                    group["item_key"] == item_key
                ]["sales_volume_group"].values[0]

            # Calculate percentiles within each sales volume group
            for sales_volume_group, subgroup in group.groupby("sales_volume_group"):
                errors = subgroup["error"].dropna()
                if errors.empty:
                    continue
                error_50th_percentile = np.percentile(errors, 50)
                error_75th_percentile = np.percentile(errors, 75)
                percentile_dict[(sale_pattern, sales_volume_group)] = (
                    error_50th_percentile,
                    error_75th_percentile,
                )

                # Assign confidence scores based on quantiles within each sales volume group and sale pattern
                for item_key in subgroup["item_key"].unique():
                    item_errors = subgroup[subgroup["item_key"] == item_key][
                        "error"
                    ].values
                    if item_errors.size == 0:
                        continue
                    error = item_errors[0]  # Assuming one error per item_key
                    confidence_score = define_confidence_score_by_quantile(
                        error, error_50th_percentile, error_75th_percentile
                    )
                    accuracy_results_dict[item_key][
                        "confidence_score"
                    ] = confidence_score

        # Assign confidence score for zero_sales items
        for item_key in merged_df[merged_df["sale_pattern"] == "zero-sales"][
            "item_key"
        ].unique():
            accuracy_results_dict[item_key]["confidence_score"] = None
            accuracy_results_dict[item_key]["sales_volume_group"] = "zero-sales"

        return accuracy_results_dict

    def _process_percentiles_by_sale_category(
        self, merged_df, accuracy_results_dict, percentile_dict
    ):
        # Group items by sales volume within each sale category group
        for sale_category, group in merged_df[
            merged_df["sale_pattern"] != "zero-sales"
        ].groupby("sale_category"):
            sales_volume_groups = group.drop_duplicates("item_key")[
                "total_sales_volume"
            ].quantile([0.25, 0.75])
            if sales_volume_groups[0.25] != sales_volume_groups[0.75]:
                bins = [
                    -np.inf,
                    sales_volume_groups[0.25],
                    sales_volume_groups[0.75],
                    np.inf,
                ]
                labels = ["low", "medium", "high"]
            else:
                bins = [-np.inf, sales_volume_groups[0.25], np.inf]
                labels = ["low", "high"]
            group["sales_volume_group"] = pd.cut(
                group["total_sales_volume"], bins=bins, labels=labels
            )
            # Update accuracy_results_dict with sales_volume_group
            for item_key in group["item_key"].unique():
                accuracy_results_dict[item_key]["sales_volume_group"] = group[
                    group["item_key"] == item_key
                ]["sales_volume_group"].values[0]

            # Calculate percentiles within each sales volume group
            for sales_volume_group, subgroup in group.groupby("sales_volume_group"):
                errors = subgroup["error"].dropna()
                if errors.empty:
                    continue
                error_50th_percentile = np.percentile(errors, 50)
                error_75th_percentile = np.percentile(errors, 75)
                percentile_dict[(sale_category, sales_volume_group)] = (
                    error_50th_percentile,
                    error_75th_percentile,
                )

                # Assign confidence scores based on quantiles within each sales volume group and sale pattern
                for item_key in subgroup["item_key"].unique():
                    item_errors = subgroup[subgroup["item_key"] == item_key][
                        "error"
                    ].values
                    if item_errors.size == 0:
                        continue
                    error = item_errors[0]  # Assuming one error per item_key
                    confidence_score = define_confidence_score_by_quantile(
                        error, error_50th_percentile, error_75th_percentile
                    )
                    accuracy_results_dict[item_key][
                        "confidence_score"
                    ] = confidence_score

        # Assign confidence score for zero_sales items
        for item_key in merged_df[merged_df["sale_pattern"] == "zero-sales"][
            "item_key"
        ].unique():
            accuracy_results_dict[item_key]["confidence_score"] = None
            accuracy_results_dict[item_key]["sales_volume_group"] = "zero-sales"

        return accuracy_results_dict
