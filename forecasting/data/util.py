"""Utility functions"""

from datetime import datetime, timedelta
from typing import Iterable, Optional, TypeVar

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.statistics import _bartlett_formula
from dateutil.relativedelta import MO, relativedelta
from scipy.signal import argrelmax
from scipy.stats import norm
from statsmodels.tsa.stattools import acf

from forecasting.configs import logger

T = TypeVar("T", str, bytes)


def iterable_to_str(iterable: Iterable) -> str:
    return "'" + "', '".join([str(item) for item in iterable]) + "'"


def verify_str_arg(
    value: T,
    arg: Optional[str] = None,
    valid_values: Optional[Iterable[T]] = None,
    custom_msg: Optional[str] = None,
) -> T:
    if not isinstance(value, str):
        if arg is None:
            msg = "Expected type str, but got type {type}."
        else:
            msg = "Expected type str for argument {arg}, but got type {type}."
        msg = msg.format(type=type(value), arg=arg)
        raise ValueError(msg)

    if valid_values is None:
        return value

    if value not in valid_values:
        if custom_msg is not None:
            msg = custom_msg
        else:
            msg = "Unknown value '{value}' for argument {arg}. Valid values are {{{valid_values}}}."
            msg = msg.format(
                value=value, arg=arg, valid_values=iterable_to_str(valid_values)
            )
        raise ValueError(msg)

    return value


def get_series_type(x_in: np.array):
    """Calculate series type based on ADI and CV2

    Args:
        x_in (np.array): Array of values of series

    Returns:
        (str, float, float): series_type, ADI, CV2
    """
    if len(x_in[x_in > 0]) == 0:
        adi = np.nan
    else:
        adi = len(x_in) / len(x_in[x_in > 0])

    if (len(x_in) > 2) and (x_in.sum() > 0):
        cov = np.std(x_in) / np.mean(x_in)
    else:
        cov = np.nan

    if np.isnan(adi) or np.isnan(cov):
        series_type = "NA"
    else:
        if adi > 1.32:
            if cov > 0.49:
                series_type = "lumpy"
            else:
                series_type = "intermittent"
        else:
            if cov > 0.49:
                series_type = "erratic"
            else:
                series_type = "smooth"
    return series_type, adi, cov


def find_sale_pattern(len_month, adi):
    if len_month > 6:
        if np.isnan(adi):
            return "zero-sales"
        if adi > 1.06:
            return "intermittent"
        return "continuous"
    return "newly-launched"


def calc_adi(x_in):
    """Calculate ADI for column whose value is greater than zero

    Args:
        x_in (array): Input Timeseries Array

    Returns:
        int: Average Demand Interval
    """
    if len(x_in[x_in > 0]) == 0:
        adi = np.nan
    else:
        adi = len(x_in) / len(x_in[x_in > 0])

    return adi


def calc_cov(x_in):
    """Calculate covariance if series has more than 2 values and
    sum greater than zero to avoid division by zero

    Args:
        x_in (array): Input Timeseries Array

    Returns:
        int: Coefficient of Variation
    """
    if (len(x_in) > 2) and (x_in.sum() > 0):
        cov = np.std(x_in) / np.mean(x_in)
    else:
        cov = np.nan
    return cov


def get_top_best_seller_items(df: pd.DataFrame):
    # sort by gross sale
    df_tmp = df.sort_values(by=["gross_sale"], ascending=False).reset_index(drop=True)
    sum_sale = df_tmp["gross_sale"].sum()
    # calculate cum sum sale
    df_tmp["cum_sum_sale"] = df_tmp["gross_sale"].transform(pd.Series.cumsum)
    # calculate cum summed percent
    df_tmp["sum_percent"] = df_tmp["cum_sum_sale"] / sum_sale
    # get top variants accounting on 80% of sales of brand
    id_ = df_tmp[df_tmp["sum_percent"] >= 0.8].index[0]
    return df_tmp.iloc[: id_ + 1]


def get_top_10(df: pd.DataFrame):
    return df.iloc[:10]


def get_top(df: pd.DataFrame, length: int = 50):
    df = df.sort_values(by=["gross_sale"], ascending=False).iloc[:length]
    return df.reset_index(drop=True)


def split_train_test(
    freq_data_path,
    train_data_save_path,
    test_data_save_path,
    test_best_seller_data_save_path,
    N_test_length,
    N_month_best_seller,
):
    """Split the <freq>_series_data.parquet to train and test"""
    df_freq_series = pd.read_parquet(
        freq_data_path, engine="pyarrow", use_nullable_dtypes=True
    )
    df_freq_series["gross_sales"] = (
        df_freq_series["quantity_order"] * df_freq_series["price"]
    )

    last_date = df_freq_series["date"].max()
    split_date = last_date - timedelta(weeks=N_test_length)
    df_train = df_freq_series[df_freq_series["date"] <= split_date]
    df_test = df_freq_series[df_freq_series["date"] > split_date]

    tmp_date = last_date - timedelta(weeks=N_month_best_seller * 4)
    df_1 = df_freq_series[(df_freq_series["date"] >= tmp_date)]

    # get best seller variants
    df_v_best = df_1[~df_1["is_product"]]
    df_v_best = (
        df_v_best.groupby(["id"])
        .agg(
            {
                "brand_name": "first",
                "variant_id": "first",
                "quantity_order": np.sum,
                "gross_sale": np.sum,
            }
        )
        .reset_index()
    )
    df_v_best = (
        df_v_best.groupby(["brand_name"])
        .apply(get_top_best_seller_items)
        .reset_index(drop=True)
    )

    # get best seller products
    df_p_best = df_1[df_1["is_product"]]
    df_p_best = (
        df_p_best.groupby(["id"])
        .agg(
            {
                "brand_name": "first",
                "product_id": "first",
                "quantity_order": np.sum,
                "gross_sale": np.sum,
            }
        )
        .reset_index()
    )
    df_p_best = (
        df_p_best.groupby(["brand_name"])
        .apply(get_top_best_seller_items)
        .reset_index(drop=True)
    )
    # make best seller test set
    df_v_best_seller = df_test[
        (~df_test["is_product"])
        & (df_test["variant_id"].isin(df_v_best["variant_id"].values.tolist()))
    ]
    df_p_best_seller = df_test[
        (df_test["is_product"])
        & (df_test["product_id"].isin(df_p_best["product_id"].values.tolist()))
    ]
    df_test_best_seller = pd.concat(
        [df_v_best_seller, df_p_best_seller], ignore_index=True
    )

    df_train.to_parquet(train_data_save_path, engine="pyarrow", index=False)
    df_test.to_parquet(test_data_save_path, engine="pyarrow", index=False)
    df_test_best_seller.to_parquet(
        test_best_seller_data_save_path, engine="pyarrow", index=False
    )


# Reference from https://github.com/unit8co/darts/blob/f0ea7e5d74152e51f7d417c2a25f1793ce26bc92/darts/utils/statistics.py#L14
# with modifications
def find_seasonality(ts: TimeSeries, max_lag: int = 24, alpha: float = 0.05):
    """
    Checks whether the TimeSeries `ts` is seasonal with period `m` or not.
    We work under the assumption that there is a unique seasonality period, which is inferred
    from the Auto-correlation Function (ACF).
    Parameters
    ----------
    ts
        The time series to check for seasonality.
    max_lag
        The maximal lag allowed in the ACF.
    alpha
        The desired confidence level (default 5%).
    Returns
    -------
    Tuple[bool, Union[List, int]]
        A tuple `(season, List)`, where season is a boolean indicating whether the series has seasonality or not
        and `List` is the list of detected seasonality periods, returns (False, 0) if no seasonality detected.
    """

    ts._assert_univariate()

    n_unique = np.unique(ts.values()).shape[0]

    if n_unique == 1:  # Check for non-constant TimeSeries
        return False, 0

    r = acf(
        ts.values(), nlags=max_lag, fft=False
    )  # In case user wants to check for seasonality higher than 24 steps.

    # Finds local maxima of Auto-Correlation Function
    candidates = argrelmax(r)[0]

    if len(candidates) == 0:
        return False, 0

    # Remove r[0], the auto-correlation at lag order 0, that introduces bias.
    r = r[1:]

    # The non-adjusted upper limit of the significance interval.
    band_upper = r.mean() + norm.ppf(1 - alpha / 2) * r.var()

    # Significance test, find all admissible value. The two '-1' below
    # compensate for the index change due to the restriction of the original r to r[1:].
    admissible_values = []
    for candidate in candidates:
        stat = _bartlett_formula(r, candidate - 1, len(ts))
        if r[candidate - 1] > stat * band_upper:
            admissible_values.append(candidate)

    if len(admissible_values) != 0:
        return True, admissible_values

    return False, 0


def get_index_from_unique_id(id_str: str, list_ts: list):
    for i, ts in enumerate(list_ts):
        u_id = ts.static_covariates.id.values[0]
        if id_str == u_id:
            return i
    return -1


def load_monthly_data_as_df(dataset):
    # Read monthly historical values from train/test set
    train_monthly_path = (
        dataset._metadata.source_data_path / "monthly_dataset/monthly_series.parquet"
    )
    train_monthly_df = pd.read_parquet(
        train_monthly_path, engine="pyarrow", use_nullable_dtypes=True
    )

    # Read monthly historical values from inference set
    infer_monthly_df = pd.read_parquet(
        dataset._infer_monthly_data_path, engine="pyarrow", use_nullable_dtypes=True
    )

    # Concat monthly df from train/test set and inference set
    train_final_month = max(train_monthly_df.date)
    filter_train_monthly_df = train_monthly_df[
        train_monthly_df.date < train_final_month
    ]
    filter_infer_monthly_df = infer_monthly_df[
        infer_monthly_df.date >= train_final_month
    ]
    monthly_df = (
        pd.concat([filter_train_monthly_df, filter_infer_monthly_df])
        .sort_values(by=["id", "date"])
        .reset_index(drop=True)
    )

    meta_df = pd.read_parquet(
        dataset._infer_metadata_path, engine="pyarrow", use_nullable_dtypes=True
    )

    monthly_df = monthly_df.merge(meta_df, on="id", how="left")
    return monthly_df


def find_leading_zeros(col, min_length=1):
    """
    Find the index of leading zero elements from a series until the first non-zero value
    or until the length of remaining elements is equal to min_length.

    Parameters
    ----------
    col : pandas Series
        Input series column.
    min_length : int, optional
        Minimum length of remaining elements. Defaults to 1.

    Returns
    -------
    array
        Array with leading zeros elements marked as True, non-zero elements are marked as False,
        ensuring that at least min_length remaining elements are preserved.
    """
    col = col.reset_index(drop=True)
    non_zero_index = (col != 0).idxmax()

    # Calculate the correct leading zeros length based on min_length constraint
    if col.eq(0).all():
        # If all elements are zero, we mark until the length of remaining elements is min_length
        leading_zeros_length = len(col) - min_length
    else:
        # Otherwise, respect the min_length constraint
        leading_zeros_length = min(non_zero_index, len(col) - min_length)

    # Create a boolean mask array based on the leading_zeros_length
    mask = [True if i < leading_zeros_length else False for i in range(len(col))]

    return np.array(mask)


def remove_leading_zeros(array, min_length=1):
    """
    Remove leading zero elements from a NumPy array until the first non-zero value
    or until the length of remaining elements is equal to min_length.

    Parameters
    ----------
    array : array_like
        Input array.
    min_length : int, optional
        Minimum length of the remaining array. Defaults to 1.

    Returns
    -------
    array
        Array with the leading zero elements removed until the first non-zero value
        or until the length of the remaining elements is equal to min_length.
    """
    if isinstance(array, list):
        array = np.array(array)

    non_zero_index = np.nonzero(array)[0]

    if len(non_zero_index) > 0:
        # Determine the index to start slicing from
        first_nonzero_idx = non_zero_index[0]
        # Ensure we do not remove beyond the length limit imposed by min_length
        if first_nonzero_idx < len(array) - min_length:
            return array[first_nonzero_idx:]
        else:
            return array[-min_length:]

    # If no non-zero elements exist, return the last min_length elements
    return array[-min_length:]


def remove_leading_zeros_maximum_6_months(array):
    """
    Remove zero elements from a NumPy array until the first non-zero value, remove maximum 6 months (time for launching)

    Parameters
    ----------
    array : array_like
        Input array.

    Returns
    -------
    array
        Array with the zero elements removed until the first non-zero value.
    """
    if isinstance(array, list):
        array = np.array(array)

    non_zero_index = np.nonzero(array)[0]

    if len(non_zero_index) > 0:
        first_nonzero_idx = non_zero_index[0]
        if first_nonzero_idx > 5:  # more than 6 months with zero-sales
            return array[6:]
        return array[first_nonzero_idx:]

    return array


def remove_leading_zeros_cumsum(array, thresh_value):
    cumsum = [sum(array[: i + 1]) for i in range(len(array))]

    # Tìm vị trí đầu tiên mà cumsum lớn hơn thresh_value
    index = -1
    for i in range(len(cumsum)):
        if cumsum[i] > thresh_value:
            index = i
            break
    # Loại bỏ phần nhiễu phía trước index
    arr_trimmed = array[index:]
    return arr_trimmed


def find_leading_zeros_cumsum(col, thresh_value=0):
    col = col.reset_index(drop=True)
    cumsum = [sum(col[: i + 1]) for i in range(len(col))]

    # Tìm vị trí đầu tiên mà cumsum lớn hơn thresh_value
    index = -1
    for i in range(len(cumsum)):
        if cumsum[i] > thresh_value:
            index = i
            break
    return np.where(np.arange(len(col)) < index, True, False)


def calc_sale_per_day(daily_ts, time_steps=90):
    if len(daily_ts) == 0:  # No historical value
        spd = 0
    else:
        if len(daily_ts) >= time_steps:
            spd = round(np.mean(daily_ts[-time_steps:]), 7)
        else:
            spd = round(np.mean(daily_ts), 7)
    return spd


def calc_created_time(inference_date, created_date):
    # Convert np.datetime64 to datetime
    unix_epoch = np.datetime64(0, "s")
    one_second = np.timedelta64(1, "s")
    seconds_since_epoch = (created_date - unix_epoch) / one_second
    date = datetime.utcfromtimestamp(seconds_since_epoch).date()

    created_time = inference_date - date
    return created_time.days


def calc_start_end_test_date(max_date, test_length, frequency):
    first_day_of_this_month = max_date.replace(day=1)
    end_test_date = first_day_of_this_month - timedelta(days=1)
    # get the middle day of the first test month
    start_test_date = end_test_date - timedelta(days=(test_length - 1) * 31 + 15)
    start_test_date = start_test_date.replace(day=1)
    start_test_date = (
        start_test_date + relativedelta(weekday=MO(-1))
        if frequency == "W-Mon"
        else start_test_date
    )
    return start_test_date, end_test_date


def create_top_selling_item_sets(daily_df, meta_df, start_test_date):
    meta_cols = [
        "id",
        "variant_id",
        "product_id",
        "channel_id",
        "brand_name",
        "is_product",
    ]
    # Merge metadata
    daily_df_full = daily_df.merge(meta_df[meta_cols], on="id", how="left")
    daily_df_full["gross_sale"] = (
        daily_df_full["quantity_order"] * daily_df_full["price"]
    )
    logger.info("Finding best seller items...")
    tmp_date = start_test_date - timedelta(weeks=24)

    # best seller items accounted on all channels
    tmp_df = daily_df_full[
        (daily_df_full["date"] >= tmp_date) & (daily_df_full["date"] < start_test_date)
    ]
    tmp_df = tmp_df[
        (tmp_df["channel_id"] == "0")
        | (tmp_df["channel_id"] == "c7d674e95445867e3488398e8b2cd2d8")
    ]

    # get best seller variants
    df_v_best = tmp_df[~tmp_df["is_product"]]
    df_v_best = (
        df_v_best.groupby(["id"])
        .agg(
            {
                "brand_name": "first",
                "variant_id": "first",
                "quantity_order": np.sum,
                "gross_sale": np.sum,
            }
        )
        .reset_index()
    )
    df_v_best = (
        df_v_best.groupby(["brand_name"])
        .apply(get_top_best_seller_items)
        .reset_index(drop=True)
    )

    # get best seller products
    df_p_best = tmp_df[tmp_df["is_product"]]
    df_p_best = (
        df_p_best.groupby(["id"])
        .agg(
            {
                "brand_name": "first",
                "product_id": "first",
                "quantity_order": np.sum,
                "gross_sale": np.sum,
            }
        )
        .reset_index()
    )
    df_p_best = (
        df_p_best.groupby(["brand_name"])
        .apply(get_top_best_seller_items)
        .reset_index(drop=True)
    )

    best_seller_ids_list = (
        df_v_best["id"].values.tolist() + df_p_best["id"].values.tolist()
    )

    logger.info("Finding top 10 items...")
    best_product_df = (
        df_p_best.groupby(["brand_name"]).apply(get_top_10).reset_index(drop=True)
    )
    top_10_ids_list = best_product_df["id"].values.tolist()

    return best_seller_ids_list, top_10_ids_list


def split_train_test_set(
    start_test_date: str,
    end_test_date: str,
    frequency_df: pd.DataFrame,
    meta_df: pd.DataFrame,
):
    start_test_date = pd.to_datetime(start_test_date)
    end_test_date = pd.to_datetime(end_test_date)
    meta_cols = ["id", "status", "created_date"]
    # start_test_date, end_test_date = calc_start_end_test_date(daily_df, test_length, frequency)
    logger.info(f"Start test date: {start_test_date}, End test date: {end_test_date}")

    frequency_df = frequency_df.merge(meta_df[meta_cols], on="id", how="left")

    active_status = [
        "active",
        "enabled",
        "disabled, enabled",
        "enabled, disabled",
        "draft",
        "archived",
        "draft, active",
        "active, draft",
    ]

    # Create new item set
    logger.info("Create new item set...")
    df_new_item = frequency_df[frequency_df["created_date"] >= start_test_date]
    df_new_item = df_new_item[df_new_item["status"].isin(active_status)].reset_index()

    # Create train, test set
    logger.info("Split train, test set...")
    df_train = frequency_df[frequency_df["date"] < start_test_date]
    df_test = frequency_df[
        (frequency_df["date"] >= start_test_date)
        & (frequency_df["date"] <= end_test_date)
    ]
    # filter activate items
    df_test = df_test[df_test["status"].isin(active_status)]
    # filter out new items
    new_item_ids = df_new_item["id"].unique()
    df_test = df_test[~df_test["id"].isin(new_item_ids)].reset_index()

    return df_train, df_test, df_new_item


def get_top_items_list(frequency_df: pd.DataFrame, length: int):

    max_date = frequency_df["date"].max()
    start_date = max_date - pd.offsets.Week(4)
    print(f"Check top {length} from week {start_date} to week {max_date}")

    df_filtered = frequency_df[frequency_df["date"].between(start_date, max_date)]
    df_filtered = df_filtered.assign(
        gross_sale=(df_filtered["quantity_order"] * df_filtered["price"])
    )

    df_tmp = (
        df_filtered.groupby(["id"])
        .agg(
            {
                "brand_name": "first",
                "channel_id": "first",
                "is_product": "first",
                "gross_sale": np.sum,
            }
        )
        .reset_index()
    )

    df_top = (
        df_tmp.groupby(["brand_name", "channel_id", "is_product"])
        .apply(lambda x: get_top(x, length))
        .reset_index(drop=True)
    )
    top_items = df_top.id.tolist()

    return top_items


def calculate_sale_pattern(
    monthly_df,
    forecast_date=None,
    brands_to_create_all_channel=None,
    unique_ids_list=None,
):
    logger.info("Finding sale_pattern...")

    # Create pivot table
    monthly_pivot = (
        pd.pivot_table(
            monthly_df,
            index="id",
            values=["quantity_order", "price", "brand_name", "platform"],
            aggfunc={
                "quantity_order": list,
                "price": "last",
                "brand_name": "first",
                "platform": "first",
            },
        )
        .rename(columns={"quantity_order": "monthly_train_ts"})
        .reset_index()
    )
    if unique_ids_list is not None:
        monthly_pivot = monthly_pivot[monthly_pivot.id.isin(unique_ids_list)]

    logger.info(
        f"Monthly_pivot before process all_channel for specified brands: {monthly_pivot.shape[0]}"
    )

    # Process for specified brands all_channel
    if brands_to_create_all_channel is not None:
        df = monthly_pivot[monthly_pivot.brand_name.isin(brands_to_create_all_channel)]
        if df.shape[0] > 0:
            logger.info(f"Number of items of specified brands: {df.shape[0] / 2}")
            df = df.assign(
                sum_id=lambda df: df.id.apply(lambda x: "_".join(x.split("_")[:-1]))
            )
            df.monthly_train_ts = df.monthly_train_ts.apply(lambda x: np.array(x))
            sum_df = (
                df.groupby("sum_id")
                .agg(
                    {
                        "monthly_train_ts": np.sum,
                        "price": np.mean,
                        "brand_name": "first",
                        "platform": "first",
                    }
                )
                .reset_index()
            )
            sum_df["id"] = sum_df.sum_id.apply(lambda x: x + "_0")
            sum_df.monthly_train_ts = sum_df.monthly_train_ts.apply(lambda x: list(x))
            monthly_pivot = pd.concat([monthly_pivot, sum_df.drop(columns=["sum_id"])])
        logger.info(
            f"Monthly_pivot after process all_channel for specified brands: {monthly_pivot.shape[0]}"
        )

    # Calculate length of month
    monthly_pivot["len_month"] = monthly_pivot.monthly_train_ts.apply(lambda x: len(x))

    # Remove leading zeros (commented out in original)
    # monthly_pivot['monthly_train_ts'] = monthly_pivot.apply(
    #     lambda row: (
    #         remove_leading_zeros_maximum_6_months(row.monthly_train_ts)
    #         if np.isnan(row.adi) == False
    #         else row.monthly_train_ts
    #     ),
    #     axis=1
    # )

    # Truncate to the last 6 or 7 months
    monthly_pivot["monthly_train_ts_truncated"] = monthly_pivot.monthly_train_ts.apply(
        lambda x: (x[-6:] if forecast_date and (forecast_date.day == 1) else x[-7:-1])
    )

    # Calculate ADI
    monthly_pivot["adi"] = monthly_pivot.monthly_train_ts_truncated.apply(
        lambda x: calc_adi(np.array(x))
    )

    # Calculate sale pattern
    monthly_pivot["sale_pattern"] = monthly_pivot.apply(
        lambda row: find_sale_pattern(row.len_month, row.adi), axis=1
    )

    # Calculate the revenue for the last 3 months
    monthly_pivot["revenue"] = monthly_pivot.apply(
        lambda row: row["price"]
        * sum(
            row["monthly_train_ts_truncated"][-3:]
            if forecast_date and (forecast_date.day == 1)
            else row["monthly_train_ts_truncated"][-4:-1]
        ),
        axis=1,
    )

    # Perform ABC analysis within each brand
    def categorize_by_revenue(group):
        group = group.sort_values(by="revenue", ascending=False)
        total_revenue = group["revenue"].sum()
        group["cumulative_revenue_pct"] = (
            group["revenue"].cumsum() / total_revenue * 100
        )
        group["sale_category"] = group["cumulative_revenue_pct"].apply(
            categorize_revenue
        )
        return group

    monthly_pivot = (
        monthly_pivot.groupby(["brand_name", "platform"])
        .apply(categorize_by_revenue)
        .reset_index(drop=True)
    )

    logger.info(f"Dataframe with sale_pattern: {monthly_pivot.shape[0]}")

    return monthly_pivot


def categorize_revenue(cumulative_revenue_pct):
    if cumulative_revenue_pct <= 80:
        return "A"
    elif cumulative_revenue_pct <= 95:
        return "B"
    else:
        return "C"
