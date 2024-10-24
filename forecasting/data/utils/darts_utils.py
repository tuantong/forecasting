import itertools
import pickle
from pathlib import Path
from typing import List, Optional, Sequence, Union

import billiard as multiprocessing
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from tqdm import tqdm

from forecasting.configs.logging_config import logger


def get_static_covariate_transformer(path) -> StaticCovariatesTransformer:
    try:
        with open(path, "rb") as f:
            static_cov_transfomer = pickle.load(f)
    except OSError as exc:
        raise ValueError(
            f"Static covariate transformer is not found at {path}"
            "You may have not fitted or saved the static covariate transformer"
        ) from exc
    return static_cov_transfomer


def extend_time_series(
    direction: str,
    data: List[TimeSeries],
    required_len: int,
    fill_value: Optional[float],
    fill_method: Optional[str],
):
    if direction not in ["past", "future"]:
        raise ValueError(
            f"Unknown direction: {direction}. Supported direction are: 'past' and 'future'."
        )

    padded_data = []

    for ts in tqdm(data):
        ts_static_cov = ts.static_covariates
        logger.debug(f"Len TS before pad: {len(ts)}")
        ts_pd = ts.pd_dataframe()
        if direction == "past":
            # Extend time series into the past for time series that has length < required_len
            # New length will be <required_len>
            if len(ts) < required_len:
                ts_pd_padded = ts_pd.reindex(
                    index=pd.date_range(
                        end=ts_pd.index.max(), freq=ts.freq, periods=required_len
                    ),
                    method=fill_method,
                    fill_value=fill_value,
                )
                ts = TimeSeries.from_dataframe(
                    ts_pd_padded, static_covariates=ts_static_cov
                )
                logger.debug(f"Len TS after pad: {len(ts)}")
        else:
            # Extend time series into the future
            # required_len in this case will be the number of points to add into the future\
            # New length will be len(ts) + required_len
            pad_length = len(ts) + required_len
            ts_pd_padded = ts_pd.reindex(
                index=pd.date_range(
                    start=ts_pd.index.min(), freq=ts.freq, periods=pad_length
                ),
                method=fill_method,
                fill_value=fill_value,
            )
            ts = TimeSeries.from_dataframe(
                ts_pd_padded, static_covariates=ts_static_cov
            )
            logger.debug(f"Len TS after pad: {len(ts)}")

        padded_data.append(ts)

    return padded_data


def split_train_valid(ts_list: Sequence[TimeSeries], VALID_LEN):
    train_list = []
    valid_list = []
    for ts in tqdm(ts_list):
        train_ts = ts[:-VALID_LEN]
        valid_ts = ts[-VALID_LEN:]

        train_list.append(train_ts)
        valid_list.append(valid_ts)

    return train_list, valid_list


def pad_short_time_series(
    freq_df: pd.DataFrame,
    required_len: int,
    freq: str,
    index_col: str,
    target_col: str,
    price_col: str,
    other_cols: List[str],
):
    """Pad short time series to a required_len"""
    freq_df_grouped = freq_df.set_index(index_col).groupby("id", observed=True)
    logger.info("Starting preprocessing step")

    freq_df_processed = []
    for u_id, df_group in tqdm(freq_df_grouped):
        logger.debug(f"Processing TS: {u_id}")
        len_ts = len(df_group)
        logger.debug(f"Len TS: {len_ts}")

        if len_ts < required_len:
            pad_length = required_len
            logger.debug(
                f"Len TS too short, padding to have len of {pad_length} points"
            )

            df_group = df_group.reindex(
                index=pd.date_range(
                    end=df_group.index.max(), freq=freq, periods=pad_length
                ),
                fill_value=pd.NA,
            )

            df_group[target_col] = df_group[target_col].fillna(0.0)
            df_group[price_col] = df_group[price_col].fillna(method="bfill")
            df_group[other_cols] = df_group[other_cols].fillna(method="bfill")

            logger.debug(f"Len TS after padding: {len(df_group)}")

        freq_df_processed.append(df_group)

    freq_df_processed = (
        pd.concat(freq_df_processed, axis=0).rename_axis(index_col).reset_index()
    )

    return freq_df_processed


def get_cat_embedding(categories_len_dict):
    categorical_embedding_sizes = {}

    for key, val in categories_len_dict.items():
        if val < 128:
            size = val
        elif 128 < val < 256:
            size = 128
        elif 256 < val < 512:
            size = 256
        elif 512 < val < 1024:
            size = 512
        else:
            size = 1024
        # if val > 2:
        #     size =  min(round(1.6 * val**0.56), 300)

        categorical_embedding_sizes[key] = (val + 1, size)
    return categorical_embedding_sizes


def process_group(args):
    """
    Process a group of data to create a TimeSeries object.

    Args:
        args (tuple): A tuple containing the following elements:
            - group: The group of data.
            - time_col: The column name representing the time index of the time series.
            - value_cols: The column name(s) representing the value(s) of the time series.
            - freq: The frequency of the time series data.
            - static_cols: The column name(s) representing the static covariates for the time series.
            - dtype: The desired data type for the time series values.

    Returns:
        TimeSeries: The processed time series object.
    """
    group, time_col, value_cols, freq, static_cols, dtype = args
    static_covariates_df = group[static_cols].head(1) if static_cols else None

    return TimeSeries.from_dataframe(
        df=group,
        time_col=time_col,
        value_cols=value_cols,
        freq=freq,
        static_covariates=static_covariates_df,
    ).astype(dtype)


def convert_df_to_darts_dataset(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    value_cols: Union[str, List[str]],
    static_cols: Union[str, List[str]],
    freq: str,
    dtype: np.float32,
    save_path: Path = None,
    num_cores: int = -1,
) -> Union[TimeSeries, List[TimeSeries]]:
    """
    Convert a pandas DataFrame into a list of univariate/multivariate time series using the Darts library.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        id_col (str): The column name representing the ID or grouping key for the time series.
        time_col (str): The column name representing the time index of the time series.
        value_cols (Union[str, List[str]]): The column name(s) representing the value(s) of the time series.
        static_cols (Union[str, List[str]]): The column name(s) representing the static covariates for the time series.
        freq (str): The frequency of the time series data (e.g., 'D' for daily, 'H' for hourly).
        dtype (np.float32): The desired data type for the time series values.
        save_path (Path, optional): The file path to save the resulting time series object(s) as a pickle file. Defaults to None.
        num_cores (int, optional): The number of CPU cores to use for parallel processing. Defaults to -1, which uses all available cores.

    Returns:
        Union[TimeSeries, List[TimeSeries]]: The converted time series object(s). If `save_path` is specified, the resulting time series
            object(s) are also saved as a pickle file.
    """

    num_cores = multiprocessing.cpu_count() if num_cores == -1 else num_cores

    groups = [group for _, group in df.groupby(id_col, sort=False)]
    total_iterations = len(groups)
    args_list = zip(
        groups,
        itertools.repeat(time_col),
        itertools.repeat(value_cols),
        itertools.repeat(freq),
        itertools.repeat(static_cols),
        itertools.repeat(dtype),
    )

    with multiprocessing.Pool(num_cores) as pool, tqdm(total=total_iterations) as pbar:
        ts_list = []
        for ts in pool.imap(process_group, args_list):
            ts_list.append(ts)
            pbar.update(1)

    if save_path:
        logger.info(f"Saving to {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(ts_list, f)

    return ts_list
