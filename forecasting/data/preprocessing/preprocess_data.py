from typing import List, Optional, Union

import numpy as np
import pandas as pd
from darts import TimeSeries
from tqdm import tqdm

from forecasting.configs.logging_config import logger
from forecasting.data.preprocessing.utils import extend_df, find_similar_items
from forecasting.data.util import get_series_type
from forecasting.data.utils.darts_utils import extend_time_series
from forecasting.data.utils.pandas_utils import clean_static_feat


def preprocess_for_training(
    df: pd.DataFrame,
    input_chunk_length: int,
    output_chunk_length: int,
    seasonal_set: list = None,
) -> pd.DataFrame:
    required_len = input_chunk_length + output_chunk_length
    seasonal_set = [] if seasonal_set is None else seasonal_set
    logger.info("Filtering out items with short historical length")

    df_processed = []
    for item_id, group in tqdm(df.groupby("id")):
        group = group[group["quantity_order"].cumsum() > 0]
        series_type, _, _ = get_series_type(group["quantity_order"])

        # if (len(group) > required_len) and (series_type != 'lumpy') or (
        #     (len(seasonal_set) > 0) and (item_id in seasonal_set)
        # ):
        if (
            (len(group) > required_len)
            and (series_type != "lumpy")
            or ((item_id in seasonal_set) and (len(group) > required_len))
        ):
            df_processed.append(group)

    df_processed = pd.concat(df_processed, ignore_index=True)

    return df_processed


def preprocess_for_inference(
    df: pd.DataFrame,
    preprocess_method: str = "fill_avg_sim",
    min_length_for_new_item: int = 12,
):
    if preprocess_method not in {"fill_zero", "fill_avg_sim"}:
        raise ValueError(
            f"Unknown value preprocess_method: {preprocess_method}, "
            "Only `fill_zero` and `fill_avg_sim` are valid values."
        )
    # Clean static feat
    clean_static_feat(df, ["category", "color"])

    # Process variants
    logger.info("Processing variants")
    df_variant = df[~df["is_product"]]
    logger.info(f"df_variant shape: {df_variant.shape}, {df_variant.id.unique().shape}")
    df_variant_meta = df_variant.drop_duplicates(subset=["id"])
    # Create unique 'text' for each item id
    df_variant_meta = df_variant_meta.assign(
        text=(
            df_variant_meta["brand_name"]
            + " "
            + df_variant_meta["channel_id"]
            + " "
            + df_variant_meta["category"]
            + " "
            + df_variant_meta["color"]
            + " "
            + df_variant_meta["name"]
        )
    )
    item_texts = df_variant_meta["text"].tolist()
    # Filter new items and old items based on min_length
    grouped = df_variant.groupby("id").agg(
        {"channel_id": "count", "quantity_order": "sum"}
    )
    mask_old_item = (grouped["channel_id"] >= min_length_for_new_item) & (
        grouped["quantity_order"] > 3
    )
    mask_new_item = grouped["channel_id"] < min_length_for_new_item
    mask_other_item = (grouped["channel_id"] >= min_length_for_new_item) & (
        grouped["quantity_order"] <= 3
    )

    old_item_df = df_variant[df_variant["id"].isin(grouped.loc[mask_old_item].index)]
    new_item_df = df_variant[df_variant["id"].isin(grouped.loc[mask_new_item].index)]
    other_item_df = df_variant[
        df_variant["id"].isin(grouped.loc[mask_other_item].index)
    ]

    logger.info(
        f"old_item_df shape: {old_item_df.shape}, {old_item_df.id.unique().shape}"
    )
    logger.info(
        f"new_item_df shape: {new_item_df.shape}, {new_item_df.id.unique().shape}"
    )
    logger.info(
        f"other_item_df shape: {other_item_df.shape}, {other_item_df.id.unique().shape}"
    )

    similar_item_dict = dict()
    if preprocess_method == "fill_avg_sim":
        logger.info("Finding similar items")
        # Find top 5 similar items for each new item from the old items
        similar_item_dict = find_similar_items(
            df_variant, old_item_df, new_item_df, item_texts
        )
        similar_old_items = list(np.unique(list(similar_item_dict.values())))
        similar_old_item_df = df_variant[
            df_variant["id"].isin(similar_old_items)
        ].reset_index(drop=True)

        # Calculate average historical quantity order of the similar old items
        similar_item_df = pd.DataFrame(
            similar_item_dict.items(), columns=["id", "similar_items"]
        )
        similar_item_df_long = similar_item_df.explode("similar_items")
        similar_item_df_long = (
            similar_item_df_long.merge(
                similar_old_item_df[["id", "date", "quantity_order"]],
                left_on="similar_items",
                right_on="id",
                how="inner",
            )
            .drop(columns=["id_y"])
            .rename(columns={"id_x": "id"})
        )
        avg_sim_df = (
            similar_item_df_long.groupby(["id", "date"])["quantity_order"]
            .mean()
            .round()
            .reset_index()
        )
    else:
        similar_item_df = None
        avg_sim_df = None
    logger.info("Extending new_item_df and fill the extended periods")
    # Extend the new items into the past and fill the extended periods
    new_item_df_extended = extend_df(
        new_item_df,
        value_col="quantity_order",
        avg_sim_df=avg_sim_df,
        required_len=min_length_for_new_item,
        direction="past",
        freq="W-MON",
        fill_method=preprocess_method,
    )
    all_variant_df_processed = pd.concat(
        [new_item_df_extended, old_item_df, other_item_df], ignore_index=True
    )
    logger.info(
        f"new_item_df_extended shape: {new_item_df_extended.shape}, {new_item_df_extended.id.unique().shape}"
    )
    logger.info(
        f"all_variant_df_processed shape: {all_variant_df_processed.shape}, {all_variant_df_processed.id.unique().shape}"
    )

    # Process for products
    logger.info("Processing products")
    all_product_meta_data = (
        df[df["is_product"]]
        .drop_duplicates(subset=["id"])
        .drop(columns=["date", "quantity_order", "price"])
        .reset_index(drop=True)
    )
    all_product_df_processed = (
        all_variant_df_processed.groupby(
            ["brand_name", "platform", "product_id", "channel_id", "date"]
        )
        .agg({"quantity_order": np.sum, "price": np.mean})
        .reset_index()
    )
    all_product_df_processed = all_product_df_processed.merge(
        all_product_meta_data,
        on=["brand_name", "platform", "product_id", "channel_id"],
        how="left",
    ).dropna(subset=["id"])
    logger.info(
        f"all_product_df_processed shape: {all_product_df_processed.shape}, {all_product_df_processed.id.unique().shape}"
    )
    final_processed_df = pd.concat(
        [all_variant_df_processed, all_product_df_processed], ignore_index=True
    )
    logger.info(
        f"final_processed_df shape: {final_processed_df.shape}, {final_processed_df.id.unique().shape}"
    )

    return final_processed_df, similar_item_dict


def preprocess_data_darts(
    stage: str,
    target_data: List[TimeSeries],
    past_cov_data: List[TimeSeries],
    input_chunk_length: int,
    output_chunk_length: int,
    prediction_length: int,
) -> Union[List[TimeSeries], Optional[List[TimeSeries]]]:
    """
    If stage == 'train':
        Remove all-zero time series and cut zero values at the beginning
    Otherwise
        Check length to apply padding for short time series and extend past covariates into the future
    """
    if stage not in {"train", "inference"}:
        raise ValueError(
            f"Unknown value stage: {stage}, Only `train` and `inference` are valid values."
        )

    if stage == "train":
        required_len = input_chunk_length + output_chunk_length
        logger.info("Filtering out all zero time series.")

        filter_list = []
        for idx, target_ts in tqdm(enumerate(target_data), total=len(target_data)):
            target_ts_values = target_ts.values().ravel()

            if target_ts_values.sum() != 0:
                # Find index of first nonzero value
                first_non_zero_index = np.nonzero(target_ts_values)[0][0]

                res = {"id": idx, "first_non_zero_index": first_non_zero_index}

                filter_list.append(res)

        logger.info(
            "Removing leading zero values for every time series, "
            f"and only keep series with length > {required_len}"
        )
        target_filtered = []
        past_cov_filtered = [] if past_cov_data else None

        for item in tqdm(filter_list):
            idx = item["id"]
            first_non_zero_index = item["first_non_zero_index"]

            target_ts = target_data[idx]
            past_cov_ts = past_cov_data[idx] if past_cov_data else None
            if first_non_zero_index != 0:
                target_ts = target_ts.drop_before(first_non_zero_index - 1)
                if past_cov_ts:
                    past_cov_ts = past_cov_ts.drop_before(first_non_zero_index - 1)

            target_ts_values = target_ts.values().ravel()
            series_type, _, _ = get_series_type(target_ts_values)

            if len(target_ts_values) > required_len:
                if series_type != "lumpy":
                    target_filtered.append(target_ts)
                    if past_cov_filtered is not None:
                        past_cov_filtered.append(past_cov_ts)

        logger.info(
            "Number of time series available for training after filtering: "
            f"({len(target_filtered)}, {len(past_cov_filtered)})"
        )

        return target_filtered, past_cov_filtered

    else:
        required_len = input_chunk_length
        min_data_len = min([len(s) for s in target_data])

        if min_data_len < required_len:
            logger.info(
                "Dataset contains some short time series, "
                "adding padding for those time series to have enough historical data "
                "in order to run inference"
            )
            target_data = extend_time_series(
                direction="past",
                data=target_data,
                required_len=required_len,
                fill_value=0.0,
                fill_method=None,
            )

        if past_cov_data:
            past_cov_data = extend_time_series(
                direction="past",
                data=past_cov_data,
                required_len=required_len,
                fill_value=None,
                fill_method="bfill",
            )

            # Extend past covariate into future if prediction_length > output_chunk_length
            if prediction_length > output_chunk_length:
                logger.info(
                    "prediction_length is bigger than output_chunk_length,"
                    "extending past_covariate into future"
                )
                pad_length = prediction_length - output_chunk_length
                past_cov_data = extend_time_series(
                    direction="future",
                    data=past_cov_data,
                    required_len=pad_length,
                    fill_value=None,
                    fill_method="ffill",
                )
        return target_data, past_cov_data
