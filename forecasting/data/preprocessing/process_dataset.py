import json
import os
from collections import Counter
from datetime import date, timedelta

import numpy as np
import pandas as pd
from dateutil.relativedelta import SU, relativedelta

from forecasting import PROCESSED_DATA_BUCKET, PROJECT_ROOT_PATH
from forecasting.configs import logger
from forecasting.data.s3_utils.aws_service import AWSStorage, aws_client

s3_data_service = AWSStorage(client=aws_client, bucket=PROCESSED_DATA_BUCKET)
STORAGE_OPTIONS = {
    "key": os.getenv("AWS_ACCESS_KEY_ID"),
    "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
}
S3_PREFIX = f"s3://{PROCESSED_DATA_BUCKET}"


def resample_brand_data(daily_df, freq, closed, label, chunk_size=1000, log_interval=5):
    """
    Incrementally resamples the input DataFrame `daily_df` by brand IDs.

    Args:
    - daily_df (pd.DataFrame): DataFrame containing daily data with at least 'id', 'datetime', 'quantity_order', and 'price'.
    - freq (str): Resampling frequency (e.g., 'M' for monthly, 'W' for weekly).
    - closed (str): Which side of the bin interval is closed ('right' or 'left').
    - label (str): How to label the bins ('right' or 'left').
    - chunk_size (int, optional): Number of IDs to process in each chunk. Default is 1000.
    - log_interval (int, optional): Interval at which to log progress. Default is 5.

    Returns:
    - pd.DataFrame: Resampled DataFrame.
    """
    # Unique IDs in the DataFrame
    unique_ids = daily_df["id"].unique()

    # Total number of chunks
    total_chunks = (len(unique_ids) + chunk_size - 1) // chunk_size

    # List to hold the resampled chunks
    resampled_chunks = []

    for i in range(0, len(unique_ids), chunk_size):
        # IDs for the current chunk
        chunk_ids = unique_ids[i : i + chunk_size]

        # Filter the DataFrame for the current chunk of IDs
        chunk_df = daily_df[daily_df["id"].isin(chunk_ids)]

        # Resample the chunk
        resampled_chunk = (
            chunk_df.groupby(
                ["id", pd.Grouper(key="date", freq=freq, closed=closed, label=label)]
            )
            .agg(
                {
                    "quantity_order": np.sum,
                    "price": np.mean,
                    "views": np.sum,
                    "gg_click": np.sum,
                    "gg_ad_spends": np.sum,
                    "fb_click": np.sum,
                    "fb_ad_spends": np.sum,
                    "ad_click": np.sum,
                    "ad_spends": np.sum,
                }
            )
            .reset_index()
        )

        # Add the resampled chunk to the list
        resampled_chunks.append(resampled_chunk)

        # Log progress after processing each chunk
        if (i // chunk_size + 1) % log_interval == 0 or (i + chunk_size) >= len(
            unique_ids
        ):
            logger.info(f"Resampled {i // chunk_size + 1} of {total_chunks} chunks")

    # Combine all the resampled chunks into a single DataFrame
    resampled_df = pd.concat(resampled_chunks, ignore_index=True)

    return resampled_df


def join_and_unique_str(series_str: pd.Series):
    s = ""
    ls = series_str.to_list()
    # get unique list
    ls = list(set(ls))
    if len(ls) > 0:
        s = ", ".join(str(x) for x in ls)
        # for tags series process
        if "," in s:
            s = list(s.split(", "))
            s = list(set(s))
            s = ", ".join(str(x) for x in s)
    return s


def get_product_replenishable_value(series_str: pd.Series):
    ls = series_str.to_list()
    # get unique list
    ls = list(set(ls))
    if len(ls) == 1:
        # If all variants of product is_replenishable = False -> False
        val = ls[0]
    else:
        val = True
    return val


def get_start_end_dates(order_df, inference_set, stop_date):
    """
    Get start and end dates for the given order data, inference set, and stop date.

    Args:
        order_df (pd.DataFrame): The order data.
        inference_set (bool): Whether the data is for inference.
        stop_date (str): The stop date for the data.

    Returns:
        tuple: A tuple containing the start and end dates.
    """
    max_date = order_df["date"].max()
    today = pd.to_datetime(date.today())
    # Get stop date of order
    if stop_date == "last_sunday":
        end_date = pd.to_datetime(today + relativedelta(weekday=SU(-1)))
    elif stop_date == "yesterday":
        end_date = pd.to_datetime(today - timedelta(days=1))
    elif stop_date == "today":
        end_date = max_date
    else:
        end_date = pd.to_datetime(stop_date)

    if inference_set:
        min_date = end_date - timedelta(weeks=24)  # last 24 weeks
    else:
        min_date = end_date - timedelta(weeks=3 * 52)  # last 3 years

    logger.info(
        f"Order data: min_date = {order_df.date.min()}, max_date = {order_df.date.max()}"
    )
    logger.info(f"Process dataset with min_date={min_date}, max_date={end_date}")
    return min_date, max_date, end_date


def read_and_prepare_data(history_dir, variant_dir, product_dir, stock_dir, ads_dir):
    """
    A function to read and prepare data from historical, variant, and product files.

    Args:
        history_dir (str): The path to the historical file.
        variant_dir (str): The path to the variant meta file.
        product_dir (str): The path to the product meta file.
        stock_dir (str): The path to the stock file.
        ads_dir(str): The path to ads_spend/clik/views file.

    Returns:
        tuple: A tuple containing three pandas DataFrames - order_df, variant_meta, product_meta and stock_df.
    """
    logger.info("Reading historical files...")
    # Read file orders
    order_columns = [
        "transaction_processed_date",
        "variant_id",
        "sku",
        "product_id",
        "channel_id",
        "quantity",
        # "location_id",
        "price",
        "from_source",
    ]

    order_dtypes = {
        "variant_id": "string",
        "sku": "string",
        "product_id": "string",
        "channel_id": "string",
        "quantity": "float32",
        "price": "float32",
        "from_source": "string",
    }
    order_df = pd.read_csv(
        history_dir,
        parse_dates=["transaction_processed_date"],
        dtype=order_dtypes,
    )
    order_df = order_df[order_df.quantity > 0]
    order_df["price"] = order_df.apply(
        lambda row: row["gross_sales"] / row["quantity"], axis=1
    )
    # drop_duplicate_subset = order_df.columns.tolist()
    # drop_duplicate_subset.remove('price')
    # order_df = order_df.drop_duplicates(subset=drop_duplicate_subset)
    order_df = order_df[order_columns]
    order_df = order_df.rename(
        columns={
            "transaction_processed_date": "date",
            "quantity": "quantity_order",
            "from_source": "platform",
        }
    )
    # order_df = order_df[order_df.location_id.isna()]
    order_df = order_df.dropna(
        subset=[
            "variant_id",
            "product_id",
            "date",
            "quantity_order",
            "channel_id",
            "platform",
        ]
    )

    # variant meta file
    variant_columns = [
        "h_key",
        "variant_id",
        "product_id",
        "sku",
        "tags",
        "color",
        "size",
        "collections",
        "name",
        "price",
        "status",
        "is_replenishable",
        "created_date",
        "image",
        "from_source",
    ]

    variant_dtypes = {
        "h_key": "string",
        "variant_id": "string",
        "product_id": "string",
        "sku": "string",
        "tags": "string",
        "color": "string",
        "size": "string",
        "collections": "string",
        "name": "string",
        "price": "float32",
        "status": "string",
        "image": "string",
        "from_source": "string",
    }
    variant_meta = pd.read_csv(
        variant_dir,
        parse_dates=["created_date"],
        usecols=variant_columns,
        dtype=variant_dtypes,
    ).dropna(
        subset=[
            "h_key",
            "product_id",
            "variant_id",
            "name",
            "created_date",
            "from_source",
        ]
    )
    variant_meta = variant_meta.rename(columns={"from_source": "platform"})

    # product meta file
    product_columns = [
        "h_key",
        "product_id",
        "category",
        "created_date",
        "name",
        "image",
        "from_source",
    ]
    product_dtypes = {
        "h_key": "string",
        "product_id": "string",
        "category": "string",
        "image": "string",
        "from_source": "string",
    }
    product_meta = pd.read_csv(
        product_dir,
        parse_dates=["created_date"],
        usecols=product_columns,
        dtype=product_dtypes,
    ).dropna(subset=["h_key", "product_id", "name", "from_source"])
    product_meta = product_meta.rename(columns={"from_source": "platform"})

    # stock file
    stock_dtypes = {"product_id": "string", "variant_id": "string", "stock": "float32"}
    stock_df = pd.read_csv(
        stock_dir,
        parse_dates=["load_date"],
        dtype=stock_dtypes,
    )
    stock_df = stock_df.rename(columns={"from_source": "platform", "load_date": "date"})

    # ads file: daily - product level, views, click/ads_spends (total = fb + gg)
    ads_dtypes = {
        "product_id": "string",
        "views": "float32",
        "gg_click": "float32",
        "gg_ad_spends": "float32",
        "fb_click": "float32",
        "fb_ad_spends": "float32",
    }
    ads_df = pd.read_csv(ads_dir, parse_dates=["date"], dtype=ads_dtypes)

    return order_df, variant_meta, product_meta, stock_df, ads_df


# TODO: Temporary fix for product's category
def prepare_variant_data(variant_meta, product_meta):
    # Map product's category from file product.csv to variant_meta
    # TODO: If a product's category has multiple values in product.csv
    # Using dict(zip()) will zip the final value only
    product_category_dict = dict(zip(product_meta.product_id, product_meta.category))
    variant_meta["category"] = variant_meta.apply(
        lambda x: (
            product_category_dict[x.product_id]
            if x.product_id in (product_category_dict.keys())
            else pd.NA
        ),
        axis=1,
    ).astype("string")

    return variant_meta


def incremental_reindexing(
    df, meta_df, min_date, end_date, chunk_size=100, log_interval=5
):
    """
    Perform incremental reindexing of a DataFrame based on a given metadata DataFrame.

    Args:
    - df (pandas.DataFrame): The main DataFrame to be reindexed.
    - meta_df (pandas.DataFrame): The metadata DataFrame containing information about the IDs and their creation dates.
    - min_date (datetime): The minimum date to start reindexing.
    - end_date (datetime): The end date to stop reindexing.
    - chunk_size (int, optional): The number of IDs to process in each chunk. Defaults to 100.
    - log_interval (int, optional): The interval at which to log progress. Defaults to 5.

    Returns:
    - df_reindexed (pandas.DataFrame): The reindexed DataFrame.

    The function performs incremental reindexing by dividing the IDs into chunks and processing each chunk separately.
    For each chunk, the function creates a temporary DataFrame for each ID in the chunk, fills in missing dates within the date range,
        and merges it with the main DataFrame to fill in available data.

    NaN values in the 'quantity_order' column are filled with 0,
    and NaN values in the 'price' column are filled by backfilling within each ID group and
        then filling remaining NaN values with values from the corresponding IDs in the metadata DataFrame.

    The function logs progress after processing each chunk, and returns the final reindexed DataFrame.
    """
    unique_ids = meta_df.id.unique()
    total_chunks = (len(unique_ids) + chunk_size - 1) // chunk_size
    reindexed_chunks = []

    for i in range(0, len(unique_ids), chunk_size):
        # Current chunk of IDs
        segment_ids = unique_ids[i : i + chunk_size]
        segment_meta = meta_df[meta_df["id"].isin(segment_ids)].drop_duplicates("id")

        # Initialize an empty list to store the DataFrame for each ID in the chunk
        temp_dfs = []

        for item_id, group in segment_meta.groupby("id", sort=False):
            created_date = max(group["created_date"].iloc[0], min_date)
            date_range = pd.date_range(start=created_date, end=end_date)

            # Create a DataFrame for the date range for this ID
            temp_df = pd.DataFrame({"id": item_id, "date": date_range}).astype(
                {"id": "string"}
            )

            # Store this temporary DataFrame
            temp_dfs.append(temp_df)

        # Concatennate temporary DataFrames for all IDs in the chunk
        chunk_df = pd.concat(temp_dfs, ignore_index=True)

        # Merge with the main DataFrame to fill in available data
        # and fill nans in quantity_order with 0
        merged_df = pd.merge(chunk_df, df, on=["id", "date"], how="left").fillna(
            value={"quantity_order": 0}
        )

        # Fill NaNs for price by performing a two-step process:
        # First backfilling NaN values within each 'id' group, and then
        # filling remaining NaN values with values from each id's price in variant metadata
        merged_df["price"] = (
            merged_df.groupby("id")["price"]
            .bfill()
            .fillna(merged_df["id"].map(segment_meta.set_index("id")["price"]))
        )

        reindexed_chunks.append(merged_df)

        # Log progress after processing each chunk
        if (i // chunk_size + 1) % log_interval == 0 or (i + chunk_size) >= len(
            unique_ids
        ):
            logger.info(f"Processed chunk {i // chunk_size + 1} of {total_chunks}")

    df_reindexed = pd.concat(reindexed_chunks, ignore_index=True)

    return df_reindexed


def process_variants(
    order_df,
    variant_meta,
    brand_name,
    source_and_channel_list,
    min_date,
    end_date,
):
    """
    Process variant data by merging variant metadata with variant orders.

    Args:
    - order_df (pandas.DataFrame): The DataFrame containing variant orders.
    - variant_meta (pandas.DataFrame): The DataFrame containing variant metadata.
    - brand_name (str): The brand name.
    - source_and_channel_list: List of dictionary with source and selected channel_list
    - min_date (datetime): The minimum date.
    - end_date (datetime): The end date.

    Returns:
    - variant_df (pandas.DataFrame): The processed variant DataFrame.
    - variant_meta (pandas.DataFrame): The processed variant metadata DataFrame.

    Loop for process data multiple-source and selected_channel_list
    The function processes variant metadata and variant orders by merging them based on variant ID and channel ID.
    It then performs aggregation on the merged DataFrame to calculate the sum of quantity orders and the mean price.
    The function also creates a DataFrame with an 'all_channel' entry to include all channels in the analysis.
    The variant DataFrame is filtered based on the selected channel IDs and unnecessary columns are dropped.
    Finally, the function performs reindexing to daily frequency using the 'incremental_reindexing' function
    and returns the processed variant DataFrame and variant metadata DataFrame.
    """
    full_variant_df = pd.DataFrame()
    full_variant_meta = pd.DataFrame()
    for source_dict in source_and_channel_list:
        source = [*source_dict][0]
        logger.info(f"Process source {source}")
        selected_channel_id_list = source_dict[source]

        source_order_df = order_df[order_df.platform == source]
        source_variant_meta = variant_meta[variant_meta.platform == source]
        # Make sure selected_channel_id_list is not empty
        if len(selected_channel_id_list) == 0:
            raise ValueError("selected_channel_id_list cannot be empty.")

        # Get the list of available channel IDs
        available_channel_id_list = source_order_df.channel_id.unique().tolist()
        # Add the all_channel string: '0' in the avilable channel_ids_list if it's not there
        # then later we filter based on the selected channels from channel_id_list
        available_channel_id_list = available_channel_id_list + ["0"]

        # Process variant meta + variant order -> product meta + order -> Concatenate them
        # Process variant metadata
        source_variant_meta = (
            source_variant_meta.assign(key=1)
            .merge(
                pd.DataFrame(
                    {"channel_id": available_channel_id_list}, dtype="string"
                ).assign(key=1),
                on="key",
            )
            .drop("key", axis=1)
            .assign(brand_name=brand_name, platform=source, is_product=False)
            .astype(
                dtype={"brand_name": "string", "platform": "string", "is_product": bool}
            )
        )

        # Create the 'id' column for reindexing
        source_variant_meta.insert(
            loc=0,
            column="id",
            value=(
                source_variant_meta["brand_name"]
                + "_"
                + source_variant_meta["platform"]
                + "_"
                + source_variant_meta["product_id"]
                + "_"
                + source_variant_meta["variant_id"]
                + "_"
                + source_variant_meta["channel_id"]
            ),
        )
        # Process variant orders
        source_order_df = source_order_df.merge(
            source_variant_meta[["variant_id", "channel_id", "created_date"]],
            on=["variant_id", "channel_id"],
            how="right",
        )
        logger.info(
            "After mering with metadata:\n"
            f"Number of variant IDs: {len(source_order_df.variant_id.unique())}; "
            f"Sum quantity: {source_order_df.quantity_order.sum()}"
        )

        agg_dict = {"quantity_order": np.sum, "price": np.mean}
        source_order_df = (
            source_order_df.loc[source_order_df.date.ge(source_order_df.created_date)]
            .groupby(
                [
                    "variant_id",
                    "channel_id",
                    pd.Grouper(
                        key="date", freq="D", label="left", closed="left", dropna=False
                    ),
                ]
            )
            .agg(agg_dict)
        )
        logger.info(
            f"List of all available channel ids: {source_order_df.reset_index().channel_id.unique().tolist()}"
        )

        # Create df with all_channel
        all_channel_df = (
            source_order_df.groupby(["variant_id", "date"])
            .agg(agg_dict)
            .reset_index()
            .assign(channel_id="0")
            .astype({"channel_id": "string"})
            .set_index(["variant_id", "channel_id", "date"])
        )

        variant_df = (
            pd.concat([source_order_df, all_channel_df]).reset_index()
            # Merge with variant_meta to get the 'id' column for reindexing
            .merge(
                source_variant_meta[["variant_id", "channel_id", "id"]],
                on=["variant_id", "channel_id"],
                how="left",
            )
        )

        # Filter selected channel_id from channel_id_list
        # Apply filtering for both order_df and meta_df
        variant_df = variant_df[variant_df.channel_id.isin(selected_channel_id_list)]
        source_variant_meta = source_variant_meta[
            source_variant_meta.channel_id.isin(selected_channel_id_list)
        ]
        logger.info(
            "Channel list of variant_df and variant_meta after filtering from selected channel_id_list:\n"
            f"{variant_df.channel_id.unique().tolist()} and "
            f"{source_variant_meta.channel_id.unique().tolist()}"
        )

        # Drop unnecessary columns
        variant_df = variant_df.drop(columns=["variant_id", "channel_id"]).reindex(
            columns=["id", "date", "quantity_order", "price"]
        )

        # Perform reindexing to daily frequency in chunks
        logger.info("Starting reindexing into daily frequency...")
        variant_df = incremental_reindexing(
            variant_df, source_variant_meta, min_date, end_date
        )

        logger.info(f"Check NaN:\n{variant_df.isna().sum()}")
        logger.info(
            "After reindexing:\n"
            f"Number of variant IDs: {len(source_variant_meta.variant_id.unique())},\n"
            f"Sum quantity: {variant_df.quantity_order.sum()}"
        )
        full_variant_df = pd.concat([full_variant_df, variant_df])
        full_variant_meta = pd.concat([full_variant_meta, source_variant_meta])

    return full_variant_df, full_variant_meta


def process_products(variant_df, variant_meta, product_meta):
    # Process product metadata
    product_meta = (
        variant_meta.groupby(by=["product_id", "platform", "channel_id"])
        .agg(
            {
                "category": join_and_unique_str,
                "tags": join_and_unique_str,
                "collections": join_and_unique_str,
                "price": np.mean,
                "color": join_and_unique_str,
                "size": join_and_unique_str,
                "status": join_and_unique_str,
                "is_replenishable": get_product_replenishable_value,
                "created_date": np.min,
                "brand_name": "first",
            }
        )
        .reset_index()
        # Join with product_meta to get the product name, image
        .join(
            product_meta[["h_key", "product_id", "name", "image", "platform"]]
            .groupby(by=["product_id", "platform"])
            .agg(
                {
                    "h_key": "first",
                    "name": "first",
                    "image": "first",
                }
            ),
            on=["product_id", "platform"],
        )
        # FIXME: Fix when bug is resolved
        # BUG: Product meta and Variant meta don't have the same number of product IDs
        # Temporary fix: fill NaNs for name column with the name of the first product
        .set_index("product_id")
        .fillna(
            {
                "name": variant_meta.groupby(
                    by=["product_id", "platform"]
                ).name.first(),
                "image": variant_meta.groupby(
                    by=["product_id", "platform"]
                ).image.first(),
            }
        )
        .reset_index()
        # Insert other columns and convert dtypes
        .assign(variant_id=pd.NA, is_product=True)
        .astype(dtype={"is_product": bool, "variant_id": pd.NA})
    )

    # Insert 'id' column to use for mapping
    product_meta.insert(
        loc=0,
        column="id",
        value=(
            product_meta["brand_name"]
            + "_"
            + product_meta["platform"]
            + "_"
            + product_meta["product_id"]
            + "_"
            + product_meta["variant_id"].fillna("NA")
            + "_"
            + product_meta["channel_id"]
        ),
    )

    # Process product orders by summing by their variants
    product_df = (
        variant_df.merge(
            variant_meta[["id", "product_id", "platform", "channel_id"]],
            on="id",
            how="left",
        )
        .groupby(by=["product_id", "platform", "channel_id", "date"])
        .agg(
            {
                "quantity_order": np.sum,
                "price": np.mean,
            }
        )
        .reset_index()
        .set_index(["product_id", "platform", "channel_id"])
        .join(product_meta.set_index(["product_id", "platform", "channel_id"])[["id"]])
        .reset_index(drop=True)
    )
    return product_df, product_meta


def process_stock_variant(
    raw_stock_df, variant_meta, brand_name, source_and_channel_list, end_date
):
    raw_stock_df = raw_stock_df[~raw_stock_df.variant_id.isna()]

    dict_variant_product = dict(zip(variant_meta.variant_id, variant_meta.product_id))
    dict_variant_platform = dict(zip(variant_meta.variant_id, variant_meta.platform))

    variant_stock_df = pd.DataFrame()
    for platform_dict in source_and_channel_list:
        platform = [*platform_dict][0]
        channel_list = platform_dict[platform]
        print(f"platform {platform} - channel_list {channel_list}")

        variant_list = variant_meta[
            variant_meta.platform == platform
        ].variant_id.unique()
        platform_stock_df = raw_stock_df[
            (raw_stock_df.platform == platform)
            & (raw_stock_df.variant_id.isin(variant_list))
        ]

        full_channel_stock_df = pd.DataFrame()
        for channel in channel_list:
            channel_df = platform_stock_df.copy()
            channel_df["channel_id"] = channel
            full_channel_stock_df = pd.concat([full_channel_stock_df, channel_df])
        full_channel_stock_df = full_channel_stock_df.set_index(
            ["variant_id", "channel_id", "date"]
        )

        date_range = pd.date_range(start=raw_stock_df.date.min(), end=end_date)
        multi_index = pd.MultiIndex.from_product(
            [variant_list, channel_list, date_range]
        )
        multi_index = multi_index.set_names(["variant_id", "channel_id", "date"])
        full_channel_stock_df = full_channel_stock_df.reindex(multi_index).reset_index()
        print(f"After reindex: {full_channel_stock_df.channel_id.unique()}")

        variant_stock_df = pd.concat([variant_stock_df, full_channel_stock_df])

    variant_stock_df["brand_name"] = brand_name
    variant_stock_df["product_id"] = variant_stock_df.variant_id.apply(
        lambda x: dict_variant_product[x]
    )
    variant_stock_df["platform"] = variant_stock_df.variant_id.apply(
        lambda x: dict_variant_platform[x]
    )

    variant_stock_df["stock_fillna"] = variant_stock_df.stock.fillna(0)
    variant_stock_df.insert(
        loc=0,
        column="id",
        value=(
            variant_stock_df["brand_name"]
            + "_"
            + variant_stock_df["platform"]
            + "_"
            + variant_stock_df["product_id"]
            + "_"
            + variant_stock_df["variant_id"]
            + "_"
            + variant_stock_df["channel_id"]
        ),
    )
    variant_stock_df["is_product"] = False

    return variant_stock_df


def process_stock_product(variant_stock_df):
    variant_stock_df["stock_fillna"] = variant_stock_df.stock.fillna(0)
    product_stock_df = (
        variant_stock_df.groupby(
            ["brand_name", "platform", "product_id", "channel_id", "date"]
        )
        .agg(
            {
                "stock_fillna": np.sum,
            }
        )
        .reset_index()
        .rename(columns={"stock_fillna": "stock"})
    )

    product_stock_df["variant_id"] = "NA"
    product_stock_df.insert(
        loc=0,
        column="id",
        value=(
            product_stock_df["brand_name"]
            + "_"
            + product_stock_df["platform"]
            + "_"
            + product_stock_df["product_id"]
            + "_"
            + product_stock_df["variant_id"]
            + "_"
            + product_stock_df["channel_id"]
        ),
    )
    product_stock_df["is_product"] = True

    variant_stock_df = variant_stock_df.drop(columns=["stock_fillna"])

    return product_stock_df, variant_stock_df


def process_ads_product(product_df, ads_df, product_meta):
    product_df = product_df.merge(
        product_meta[["id", "product_id"]], how="left", on="id"
    )
    product_df = product_df.merge(ads_df, how="left", on=["date", "product_id"])

    fillna_columns = ["views", "gg_click", "gg_ad_spends", "fb_click", "fb_ad_spends"]
    product_df[fillna_columns] = product_df[fillna_columns].fillna(value=0)

    # Sum total ad_click / ad_spends
    product_df["ad_click"] = product_df.apply(
        lambda row: row["gg_click"] + row["fb_click"], axis=1
    )
    product_df["ad_spends"] = product_df.apply(
        lambda row: row["gg_ad_spends"] + row["fb_ad_spends"], axis=1
    )

    product_df = product_df.drop(columns=["product_id"])

    return product_df


def process_daily_series_data(
    inference_set: bool,
    brand_name: str,
    source_and_channel_list: list,
    history_dir: str,
    variant_dir: str,
    product_dir: str,
    stock_dir: str,
    ads_dir: str,
    stop_date: str = "last_sunday",
    mode: str = "full",
):
    assert mode in ["full", "incremental"], "Mode must be 'full' or 'incremental'"

    order_df, variant_meta, product_meta, raw_stock_df, ads_df = read_and_prepare_data(
        history_dir, variant_dir, product_dir, stock_dir, ads_dir
    )

    variant_meta = prepare_variant_data(variant_meta, product_meta)

    # Get the min_date and end_date from the raw order file
    min_date, _, end_date = get_start_end_dates(order_df, inference_set, stop_date)
    if mode == "incremental":
        stop_date = "yesterday"
        min_date, _, end_date = get_start_end_dates(order_df, inference_set, stop_date)
        # Only get last 7 days of orders to process
        min_date = end_date - timedelta(days=7)

    # Mapping created_date by SKU
    if brand_name == "melinda_maria":
        logger.info("Mapping created_date by SKU for Melinda...")
        counter = Counter(variant_meta.sku.dropna())
        duplicate_sku = {item for item, count in counter.items() if count > 1}

        sku_created_date_dict = dict()
        for sku in duplicate_sku:
            min_created_date = variant_meta[variant_meta.sku == sku].created_date.min()
            sku_created_date_dict[sku] = min_created_date

        variant_meta.created_date = variant_meta.apply(
            lambda x: (
                sku_created_date_dict[x.sku]
                if x.sku in duplicate_sku
                else x.created_date
            ),
            axis=1,
        )

    # Filter between min_date and end_date
    order_df = order_df.loc[order_df.date.between(min_date, end_date)].dropna(
        subset=["variant_id", "product_id", "date", "channel_id"]
    )
    logger.info(
        f"After filtering order_df: Sum quantity: {order_df.quantity_order.sum()}"
    )

    logger.info("Processing variants meta and orders...")
    variant_df, variant_meta = process_variants(
        order_df,
        variant_meta,
        brand_name,
        source_and_channel_list,
        min_date,
        end_date,
    )

    # Concat daily_history by SKU
    if brand_name == "melinda_maria":
        logger.info("Concat daily_history by SKU for Melinda...")
        variant_df = variant_df.merge(
            variant_meta[["id", "sku", "channel_id", "status"]], on="id", how="left"
        )

        duplicate_variant_df = variant_df[variant_df.sku.isin(duplicate_sku)]
        non_duplicate_variant_df = variant_df[~variant_df.sku.isin(duplicate_sku)]

        sku_df = (
            duplicate_variant_df.groupby(by=["sku", "channel_id", "date"])
            .agg(
                {
                    "quantity_order": np.sum,
                }
            )
            .reset_index()
            .set_index(["sku", "channel_id", "date"])
        )

        map_dict = dict(zip(sku_df.index, sku_df.quantity_order))
        duplicate_variant_df.quantity_order = duplicate_variant_df.apply(
            lambda x: (
                map_dict[(x["sku"], x["channel_id"], x["date"])]
                if x["status"] != "deleted"
                else x["quantity_order"]
            ),
            axis=1,
        )
        variant_df = pd.concat([duplicate_variant_df, non_duplicate_variant_df])[
            ["id", "date", "quantity_order", "price"]
        ].reset_index(drop=True)

    logger.info("Processing products meta and orders...")
    product_df, product_meta = process_products(variant_df, variant_meta, product_meta)

    # Merge ads data (view, clik, ad_spends) to product_df
    logger.info("Process ads data...")
    product_df = process_ads_product(product_df, ads_df, product_meta)

    # Concat variant and product to get final results
    df_meta = pd.concat([variant_meta, product_meta], ignore_index=True)
    df_series = pd.concat([variant_df, product_df], ignore_index=True)

    # Post process
    # Fix the values in `status` column
    logger.info("Performing post-processing...")
    df_meta.loc[:, "status"] = (
        df_meta["status"]
        .apply(lambda x: "active" if "enabled" in x or "active" in x else x)
        .astype({"status": "string"})
    )
    # Remove outliers
    logger.info("Removing outliers...")
    outlier_file_path = os.path.join(
        PROJECT_ROOT_PATH, "misc", "adjustments", "outlier.txt"
    )
    with open(outlier_file_path, encoding="utf-8") as file:
        file_content = file.read()
        outlier_dict = json.loads(file_content)

    prod_list = outlier_dict.get(brand_name)

    if prod_list:
        prod_list_expr = "|".join(prod_list)
        id_list = (
            df_meta[
                (df_meta.id.str.contains(brand_name))
                & (df_meta.id.str.contains(prod_list_expr))
            ]
            .id.unique()
            .tolist()
        )
        df_meta = df_meta[~df_meta.id.isin(id_list)]
        df_series = df_series[~df_series.id.isin(id_list)]

    logger.info(
        "Final results: \n"
        f"Total number of IDs = {len(df_series.id.unique())},\n"
        f"Number of variant IDs = {len(variant_meta.id.unique())},\n"
        f"Number of product IDs = {len(product_meta.id.unique())},\n"
        f"Sum quantity = {df_series.quantity_order.sum()},\n"
        f"Shape = {df_series.shape}"
    )

    df_series.reset_index(drop=True, inplace=True)
    df_meta.reset_index(drop=True, inplace=True)

    # Process stock data
    # Load processed stock_df from S3
    stock_S3_path = (
        os.path.join(
            "forecasting",
            "processed_data",
            brand_name,
            "inference_set",
            "stock_ts.parquet",
        )
        if inference_set
        else os.path.join(
            "forecasting", "processed_data", brand_name, "stock_ts.parquet"
        )
    )
    if s3_data_service.file_exists(stock_S3_path) is True:
        stock_dataset = pd.read_parquet(os.path.join(S3_PREFIX, stock_S3_path))

        stock_min_date = stock_dataset.date.max()
        raw_stock_df = raw_stock_df[raw_stock_df.date > stock_min_date]
    else:
        stock_dataset = None

    if raw_stock_df.date.min() < end_date:
        # Process stock_df at recent time for updating to existed stock dataset
        variant_meta = df_meta[df_meta.is_product == False]
        variant_stock_df = process_stock_variant(
            raw_stock_df, variant_meta, brand_name, source_and_channel_list, end_date
        )
        product_stock_df, variant_stock_df = process_stock_product(variant_stock_df)
        update_stock_df = pd.concat([variant_stock_df, product_stock_df])
        update_stock_df["is_stockout"] = update_stock_df.stock.apply(
            lambda x: 0 if (x > 0) or (pd.isna(x) == True) else 1
        )
        logger.info(
            f"Update_stock_df: min_date={update_stock_df.date.min()}, max_date={update_stock_df.date.max()}"
        )

        if stock_dataset is not None:
            full_stock_df = pd.concat([stock_dataset, update_stock_df])
        else:
            full_stock_df = update_stock_df

    else:
        full_stock_df = stock_dataset

    full_stock_df = full_stock_df.sort_values(by=["id", "date"])
    logger.info(
        f"Final stock_df: min_date={full_stock_df.date.min()}, max_date={full_stock_df.date.max()}"
    )

    return df_series, df_meta, full_stock_df
