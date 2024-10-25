import argparse
import os
import subprocess
import time

import pandas as pd

# import dask.dataframe as dd
import pyarrow.parquet as pq
import yaml

from forecasting import (
    DAILY_SERIES_FILENAME,
    METADATA_FILENAME,
    MONTHLY_SERIES_FILENAME,
    PROCESSED_DATA_BUCKET,
    PROJECT_ROOT_PATH,
    S3_RAW_DATA_DIR,
    STOCK_SERIES_FILENAME,
    TEST_BEST_SELLER_FILENAME,
    TEST_FILENAME,
    TEST_LENGTH,
    TEST_NEW_ITEM_FILENAME,
    TEST_TOP10_FILENAME,
    TRAIN_FILENAME,
    WEEKLY_SERIES_FILENAME,
)
from forecasting.configs.logging_config import logger
from forecasting.data.preprocessing.preprocess_data import preprocess_for_inference
from forecasting.data.preprocessing.process_dataset import (
    process_daily_series_data,
    resample_brand_data,
)
from forecasting.data.s3_utils import aws_service
from forecasting.data.util import (
    calc_start_end_test_date,
    create_top_selling_item_sets,
    split_train_test_set,
)
from forecasting.data.utils.pandas_utils import (
    SaveDataException,
    clean_static_feat,
    save_df_to_parquet,
)
from forecasting.util import FREQ_ENUM, get_formatted_duration
from forecasting.utils.common_utils import load_dict_from_json, save_dict_to_json

# Constants
DEFAULT_YAML_CONFIG_PATH = os.path.join(
    PROJECT_ROOT_PATH,
    "forecasting",
    "data",
    "yaml_configs",
    "unified_model",
    "config_multiple_sources.yaml",
)

DOWNLOAD_SCRIPT_PATH = os.path.join(
    PROJECT_ROOT_PATH, "forecasting", "data", "s3_utils", "download_raw_data.py"
)
DOWNLOAD_DATA_DIR = os.path.join(
    PROJECT_ROOT_PATH, "data", "downloaded_multiple_datasource"
)
PROCESSED_DATA_S3_DIR = "forecasting/processed_data"
LOCAL_DIR = os.path.join(PROJECT_ROOT_PATH, "data", "processed_multiple_datasource")


# Functions
def _setup_parser():
    """Setup Argument Parser"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config_path", default=DEFAULT_YAML_CONFIG_PATH)
    parser.add_argument(
        "--inference_set",
        action="store_true",
        help="Whether to generate inference set or training set",
    )
    parser.add_argument(
        "--save_s3",
        action="store_true",
        help="Whether to save the processed data to S3",
    )
    return parser


def download_raw_data(download_script_path, raw_data_s3_dir, brand_name, data_dir):
    """
    Function to download raw data from S3.

    Args:
        download_script_path (str): Path to the script for downloading raw data.
        raw_data_s3_dir (str): S3 directory for raw data.
        brand_name (str): Brand name.
        data_dir (str): Local directory for downloaded data.
    """
    logger.info(f"Downloading raw data for brand {brand_name}")
    brand_name = "melinda" if brand_name == "melinda_maria" else brand_name
    command = [
        "python",
        download_script_path,
        "--s3_dir",
        raw_data_s3_dir,
        "--brand_name",
        brand_name,
        "--data_dir",
        data_dir,
        "--skip_date_check",
    ]
    subprocess.run(command, check=True)


def create_preprocessed_data(inference_set, config_dict, local_save_dir):
    """
    Create the preprocessed data in daily, weekly, and monthly frequencies.

    1. Daily orders will be loaded and process into daily frequency format data along with a metadata
    2. Weekly, and Monthly format data are resampled from the daily data.

    Args:
        inference_set (bool): Flag indicating whether to generate the inference set or training set.
        config_dict (dict): Configuration dictionary.
        local_save_dir (str): Local directory to save the data

    """
    order_path = os.path.join(DOWNLOAD_DATA_DIR, config_dict["brand_name"], "order.csv")
    variant_meta_path = os.path.join(
        DOWNLOAD_DATA_DIR, config_dict["brand_name"], "variant.csv"
    )
    product_meta_path = os.path.join(
        DOWNLOAD_DATA_DIR, config_dict["brand_name"], "product.csv"
    )
    stock_path = os.path.join(
        DOWNLOAD_DATA_DIR, config_dict["brand_name"], "product-ts.csv"
    )
    ads_path = os.path.join(DOWNLOAD_DATA_DIR, config_dict["brand_name"], "ads.csv")

    df_series, df_meta, df_stock = process_daily_series_data(
        inference_set,
        config_dict["brand_name"],
        config_dict["source_and_channel_list"],
        order_path,
        variant_meta_path,
        product_meta_path,
        stock_path,
        ads_path,
    )
    # Save metadata file
    save_df_to_parquet(df_meta, os.path.join(local_save_dir, METADATA_FILENAME))
    save_df_to_parquet(df_stock, os.path.join(local_save_dir, STOCK_SERIES_FILENAME))

    # Save time series file
    data_info = [
        ("D", DAILY_SERIES_FILENAME),
        ("W-Mon", WEEKLY_SERIES_FILENAME),
        ("M", MONTHLY_SERIES_FILENAME),
    ]
    # Step 2.1: Resample the daily series into weekly and monthly frequency
    for freq, ts_filename in data_info:
        local_folder = (
            os.path.join(local_save_dir, f"{FREQ_ENUM[freq.upper()]}_dataset")
            if not inference_set
            else local_save_dir
        )
        os.makedirs(local_folder, exist_ok=True)
        if freq != "D":
            closed, label = ("left", "left") if freq == "W-Mon" else ("right", "right")
            df_resampled = resample_brand_data(df_series, freq, closed, label)

        save_df_to_parquet(
            df=(df_series if freq == "D" else df_resampled),
            local_path=os.path.join(local_folder, ts_filename),
            index_cols=["id", "date"],
        )


def create_train_test_data(
    freq_ts_filepath, metadata_filepath, split_info_path, local_save_dir
):
    # Reload the freq_df and meta_df from local directory
    freq_df = pd.read_parquet(freq_ts_filepath).reset_index()
    meta_df = pd.read_parquet(metadata_filepath)

    # Reload the split info
    split_info = load_dict_from_json(split_info_path)

    # Split train, test set for dataframe
    selected_cols = [
        "id",
        "date",
        "quantity_order",
        "price",
        "views",
        "gg_click",
        "gg_ad_spends",
        "fb_click",
        "fb_ad_spends",
        "ad_click",
        "ad_spends",
    ]
    df_train, df_test, df_new_item = split_train_test_set(
        start_test_date=split_info["start_test_date"],
        end_test_date=split_info["end_test_date"],
        frequency_df=freq_df,
        meta_df=meta_df,
    )

    datasets_mapping = {
        TRAIN_FILENAME: df_train,
        TEST_FILENAME: df_test,
        TEST_NEW_ITEM_FILENAME: df_new_item,
        TEST_BEST_SELLER_FILENAME: split_info["best_seller_ids"],
        TEST_TOP10_FILENAME: split_info["top_10_ids"],
    }

    for filename, data in datasets_mapping.items():
        # Check if data is a dataframe or a list of IDs
        if isinstance(data, pd.DataFrame):
            # Directly save the dataframe if it's already one
            save_df_to_parquet(
                data[selected_cols],
                os.path.join(local_save_dir, filename),
                index_cols=["id", "date"],
            )
        else:
            # Handle ID lists by filtering df_test and saving the result
            df_subset = df_test[df_test["id"].isin(data)].reset_index(drop=True)
            save_df_to_parquet(
                df_subset[selected_cols],
                os.path.join(local_save_dir, filename),
                index_cols=["id", "date"],
            )


def create_splitting_info(test_length, local_save_dir):

    logger.info(f"Creating splitting info for test_length={test_length} months")

    logger.info("Reloading daily_df and meta_df...")
    # Load the daily series and meta data
    daily_df = pd.read_parquet(
        os.path.join(local_save_dir, "daily_dataset", DAILY_SERIES_FILENAME),
        engine="pyarrow",
    ).reset_index()
    meta_df = pd.read_parquet(
        os.path.join(local_save_dir, METADATA_FILENAME), engine="pyarrow"
    )

    max_date = pd.to_datetime(daily_df["date"]).max()
    logger.info(f"Calculating test periods with max_date={max_date}")

    start_test_date, end_test_date = calc_start_end_test_date(
        max_date=max_date, test_length=test_length, frequency="D"
    )
    logger.info(f"Start_test_date: {start_test_date}, End_test_date: {end_test_date}")
    # Get top selling item ids based on daily data for consistency
    best_seller_ids_list, top_10_ids_list = create_top_selling_item_sets(
        daily_df=daily_df, meta_df=meta_df, start_test_date=start_test_date
    )

    split_info = {
        "start_test_date": start_test_date.strftime("%Y-%m-%d"),
        "end_test_date": end_test_date.strftime("%Y-%m-%d"),
        "best_seller_ids": best_seller_ids_list,
        "top_10_ids": top_10_ids_list,
    }
    # Save the splitting info later use
    save_dict_to_json(split_info, os.path.join(local_save_dir, "split_info.json"))


def process_all_brand_data(brand_names, inference_set):
    """Create an all-brand dataframe by combining the parquet files into one

    Args:
        brand_names (list): List of brand names
        inference_set (bool): Flag indicating whether to generate the inference set or training set.
    """

    def table_generator(file_paths, schema):
        for file_path in file_paths:
            yield pq.read_table(file_path, schema=schema)

    def combine_and_save(brand_paths, local_save_path):
        """Combine same file from each brand into one dataframe as save as parquet"""
        try:
            logger.info(f"Combining files and saving dataframe to {local_save_path}")
            schema = pq.ParquetFile(brand_paths[0]).schema_arrow
            with pq.ParquetWriter(local_save_path, schema=schema) as writer:
                for table in table_generator(brand_paths, schema):
                    writer.write_table(table)
                    del table

            # full_df.to_parquet(local_save_path, engine='pyarrow', index=True)
            # save_df_to_parquet(full_df, local_save_path, index_cols=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise SaveDataException(
                f"Failed to combine and save parquet files. Reason: {repr(e)}"
            ) from e

    def combine_and_save_split_info(brand_paths, local_save_path):
        """Combine split_info from multiple directories and save as a single JSON file."""
        combined_info = {"best_seller_ids": [], "top_10_ids": []}
        dates = {}
        try:
            for path in brand_paths:
                # Attempt to load the JSON file
                info = load_dict_from_json(path)
                combined_info["best_seller_ids"].extend(info["best_seller_ids"])
                combined_info["top_10_ids"].extend(info["top_10_ids"])
                # Assuming date fields are consistent across files
                dates = {k: info[k] for k in ("start_test_date", "end_test_date")}
            combined_info.update(dates)
            # Attempt to save the combined dictionary to a JSON file
            save_dict_to_json(combined_info, local_save_path)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise SaveDataException(
                f"Failed to combine and save split_info.json files. Reason: {repr(e)}"
            ) from e

    # Look for local directory for each brand
    folder_name = "all_brand"

    local_dir = (
        os.path.join(LOCAL_DIR, folder_name)
        if not inference_set
        else os.path.join(LOCAL_DIR, folder_name, "inference_set")
    )
    os.makedirs(local_dir, exist_ok=True)

    if inference_set:
        for filename in [
            STOCK_SERIES_FILENAME,
            METADATA_FILENAME,
            DAILY_SERIES_FILENAME,
            WEEKLY_SERIES_FILENAME,
            MONTHLY_SERIES_FILENAME,
        ]:
            brand_paths = [
                os.path.join(LOCAL_DIR, brand_name, "inference_set", filename)
                for brand_name in brand_names
            ]
            local_save_path = os.path.join(local_dir, filename)
            combine_and_save(brand_paths, local_save_path)

            if filename == METADATA_FILENAME:
                logger.info("Preprocess meta_df...")
                meta_df = pd.read_parquet(
                    local_save_path, engine="pyarrow", use_nullable_dtypes=True
                )
                if "price" in meta_df.columns:
                    meta_df = meta_df.drop(columns=["price"])

                static_cov_cols = [
                    "id",
                    "brand_name",
                    "is_product",
                    "platform",
                    "category",
                ]
                clean_static_feat(meta_df, static_cov_cols)
                active_ids = meta_df[meta_df.status != "deleted"].id.tolist()
                meta_df = meta_df[meta_df.id.isin(active_ids)]

            # After process all_brand inference_set -> preprocess_infer_df and save
            if filename == WEEKLY_SERIES_FILENAME:
                logger.info("Preprocess infer df...")
                time_series_df = pd.read_parquet(
                    local_save_path, engine="pyarrow", use_nullable_dtypes=True
                )
                if not isinstance(time_series_df.index, pd.RangeIndex):
                    time_series_df.reset_index(inplace=True)

                time_series_df = time_series_df[time_series_df.id.isin(active_ids)]
                full_df = time_series_df.merge(
                    meta_df, on="id", how="left"
                ).reset_index(drop=True)

                preprocess_infer_df, _ = preprocess_for_inference(
                    df=full_df,
                    preprocess_method="fill_zero",
                    min_length_for_new_item=12,
                )
                local_save_path = os.path.join(local_dir, "preprocess_infer_df.parquet")
                preprocess_infer_df.to_parquet(local_save_path, index=False)

    else:
        # Combine the split_info.json files
        split_info_brand_paths = [
            os.path.join(LOCAL_DIR, brand_name, "split_info.json")
            for brand_name in brand_names
        ]
        combine_and_save_split_info(
            brand_paths=split_info_brand_paths,
            local_save_path=os.path.join(local_dir, "split_info.json"),
        )

        # Combine metadata files
        metadata_brand_paths = [
            os.path.join(LOCAL_DIR, brand_name, METADATA_FILENAME)
            for brand_name in brand_names
        ]
        combine_and_save(
            brand_paths=metadata_brand_paths,
            local_save_path=os.path.join(local_dir, METADATA_FILENAME),
        )

        # Combine stock files
        stock_brand_paths = [
            os.path.join(LOCAL_DIR, brand_name, STOCK_SERIES_FILENAME)
            for brand_name in brand_names
        ]
        combine_and_save(
            brand_paths=stock_brand_paths,
            local_save_path=os.path.join(local_dir, STOCK_SERIES_FILENAME),
        )

        for series_filename, freq_str in zip(
            [DAILY_SERIES_FILENAME, WEEKLY_SERIES_FILENAME, MONTHLY_SERIES_FILENAME],
            ["D", "W-Mon", "M"],
        ):
            all_filenames = [
                series_filename,
                TRAIN_FILENAME,
                TEST_FILENAME,
                TEST_NEW_ITEM_FILENAME,
                TEST_BEST_SELLER_FILENAME,
                TEST_TOP10_FILENAME,
            ]
            # Convert freq_str from short format to long format
            freq_folder = f"{FREQ_ENUM[freq_str.upper()]}_dataset"

            # Combine each files in each frequency folder data for each brand
            for filename in all_filenames:
                brand_paths = [
                    os.path.join(LOCAL_DIR, brand_name, freq_folder, filename)
                    for brand_name in brand_names
                ]
                local_dir = os.path.join(LOCAL_DIR, folder_name, freq_folder)
                os.makedirs(local_dir, exist_ok=True)
                local_save_path = os.path.join(local_dir, filename)
                combine_and_save(brand_paths, local_save_path)


def upload_data_to_s3(brand_folders, aws_storage, inference_set):
    """
    Upload processed data to S3.

    Args:
        brand_folders (list): List of local directories for each brand.
        aws_storage (AWSStorage): AWSStorage instance for interacting with S3.
        inference_set (bool): Flag indicating whether to upload the inference set or training set.

    Returns:
        None
    """
    # Iterate over each brand folder
    for brand_folder_name in sorted(brand_folders):
        # Set the local and S3 folder paths
        local_folder = os.path.join(LOCAL_DIR, brand_folder_name)
        s3_folder = os.path.join(PROCESSED_DATA_S3_DIR, brand_folder_name)

        # Upload only the inference_set folder if inference_set is True
        if inference_set:
            local_folder = os.path.join(local_folder, "inference_set")
            s3_folder = os.path.join(s3_folder, "inference_set")

        # Upload all folders and files except the `inference_set` folder
        for f_name in os.listdir(local_folder):
            if f_name != "inference_set":
                f_path = os.path.join(local_folder, f_name)
                # Check if the path is a file or a folder
                if os.path.isfile(f_path):
                    aws_storage.upload(f_path, os.path.join(s3_folder, f_name))
                else:
                    aws_storage.upload_folder(f_path, os.path.join(s3_folder, f_name))


def process_data_flow(data_config, inference_set, save_s3):
    """
    Main processing flow for forecasting data.

    Args:
        data_config (list): List of configuration dictionaries for different brands.
        INFERENCE_SET (bool): Flag indicating whether to generate the inference set or training set.
        storage_options (dict): Storage options for S3.
        local_dir (str): Local directory for saving data.

    Returns:
        None
    """
    logger.info("Starting processing flow")
    # Start process flow
    start = time.time()
    # Process each brand
    for config_dict in data_config:
        brand_name = config_dict["brand_name"]
        logger.info(f"Processing brand {brand_name.upper()}")

        # Step 1: Download raw data
        download_raw_data(
            DOWNLOAD_SCRIPT_PATH, S3_RAW_DATA_DIR, brand_name, DOWNLOAD_DATA_DIR
        )

        # Step 2: Create a daily, weekly, and monthly series dataframe
        brand_processed_local_dir = (
            os.path.join(LOCAL_DIR, brand_name)
            if not inference_set
            else os.path.join(LOCAL_DIR, brand_name, "inference_set")
        )
        os.makedirs(brand_processed_local_dir, exist_ok=True)
        create_preprocessed_data(inference_set, config_dict, brand_processed_local_dir)

        # Step 3 + 4: Only for generating training data
        if not inference_set:
            # Step 3: Create the splitting information (best seller sets, start and end test periods)
            create_splitting_info(TEST_LENGTH, brand_processed_local_dir)

            # Step 4: Create train_test set
            for freq, ts_filename in zip(
                ["D", "W-Mon", "M"],
                [
                    DAILY_SERIES_FILENAME,
                    WEEKLY_SERIES_FILENAME,
                    MONTHLY_SERIES_FILENAME,
                ],
            ):
                freq_folder = f"{FREQ_ENUM[freq.upper()]}_dataset"

                create_train_test_data(
                    freq_ts_filepath=os.path.join(
                        brand_processed_local_dir, freq_folder, ts_filename
                    ),
                    metadata_filepath=os.path.join(
                        brand_processed_local_dir, METADATA_FILENAME
                    ),
                    split_info_path=os.path.join(
                        brand_processed_local_dir, "split_info.json"
                    ),
                    local_save_dir=os.path.join(brand_processed_local_dir, freq_folder),
                )

    brand_names = [config["brand_name"] for config in data_config]
    logger.info("Processing all brand data set")
    process_all_brand_data(brand_names, inference_set)

    if save_s3:
        logger.info("Uploading data to S3 storage")
        client = aws_service.create_client()
        aws_storage = aws_service.AWSStorage(client, PROCESSED_DATA_BUCKET)
        brand_folders = brand_names + ["all_brand"]

        upload_data_to_s3(brand_folders, aws_storage, inference_set)

    total_ela = time.time() - start
    logger.info(f"Total run time: {get_formatted_duration(total_ela)}")


def main():
    """
    The flow is as follows:

        - Download raw data from S3
        - Process daily dataframe
        - Save to local
        - Upload to S3 (Optional)
    """
    parser = _setup_parser()
    args = parser.parse_args()

    kwargs = vars(args)
    data_config_path = kwargs["data_config_path"]
    inference_set = kwargs["inference_set"]
    save_s3 = kwargs["save_s3"]

    if save_s3:
        logger.info(
            "Argument --save_s3 is passed, data will be uploaded to s3 storage after processing"
        )

    with open(data_config_path, encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    data_config = configs["data"]
    os.makedirs(LOCAL_DIR, exist_ok=True)
    process_data_flow(data_config, inference_set, save_s3)


if __name__ == "__main__":
    main()
