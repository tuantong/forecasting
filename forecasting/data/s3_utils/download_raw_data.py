import argparse
import glob
import os
import shutil
import tempfile
from typing import List

import pandas as pd
from tqdm import tqdm

from forecasting.configs import logger
from forecasting.data.s3_utils.aws_service import s3_data_service


def process_brand_folders(
    s3_dir: str,
    brand_folders: List[str],
    downloaded_dir: str,
) -> None:
    """
    Process each brand folder to download files from AWS S3.

    Parameters:
    - s3_dir (str): The root folder on S3 to download from.
    - brand_folders (List[str]): List of brand folders to download.
    - downloaded_dir (str): The local directory to save downloaded files.

    This function will create a separate folder for each brand in the downloaded_dir
    and save the data in separate files within each brand folder. The files will
    be named after the S3 folder they were downloaded from.
    """
    for brand_name in tqdm(brand_folders):
        s3_folder = os.path.join(s3_dir, f"{brand_name}")

        brand_name = "melinda_maria" if brand_name == "melinda" else brand_name
        brand_data_save_dir = os.path.join(downloaded_dir, brand_name)

        if os.path.exists(brand_data_save_dir):
            logger.info(f"Deleting existing data at {brand_data_save_dir}")
            shutil.rmtree(brand_data_save_dir)

        logger.info(f"Creating directory {brand_data_save_dir}")
        os.makedirs(brand_data_save_dir, exist_ok=True)

        # Create temp directory to process the chunks
        with tempfile.TemporaryDirectory() as tmp_dir:
            brand_tmp_dir = os.path.join(tmp_dir, brand_name)
            # Download the data to the temporary directory
            logger.info(f"Downloading folder {s3_folder} to {brand_tmp_dir}")
            s3_data_service.download_folder(s3_folder, brand_tmp_dir)

            logger.info(f"Processing {brand_tmp_dir} to {brand_data_save_dir}")
            # Process the chunks into a single file
            brand_data_tmp_folders = os.listdir(brand_tmp_dir)
            for f_name in brand_data_tmp_folders:
                source_path = os.path.join(brand_tmp_dir, f_name)
                # Perform merging chunks if f_name is a folder, otherwise copy the file
                if os.path.isdir(source_path):
                    if f_name == "daily":
                        logger.info(f"Copying entire 'daily' folder {source_path}")
                        shutil.copytree(
                            source_path, os.path.join(brand_data_save_dir, f_name)
                        )
                    else:
                        logger.info(f"Processing folder {f_name}")
                        dtypes = {"variant_id": "string", "product_id": "string"}
                        if f_name == "order":
                            dtypes["channel_id"] = "string"

                        df = pd.concat(
                            (
                                pd.read_csv(f, dtype=dtypes)
                                for f in glob.glob(
                                    os.path.join(brand_tmp_dir, f_name, "*.csv")
                                )
                            ),
                            ignore_index=True,
                        )
                        logger.info(
                            f"Saving dataframe to {os.path.join(brand_data_save_dir, f'{f_name}.csv')}"
                        )
                        df.to_csv(
                            os.path.join(brand_data_save_dir, f"{f_name}.csv"),
                            index=False,
                        )
                else:
                    logger.info(f"Copying file {f_name}")
                    shutil.copyfile(
                        os.path.join(brand_tmp_dir, f_name),
                        os.path.join(brand_data_save_dir, f_name),
                    )


def main(args):
    """
    Main function to download files from AWS S3 based on specified parameters.

    Parameters:
    - args (argparse.Namespace): Command-line arguments parsed by argparse.

    Returns:
    - None
    """
    s3_dir = args.s3_dir if args.s3_dir is not None else os.getenv("S3_RAW_DATA_DIR")
    brand_name = args.brand_name
    data_dir = args.data_dir
    # skip_date_check = args.skip_date_check

    if brand_name == "all":
        # Look up the directory of brand folder
        api_response = s3_data_service.list_objects(s3_dir)
        common_prefixes = api_response["CommonPrefixes"]
        assert len(common_prefixes) != 0, f"Found 0 folder(s) in {s3_dir}"
        brand_folders = [
            os.path.basename(os.path.dirname(item["Prefix"]))
            for item in common_prefixes
        ]
        # brand_folders = ['mizmooz', 'chanluu', 'hammitt', 'melinda', 'raquelallegra', 'as98']
    else:
        brand_folders = [brand_name]

    # Print out the arguments
    logger.info(
        f"Downloading data from {s3_dir} for {len(brand_folders)} brands: {brand_folders}"
    )
    logger.info(f"Downloaded data will be saved to {data_dir}")

    # today_date = datetime.now(tz=tzutc()).date()
    process_brand_folders(s3_dir, brand_folders, data_dir)
    logger.info("Download process completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s3_dir",
        default=None,
        help="S3 directory containing brands folder (must have / at the end)",
    )
    parser.add_argument("--brand_name", default="all", help="Brand name")
    parser.add_argument(
        "--data_dir", default="data/downloaded", help="Local dataset dir"
    )
    parser.add_argument(
        "--skip_date_check",
        action="store_true",
        help="Whether to ignore the last modified date check of remote files on s3",
    )

    arguments = parser.parse_args()

    main(arguments)
