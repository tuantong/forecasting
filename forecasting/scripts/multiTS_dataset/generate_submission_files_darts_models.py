import argparse
import glob
import os
import pickle
import time

import pandas as pd
import yaml
from tqdm import tqdm

from forecasting.configs.logging_config import logger
from forecasting.data.data_handler import DataHandler
from forecasting.util import get_formatted_duration


def main(**kwargs):
    """
    Clean up predictions:
     + Clip values to 0 and rounding
     + Drop redudant values to match test length for every time series

    Args:
        model_result_path: The raw test dataframe
        unified_dataloader: The dataloader for unified_dataset

    """
    total_start = time.time()
    work_dir = kwargs["work_dir"]
    model_predictions_path = kwargs["model_predictions_path"]
    submission_save_path = kwargs["submission_save_path"]

    # Get config file from work_dir
    config_file = glob.glob(os.path.join(work_dir, "*.yaml"))

    assert (
        len(config_file) == 1
    ), f"There should be 1 config file, found {len(config_file)}."

    config_file = config_file[0]

    with open(config_file, encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    # Load config
    data_config = configs["data"]
    # Data configs
    data_configs = data_config["configs"]

    id_col = data_configs["id_col"]
    time_col = data_configs["time_col"]
    target_col = data_configs["target_col"]

    # Load data
    logger.info("Loading data")
    test_data_handler = DataHandler(configs, "test")

    test_df = test_data_handler.load_data()

    # Read predictions
    logger.info("Reading predictions file")
    with open(model_predictions_path, "rb") as f:
        pred_target = pickle.load(f)

    # Make sure the number of predictions are equal to the number of test items
    assert len(pred_target) == len(
        test_df["id"].unique().tolist()
    ), f"Number of predictions are not equal to number of test items: {len(pred_target)} != {len(test_df['id'].unique().tolist())}"

    ids_list = []
    date_list = []
    pred_list = []

    # Generate submission
    # Must sort the test_df by 'id' because we sorted the items before running `model.predict`
    # so the order of the predictions are based on the sorted ids from the filtered_train_df
    logger.info("Generating submission")
    for (item_id, item_test_df), pred_ts in tqdm(
        zip(test_df.groupby("id", sort=True), pred_target), total=len(pred_target)
    ):
        len_test_ts = len(item_test_df)

        ts_date = pred_ts.time_index

        # clip predictions to 0.0 and round values
        ts_values = pred_ts.univariate_values().clip(0.0).round()

        # drop redundant values in pred_set to match test_set length of each
        ts_date = ts_date[:len_test_ts]
        ts_values = ts_values[:len_test_ts]

        ids_list.extend([item_id] * len_test_ts)
        date_list.extend(ts_date)
        pred_list.extend(ts_values)

    submission_df = pd.DataFrame(
        {id_col: ids_list, time_col: date_list, "predictions": pred_list}
    )

    submission_df = (
        test_df[[id_col, time_col, target_col]]
        .set_index([id_col, time_col])
        .join(submission_df.set_index([id_col, time_col]))
        .reset_index()
    )

    logger.info("Saving submission result")
    submission_df.to_parquet(submission_save_path)

    # Record time
    total_ela = time.time() - total_start
    logger.info(f"Total run time: {get_formatted_duration(total_ela)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate submission files for test set from model predictions"
    )

    parser.add_argument("-wd", "--work_dir", required=True)
    parser.add_argument("-p", "--model_predictions_path", required=True)
    parser.add_argument("-s", "--submission_save_path", required=True)

    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)
    main(**kwargs)
