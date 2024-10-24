import argparse
import glob
import os
import pickle
import time

import yaml
from tqdm import tqdm

from forecasting.configs.logging_config import logger
from forecasting.data.base_dataset import DatasetMetadata
from forecasting.data.multiTS_unified_dataset import PMUnifiedDatasetLoader
from forecasting.util import get_formatted_duration


def main(**kwargs):
    """Generate predictions for test set"""
    total_start = time.time()

    model_dir = kwargs["model_dir"]
    prediction_save_path = kwargs["prediction_save_path"]

    # Get config file from model_dir
    config_file = glob.glob(os.path.join(model_dir, "*.yaml"))

    assert len(config_file) == 1, "Found more than 1 config file"

    config_file = config_file[0]

    with open(config_file, encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    # Load config
    data_config = configs["data"]
    model_config = configs["model"]

    model_class_name = model_config["class"]
    # prediction_length = data_config["prediction_length"]

    # Load data
    logger.info("Loading data")

    unified_dataset_metadata = DatasetMetadata(**data_config)

    unified_data = PMUnifiedDatasetLoader(unified_dataset_metadata)

    total_train_set, test_set = unified_data.load_darts(required_test=True)

    # Extract target and covariate columns
    # logger.info("Extracting target and covariate columns")
    # target_col = unified_data._metadata.target_col
    # total_train_target = [ts[target_col] for ts in tqdm(total_train_set)]

    # Load model
    logger.info(f"Loading trained model from {model_dir}")
    model_save_path = os.path.join(model_dir, model_class_name, "_model.pkl")
    with open(model_save_path, "rb") as f:
        all_models = pickle.load(f)

    # Make predictions
    logger.info("Predicting")
    test_prediction_length = max([len(ts) for ts in test_set])
    test_preds = []
    for model in tqdm(all_models):
        test_pred = model.predict(n=test_prediction_length)
        test_preds.append(test_pred)

    # Save results
    logger.info(f"Saving predictions at {prediction_save_path}")
    with open(prediction_save_path, "wb") as f:
        pickle.dump(test_preds, f)

    # Record time
    total_ela = time.time() - total_start
    logger.info(f"Total run time: {get_formatted_duration(total_ela)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to generate predictions for test set"
    )

    parser.add_argument(
        "-wd",
        "--model_dir",
        default="forecasting/saved_models/Baseline_2022_11_01-18_43_00",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--prediction_save_path",
        default="forecasting/saved_models/Baseline_2022_11_01-18_43_00/results/Baseline/test_set_predictions.pkl",
        required=False,
    )
    args = vars(parser.parse_args())

    main(**args)
