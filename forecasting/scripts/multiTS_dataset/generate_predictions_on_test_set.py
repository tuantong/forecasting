import argparse
import glob
import os
import pickle
import time

import pytorch_lightning as pl
import yaml

from forecasting.configs.logging_config import logger
from forecasting.data.data_handler import DataHandler
from forecasting.util import get_formatted_duration, import_class


def main(**kwargs):
    """Generate predictions for test set"""
    total_start = time.time()

    model_dir = kwargs["model_dir"]
    prediction_save_path = kwargs["prediction_save_path"]

    # Get config file from model_dir
    config_file = glob.glob(os.path.join(model_dir, "*.yaml"))

    assert (
        len(config_file) == 1
    ), f"There should be 1 config file, found {len(config_file)}."

    config_file = config_file[0]

    with open(config_file, encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    # Load config
    model_config = configs["model"]

    model_class_str = model_config["class"]

    # Model configs
    basic_configs = model_config["configs"]
    model_name = basic_configs.get("name")
    num_loader_workers = basic_configs.get("num_loader_workers")
    output_chunk_length = model_config["parameters"]["output_chunk_length"]

    # Load data
    logger.info("Loading data")
    start = time.time()

    train_data_handler = DataHandler(configs, "train")
    test_data_handler = DataHandler(configs, "test")

    train_df = train_data_handler.load_data()
    test_df = test_data_handler.load_data()
    prediction_length = test_df.groupby("id", sort=False).cumcount().max() + 1
    logger.info(f"Prediction length: {prediction_length}")
    train_data_handler.dataset._metadata.prediction_length = prediction_length

    # Preprocess data
    train_target_df, train_past_cov_df, _ = train_data_handler.preprocess_data(
        df=train_df,
        stage="inference",
        preprocess_method="fill_zero",
        output_chunk_length=output_chunk_length,
    )

    # Filter to get only items that are in the test_df to perform inference
    logger.info(
        "Filtering train_target_df and train_past_cov_df to get only items that are in the test_df"
    )
    train_target_df = train_target_df[train_target_df["id"].isin(test_df["id"])]
    train_past_cov_df = train_past_cov_df[train_past_cov_df["id"].isin(test_df["id"])]
    logger.info(
        f"Filtered train_target_df shape: {train_target_df.shape}, {train_target_df.id.unique().shape}"
    )
    logger.info(
        f"Filtered train_past_cov_df shape: {train_past_cov_df.shape}, {train_past_cov_df.id.unique().shape}"
    )

    # Sort train_target_df and train_past_cov_df by id and date for consistency later
    logger.info("Sorting train_target_df and train_past_cov_df by `id` and `date`")
    train_target_df = train_target_df.sort_values(by=["id", "date"]).reset_index(
        drop=True
    )
    train_past_cov_df = train_past_cov_df.sort_values(by=["id", "date"]).reset_index(
        drop=True
    )

    # Convert to darts format
    logger.info("Converting data to darts format")
    train_target, train_past_cov = train_data_handler.convert_df_to_darts_format(
        train_target_df, train_past_cov_df
    )

    # Transform (scaling) data
    if train_data_handler.dataset.pipeline:
        logger.info("Transforming target (and past covariates)")
        train_target = train_data_handler.dataset.pipeline.fit_transform(train_target)
        train_past_cov = (
            train_data_handler.dataset.pipeline.fit_transform(train_past_cov)
            if train_past_cov
            else None
        )

    # Transform static covariates
    if train_data_handler.dataset._metadata.static_cov_cols:
        logger.info(f"Loading static covariates transformer from {model_dir}")
        with open(os.path.join(model_dir, "static_cov_transformer.pkl"), "rb") as f:
            static_cov_transformer = pickle.load(f)

        logger.info("Transforming static covariates")
        train_target = static_cov_transformer.transform(train_target)

    ela = time.time() - start
    logger.info(
        f"Runtime for loading and preparing data: {get_formatted_duration(ela)}"
    )
    # Load model class from configs
    logger.info(f"Initiating model class {model_class_str}")
    model_class = import_class(f"forecasting.models.{model_class_str}")

    # Load trained model
    logger.info(f"Loading trained model from {model_dir}")

    model = model_class.load_from_checkpoint(
        model_name=model_name, work_dir=model_dir, best=False
    )

    trainer_params = model.trainer_params
    trainer_params.update({"logger": False, "enable_model_summary": False})
    # Make predictions
    test_pred = model.predict(
        n=prediction_length,
        series=train_target,
        past_covariates=train_past_cov,
        num_loader_workers=num_loader_workers,
        trainer=pl.Trainer(**trainer_params),
    )
    # Inverse scale the prediction
    if train_data_handler.dataset.pipeline:
        logger.info("Inverse transform predictions")
        test_pred_inverse = train_data_handler.dataset.pipeline.inverse_transform(
            test_pred
        )

    # Save results
    logger.info(f"Saving predictions at {prediction_save_path}")
    with open(prediction_save_path, "wb") as f:
        pickle.dump(test_pred_inverse, f)

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
        default="forecasting/saved_models/NHiTS_2022_10_25-17_24_00",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--prediction_save_path",
        default="forecasting/saved_models/NHiTS_2022_10_25-17_24_00/results/NHiTS/test_set_predictions.pkl",
        required=False,
    )
    args = vars(parser.parse_args())

    main(**args)
