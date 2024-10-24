import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from forecasting.configs.logging_config import logger
from forecasting.evaluation._evaluator import Evaluator, aggregate_valid
from forecasting.evaluation.aggregation_utils import (
    aggregate_result_df_to_monthly,
    build_result_df_from_pd,
)
from forecasting.metadata.shared import SOURCE_DATA_DIRNAME
from forecasting.util import get_formatted_duration

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _setup_parser():
    """Setup Argument Parser"""
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("-f", "--data_config_path", required=True)
    parser.add_argument("-droot", "--data_dir", default=SOURCE_DATA_DIRNAME)
    parser.add_argument("-p", "--results_dir", required=True)

    parser.add_argument("--help", "-h", action="help")

    return parser


def main():
    """Main function"""
    parser = _setup_parser()
    args = parser.parse_args()

    data_config_path = args.data_config_path
    data_dir = args.data_dir
    results_dir = args.results_dir

    logger.info(f"Directory for evaluation is {os.path.abspath(results_dir)}")
    logger.info(f"Reading data config from {os.path.abspath(data_config_path)}")

    with open(data_config_path, encoding="utf-8") as f:
        configs = yaml.safe_load(f)

    # Load data config
    data_config = configs["data"]
    data_configs = data_config["configs"]
    freq = data_configs["freq"]
    id_col = data_configs["id_col"]

    # Load data

    if freq.startswith("W"):
        full_freq_str = "weekly"
    elif freq.startswith("M"):
        full_freq_str = "monthly"
    else:
        raise ValueError(f"Expected freq to start with either 'W' or 'M', found {freq}")

    raw_data_dir = (
        Path(data_dir) / data_configs["version"] / (full_freq_str + "_dataset")
    )

    logger.info(f"Loading data from {raw_data_dir}")
    metadata_df = pd.read_parquet(
        data_dir / data_configs["version"] / "meta_data.parquet"
    )
    # Load data
    freq_series = pd.read_parquet(
        raw_data_dir / f"{full_freq_str}_series.parquet",
        engine="pyarrow",
        use_nullable_dtypes=True,
    )
    train_series = pd.read_parquet(
        raw_data_dir / "train.parquet", engine="pyarrow", use_nullable_dtypes=True
    )
    test_series = pd.read_parquet(
        raw_data_dir / "test.parquet", engine="pyarrow", use_nullable_dtypes=True
    )

    freq_df = freq_series.merge(metadata_df, on=id_col, how="left")
    train_df = train_series.merge(metadata_df, on=id_col, how="left")
    test_df = test_series.merge(metadata_df, on=id_col, how="left")
    seasonal_df = pd.read_parquet(
        raw_data_dir / "test_seasonality.parquet",
        engine="pyarrow",
        use_nullable_dtypes=True,
    )
    best_seller_df = pd.read_parquet(
        raw_data_dir / "test_best_seller.parquet",
        engine="pyarrow",
        use_nullable_dtypes=True,
    )

    # Remove items in train_df but not in test_df (archived, draft, disable items)
    logger.info("Removing archived, draft, disabled items")
    train_df = train_df[train_df.id.isin(test_df.id.unique())]
    logger.info(
        f"freq_df shape: {freq_df.shape}, freq_df unique ids: {freq_df.id.unique().shape}"
    )

    full_ids, best_seller_ids, seasonal_ids = (
        freq_df.id.unique(),
        best_seller_df.id.unique(),
        seasonal_df.id.unique(),
    )

    # Remove the hidden item
    hidden_product_id = "6682416578606"
    train_df = train_df[train_df.product_id != hidden_product_id]

    logger.info(f"Numboer of all ids: {len(full_ids)}")
    logger.info(f"Number of ids to evaluate: {len(train_df.id.unique())}")

    # Get model result folders
    model_name = configs["config_name"]
    metric_list = ["MAE", "RMSE", "MAPE"]
    evaluate_ids_list = ["full", "best_seller", "seasonal"]
    frequency = "M"  # aggregate result to monthly before evaluate

    start = time.time()
    final_evaluate_df_dict = dict.fromkeys(evaluate_ids_list, pd.DataFrame())
    final_item_df = pd.DataFrame()
    forecast_result_df = pd.DataFrame()

    logger.info(f"Evaluating for model {model_name} at frequency: {frequency}")

    # Convert prediction_df to wide format (list of time series)
    model_result_path = f"{results_dir}/test_submission.parquet"
    pred_df = pd.read_parquet(model_result_path).reset_index()
    pred_pivot = build_result_df_from_pd(train_df, pred_df)
    # Add metadata
    pred_pivot = pred_pivot.merge(metadata_df, on=id_col, how="left")

    if frequency == "M":
        logger.info("Resampling to monthly...")
        monthly_pred_pivot = aggregate_result_df_to_monthly(pred_pivot)
        result_df = monthly_pred_pivot.reset_index()
    if frequency == "W":
        result_df = pred_pivot.reset_index()

    # save all forecast results for plotting
    fc_result_df = result_df
    if forecast_result_df.shape[1] == 0:  # Init forecast_result_df
        fc_result_df = fc_result_df.rename(columns={"pred_ts": model_name})
        forecast_result_df = fc_result_df
    else:  # Add pred_ts field and rename
        forecast_result_df[model_name] = list(fc_result_df.pred_ts)

    for ids_list in evaluate_ids_list:
        if ids_list == "full":
            agg_result_df = result_df
        elif ids_list == "best_seller":
            agg_result_df = result_df[result_df.id.isin(best_seller_ids)]
        elif ids_list == "seasonal":
            agg_result_df = result_df[result_df.id.isin(seasonal_ids)]

        # assign list of brands
        brand_list = agg_result_df.brand_name.unique().tolist()
        if len(brand_list) > 1:
            brand_list.append("all")

        for brand in brand_list:
            brand_df = (
                agg_result_df
                if brand == "all"
                else agg_result_df[agg_result_df.brand_name == brand]
            )
            # assign list of levels
            if len(brand_df.is_product.unique()) < 2:
                if brand_df.is_product.unique()[0] is False:  # all variant IDs
                    level_list = ["variant"]
                if brand_df.is_product.unique()[0] is True:  # all product IDs
                    level_list = ["product"]
            else:
                level_list = ["variant", "product", "all"]

            for level in level_list:
                if level == "product":
                    df = brand_df[brand_df.is_product is True]
                elif level == "variant":
                    df = brand_df[brand_df.is_product is False]
                else:
                    df = brand_df

                df = df.set_index("id")
                evaluator = Evaluator(
                    seasonality=1,
                    aggregation_strategy=aggregate_valid,
                    ignore_invalid_values=True,
                )
                agg_metrics, item_metrics = evaluator(df)

                # Create evaluate_df for each model
                evaluate_df = pd.DataFrame(
                    {"model": model_name, "brand_name": brand, "level": level},
                    index=[0],
                ).reset_index(drop=True)
                evaluate_df[metric_list] = [
                    np.round(agg_metrics[metric], 3) for metric in metric_list
                ]
                final_evaluate_df_dict[ids_list] = pd.concat(
                    [final_evaluate_df_dict[ids_list], evaluate_df]
                ).reset_index(drop=True)

                # Save metrics of all items in brand
                if (
                    brand != "all"
                    and (level == "all" or len(level_list) == 1)
                    and (ids_list == "full")
                ):
                    item_df = item_metrics[metric_list].reset_index()
                    item_df["model"] = model_name
                    item_df["brand_name"] = brand
                    final_item_df = pd.concat([final_item_df, item_df])

    final_item_df = final_item_df.reset_index(drop=True)
    forecast_result_df = forecast_result_df.reset_index(drop=True)

    # Save forecast predictions results
    forecast_res_save_path = os.path.join(results_dir, "forecast_results_df.xlsx")
    logger.info(f"Saving forecast predictions results to {forecast_res_save_path}")
    forecast_result_df.to_excel(forecast_res_save_path, index=False)

    # Save per item result metrics
    final_item_df = final_item_df.set_index(
        ["model", "brand_name", "item_id"]
    ).sort_index()
    item_df_save_path = os.path.join(results_dir, "final_item_metrics_df.xlsx")
    logger.info(f"Saving per item evaluation results to {item_df_save_path}")
    final_item_df.to_excel(item_df_save_path)

    # Aggregated result metrics for all items
    for ids_list in evaluate_ids_list:
        evaluate_df_save_path = os.path.join(
            results_dir, f"{ids_list}_evaluate_df.xlsx"
        )

        final_evaluate_df = (
            final_evaluate_df_dict[ids_list]
            .set_index(["brand_name", "level", "model"])
            .sort_index()
        )
        logger.info(
            f"Saving evaluation results for {ids_list} test set to {evaluate_df_save_path}"
        )
        final_evaluate_df.to_excel(evaluate_df_save_path)

    ela = time.time() - start
    logger.info(f"Time for running evaluation: {get_formatted_duration(ela)}")


if __name__ == "__main__":
    main()
