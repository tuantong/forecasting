from typing import Callable, Dict, List, Mapping, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from forecasting.evaluation.metrics import (
    abs_error,
    abs_target_mean,
    abs_target_sum,
    calculate_seasonal_error,
    calculate_seasonal_squared_error,
    fa,
    mae,
    mape_original,
    mase,
    mse,
    rmse,
    rmsse,
    smape,
)


def worker_function(evaluator: "Evaluator", inp: tuple):
    ts, forecast = inp
    return evaluator.get_metrics_per_ts(ts, forecast)


def aggregate_all(
    metric_per_ts: pd.DataFrame, agg_funs: Dict[str, str]
) -> Dict[str, float]:
    """
    No filtering applied

    Both `nan` and `inf` possible in aggregate metrics.
    """
    return {
        key: metric_per_ts[key].agg(agg, skipna=False) for key, agg in agg_funs.items()
    }


def aggregate_no_nan(
    metric_per_ts: pd.DataFrame, agg_funs: Dict[str, str]
) -> Dict[str, float]:
    """
    Filter all `nan` but keep `inf`.
    `nan` is only possible in the aggregate metric if all timeseries for a
    metric resulted in `nan`.
    """
    return {
        key: metric_per_ts[key].agg(agg, skipna=True) for key, agg in agg_funs.items()
    }


def aggregate_valid(
    metric_per_ts: pd.DataFrame, agg_funs: Dict[str, str]
) -> Dict[str, Union[float, np.ma.core.MaskedConstant]]:
    """
    Filter all `nan` & `inf` values from `metric_per_ts`.
    If all metrics in a column of `metric_per_ts` are `nan` or `inf` the result
    will be `np.ma.masked` for that column.
    """
    metric_per_ts = metric_per_ts.apply(np.ma.masked_invalid)
    return {
        key: metric_per_ts[key].agg(agg, skipna=True) for key, agg in agg_funs.items()
    }


class Evaluator:
    """Evaluation class to compute accuracy metrics"""

    def __init__(
        self,
        seasonality: Optional[int] = None,
        aggregation_strategy: Callable = aggregate_no_nan,
        ignore_invalid_values: bool = True,
    ) -> None:

        self.seasonality = seasonality
        self.aggregation_strategy = aggregation_strategy
        self.ignore_invalid_values = ignore_invalid_values

    def __call__(
        self,
        result_df: pd.DataFrame,
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Compute accuracy metrics by comparing actual data to the forecasts

        Args:
            result_df:
                pd.DataFrame containing 3 columns, true_ts, pred_ts, and train_ts

        Returns:

            dict
                Dictionary of aggregated metrics
            pd.DataFrame
                DataFrame containing per-time-series metrics
        """
        assert all(
            col in result_df.columns for col in ["train_ts", "test_ts", "pred_ts"]
        ), "result_df must have 3 columns 'train_ts', 'test_ts', 'pred_ts'."

        item_results = []

        for row in tqdm(
            result_df.itertuples(), total=len(result_df), desc="Running evaluation"
        ):
            item_id = row.Index
            y_train = row.train_ts
            y_true = row.test_ts
            y_pred = row.pred_ts
            item_results.append(
                self.get_metrics_per_ts(item_id, y_true, y_pred, y_train)
            )

        metrics_per_ts = pd.DataFrame.from_records(item_results, index="item_id")

        # If all entries of a target array are NaNs, the resulting metric will
        # have value "masked". Pandas does not handle masked values correctly.
        # Thus we set dtype=np.float64 to convert masked values back to NaNs
        # which are handled correctly by pandas Dataframes during
        # aggregation.
        metrics_per_ts = metrics_per_ts.astype(np.float64)
        return self.get_aggregate_metrics(metrics_per_ts)

    def get_metrics_per_ts(
        self,
        item_id: str,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        y_train: Union[np.ndarray, List],
    ) -> Mapping[str, Union[float, str, None, np.ma.core.MaskedConstant]]:

        if isinstance(y_true, List):
            y_true = np.array(y_true)
        if isinstance(y_pred, List):
            y_pred = np.array(y_pred)
        if isinstance(y_train, List):
            y_train = np.array(y_train)

        if self.ignore_invalid_values:
            y_true = np.ma.masked_invalid(y_true)
            y_pred = np.ma.masked_invalid(y_pred)
            y_train = np.ma.masked_invalid(y_train)

        seasonal_error = calculate_seasonal_error(
            past_data=y_train, freq=None, seasonality=self.seasonality
        )
        seasonal_squared_error = calculate_seasonal_squared_error(
            past_data=y_train, freq=None, seasonality=self.seasonality
        )

        metrics: Dict[str, Union[float, str, None]] = {
            "item_id": item_id,
            "abs_error": abs_error(y_true, y_pred),
            "abs_target_sum": abs_target_sum(y_true),
            "abs_target_mean": abs_target_mean(y_true),
            "MAE": mae(y_true, y_pred),
            "MSE": mse(y_true, y_pred),
            "RMSE": rmse(y_true, y_pred),
            "MASE": mase(y_true, y_pred, seasonal_error),
            "RMSSE": rmsse(y_true, y_pred, seasonal_squared_error),
            "FA": fa(y_true, y_pred),
            "MAPE": mape_original(y_true, y_pred),
            "sMAPE": smape(y_true, y_pred),
        }

        # Compute metrics for all items

        metrics["ND"] = cast(float, metrics["abs_error"]) / cast(
            float, metrics["abs_target_sum"]
        )

        return metrics

    def get_aggregate_metrics(
        self, metric_per_ts: pd.DataFrame
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        # Define how to aggregate metrics
        agg_funs = {
            "abs_error": "sum",
            "abs_target_sum": "sum",
            "abs_target_mean": "mean",
            "MAE": "mean",
            "MSE": "mean",
            "RMSE": "mean",
            "MASE": "mean",
            "RMSSE": "mean",
            "FA": "mean",
            "MAPE": "mean",
            "sMAPE": "mean",
        }

        assert (
            set(metric_per_ts.columns) >= agg_funs.keys()
        ), "Some of the requested item metrics are missing."

        # Compute the aggregation
        totals = self.aggregation_strategy(
            metric_per_ts=metric_per_ts, agg_funs=agg_funs
        )

        # Compute derived metrics
        total_rmse = np.sqrt(totals["MSE"])
        totals["NRMSE"] = total_rmse / totals["abs_target_mean"]

        totals["ND"] = totals["abs_error"] / totals["abs_target_sum"]

        return totals, metric_per_ts
