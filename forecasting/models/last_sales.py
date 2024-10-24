import numpy as np
import pandas as pd
from tqdm import tqdm

from forecasting.models.base_model import BaseModel


class LastSalesModel(BaseModel):
    """
    Baseline model that uses average values of the last K periods to make predictions for multiple series.

    Parameters
    ----------
    global_window : int
        The default number of last time steps of the training set to take the average.
    item_configs : dict (Optional)
        A dictionary that contains the configurations for each item.
        The keys are the item IDs and the values are dictionaries containing the configuration values.

        Example: {"item_1": {"window": 10}, "item_2": {"window": 20}}
    """

    def __init__(self, global_window: int, item_configs: dict = {}):
        super().__init__()
        self.global_window = global_window
        self.item_configs = item_configs

    def fit(self, df: pd.DataFrame, **kwargs):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if not {"date", "id", "quantity_order"}.issubset(df.columns):
            raise ValueError(
                "DataFrame must contain 'date', 'id', and 'quantity_order' columns."
            )

        verbose = kwargs.get("verbose", True)

        grouped = df.groupby("id")
        total_series = len(grouped)

        for series_id, group in tqdm(
            grouped, total=total_series, desc="Fitting progress", disable=not verbose
        ):
            self._fit_series(group, series_id)

        return self

    def _fit_series(self, group, series_id):
        group = group.set_index(
            "date"
        ).sort_index()  # Ensure the group is sorted by date
        series = group["quantity_order"].dropna()
        self.last_dates[series_id] = series.index[-1]

        window = self.get_item_config(series_id, "window", self.global_window)
        self.mean_vals[series_id] = series[-window:].mean()

    def refit_item(self, df: pd.DataFrame, item_id: str, **kwargs):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        verbose = kwargs.get("verbose", True)

        df["date"] = pd.to_datetime(df["date"])
        group = df[df["id"] == item_id]
        if group.empty:
            raise ValueError(f"No data found for item_id: {item_id}")

        self._fit_series(group, item_id)
        if verbose:
            print(f"Refitted model for item {item_id}")
        return self

    def predict(self, n: int, freq="D", **kwargs):
        if not self.mean_vals:
            raise ValueError("The model must be fitted before making predictions.")

        verbose = kwargs.get("verbose", True)
        predictions = []
        for series_id, mean_val in tqdm(
            self.mean_vals.items(),
            total=len(self.mean_vals),
            desc="Predicting progress",
            disable=not verbose,
        ):
            predictions.extend(self._predict_for_series(series_id, mean_val, n, freq))

        return pd.DataFrame(predictions).astype(
            {"id": "string", "predictions": np.float32}
        )

    def _predict_for_series(self, series_id, mean_val, n, freq):
        last_date = self.last_dates[series_id]
        future_dates = pd.date_range(start=last_date, periods=n + 1, freq=freq)[1:]

        predictions = [
            {"date": date, "id": series_id, "predictions": mean_val}
            for date in future_dates
        ]

        return predictions

    def predict_for_item(self, item_id: str, n: int, freq="D"):
        if item_id not in self.mean_vals:
            raise ValueError(
                f"The model must be fitted for item {item_id} before making predictions."
            )

        mean_val = self.mean_vals[item_id]
        predictions = self._predict_for_series(item_id, mean_val, n, freq)
        return pd.DataFrame(predictions).astype(
            {"id": "string", "predictions": np.float32}
        )
