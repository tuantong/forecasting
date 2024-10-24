import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm


class SeasonalModel:
    """
    Seasonal model that considers the same months in prior years as the sales reference period.
    It is beneficial for holiday and seasonally driven items with 12+ months of history.

    Parameters
    ----------
    None
    """

    def __init__(self, freq="W-MON", default_replenishable=True):
        self.freq = freq
        self.default_replenishable = default_replenishable
        self.seasonal_factors = {}
        self.mean_vals = {}
        self.last_dates = {}

    def __str__(self):
        return "Seasonal model"

    def fit(self, df: pd.DataFrame, freq="M"):
        """
        Fits the model using the provided DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing multiple time series in long format to fit the model on.
            It must have 'date', 'id', 'category', 'quantity_order', and 'replenishable' columns.
        freq : str
            The frequency of the data ('D', 'W-MON', or 'M').
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        if self.default_replenishable and "replenishable" not in df.columns:
            df["replenishable"] = True

        grouped = df.groupby("id")
        total_series = len(grouped)

        for series_id, group in tqdm(
            grouped, total=total_series, desc="Fitting progress"
        ):
            group = group.set_index("date").sort_index()
            series = group["quantity_order"].dropna()
            self.last_dates[series_id] = series.index[-1]

            if self._has_sufficient_data(series):
                last_12_periods, previous_12_periods = (
                    self._get_last_and_previous_periods(series)
                )
                trend = (
                    last_12_periods.sum() - previous_12_periods.sum()
                ) / previous_12_periods.sum()
                self.mean_vals[series_id] = last_12_periods.mean() * (1 + trend)

                # Calculate seasonal factors for each period
                period_factors = (
                    last_12_periods.groupby(
                        last_12_periods.index.to_period(freq)
                    ).mean()
                    / last_12_periods.mean()
                )
                self.seasonal_factors[series_id] = period_factors
            else:
                # Handle cases with less than 24 periods of data at the category level
                category = group["category"].iloc[0]
                replenishable = group["replenishable"].iloc[0]
                if replenishable:
                    self._fit_category_level(df, category, series_id, series)

        return self

    def _has_sufficient_data(self, series):
        """
        Check if the series has sufficient data for the given frequency.

        Parameters
        ----------
        series : pd.Series
            The sales series for the item.

        Returns
        -------
        bool
            True if there is sufficient data, False otherwise.
        """
        if self.freq == "D":
            return len(series) >= 730  # 2 years of daily data
        elif self.freq.startswith("W"):
            return len(series) >= 104  # 2 years of weekly data
        elif self.freq == "M":
            return len(series) >= 24  # 2 years of monthly data
        else:
            raise ValueError(f"Unsupported frequency: {self.freq}")

    def _get_last_and_previous_periods(self, series):
        """
        Get the last 12 periods and the previous 12 periods based on the frequency.

        Parameters
        ----------
        series : pd.Series
            The sales series for the item.

        Returns
        -------
        tuple
            The last 12 periods and the previous 12 periods.
        """
        if self.freq == "D":
            last_12_periods = series[-365:]
            previous_12_periods = series[-730:-365]
        elif self.freq.startswith("W"):
            last_12_periods = series[-52:]
            previous_12_periods = series[-104:-52]
        elif self.freq == "M":
            last_12_periods = series[-12:]
            previous_12_periods = series[-24:-12]
        else:
            raise ValueError(f"Unsupported frequency: {self.freq}")

        return last_12_periods, previous_12_periods

    def _fit_category_level(self, df, category, series_id, series):
        """
        Fits the model at the category level when there are less than 24 periods of data for the series.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the sales data.
        category : str
            The category of the item.
        series_id : str
            The identifier of the series.
        series : pd.Series
            The sales series for the item.
        """
        category_group = df[df["category"] == category]
        last_2_months = category_group[
            category_group["date"]
            >= (category_group["date"].max() - pd.DateOffset(months=2))
        ]
        category_sales = last_2_months.groupby("id")["quantity_order"].sum()
        total_category_sales = category_sales.sum()

        # Compute the contribution of each variant
        variant_contribution = category_sales / total_category_sales

        # Calculate the seasonal forecast at the category level
        category_seasonal_forecast = self._compute_category_seasonal_forecast(
            category_group
        )

        # Distribute the category forecast to the variants
        self.mean_vals[series_id] = (
            category_seasonal_forecast * variant_contribution.get(series_id, 0)
        )

    def _compute_category_seasonal_forecast(self, category_group):
        """
        Computes the seasonal forecast at the category level.

        Parameters
        ----------
        category_group : pd.DataFrame
            The DataFrame containing the sales data for the category.

        Returns
        -------
        float
            The seasonal forecast for the category.
        """
        last_12_periods, previous_12_periods = self._get_last_and_previous_periods(
            category_group.set_index("date")["quantity_order"]
        )

        trend = (
            last_12_periods.sum() - previous_12_periods.sum()
        ) / previous_12_periods.sum()
        category_mean = last_12_periods.mean() * (1 + trend)

        return category_mean

    def predict(self, n: int, freq=None):
        """
        Predicts the next n values for each series using the fitted model.

        Parameters
        ----------
        n : int
            The number of future values to predict.
        freq : str
            The frequency of the time index for the predictions.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame in long format containing the predicted values for each series with a time index.
        """
        if not self.mean_vals:
            raise ValueError("The model must be fitted before making predictions.")

        if freq is None:
            freq = self.freq

        predictions = []
        for series_id, mean_val in tqdm(self.mean_vals.items()):
            last_date = self.last_dates[series_id]
            future_dates = pd.date_range(start=last_date, periods=n + 1, freq=freq)[1:]

            for date in future_dates:
                period = date.to_period(freq)
                seasonal_factor = self.seasonal_factors.get(
                    series_id, pd.Series([1] * 12, index=range(1, 13))
                ).get(period.month, 1)
                prediction = mean_val * seasonal_factor

                predictions.append(
                    {"date": date, "id": series_id, "predictions": prediction}
                )

        return pd.DataFrame(predictions).astype(
            {"id": "string", "predictions": np.float32}
        )

    def save(self, filepath):
        """
        Saves the fitted model to a file.

        Parameters
        ----------
        filepath : str
            The path to the file where the model should be saved.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """
        Loads the model from a file.

        Parameters
        ----------
        filepath : str
            The path to the file from which the model should be loaded.

        Returns
        -------
        RecentSalesModel
            The loaded model.
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)
