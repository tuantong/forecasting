import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm


class RecentSalesAndTrendModel:
    """
    RecentSalesAndTrend model that uses average sales of the last `window` periods and
    applies seasonal increases for Black Friday and Christmas if enabled.

    Parameters
    ----------
    window: int
        The number of last time steps of the training set to take average.
    enable_black_friday: bool
        Whether to enable the detection of Black Friday increase.
    enable_christmas: bool
        Whether to enable the detection of Christmas increase.
    trend_months: int
        The number of months to consider for trend calculation.
    freq: str
        The frequency of the data (e.g., 'D' for daily, 'W' for weekly).
    """

    def __init__(
        self,
        window,
        enable_black_friday=True,
        enable_christmas=True,
        trend_months=3,
        freq="D",
        trend_cap=4,
    ):
        self.window = window
        self.enable_black_friday = enable_black_friday
        self.enable_christmas = enable_christmas
        self.trend_months = trend_months
        self.freq = freq
        self.trend_cap = trend_cap
        self.mean_vals = {}
        self.last_dates = {}
        self.black_friday_increase = {}
        self.christmas_increase = {}
        self.trends = {}

    def __str__(self):
        return "RecentSalesAndTrend predictor model"

    def fit(self, df: pd.DataFrame):
        """
        Fits the model using the provided DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing multiple time series in long format to fit the model on.
            It must have 'date', 'id', and 'quantity_order' columns.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        self.mean_vals = {}
        self.last_dates = {}
        self.black_friday_increase = {}
        self.christmas_increase = {}
        self.trends = {}

        df["date"] = pd.to_datetime(df["date"])
        grouped = df.groupby("id")
        total_series = len(grouped)
        for series_id, group in tqdm(
            grouped, total=total_series, desc="Fitting progress"
        ):
            group = group.set_index(
                "date"
            ).sort_index()  # Ensure the group is sorted by date
            series = group["quantity_order"].dropna()
            self.last_dates[series_id] = series.index[-1]

            # Determine the last complete year in the dataset
            last_complete_year = series.index.max().year - 1

            if self.enable_black_friday:
                # Calculate the total sales for November of the last complete year
                november_sales = group[
                    (group.index.year == last_complete_year) & (group.index.month == 11)
                ]["quantity_order"].sum()
                # Calculate the total sales for the three months before November of the last complete year
                total_sales_before_nov = group[
                    (group.index.year == last_complete_year)
                    & (group.index.month >= 8)
                    & (group.index.month <= 10)
                ]["quantity_order"].sum()
                # Calculate the average sales per month for the three months before November
                avg_sales_before_nov = total_sales_before_nov / 3
                # Calculate the percentage increase
                self.black_friday_increase[series_id] = (
                    (november_sales / avg_sales_before_nov) - 1
                    if avg_sales_before_nov > 0
                    else 0
                )

            if self.enable_christmas:
                # Calculate the total sales for December of the last complete year
                december_sales = group[
                    (group.index.year == last_complete_year) & (group.index.month == 12)
                ]["quantity_order"].sum()
                # Calculate the total sales for the three months before December of the last complete year
                total_sales_before_dec = group[
                    (group.index.year == last_complete_year)
                    & (group.index.month >= 8)
                    & (group.index.month <= 10)
                ]["quantity_order"].sum()
                # Calculate the average sales per month for the three months before December
                avg_sales_before_dec = total_sales_before_dec / 3
                # Calculate the percentage increase
                self.christmas_increase[series_id] = (
                    (december_sales / avg_sales_before_dec) - 1
                    if avg_sales_before_dec > 0
                    else 0
                )

            # Calculate the trend using the last `trend_months` plus the current month
            if self.freq.startswith("D"):
                trend_periods = self.trend_months * 30  # 30 days per month
            elif self.freq.startswith("W"):
                trend_periods = self.trend_months * 4  # 4 weeks per month
            elif self.freq.startswith("M"):
                trend_periods = self.trend_months * 1  # 1 month per month
            else:
                raise ValueError(f"Unsupported frequency: {self.freq}")

            recent_periods_sales = series[-trend_periods:]

            if len(recent_periods_sales) >= self.trend_months + 1:
                x = np.arange(1, len(recent_periods_sales) + 1)
                y = recent_periods_sales.values
                slope, _ = np.polyfit(x, y, 1)
                if self.freq.startswith("D"):
                    daily_trend = slope  # No division needed for daily data
                elif self.freq.startswith("W"):
                    daily_trend = slope / 7  # Convert weekly trend to daily
                elif self.freq.startswith("M"):
                    daily_trend = slope / 30  # Convert monthly trend to daily
                self.trends[series_id] = daily_trend
                print(f"Trend for series {series_id}: {daily_trend:.5f} sales/day")

                # Calculate the number of periods for the average based on the frequency
                if self.freq.startswith("D"):
                    periods = 30
                elif self.freq.startswith("W"):
                    periods = 4
                elif self.freq.startswith("M"):
                    periods = 1
                else:
                    raise ValueError(f"Unsupported frequency: {self.freq}")

                # Calculate mean_val based on the last periods if a trend is detected
                last_periods_sales = series[-periods:]
                self.mean_vals[series_id] = last_periods_sales.mean()
                print(
                    f"Mean value for series {series_id} based on the last {periods} periods: {self.mean_vals[series_id]:.2f}"
                )
            else:
                # Calculate mean_val based on the given window size
                self.mean_vals[series_id] = series[-self.window :].mean()
                print(
                    f"Mean value for series {series_id} based on the window size {self.window}: {self.mean_vals[series_id]:.2f}"
                )

        return self

    def predict(self, n: int, freq="D"):
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

        predictions = []
        for series_id, mean_val in tqdm(self.mean_vals.items()):
            last_date = self.last_dates[series_id]
            future_dates = pd.date_range(start=last_date, periods=n + 1, freq=freq)[1:]
            trend = self.trends.get(series_id, 0)

            for date in future_dates:
                days_from_start = (date - future_dates[0]).days
                if freq.startswith("D"):
                    period_trend = trend * min(
                        days_from_start, self.trend_cap * 30
                    )  # Daily trend, capped
                elif freq.startswith("W"):
                    period_trend = trend * min(
                        days_from_start / 7, self.trend_cap * 4
                    )  # Weekly trend, capped
                elif freq.startswith("M"):
                    period_trend = trend * min(
                        days_from_start / 30, self.trend_cap
                    )  # Monthly trend, capped
                else:
                    raise ValueError(f"Unsupported frequency: {freq}")

                prediction = mean_val + period_trend

                # Apply Black Friday and Christmas adjustments
                if self.enable_black_friday and date.month == 11:
                    black_friday_increase = self.black_friday_increase.get(series_id, 0)
                    if black_friday_increase >= 0:
                        prediction *= 1 + self.black_friday_increase.get(series_id, 0)
                        print(
                            f"Black Friday Adjustment Applied on {date}: {prediction}"
                        )

                if self.enable_christmas and date.month == 12:
                    christmas_increase = self.christmas_increase.get(series_id, 0)
                    if christmas_increase >= 0:
                        prediction *= 1 + christmas_increase
                        print(f"Christmas Adjustment Applied on {date}: {prediction}")

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
