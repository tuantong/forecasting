"""
Baseline model.
"""

import numpy as np
from darts.models.forecasting.forecasting_model import LocalForecastingModel
from darts.timeseries import TimeSeries


class Baseline(LocalForecastingModel):
    """
    Baseline model that uses average sales of K previous periods to make prediction.

    Parameters
    ----------
    K
        the number of last time steps of the training set to take average.

    """

    def __init__(self, window):
        super().__init__()
        self.window = window
        self.mean_val = None

    def __str__(self):
        return "Baseline predictor model"

    @property
    def min_train_series_length(self) -> int:
        return 1

    def fit(self, series: TimeSeries):
        super().fit(series)
        self.mean_val = np.sum(series.univariate_values()[-self.window :]) / self.window
        return self

    def predict(self, n: int, num_samples: int = 1):
        super().predict(n, num_samples)
        forecast = np.array([self.mean_val for _ in range(n)])
        return self._build_forecast_series(forecast)
