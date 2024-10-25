from typing import Tuple

from darts.models import StatsForecastAutoARIMA
from darts.timeseries import TimeSeries

from forecasting.models.local_base_model import LocalBaseModel


class SFAutoARIMA(LocalBaseModel):

    def __init__(self, **autoarima_kwargs):
        super().__init__()

    @staticmethod
    def _get_model_params(series: TimeSeries, model_params):
        values = series.values()[:, 0]
        params = model_params["autoarima_kwargs"]
        period = params["season_length"]
        max_p = params["max_p"]
        max_P = params["max_P"]
        max_d = params["max_d"]
        max_D = params["max_D"]

        # check method with short timeseries, change method to Maximize Likelihood
        if len(values) <= max_p + max_d + (max_P + max_D) * period:
            params["method"] = "ML"

        return params

    @staticmethod
    def _fit_single_timeseries(
        series: TimeSeries, model_params
    ) -> Tuple[str, "LocalBaseModel"]:
        if len(series) < 10:
            return None
        params = SFAutoARIMA._get_model_params(series, model_params)
        item_id = series.static_covariates.iat[0, 0]
        try:
            model = StatsForecastAutoARIMA(**params)
            model.fit(series)
            return item_id, model
        except Exception as e:
            print(f"Error: {e} at {item_id}")
            return None

    @staticmethod
    def _predict_single_timeseries(
        item_id, model, n, num_samples: int = 1, n_jobs: int = 2
    ) -> Tuple[str, TimeSeries]:
        return item_id, model.predict(n, num_samples=num_samples) if model else None
