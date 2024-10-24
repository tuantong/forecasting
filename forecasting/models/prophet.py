from typing import List, Optional, Sequence, Tuple, Union

from darts.models import Prophet
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.timeseries import TimeSeries


class GlobalProphet(GlobalForecastingModel):

    def __init__(
        self,
        add_seasonalities: Optional[Union[dict, List[dict]]] = None,
        country_holidays: Optional[str] = None,
        suppress_stdout_stderror: bool = True,
        **prophet_kwargs
    ):
        super().__init__()

    def fit(
        self,
        series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
    ) -> "GlobalForecastingModel":
        super().fit(series, past_covariates, future_covariates)
        if isinstance(series, TimeSeries):
            series = [series]
        for ts in series:
            if len(ts) < 10:
                break
            model = Prophet(**self.model_params["autoarima_kwargs"])
            model.fit(ts)
            self.models.append(model)

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        super().predict(n, series, past_covariates, future_covariates, num_samples)
        forecasts = []
        for model in self.models:
            forecast = model.predict(n, future_covariates, num_samples)
            forecasts.append(forecast)
        return forecasts

    def _model_encoder_settings(self) -> Tuple[int, int, bool, bool]:
        return None
