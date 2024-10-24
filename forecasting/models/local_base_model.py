from abc import abstractmethod
from typing import Optional, Sequence, Tuple, Union

from darts import concatenate
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.timeseries import TimeSeries
from joblib import delayed

from forecasting.util import ProgressParallel


class LocalBaseModel(GlobalForecastingModel):

    def __init__(self) -> None:
        self.models = {}

    @abstractmethod
    def _fit_single_timeseries(
        series: TimeSeries, model_params
    ) -> Tuple[str, "LocalBaseModel"]:
        pass

    def fit(
        self, series: Union[TimeSeries, Sequence[TimeSeries]], n_jobs: int = 2
    ) -> "GlobalForecastingModel":
        super().fit(series)
        if isinstance(series, TimeSeries):
            series = [series]

        for ts in series:
            num_series = ts.n_components
            brand_name = ts.static_covariates.brand_name.iat[0]
            print(f"Fitting for {brand_name}")
            results = ProgressParallel(n_jobs=n_jobs, use_tqdm=True, total=num_series)(
                delayed(self._fit_single_timeseries)(
                    ts.univariate_component(i), self.model_params
                )
                for i in range(num_series)
            )
            self.models[brand_name] = dict(results)

    @abstractmethod
    def _predict_single_timeseries(
        item_id, model, n, num_samples: int = 1, n_jobs: int = 2
    ) -> Tuple[str, TimeSeries]:
        pass

    def predict(
        self,
        n: int,
        series: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        num_samples: int = 1,
        n_jobs: int = 2,
    ) -> Union[TimeSeries, Sequence[TimeSeries]]:
        super().predict(n, series, past_covariates, future_covariates, num_samples)

        if isinstance(series, TimeSeries):
            series = [series]

        input_brands = (
            [ts.static_covariates.brand_name.iat[0] for ts in series]
            if series is not None
            else None
        )
        brand_predicts = []
        for brand, models in self.models.items():
            print(f"Predict for {brand}")
            input_brand_ts = (
                next(
                    (
                        x
                        for x in series
                        if x.static_covariates.brand_name.iat[0] == brand
                    ),
                    None,
                )
                if series is not None
                else None
            )
            if input_brands is None or brand in input_brands:
                forecasts = dict(
                    ProgressParallel(
                        n_jobs=n_jobs, use_tqdm=True, total=len(models.items())
                    )(
                        delayed(self._predict_single_timeseries)(
                            key, model, n, num_samples, n_jobs
                        )
                        for key, model in models.items()
                    )
                )
                results = []
                if input_brand_ts is not None:
                    for key in input_brand_ts.columns:
                        ts = forecasts[key]
                        results.append(ts)
                else:
                    results = list(forecasts.values())
                brand_ts = concatenate(results, axis=1)
                if input_brand_ts is not None:
                    brand_ts.with_hierarchy(input_brand_ts.hierarchy)
                brand_predicts.append(brand_ts)
                print(
                    f"Added brand predict result to results. len(results) = {len(brand_predicts)}"
                )
            elif input_brands is not None:
                print(
                    f"Ignore brand {brand} in models because it is not in the input series: {input_brands}"
                )

        return brand_predicts

    def _model_encoder_settings(self) -> Tuple[int, int, bool, bool]:
        return None
