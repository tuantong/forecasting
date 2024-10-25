# Influence from gluonTS implementation with modifications
# https://github.com/awslabs/gluon-ts/blob/dev/src/gluonts/evaluation/metrics.py
from typing import Optional

import numpy as np

from forecasting.evaluation.utils import get_seasonality


def calculate_seasonal_error(
    past_data: np.ndarray,
    freq: Optional[str] = None,
    seasonality: Optional[int] = None,
):
    r"""
    .. math::
        seasonal\_error = mean(|Y[t] - Y[t-m]|)
    where m is the seasonal frequency.
    """
    # Check if the length of the time series is larger than the seasonal
    # frequency
    if not seasonality:
        assert freq is not None, "Either freq or seasonality must be provided"
        seasonality: int = get_seasonality(freq)

    if seasonality < len(past_data):
        forecast_freq: int = seasonality
    else:
        # edge case: the seasonal freq is larger than the length of ts
        # revert to freq=1

        # logging.info('The seasonal frequency is larger than the length of the
        # time series. Reverting to freq=1.')
        forecast_freq: int = 1

    y_t = past_data[:-forecast_freq]
    y_tm = past_data[forecast_freq:]

    return np.mean(abs(y_t - y_tm))


def calculate_seasonal_squared_error(
    past_data: np.ndarray,
    freq: Optional[str] = None,
    seasonality: Optional[int] = None,
):
    r"""
    .. math::
        seasonal\_squared\_error = mean((Y[t] - Y[t-m])^2)
    where m is the seasonal frequency.
    """
    # Check if the length of the time series is larger than the seasonal
    # frequency
    if not seasonality:
        assert freq is not None, "Either freq or seasonality must be provided"
        seasonality = get_seasonality(freq)

    if seasonality < len(past_data):
        forecast_freq = seasonality
    else:
        # edge case: the seasonal freq is larger than the length of ts
        # revert to freq=1

        # logging.info('The seasonal frequency is larger than the length of the
        # time series. Reverting to freq=1.')
        forecast_freq = 1

    y_t = past_data[:-forecast_freq]
    y_tm = past_data[forecast_freq:]

    return np.mean(np.square(y_t - y_tm))


# target: actual, forecast: prediction
# Scaled-dependent errors
def mae(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    Mean Absolute Error
        mae = mean(|Y - \hat{Y}|)
    """
    return np.mean(np.abs(target - forecast))


def mse(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    Mean Squared Error
        mse = mean((Y - \hat{Y})^2)
    """
    return np.mean(np.square(target - forecast))


def msle(target: np.ndarray, forecast: np.ndarray) -> float:
    """
    Mean Squared Logarithmic Error
    """
    return np.mean(np.square(np.log1p(target) - np.log1p(forecast)))


def rmse(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    Root Mean Squared Error
        rmse = sqrt(mean(Y - \hat{Y})^2)
    """
    return np.sqrt(mse(target, forecast))


# Scaled errors
# scaling the errors based on the training MAE from a simple forecast method
def mase(
    target: np.ndarray,
    forecast: np.ndarray,
    seasonal_error: float,
) -> float:
    r"""
    Mean Absolute Scaled Error
        mase = mean(|Y - \hat{Y}|) / seasonal\_error
    """
    return np.mean(np.abs(target - forecast)) / seasonal_error


def rmsse(
    target: np.ndarray,
    forecast: np.ndarray,
    seasonal_squared_error: float,
) -> float:
    r"""
    Root Mean Squared Scaled Error
        rmsse = sqrt(mean((Y - \hat{Y})^2) / seasonal\_squared\_error)
    """
    return np.sqrt(np.mean(np.square(target - forecast)) / seasonal_squared_error)


# Percentage errors
def mape(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    Mean Absolute Percentage Error
        mape = mean(|Y - \hat{Y}| / |Y|))
    """
    if target == 0 and forecast == 0:
        return 0
    elif target == 0 and forecast != 0:
        return 1
    else:
        return np.mean(np.abs(target - forecast) / np.abs(target))


# Percentage errors
def mape_original(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    Mean Absolute Percentage Error
        mape = mean(|Y - \hat{Y}| / |Y|))
    """
    return np.mean(np.abs(target - forecast) / np.abs(target))


def fa(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    Forecast Accuracy
        fa = mean(1 - (|Y - \hat{y}| / |Y|))
    """
    return np.mean(1 - np.abs(target - forecast) / np.abs(target))


def smape(target: np.ndarray, forecast: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error
    """
    return 2 * np.mean(np.abs(target - forecast) / (np.abs(target) + np.abs(forecast)))


# Quantile Loss
def quantile_loss(target: np.ndarray, forecast: np.ndarray, q: float) -> float:
    r"""
    Quantile Loss
        quantile\_loss = 2 * sum(|(Y - \hat{Y}) * (Y <= \hat{Y}) - q|)
    """
    return 2 * np.sum(np.abs((target - forecast) * ((target <= forecast) - q)))


# Some errors metrics for aggregation
def abs_error(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    Sum Absolute Error
        abs\_error = sum(|Y - \hat{Y}|)
    """
    return np.sum(np.abs(target - forecast))


def abs_target_sum(target: np.ndarray) -> float:
    r"""
    Sum Absolute Target
        abs\_target\_sum = sum(|Y|)
    """
    return np.sum(np.abs(target))


def abs_target_mean(target: np.ndarray) -> float:
    r"""
    Mean Absolute Target
        abs\_target\_mean = mean(|Y|)
    """
    return np.mean(np.abs(target))


def signed_error(target, forecast) -> float:
    r"""
    Signed error
        signed\_error = sum(Y - \hat{Y})
    """
    return np.sum(target - forecast)


def cum_error(target, forecast) -> float:
    """
    Cummulative error
    """
    return np.cumsum(target - forecast)


# Intermittent metrics
def nos_p(y_true, y_pred):
    c = np.cumsum(y_true - y_pred)
    mask = y_pred > 0
    return np.sum(c[mask] > 0) / np.sum(mask)


def pis(y_true, y_pred):
    cfe_t = np.cumsum(y_true - y_pred)
    return np.sum(-1 * cfe_t)


def mean_relative_abs_error(y_true, y_pred, seasonal_error):
    return np.mean(np.abs(y_true - y_pred) / np.abs(y_true - seasonal_error))


def percent_better(y_true, y_pred, seasonal_error):
    mae = np.abs(y_true - y_pred)
    mae_star = np.abs(y_true - seasonal_error)
    pb = mae > mae_star
    return np.sum(pb) / len(pb)


def spec(y_true, y_pred, a1=0.75, a2=0.25):
    """Stock-keeping-oriented Prediction Error Costs (SPEC)
    Read more in the :ref:`https://arxiv.org/abs/2004.10537`.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    a1 : opportunity costs weighting parameter
        a1 ∈ [0, ∞]. Default value is 0.75.
    a2 : stock-keeping costs weighting parameter
        a2 ∈ [0, ∞]. Default value is 0.25.
    Returns
    -------
    loss : float
        SPEC output is non-negative floating point. The best value is 0.0.
    Examples
    --------
    >>> from spec_metric import spec
    >>> y_true = [0, 0, 5, 6, 0, 5, 0, 0, 0, 8, 0, 0, 6, 0]
    >>> y_pred = [0, 0, 5, 6, 0, 5, 0, 0, 8, 0, 0, 0, 6, 0]
    >>> spec(y_true, y_pred)
    0.1428...
    >>> spec(y_true, y_pred, a1=0.1, a2=0.9)
    0.5142...
    """
    assert len(y_true) > 0 and len(y_pred) > 0
    assert len(y_true) == len(y_pred)

    sum_n = 0
    for t in range(1, len(y_true) + 1):
        sum_t = 0
        for i in range(1, t + 1):
            delta1 = np.sum([y_k for y_k in y_true[:i]]) - np.sum(
                [f_j for f_j in y_pred[:t]]
            )
            delta2 = np.sum([f_k for f_k in y_pred[:i]]) - np.sum(
                [y_j for y_j in y_true[:t]]
            )

            sum_t = sum_t + np.max(
                [
                    0,
                    a1 * np.min([y_true[i - 1], delta1]),
                    a2 * np.min([y_pred[i - 1], delta2]),
                ]
            ) * (t - i + 1)
        sum_n = sum_n + sum_t
    return sum_n / len(y_true)
