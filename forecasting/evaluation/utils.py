import logging

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SEASONALITIES = {
    "S": 3600,  # 1 hour
    "T": 1440,  # 1 day
    "H": 24,  # 1 day
    "D": 1,  # 1 day
    "W": 1,  # 1 week
    "M": 12,
    "B": 5,
    "Q": 4,
}


def norm_freq_str(freq_str: str) -> str:
    return freq_str.split("-")[0]


def get_seasonality(freq: str, seasonalities=None) -> int:
    """
    Return the seasonality of a given frequency:
    >>> get_seasonality("2H")
    12
    """
    if seasonalities is None:
        seasonalities = DEFAULT_SEASONALITIES
    offset = pd.tseries.frequencies.to_offset(freq)

    base_seasonality = seasonalities.get(norm_freq_str(offset.name), 1)

    seasonality, remainder = divmod(base_seasonality, offset.n)
    if not remainder:
        return seasonality

    logger.warning(
        f"Multiple {offset.n} does not divide base seasonality "
        f"{base_seasonality}. Falling back to seasonality 1."
    )
    return 1


if __name__ == "__main__":
    print(get_seasonality("W-Mon"))
