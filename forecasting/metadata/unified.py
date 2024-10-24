from typing import List

from pydantic.dataclasses import dataclass

from forecasting.metadata.shared import Metadata


@dataclass
class UnifiedMetadata(Metadata):
    """
    This class holds information about the configuration of the dataset
    """

    # past covariates columns
    past_cov_cols: List[str] = None
    # static covariates columns
    static_cov_cols: List[str] = None
