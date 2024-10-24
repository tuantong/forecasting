import pickle
from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    """
    Abstract base class for different time series models.
    """

    def __init__(self):
        self.mean_vals = {}
        self.last_dates = {}
        self.item_configs = {}

    @abstractmethod
    def fit(self, df: pd.DataFrame, **kwargs):
        pass

    @abstractmethod
    def predict(self, n: int, freq="D", **kwargs):
        pass

    def set_item_config(self, item_id: str, config: dict):
        """
        Sets individual configuration for a specific item.

        Parameters
        ----------
        item_id : str
            The ID of the item.
        config : dict
            The configuration for the item.
        """
        self.item_configs[item_id] = config

    def get_item_config(self, item_id: str, config_key: str, default_value):
        """
        Gets the configuration value for a specific item, falling back to the default value if not set.

        Parameters
        ----------
        item_id : str
            The ID of the item.
        config_key : str
            The key of the configuration value.
        default_value : any
            The default value to use if the configuration key is not set for the item.

        Returns
        -------
        any
            The configuration value for the item.
        """
        return self.item_configs.get(item_id, {}).get(config_key, default_value)

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
        BaseModel
            The loaded model.
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)
