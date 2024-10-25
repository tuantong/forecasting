import inspect
import os
import pickle
from typing import Dict

# import mlflow
import numpy as np
import torch
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from darts.utils.likelihood_models import QuantileRegression
from sklearn.preprocessing import OrdinalEncoder

import forecasting.models.mlflow.darts_flavor as mlflow_darts
from forecasting import PROJECT_ROOT_PATH
from forecasting.configs.config_helpers import validate_config
from forecasting.configs.logging_config import logger
from forecasting.data.data_handler import DataHandler
from forecasting.data.utils.darts_utils import get_cat_embedding
from forecasting.models import (
    BlockRNNModel,
    DLinearModel,
    NBEATSModel,
    NHiTSModel,
    NLinearModel,
    RNNModel,
    TCNModel,
    TFTModel,
    TransformerModel,
)
from forecasting.util import import_class


class ModelHandler:
    """Handles the configuration, loading, saving and training of forecasting models."""

    _static_cov_transformer_filename = "static_cov_transformer.pkl"

    def __init__(self, configs: Dict, model_dir: str):
        """
        Initializes the ModelHandler instance.

        Parameters:
        - configs (Dict): Dictionary containing model configurations.
        - model_dir (str): Directory where the model will be saved or loaded.
        """
        # Assuming we already validated the outer fields ['data', 'model', 'config_name']
        self.data_metadata = DataHandler.init_metadata(
            configs["data"]["metadata_class"], configs["data"]["configs"]
        )
        # Validate the `model` config
        validate_config(
            config_data=configs["model"],
            required_fields=["class", "model_type", "configs"],
        )
        self.model_dir = model_dir
        self.static_cov_path = os.path.join(
            self.model_dir, self._static_cov_transformer_filename
        )
        self._parse_configs(configs["model"])
        self.model_format = self._get_model_format()
        # Create the static covariate transformer if needed
        self.static_cov_transformer = (
            self._create_or_load_static_cov_transformer()
            if self.data_metadata.static_cov_cols and self.configs["use_static_cov"]
            else None
        )
        self.model = None

    def _parse_configs(self, config_dict: Dict):
        """Parses and sets the model configurations."""
        self.model_class = import_class(f"forecasting.models.{config_dict['class']}")
        self.model_type = config_dict["model_type"]

        # Parse model configs
        self.configs = config_dict["configs"]

        # Parse model hyper-parameters
        self.parameters = config_dict["parameters"]
        # Fix parameters for global model
        if self.model_type == "global":
            learning_rate = self.parameters.pop("lr")
            n_epochs = self.parameters.get("n_epochs")
            loss_fn, likelihood_fn = self._parse_loss_function()
            self.parameters.update(
                {
                    "pl_trainer_kwargs": {
                        "enable_progress_bar": bool(self.configs.get("verbose")),
                        "accelerator": "gpu" if self.configs.get("use_gpu") else "cpu",
                        "devices": (
                            [0] if self.configs.get("use_gpu") else None
                        ),  # TODO: Fix this later
                    },
                    "optimizer_cls": torch.optim.AdamW,
                    "optimizer_kwargs": {"lr": learning_rate},
                    "lr_scheduler_cls": torch.optim.lr_scheduler.CosineAnnealingLR,
                    "lr_scheduler_kwargs": {"T_max": n_epochs, "eta_min": 1e-7},
                    "model_name": self.configs.get("model_name"),
                    "work_dir": self.model_dir,
                    "loss_fn": loss_fn,
                    "likelihood": likelihood_fn,
                }
            )
            # self._parse_categorical_embedding_size()
        else:
            # TODO: Add any updates here for model_type: `local`
            pass

    # def transform_static_cov

    def _create_or_load_static_cov_transformer(self):
        # Load the static cov transformer if it exists or create a new one
        if os.path.exists(self.static_cov_path):
            logger.info(
                f"Loading static covariates transformer from {self.static_cov_path}"
            )
            with open(self.static_cov_path, "rb") as f:
                return pickle.load(f)

        logger.info("Static covariates transformer not found, creating a new one")
        return StaticCovariatesTransformer(
            transformer_cat=OrdinalEncoder(
                dtype=np.float32,
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
            ),
            cols_num=[],
            cols_cat=self.data_metadata.static_cov_cols,
            n_jobs=-1,
            verbose=True,
        )

    def save_static_cov_transformer(self):
        # TODO: Add warning checks if the transformer is not fitted before saving
        logger.info(f"Saving static covariate transformer at {self.static_cov_path}")
        if "_fit_called" not in self.static_cov_transformer.__dict__:
            logger.warning(
                "Saving an unfitted Static Covariate Transformer. Please check to make sure .fit() method is called."
            )
        with open(self.static_cov_path, "wb") as f:
            pickle.dump(self.static_cov_transformer, f)

    def create_model(self):
        """Create a new model"""
        if self.static_cov_transformer:
            # Add embedding size parameters for supported models
            self._parse_categorical_embedding_size()
        logger.info(f"Initiating model {self.model_class}")
        self.model = self.model_class(**self.parameters)

    def load_model(self):
        """Load the model from the specified directory."""
        logger.info(f"Loading model from {self.model_dir}")
        self.model = self._load_model()
        logger.info("Model loaded.")

    def _load_model(self):
        """Internal method for loading the model."""
        return mlflow_darts.load_model(os.path.join(self.model_dir, "model"))

    def log_model(self):
        """Log and save the model to the MLFlow artifacts directory"""
        return mlflow_darts.log_model(
            darts_model=self.model,
            artifact_path="model",
            model_info={
                "model_format": self.model_format,
                "model_class": self.model_class,
            },
            pip_requirements=os.path.join(
                PROJECT_ROOT_PATH, "requirements", "requirements-dev.txt"
            ),
        )

    def train_model(self, **kwargs):
        if self.model_type not in ["global", "local"]:
            raise ValueError(f"Unsupported training for model_type: {self.model_type}")

        logger.info("Starting training")
        if self.model_type == "global":
            kwargs.update(
                {
                    "max_samples_per_ts": self.configs["max_samples_per_ts"],
                    "num_loader_workers": self.configs["num_loader_workers"],
                    "verbose": self.configs["verbose"],
                }
            )
            self.model.fit(**kwargs)

        # TODO: Implement evaluation flow
        # def evaluate_model(
        #     self, prediction_length, series, past_covariates, test_df, **kwargs
        # ):
        #     test_pred = self.model.predict(**kwargs)
        #     mlflow.evaluate()

    def _parse_categorical_embedding_size(self):
        use_static_cov = self.configs.get("use_static_cov")
        static_cols = self.data_metadata.static_cov_cols
        # Add categorical embedding for supported models
        if self.model_class in [TFTModel] and use_static_cov:
            # Make sure the static cov transformer is fitted
            static_categories_list = self.static_cov_transformer._fitted_params[0][
                "transformer_cat"
            ].categories_
            categories_len = {
                cat_col: len(cat_list)
                for cat_col, cat_list in zip(static_cols, static_categories_list)
            }
            categorical_embedding_sizes = (
                get_cat_embedding(categories_len) if use_static_cov else None
            )
            # Update the parameters dictionary
            self.parameters.update(
                {"categorical_embedding_sizes": categorical_embedding_sizes}
            )

    def _parse_loss_function(self):
        loss_dict = self.parameters.pop("loss")
        loss_str = loss_dict["name"]
        loss_args = loss_dict["args"]
        if loss_str == "l1":
            loss_fn = torch.nn.L1Loss()
            likelihood = None
        elif loss_str == "l2":
            loss_fn = torch.nn.MSELoss()
            likelihood = None
        elif loss_str == "quantile":
            loss_fn = None
            likelihood = QuantileRegression(quantiles=loss_args)
        else:
            raise ValueError(f"Unrecognised loss_str={loss_str}")

        return loss_fn, likelihood

    def _get_model_format(self):
        assert inspect.isclass(
            self.model_class
        ), "self.model_class must be a class type"
        imported_classes = (
            BlockRNNModel,
            DLinearModel,
            NBEATSModel,
            NHiTSModel,
            NLinearModel,
            RNNModel,
            TCNModel,
            TFTModel,
            TransformerModel,
        )
        if self.model_class in imported_classes:
            return ".pt"
        return ".pkl"
