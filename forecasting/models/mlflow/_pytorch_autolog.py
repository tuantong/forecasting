import logging
import os
import shutil
import tempfile

import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
from mlflow.pytorch._lightning_autolog import (
    BatchMetricsLogger,
    MlflowAutologgingQueueingClient,
    ModelSummary,
    __MLflowPLCallback,
    _log_early_stop_metrics,
    _log_early_stop_params,
    get_autologging_config,
)
from packaging.version import Version

import forecasting.models.mlflow.darts_flavor as mlflow_darts

logging.basicConfig(level=logging.ERROR)


def patched_fit(original, self, *args, **kwargs):
    """
    A patched implementation of `pytorch_lightning.Trainer.fit` which enables logging the
    following parameters, metrics and artifacts:

    - Training epochs
    - Optimizer parameters
    - `EarlyStoppingCallback`_ parameters
    - Metrics stored in `trainer.callback_metrics`
    - Model checkpoints
    - Trained model

    .. _EarlyStoppingCallback:
        https://pytorch-lightning.readthedocs.io/en/latest/early_stopping.html
    """
    run_id = mlflow.active_run().info.run_id
    tracking_uri = mlflow.get_tracking_uri()
    client = MlflowAutologgingQueueingClient(tracking_uri)
    metrics_logger = BatchMetricsLogger(run_id, tracking_uri)

    log_models = get_autologging_config(mlflow_darts.FLAVOR_NAME, "log_models", True)
    log_every_n_epoch = get_autologging_config(
        mlflow_darts.FLAVOR_NAME, "log_every_n_epoch", 1
    )
    log_every_n_step = get_autologging_config(
        mlflow_darts.FLAVOR_NAME, "log_every_n_step", None
    )
    model_info = get_autologging_config(mlflow_darts.FLAVOR_NAME, "model_info")

    early_stop_callback = None
    for callback in self.callbacks:
        if isinstance(callback, pl.callbacks.early_stopping.EarlyStopping):
            early_stop_callback = callback
            _log_early_stop_params(early_stop_callback, client, run_id)

    if not any(
        isinstance(callbacks, __MLflowPLCallback) for callbacks in self.callbacks
    ):
        self.callbacks += [
            __MLflowPLCallback(
                client,
                metrics_logger,
                run_id,
                log_models,
                log_every_n_epoch,
                log_every_n_step,
            )
        ]

    client.flush(synchronous=False)

    result = original(self, *args, **kwargs)

    if early_stop_callback is not None:
        _log_early_stop_metrics(early_stop_callback, client, run_id)

    if Version(pl.__version__) < Version("1.4.0"):
        # pylint: disable-next=unexpected-keyword-arg
        summary = str(ModelSummary(self.model, mode="full"))
    else:
        summary = str(ModelSummary(self.model, max_depth=-1))

    tempdir = tempfile.mkdtemp()
    try:
        summary_file = os.path.join(tempdir, "model_summary.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary)

        mlflow.log_artifact(local_path=summary_file)
    finally:
        shutil.rmtree(tempdir)

    # BUG: log_models=True not working yet
    # **Note**: Autologging is only supported for PyTorch Lightning models, i.e., models that subclass
    # pytorch_lightning.LightningModule <https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html>_.
    # In particular, autologging support for vanilla PyTorch models that only subclass
    # torch.nn.Module
    if log_models:
        registered_model_name = get_autologging_config(
            mlflow_darts.FLAVOR_NAME, "registered_model_name", None
        )
        mlflow_darts.log_model(
            darts_model=self.model,
            artifact_path="model",
            model_info=model_info,
            registered_model_name=registered_model_name,
        )

        if early_stop_callback is not None and self.checkpoint_callback.best_model_path:
            mlflow.log_artifact(
                local_path=self.checkpoint_callback.best_model_path,
                artifact_path="restored_model_checkpoint",
            )

    client.flush(synchronous=True)

    return result
