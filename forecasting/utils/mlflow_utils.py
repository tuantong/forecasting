import mlflow
from mlflow import MlflowClient

from forecasting import MLFLOW_ARTIFACTS_DESTINATION, MLFLOW_TRACKING_URI
from forecasting.configs import logger


class MlflowManager:
    """
    A utility class for managing MLflow experiments and logging information.

    Attributes:
        tracking_uri (str): The MLflow tracking URI.
        default_experiment_name (str): The default name for MLflow experiments.
        artifacts_destination (str): The destination for MLflow artifacts.
        client (MlflowClient): An instance of the MLflow tracking client.

    Methods:
        init_experiment(experiment_name=None):
            Initializes an MLflow experiment. If the experiment does not exist, it creates a new one.

        print_auto_logged_info(run):
            Prints auto-logged information for a given MLflow run.

    Example Usage:
        mlflow_manager = MlflowManager(tracking_uri="http://localhost:5000",
                                       default_experiment_name="default_exp",
                                       artifacts_destination="/path/to/artifacts")

        experiment_id = mlflow_manager.init_experiment("my_experiment")
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_param("param1", 123)
            mlflow.log_metric("metric1", 0.85)
            mlflow_manager.print_auto_logged_info(mlflow.active_run())
    """

    def __init__(
        self,
        tracking_uri=MLFLOW_TRACKING_URI,
        artifacts_destination=MLFLOW_ARTIFACTS_DESTINATION,
        default_experiment_name="default_exp",
    ):
        """
        Initializes the MlflowManager with the specified tracking URI,
        default experiment name, and artifacts destination.

        Args:
            tracking_uri (str): The MLflow tracking URI.
            default_experiment_name (str): The default name for MLflow experiments.
            artifacts_destination (str): The destination for MLflow artifacts.
        """
        self.tracking_uri = tracking_uri
        self.default_experiment_name = default_experiment_name
        self.artifacts_destination = artifacts_destination
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        mlflow.set_tracking_uri(self.tracking_uri)

    def init_experiment(self, experiment_name=None):
        """
        Initializes an MLflow experiment. If the experiment does not exist, it creates a new one.

        Args:
            experiment_name (str, optional): The name of the experiment.
                                             If not provided, uses the default experiment name.

        Returns:
            str: The experiment ID.
        """
        if not experiment_name:
            experiment_name = self.default_experiment_name

        experiment_id = self._create_experiment_if_not_exists(experiment_name)
        return experiment_id

    def get_latest_run(self, experiment_name):
        """Get the latest run of a given experiment.

        Args:
            experiment_name (str): The name of the experiment.
        """
        experiment_id = self.client.get_experiment_by_name(
            experiment_name
        ).experiment_id
        runs = self.client.search_runs(experiment_ids=[experiment_id])
        return runs[0]

    def _create_experiment_if_not_exists(self, experiment_name):
        """
        Creates an MLflow experiment if it does not already exist.

        Args:
            experiment_name (str): The name of the experiment.

        Returns:
            str: The experiment ID.
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.info(
                "Experiment with name '%s' does not exist. Creating a new experiment.",
                experiment_name,
            )
            experiment_id = self.client.create_experiment(
                experiment_name, self.artifacts_destination
            )
            return experiment_id
        else:
            mlflow.set_experiment(experiment_name)
            return experiment.experiment_id

    def print_auto_logged_info(self, run_id: str):
        """
        Prints auto-logged information for a given MLflow run.

        Args:
            run: An MLflow run object.
        """
        run = self.client.get_run(run_id)
        tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
        artifacts = [
            f.path for f in self.client.list_artifacts(run.info.run_id, "model")
        ]
        logger.info(f"run_id: {run.info.run_id}")
        logger.info(f"artifacts: {artifacts}")
        logger.info(f"params: {run.data.params}")
        logger.info(f"metrics: {run.data.metrics}")
        logger.info(f"tags: {tags}")
