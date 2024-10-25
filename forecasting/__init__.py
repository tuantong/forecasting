import os

from dotenv import load_dotenv
from mlflow.utils.lazy_load import LazyLoader


# Load env variables from .env
def load_environment_vars(root_path: str):
    """
    Load environment variables from a specified .env file based on the ENV or default to dev
    """
    env = os.getenv("ENV", "dev")
    dotenv_file = ".env" if env == "" else f".env.{env}"
    dotenv_path = os.path.join(root_path, dotenv_file)
    load_dotenv(dotenv_path=dotenv_path, verbose=True, override=True)
    print(f"Loaded environment variables from {dotenv_path}")
    for key, value in os.environ.items():
        print(f"{key}: {value}")


darts_flavor = LazyLoader(
    "forecasting.models.mlflow.darts_flavor",
    globals(),
    "forecasting.models.mlflow.darts_flavor",
)

# Define some variables to use accross the project
PROJECT_ROOT_PATH = "/tmp" # custom-event
PROJECT_SRC_PATH = os.path.join(PROJECT_ROOT_PATH, "forecasting")
PROJECT_DATA_PATH = os.path.join(PROJECT_ROOT_PATH, "data")
PROJECT_CONFIG_PATH = os.path.join(PROJECT_ROOT_PATH, "config") # custom-event
PROJECT_DEBUG_PATH = os.path.join(PROJECT_ROOT_PATH, "debug") # custom-event

load_environment_vars(PROJECT_ROOT_PATH)

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_REGION = os.environ["AWS_REGION"]
AWS_FORECAST_DATA_BUCKET = os.environ["AWS_BUCKET_FORECAST_NAME"]
RAW_DATA_BUCKET = os.environ["AWS_DATA_BUCKET"]
S3_RAW_DATA_DIR = os.environ["S3_RAW_DATA_DIR"]
PROCESSED_DATA_BUCKET = os.environ["AWS_BUCKET_NAME"]
S3_PROCESSED_DATA_URI = os.environ["S3_PROCESSED_DATA_URI"]
MODEL_DIR = os.environ["MODEL_DIR"]
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MLFLOW_ARTIFACTS_DESTINATION = os.environ["MLFLOW_ARTIFACTS_DESTINATION"]
STOCK_SERIES_FILENAME = os.environ["STOCK_SERIES_FILENAME"]
METADATA_FILENAME = os.environ["METADATA_FILENAME"]
DAILY_SERIES_FILENAME = os.environ["DAILY_SERIES_FILENAME"]
WEEKLY_SERIES_FILENAME = os.environ["WEEKLY_SERIES_FILENAME"]
MONTHLY_SERIES_FILENAME = os.environ["MONTHLY_SERIES_FILENAME"]
TRAIN_FILENAME = os.environ["TRAIN_FILENAME"]
TEST_FILENAME = os.environ["TEST_FILENAME"]
TEST_NEW_ITEM_FILENAME = os.environ["TEST_NEW_ITEM_FILENAME"]
TEST_BEST_SELLER_FILENAME = os.environ["TEST_BEST_SELLER_FILENAME"]
TEST_TOP10_FILENAME = os.environ["TEST_TOP10_FILENAME"]
TEST_LENGTH = int(os.environ["TEST_LENGTH"])
AWS_FORECAST_HISTORY_BUCKET_NAME = os.environ["AWS_FORECAST_HISTORY_BUCKET_NAME"]
S3_FORECAST_DATA_DIR = os.environ["S3_FORECAST_DATA_DIR"]
