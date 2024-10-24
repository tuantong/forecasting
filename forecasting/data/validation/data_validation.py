import os

import pandas as pd
import pandera as pa
from pandera import DataFrameSchema

from forecasting.configs import logger


class ValidateDataException(BaseException):
    """Class for handling data validation exception"""


def validate_data(data_path: str, schema_json_path: str):
    """Validates the data with the specified Schema as JSON"""
    assert (
        os.path.splitext(data_path)[-1] == ".csv"
    ), "The input data path must be a CSV."
    assert (
        os.path.splitext(schema_json_path)[-1] == ".json"
    ), "The schema path must be a JSON."

    try:
        logger.info(f"Validating file {data_path} with Schema file {schema_json_path}")
        try:
            # Read the data as dataframe
            df = pd.read_csv(data_path)
        except Exception as e:
            logger.error(f"Failed to read CSV file {data_path}: {e}")
            raise ValidateDataException(
                f"Failed to read CSV file {data_path}: {repr(e)}"
            ) from e
        try:
            # Read the Schema
            schema: DataFrameSchema = pa.DataFrameSchema.from_json(schema_json_path)
        except Exception as e:
            logger.error(f"Failed to read or parse Schema file {schema_json_path}: {e}")
            raise ValidateDataException(
                f"Failed to read or parse Schema file {schema_json_path}: {repr(e)}"
            ) from e

        # Validate the DataFrame against the schema
        schema.validate(df, lazy=True)
        logger.info(
            f"Validation successful for file {data_path} with Schema file {schema_json_path}"
        )

    except pa.errors.SchemaErrors as err:
        logger.error(
            "Schema errors and failure cases:\n"
            f"{err.failure_cases}\n"
            f"DataFrame object that failed validation:\n"
            f"{err.data}"
        )
        raise ValidateDataException(
            f"Failed to validate data. Reason {repr(err)}"
        ) from err
