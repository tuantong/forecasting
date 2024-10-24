import numpy as np
import pandas as pd

from forecasting.configs import logger


class SaveDataException(BaseException):
    """Class for handling saving file exception"""


def clean_static_feat(df, cols_to_clean):
    """Feature engineer for static covariate columns

    Args:
        df (pd.DataFrame): Dataframe containing the columns to clean
        cols_to_clean (list[str]): List of columns to be cleaned
    """
    fill_missing_values(df, cols_to_clean)

    exclude_cols = [
        "id",
        "variant_id",
        "product_id",
        "channel_id",
        "brand_name",
        "platform",
    ]

    for col in cols_to_clean:
        if col not in exclude_cols and pd.api.types.is_string_dtype(df[col].dtype):
            df[col] = (
                df[col].str.replace("<NA>", "").str.strip().str.lstrip(", ").str.lower()
            )


def fill_missing_values(df, cols_to_fill):
    for col in cols_to_fill:
        if pd.api.types.is_string_dtype(df[col].dtype):
            df[col] = df[col].fillna("<NA>")
        else:
            df[col] = df[col].fillna(np.nan)


def save_df_to_parquet(df, local_path, index_cols=None):
    """
    Save DataFrame to parquet format.

    Args:
        df (pd.DataFrame): DataFrame to be saved.
        local_path (str): Local path for saving the DataFrame.
        index_cols (None, bool, list): If None or False, the DataFrame index is not included.
                                       If True, the current DataFrame index is included.
                                       If a list, the columns specified are set as the index before saving.
    Returns:
        None
    """
    assert isinstance(
        index_cols, (type(None), bool, list)
    ), "index_cols must be None, a boolean, or a list of strings"
    try:
        logger.info(f"Saving parquet to {local_path}")
        if isinstance(index_cols, list):
            df.set_index(index_cols).to_parquet(
                local_path, engine="pyarrow", index=True
            )
        elif index_cols is True:
            df.to_parquet(local_path, engine="pyarrow", index=True)
        else:
            df.to_parquet(local_path, engine="pyarrow", index=False)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise SaveDataException(
            f"Failed to save the parquet file. Reason: {repr(e)}"
        ) from e
