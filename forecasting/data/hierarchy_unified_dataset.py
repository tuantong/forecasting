import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from darts.timeseries import TimeSeries

from forecasting.configs.logging_config import logger
from forecasting.data.base_dataset import (
    BaseDataset,
    _copy_raw_dataset,
    load_and_print_info,
)
from forecasting.metadata.hierarchy_unified import HierarchyUnifiedMetadata

warnings.filterwarnings("ignore", category=UserWarning)


class HierarchyUnifiedDataset(BaseDataset[HierarchyUnifiedMetadata]):

    def __init__(
        self, metadata: Union[HierarchyUnifiedMetadata, Dict], download: bool = False
    ):
        if isinstance(metadata, Dict):
            metadata = HierarchyUnifiedMetadata(**metadata)
        super().__init__(metadata)
        self.download = download
        self.train_data: List[TimeSeries]
        self.test_data: List[TimeSeries]

    def __repr__(self):
        basic = f"{self.brand_name} Dataset\nConfig: {self.args}\n"
        if self.data_train is None and self.data_test is None:
            return basic

        data = (
            f"Train/test sizes: {len(self.data_train)}, {len(self.data_test)}\n"
            f"Number of components: {self.data_train.n_components}\n"
            f"Number of variants: {len(self.data_train.hierarchy.keys())}\n"
            f"Number of products: {len(set().union(*self.data_train.hierarchy.values())) - 1}\n"
        )
        return basic + data

    def prepare_data(self):
        """Prepare raw data for loading"""

        if self.download or self._metadata.raw_train_data_path.exists() is False:
            logger.info("Download train file")
            _copy_raw_dataset(
                self._metadata.source_train_path, self._metadata.raw_train_data_path
            )

        if self.download or self._metadata.raw_test_data_path.exists() is False:
            logger.info("Download test file")
            _copy_raw_dataset(
                self._metadata.source_test_path, self._metadata.raw_test_data_path
            )

        if self.download or self._metadata.meta_data_path.exists() is False:
            logger.info("Download metadata file")
            _copy_raw_dataset(
                self._metadata.source_metadata_path, self._metadata.meta_data_path
            )

        if (
            self._metadata.processed_train_data_path.exists() is False
            or self._metadata.processed_test_data_path.exists() is False
        ):
            train_df = pd.read_parquet(
                self._metadata.raw_train_data_path,
                engine="pyarrow",
                use_nullable_dtypes=True,
            )
            test_df = pd.read_parquet(
                self._metadata.raw_test_data_path,
                engine="pyarrow",
                use_nullable_dtypes=True,
            )
            meta_df = pd.read_parquet(
                self._metadata.meta_data_path,
                engine="pyarrow",
                use_nullable_dtypes=True,
            )
            logger.info("Process train data")
            self._process_raw_data(
                train_df, meta_df, self._metadata.processed_train_data_path
            )
            logger.info("Process test data")
            self._process_raw_data(
                test_df, meta_df, self._metadata.processed_test_data_path
            )

    def setup(self, stage: Optional[str] = None) -> None:

        if stage is None or ("fit" in stage):
            with open(self._metadata.processed_train_data_path, mode="rb") as f:
                timeseries = pickle.load(f)
            self.train_data = []
            for ts in timeseries:
                if self.pipeline is not None:
                    ts_transformed = self.pipeline.fit_transform(ts)
                    self.train_data.append(ts_transformed)

        if stage == "test" or stage is None:
            with open(self._metadata.processed_test_data_path, mode="rb") as f:
                timeseries = pickle.load(f)
            self.test_data = []
            for ts in timeseries:
                if self.pipeline is not None:
                    ts_transformed = self.pipeline.fit_transform(ts)
                    self.test_data.append(ts_transformed)

    def _process_raw_data(self, raw_df, meta_df, output_path: Path):
        brands_name = meta_df.brand_name.unique()
        timeseries = []
        for brand in brands_name:
            logger.info(f"Process brand: {brand}")
            brand_meta_df = meta_df[meta_df.id.str.startswith(brand)]
            brand_raw_df = raw_df[raw_df.id.str.startswith(brand)]
            brand_timeseries = self._process_brand_data(brand_raw_df, brand_meta_df)
            timeseries.append(brand_timeseries)

        output_path.parent.mkdir(mode=0o775, parents=True, exist_ok=True)
        logger.info("Save processed data")
        with open(output_path, mode="wb") as f:
            pickle.dump(timeseries, f)

    def _process_brand_data(self, raw_df, meta_df):
        # Filter meta data
        raw_ids = raw_df.id.unique()
        filtered_meta_df = meta_df[meta_df.id.isin(raw_ids)]

        # Prepare static covariates dataframe
        static_covariates_df = pd.DataFrame(filtered_meta_df, copy=True)
        empty_row = pd.DataFrame(columns=static_covariates_df.columns, index=[1])
        # At root node of hiearchy dict
        empty_row["id"] = "total"
        static_covariates_df = pd.concat(
            [static_covariates_df, empty_row], ignore_index=True
        )

        # Prepare hierarchy
        # Create product meta dict
        product_meta_df = filtered_meta_df[filtered_meta_df.variant_id.isnull()][
            ["id", "product_id"]
        ]
        product_meta_df["product_full_id"] = product_meta_df["id"]
        product_meta_df = product_meta_df.drop(columns=["id"])
        # Cheat to add hierarchy to Timeseries object
        product_meta_df["total"] = "total"
        product_dict = (
            product_meta_df[["product_full_id", "total"]]
            .set_index("product_full_id")
            .to_dict()["total"]
        )

        # Create variant meta dict
        variant_meta_df = filtered_meta_df[filtered_meta_df.variant_id.notnull()][
            [self._metadata.id_col, "variant_id", "product_id"]
        ]
        variant_meta_df = variant_meta_df.join(
            product_meta_df.set_index("product_id"), on="product_id"
        )
        hierarchy_dict = (
            variant_meta_df[["id", "product_full_id"]]
            .set_index("id")
            .to_dict()["product_full_id"]
        )

        # Join to meta dict
        hierarchy_dict.update(product_dict)
        # Convert parent value to list
        hierarchy_dict = {k: [v] for k, v in hierarchy_dict.items()}

        # Prepare data dataframe
        # Reshape raw dataframe with columns are variants and products (components of Timeseries object)
        # Convert to float to change pd.NA value to np.nan
        pivot_df = pd.pivot_table(
            raw_df,
            values=self._metadata.target_col,
            index=self._metadata.time_col,
            columns=self._metadata.id_col,
        ).astype(float)
        # Add total component to fix root node in hierarchy
        pivot_df["total"] = pivot_df.sum(axis=1)
        # Use DataArray to fix the Timeseries.from_dataframe doesn't work with columns from pivot operator (dtype=Index)
        xa = xr.DataArray(
            pivot_df.values[:, :, np.newaxis],
            dims=("date", "component", "sample"),
            coords={"date": pivot_df.index, "component": pivot_df.columns.values},
            attrs={
                "static_covariates": static_covariates_df,
                "hierarchy": hierarchy_dict,
            },
        )
        print(
            f"component size: {pivot_df.columns.values.shape}, "
            f"static_covariates: {static_covariates_df.shape}, hierarchy: {len(hierarchy_dict)}"
        )
        # Finally, create Timeseries and save it to disk with pkl format
        timeseries = TimeSeries.from_xarray(
            xa, fill_missing_dates=True, freq=self._metadata.freq, fillna_value=np.nan
        )
        return timeseries


if __name__ == "__main__":
    load_and_print_info(HierarchyUnifiedDataset)
