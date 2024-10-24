import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from darts import TimeSeries

from forecasting.data.base_dataset import BaseDataset, load_and_print_info
from forecasting.metadata.shared import Metadata


class BrandDataset(BaseDataset[Metadata]):

    def __init__(self, args: dict, root_path: Path = None):
        super().__init__(args, root_path)
        brand_info = self.args["brand_info"]
        self.brand_name: str = brand_info["brand_name"]
        self.dataset_name: str = brand_info["name"]

        self.brand_path: Path = self.data_path / self.brand_name
        self.dataset_path: Path = self.brand_path / self.dataset_name

        metadata_filename = self.args["metadata_filename"]
        raw_filename = self.args["raw_filename"]
        processed_filename = self.args["processed_filename"]
        self.meta_data_path: Path = self.brand_path / "raw_data" / metadata_filename
        self.raw_path: Path = self.brand_path / "raw_data" / raw_filename
        self.processed_path: Path = self.dataset_path / processed_filename

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
        if not self.processed_path.exists():
            self._process_raw_data()

    def setup(self, stage: Optional[str] = None) -> None:
        with open(self.processed_path, mode="rb") as f:
            timeseries = pickle.load(f)
        if self.pipeline is not None:
            timeseries = self.pipeline.fit_transform(timeseries.copy())
        if stage is None or ("fit" in stage):
            self.data_train = timeseries[: -self.prediction_length]

        if stage == "test" or stage is None:
            self.data_test = timeseries[-self.prediction_length :]

    def _process_raw_data(self):
        # Load raw dataframe & meta data dataframe from parquet file
        raw_df = pd.read_parquet(
            self.raw_path, engine="pyarrow", use_nullable_dtypes=True
        )
        meta_df = pd.read_parquet(
            self.meta_data_path, engine="pyarrow", use_nullable_dtypes=True
        )

        # Prepare static covariates dataframe
        static_covariates_df = pd.DataFrame(meta_df, copy=True)
        static_covariates_df.loc[static_covariates_df.shape[0]] = None
        # At root node of hiearchy dict
        static_covariates_df.at[static_covariates_df.shape[0] - 1, "id"] = "total"

        # Prepare hierarchy
        # Create product meta dict
        product_meta_df = meta_df[meta_df.variant_id.isnull()][["id", "product_id"]]
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
        variant_meta_df = meta_df[meta_df.variant_id.notnull()][
            ["id", "variant_id", "product_id"]
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
            raw_df, values="quantity_order", index="date", columns="id"
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

        # Finally, create Timeseries and save it to disk with pkl format
        timeseries = TimeSeries.from_xarray(
            xa, fill_missing_dates=True, freq=self.frequency, fillna_value=np.nan
        )
        self.processed_path.parent.mkdir(mode=0o775, parents=True, exist_ok=True)
        timeseries.to_pickle(self.processed_path)


if __name__ == "__main__":
    load_and_print_info(BrandDataset)
