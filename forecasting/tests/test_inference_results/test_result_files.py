import json
import os

from forecasting.tests.support.assertions import assert_valid_schema


def test_inference_variant_results(forecast_data_test_dir):
    brand_folders = os.listdir(forecast_data_test_dir)
    for brand_folder in brand_folders:
        variant_results_path = os.path.join(
            forecast_data_test_dir, brand_folder, "variant_result_forecast.json"
        )
        with open(variant_results_path) as f:
            variant_results = json.load(f)
        assert_valid_schema(variant_results, "variant_result_forecast.json")


def test_inference_product_results(forecast_data_test_dir):
    brand_folders = os.listdir(forecast_data_test_dir)
    for brand_folder in brand_folders:
        product_results_path = os.path.join(
            forecast_data_test_dir, brand_folder, "product_result_forecast.json"
        )
        with open(product_results_path) as f:
            product_results = json.load(f)
        assert_valid_schema(product_results, "product_result_forecast.json")
