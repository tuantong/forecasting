import pandas as pd
import pytest

from forecasting.data.preprocessing.process_dataset import resample_brand_data


class TestResampleBrandData:
    # Test that resampling from daily to weekly works
    def test_resample_daily_to_weekly(self):
        freq, closed, label = "W-MON", "left", "left"
        # Create a sample DataFrame
        daily_df = pd.DataFrame(
            {
                "id": [1, 1, 2, 2, 3, 3],
                "date": pd.date_range(start="2020-01-01", periods=6),
                "quantity_order": [10, 20, 30, 40, 50, 60],
                "price": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            }
        )

        expected_result = (
            daily_df.set_index("date")
            .groupby(["id", pd.Grouper(freq=freq, closed=closed, label=label)])
            .agg({"quantity_order": "sum", "price": "mean"})
            .reset_index()
        )

        # Resample the DataFrame by brand IDs
        resampled_df = resample_brand_data(daily_df, freq, closed, label)

        # Check if the result is as expected
        pd.testing.assert_frame_equal(resampled_df, expected_result)

    # Test the resampling daily to monthly works
    def test_resample_daily_to_monthly(self):
        freq, closed, label = "M", "right", "right"
        # Create a sample DataFrame
        daily_df = pd.DataFrame(
            {
                "id": [1, 1, 2, 2, 3, 3],
                "date": pd.date_range(start="2020-01-01", periods=6),
                "quantity_order": [10, 20, 30, 40, 50, 60],
                "price": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            }
        )

        expected_result = (
            daily_df.set_index("date")
            .groupby(["id", pd.Grouper(freq=freq, closed=closed, label=label)])
            .agg({"quantity_order": "sum", "price": "mean"})
            .reset_index()
        )

        # Resample the DataFrame by brand IDs
        resampled_df = resample_brand_data(daily_df, freq, closed, label)

        # Check if the result is as expected
        pd.testing.assert_frame_equal(resampled_df, expected_result)


if __name__ == "__main__":
    pytest.main()
