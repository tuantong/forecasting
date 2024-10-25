# Data Preprocessing

This section outlines the data processing workflow for each brand, which should be scheduled to run weekly and monthly.

## Procedure for Brand-Specific Dataset

`FOR EACH brand (brand_name)`:

1. **Download raw data from S3:** Download raw data files (e.g., `orders.csv`, `variants.csv`, `products.csv`, etc.) from S3 to the local folder.

2. **Convert to Daily DataFrame:** Convert the raw CSV files into a DataFrame with a *Daily* frequency format.

3. **Process to Weekly and Monthly DataFrames:** Process the *Daily* DataFrame into *Weekly* and *Monthly* frequency format DataFrames.

4. ***MONTHLY* Task - Create Sets for Evaluation:**
   - Create training and testing sets.
   - Generate sets of top-selling items (top10, top50, best seller) for evaluation purposes.

5. ***WEEKLY* Task - Create Inference Sets:** Create inference sets by processing the latest 25 weeks from step 2.

6. **Save Datasets:**
   - Save the *Daily*, *Weekly* and *Monthly* data in separate folders named "daily_dataset", "weekly_dataset" and "monthly_dataset." These folders will share the same *metadata.parquet* file stored at the parent folder.

7. **Store Inference Sets:**
   - Store the *inference set*, generated weekly, in the "inference_set" folder with three files: *metadata.parquet*, *weekly_series.parquet*, and *monthly_series.parquet*.

`END FOR`

## Procedure for All-Brand Dataset

After creating brand-specific datasets, an "all_brand" folder will be created by concatenating every dataset file from each brand. This procedure is also repeated weekly and monthly.

## Script Guidance

To run the processing flow, use the script [forecasting/data/preprocessing/process_dataset_flow.py](process_dataset_flow.py).

- Use the `--inference_set` flag to generate the inference set. Omit it to generate the full dataset.
- Use the `--save_s3` flag to store the generated files in S3 storage.

**Example:**

```bash
python forecasting/data/preprocessing/process_dataset_flow.py --inference_set --save_s3
```
