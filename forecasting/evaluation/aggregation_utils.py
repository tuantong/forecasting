import itertools
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm


def build_result_df_from_pd(
    train_df,
    pred_df,
    pred_cols: List[str],
    target_col: str = "quantity_order",
):
    """
    Build result df that contains train_set, (test_set if available) and pred_set
    for each time series

    Returns:
        result_df: Result df in wide format:
            "id", "train_ts", ("test_ts"), "pred_ts", "first_train_date", "first_pred_date"
    """

    # create ids and train, test, pred list for every time series

    if target_col in pred_df.columns:
        pred_df_pivot = pd.pivot_table(
            pred_df,
            index="id",
            values=[target_col] + pred_cols,
            aggfunc=lambda x: list(x),
        ).rename(columns={target_col: "test_ts"})

    else:

        pred_df_pivot = pd.pivot_table(
            pred_df, index="id", values=pred_cols, aggfunc=lambda x: list(x)
        )

    # add metadata
    meta_pred_pivot = (
        pred_df.groupby(by="id")
        .agg(
            {
                "date": "first",
                # 'brand_name': 'first',
                # 'product_id': 'first',
                # 'variant_id': 'first',
                # 'channel_id': 'first',
                # 'is_product': 'first'
            }
        )
        .rename(columns={"date": "first_pred_date"})
    )

    train_df_pivot = pd.pivot_table(
        train_df, index="id", values=[target_col], aggfunc=lambda x: list(x)
    ).rename(columns={target_col: "train_ts"})

    meta_train_pivot = (
        train_df.groupby(by="id")
        .agg({"date": "first"})
        .rename(columns={"date": "first_train_date"})
    )

    result_df = pred_df_pivot.join(train_df_pivot)
    result_df = result_df.join(meta_pred_pivot)
    result_df = result_df.join(meta_train_pivot)

    # Fill NA with empty list in train_data
    result_df["train_ts"] = result_df["train_ts"].apply(
        lambda d: d if isinstance(d, list) else []
    )

    return result_df


def load_monthly_history_data(data):
    # Read monthly historical values from train/test set
    train_monthly_path = (
        data._metadata.source_data_path / "monthly_dataset/monthly_series.parquet"
    )
    train_monthly_df = pd.read_parquet(
        train_monthly_path, engine="pyarrow", use_nullable_dtypes=True
    )

    # Read monthly historical values from inference set
    infer_monthly_df = pd.read_parquet(
        data._infer_monthly_data_path, engine="pyarrow", use_nullable_dtypes=True
    )

    # Concat monthly df from train/test set and inference set
    train_final_month = max(train_monthly_df.date)
    filter_train_monthly_df = train_monthly_df[
        train_monthly_df.date < train_final_month
    ]
    filter_infer_monthly_df = infer_monthly_df[
        infer_monthly_df.date >= train_final_month
    ]
    monthly_df = (
        pd.concat([filter_train_monthly_df, filter_infer_monthly_df])
        .sort_values(by=["id", "date"])
        .reset_index(drop=True)
    )

    return monthly_df


def aggregate_result_df_to_monthly(result_df):
    """
    Aggregate result_df (wide format): pred_ts, test_ts, train_ts to monthly
    """
    monthly_result_df = result_df.copy()
    monthly_pred_list = []
    monthly_test_list = []
    monthly_train_list = []
    first_train_date_list = []
    first_pred_date_list = []

    for item in tqdm(result_df.index):
        df = result_df.loc[item]
        weekly_pred = df.pred_ts
        weekly_test = df.test_ts
        weekly_train = df.train_ts

        # Convert pred and test_ts
        # divide weekly_ts to daily_ts
        daily_pred = list(
            itertools.chain(
                *[np.repeat(round(value / 7, 1), 7).tolist() for value in weekly_pred]
            )
        )
        daily_test = list(
            itertools.chain(
                *[np.repeat(round(value / 7, 1), 7).tolist() for value in weekly_test]
            )
        )

        # aggregate daily_ts to monthly_ts
        first_pred_date = df.first_pred_date
        pred_horizon = pd.date_range(first_pred_date.date(), periods=len(daily_pred))
        pred_and_test_df = pd.DataFrame(
            {"date": pred_horizon, "pred_ts": daily_pred, "test_ts": daily_test}
        )

        # filter out the rows of test set with days of month <= 15 days
        pred_and_test_df["month"] = pred_and_test_df["date"].dt.month
        count_m = pred_and_test_df["month"].value_counts()
        count_m = count_m[count_m < 15]
        removed_m = count_m.index.tolist()
        pred_and_test_df = pred_and_test_df[~pred_and_test_df["month"].isin(removed_m)]

        monthly_pred_df = (
            pred_and_test_df[["date", "pred_ts"]].resample("M", on="date").sum()
        )
        monthly_test_df = (
            pred_and_test_df[["date", "test_ts"]].resample("M", on="date").sum()
        )
        monthly_pred_list.append(
            list(np.round(monthly_pred_df.pred_ts.values.tolist()))
        )
        monthly_test_list.append(
            list(np.round(monthly_test_df.test_ts.values.tolist()))
        )
        first_pred_date_list.append(monthly_pred_df.index[0])

        # Convert train_ts
        if len(weekly_train) > 0:  # check if item in tran set
            daily_train = list(
                itertools.chain(
                    *[
                        np.repeat(round(value / 7, 1), 7).tolist()
                        for value in weekly_train
                    ]
                )
            )

            first_train_date = df.first_train_date
            train_horizon = pd.date_range(
                first_train_date.date(), periods=len(daily_train)
            )
            item_train_df = pd.DataFrame(
                {"date": train_horizon, "train_ts": daily_train}
            )
            monthly_train_df = item_train_df.resample("M", on="date").sum()
            monthly_train_list.append(
                list(np.round(monthly_train_df.train_ts.values.tolist()))
            )
            first_train_date_list.append(monthly_train_df.index[0])
        else:
            monthly_train_list.append([])
            first_train_date_list.append(np.nan)

    monthly_result_df.pred_ts = monthly_pred_list
    monthly_result_df.test_ts = monthly_test_list
    monthly_result_df.train_ts = monthly_train_list
    monthly_result_df.first_train_date = first_train_date_list
    monthly_result_df.first_pred_date = first_pred_date_list

    return monthly_result_df


def aggregate_bottom_up(
    pred_pivot_df: pd.DataFrame,
    pred_column: str,  # Predictions column name used for aggregating
):
    """
    Sum predictions of variant_level for predictions of product_level
    """
    # get all variants' forecasts of the variant level group
    variant_df_forecast = pred_pivot_df[~pred_pivot_df.is_product]
    variant_df_forecast[pred_column] = variant_df_forecast[pred_column].apply(
        lambda x: np.array(x) if isinstance(x, list) else x
    )

    # sum predictions from variant_level
    variant_df_sum = (
        variant_df_forecast.groupby(
            by=["product_id", "channel_id", "brand_name", "platform"]
        )
        .agg({pred_column: np.sum})
        .sort_index()
    )

    # get product forecast level of the product level group
    product_df_forecast = (
        pred_pivot_df[pred_pivot_df.is_product]
        .set_index(["product_id", "channel_id", "brand_name", "platform"])
        .sort_index()
    )

    # Assign product level forecast df with real product level values
    product_df_forecast[pred_column] = variant_df_sum[pred_column]
    product_df_forecast = product_df_forecast.reset_index()

    final_df = pd.concat([variant_df_forecast, product_df_forecast])
    final_df[pred_column] = final_df[pred_column].apply(
        lambda x: x if isinstance(x, float) else list(np.array(x))
    )

    return final_df


def aggregate_bottom_up_update(
    pred_pivot_df: pd.DataFrame,
    pred_column: str,  # Predictions column name used for aggregating
):
    """
    Sum predictions of variant_level for predictions of product_level
    """

    def add_zeros_for_historical_val(array, max_len):
        len_zeros = max_len - len(array)
        added_array = np.concatenate([np.zeros(len_zeros), array])
        return added_array

    # get all variants' forecasts of the variant level group
    variant_df_forecast = pred_pivot_df[~pred_pivot_df.is_product]
    variant_df_forecast[pred_column] = variant_df_forecast[pred_column].apply(
        lambda x: np.array(x) if isinstance(x, list) else x
    )

    # Reshape pred_column values (concat zeros at the front)
    if isinstance(variant_df_forecast[pred_column].values[0], np.ndarray):
        variant_df_forecast = variant_df_forecast.set_index(
            ["product_id", "channel_id", "brand_name", "platform"]
        )
        max_len_historical_val_dict = dict()
        for index in variant_df_forecast.index.unique():
            max_len = max(
                [len(x) for x in variant_df_forecast.loc[[index]][pred_column]]
            )
            max_len_historical_val_dict[index] = max_len
        variant_df_forecast = variant_df_forecast.reset_index()
        variant_df_forecast["max_len"] = variant_df_forecast.apply(
            lambda row: max_len_historical_val_dict[
                (
                    row["product_id"],
                    row["channel_id"],
                    row["brand_name"],
                    row["platform"],
                )
            ],
            axis=1,
        )

        variant_df_forecast[pred_column] = variant_df_forecast.apply(
            lambda row: (
                add_zeros_for_historical_val(row[pred_column], row["max_len"])
                if len(row[pred_column]) < row["max_len"]
                else row[pred_column]
            ),
            axis=1,
        )

    # sum predictions from variant_level
    variant_df_sum = (
        variant_df_forecast.groupby(
            by=["product_id", "channel_id", "brand_name", "platform"]
        )
        .agg({pred_column: np.sum})
        .sort_index()
    )

    # get product forecast level of the product level group
    product_df_forecast = (
        pred_pivot_df[pred_pivot_df.is_product]
        .set_index(["product_id", "channel_id", "brand_name", "platform"])
        .sort_index()
    )

    # Assign product level forecast df with real product level values
    product_df_forecast[pred_column] = variant_df_sum[pred_column]
    product_df_forecast = product_df_forecast.reset_index()

    final_df = pd.concat([variant_df_forecast, product_df_forecast])
    final_df[pred_column] = final_df[pred_column].apply(
        lambda x: x if isinstance(x, float) else list(np.array(x))
    )

    return final_df


def aggregate_top_down(
    pred_pivot_df: pd.DataFrame,
    pred_column: str,  # Predictions column name used for aggregating
):
    """
    Split predictions of product_level for predictions of variant_level by proportions
    """

    # get all variants' forecasts of the variant level group
    var_df_forecast = pred_pivot_df[~pred_pivot_df["is_product"]]
    var_df_forecast[pred_column] = var_df_forecast[pred_column].apply(
        lambda x: np.array(x)
    )

    # sum forecast values of variants by product
    var_df_sum = var_df_forecast.groupby(["product_id", "channel_id", "brand_name"])[
        pred_column
    ].transform("sum")

    # calculate proportion of each variant by div its forecast value by the forecasting sum value.
    var_df_prop = var_df_forecast[pred_column].div(var_df_sum)

    # Assign variant_level forecast df with proportions
    var_df_forecast.loc[var_df_forecast.index, pred_column] = var_df_prop.values

    # get product forecast level of the product level group
    prod_df_forecast = pred_pivot_df[pred_pivot_df["is_product"]]
    prod_df_forecast[pred_column] = prod_df_forecast[pred_column].apply(
        lambda x: np.array(x)
    )

    tmp1 = var_df_forecast.set_index(["product_id", "channel_id", "brand_name"])
    tmp2 = prod_df_forecast.set_index(["product_id", "channel_id", "brand_name"])

    # Multiply proportions with product level forecast to get real variant level values
    tmp3 = tmp1[pred_column] * tmp2[pred_column]
    # print(tmp3[~tmp3.index.isin(tmp1.index)].index)
    tmp3 = tmp3[tmp3.index.isin(tmp1.index)]

    # Assign variant level forecast df with real variant level values
    tmp1 = tmp1.sort_index()
    tmp3 = tmp3.sort_index()
    tmp1[pred_column] = tmp3.values
    # Fill Nan pred values because prod_sum=0 -> props = nan
    tmp1[pred_column] = tmp1[pred_column].apply(lambda x: np.nan_to_num(x))

    ####### Fix bugs related to sum(variant_fc) != product_fc #####################
    tmp1_idx_list = tmp1.index.to_list()
    tmp2_idx_list = tmp2.index.to_list()
    # Count number of variants in 1 product
    tmp2["no_variant"] = [tmp1_idx_list.count(index) for index in tmp2_idx_list]

    # Fill nan = product_fc/no.variants
    tmp4 = tmp3 * 0
    tmp4 = tmp4.apply(lambda x: np.nan_to_num(x, nan=1))
    tmp4 = tmp4 * tmp2[pred_column]
    tmp4 = tmp4 / tmp2.no_variant

    tmp1[pred_column] = tmp1[pred_column] + tmp4.values
    ###############################################################################

    final_df = pd.concat([tmp1.reset_index(), prod_df_forecast]).reset_index(drop=True)

    final_df[pred_column] = final_df[pred_column].apply(lambda x: list(x))

    return final_df


def aggregate_top_down_based_on_sale_distribution(
    pred_pivot_df: pd.DataFrame,
    pred_column: str,  # Predictions column name used for aggregating
):
    """
    Split predictions of product_level for predictions of variant_level by previous year's sale_distribution
    """

    def rearrange_order_of_monthly_ts(pred_date, pred_month, monthly_ts):
        if pred_date < 15:  #
            monthly_ts = monthly_ts[:-1]
            pred_month = pred_month - 1 if pred_month != 1 else 12

        avg_qty_last_3_months = np.sum(monthly_ts[-3:]) / 3

        if len(monthly_ts) < 12:
            monthly_ts = [np.nan] * (12 - len(monthly_ts)) + monthly_ts

        if pred_date != 1:
            ts = monthly_ts[-pred_month:] + monthly_ts[-12:-pred_month]
        else:
            if pred_month != 1:
                ts = monthly_ts[-pred_month + 1 :] + monthly_ts[-12 : -pred_month + 1]
            else:
                ts = monthly_ts[-12:]

        # Fill Nan (history < 12 months) by last month value
        # for time in range(pd.isna(ts).sum()):
        #     ts = [ts[i-1] if pd.isna(ts[i]) else ts[i] for i in range(len(ts))]

        # Fill Nan (history < 12 months) by average quantity of last_3_months
        ts = [avg_qty_last_3_months if pd.isna(val) else val for val in ts]

        return np.array(ts)

    # get variant's dataframe
    var_df_forecast = pred_pivot_df[~pred_pivot_df["is_product"]]
    var_df_forecast[pred_column] = var_df_forecast[pred_column].apply(
        lambda x: np.array(x)
    )

    var_df_forecast["monthly_ts"] = var_df_forecast.apply(
        lambda x: rearrange_order_of_monthly_ts(
            x.first_pred_date.date().day,
            x.first_pred_date.date().month,
            x.monthly_train_ts,
        ),
        axis=1,
    )
    # sum history values of variants by product
    var_df_forecast["month_sum"] = var_df_forecast.groupby(
        ["product_id", "channel_id", "brand_name", "platform"]
    )["monthly_ts"].transform("sum")

    # calculate proportion of each variant by monthly_sale distribution
    var_df_forecast["month_prop"] = var_df_forecast.apply(
        lambda x: x.monthly_ts / x.month_sum, axis=1
    )

    # Assign variant_level daily forecast df with proportions
    first_pred_date = pd.to_datetime(var_df_forecast.first_pred_date.values[0])
    pred_time_range = pd.date_range(
        first_pred_date, periods=len(var_df_forecast.daily_pred_ts.values[0]), freq="D"
    ).tolist()
    pred_month_list = [pred_date.date().month for pred_date in pred_time_range]

    var_df_forecast["daily_prop"] = var_df_forecast["month_prop"].apply(
        lambda x: [x[pred_date - 1] for pred_date in pred_month_list]
    )

    # get product's dataframe
    prod_df_forecast = pred_pivot_df[pred_pivot_df["is_product"]]
    prod_df_forecast[pred_column] = prod_df_forecast[pred_column].apply(
        lambda x: np.array(x)
    )

    tmp1 = var_df_forecast.set_index(
        ["product_id", "channel_id", "brand_name", "platform"]
    )
    tmp2 = prod_df_forecast.set_index(
        ["product_id", "channel_id", "brand_name", "platform"]
    )

    # Multiply proportions with product level forecast to get real variant level values
    tmp3 = tmp1["daily_prop"] * tmp2[pred_column]
    tmp3 = tmp3[tmp3.index.isin(tmp1.index)]

    # Assign variant level forecast df with real variant level values
    tmp1 = tmp1.sort_index()
    tmp3 = tmp3.sort_index()
    tmp1[pred_column] = tmp3.values
    # Fill Nan pred values because prod_sum=0 -> props = nan
    tmp1[pred_column] = tmp1[pred_column].apply(lambda x: np.nan_to_num(x))

    ####### Fix bugs related to sum(variant_fc) != product_fc #####################
    tmp1_idx_list = tmp1.index.to_list()
    tmp2_idx_list = tmp2.index.to_list()
    # Count number of variants in 1 product
    tmp2["no_variant"] = [tmp1_idx_list.count(index) for index in tmp2_idx_list]

    # Fill nan = product_fc/no.variants
    tmp4 = tmp3 * 0
    tmp4 = tmp4.apply(lambda x: np.nan_to_num(x, nan=1))
    tmp4 = tmp4 * tmp2[pred_column]
    tmp4 = tmp4 / tmp2.no_variant

    tmp1[pred_column] = tmp1[pred_column] + tmp4.values
    ###############################################################################

    final_df = pd.concat([tmp1.reset_index(), prod_df_forecast]).reset_index(drop=True)

    final_df[pred_column] = final_df[pred_column].apply(lambda x: list(x))

    return final_df


def aggregate_and_unpad_results(pred_pivot_df: pd.DataFrame, aggregate_type: str):
    if aggregate_type == "top_down":
        agg_result_df = aggregate_top_down(pred_pivot_df)
    if aggregate_type == "bottom_up":
        agg_result_df = aggregate_bottom_up(pred_pivot_df)

    # Create dictionary of id:test_length
    id_list = pred_pivot_df.id.tolist()
    test_len_list = [
        len(pred_pivot_df.iloc[i].test_ts) for i in range(pred_pivot_df.shape[0])
    ]
    test_len_dict = dict(zip(id_list, test_len_list))

    agg_result_df = agg_result_df.set_index("id")
    # Unpad pred_ts and test_ts

    pred_list = [agg_result_df.loc[item].pred_ts for item in agg_result_df.index]
    pred_list = [
        agg_result_df.loc[item].pred_ts[-test_len_dict[item] :]
        for item in agg_result_df.index
    ]
    agg_result_df.pred_ts = pred_list
    agg_result_df = agg_result_df.reset_index()

    return agg_result_df


def preprocess_results(pred_pivot_df: pd.DataFrame, aggregate_type: str):
    # Padding result_ts -> Aggregate top-down/bottom-up -> unpadding

    # Assign length of pred_ts for padding
    pred_len = max([len(pred_ts) for pred_ts in pred_pivot_df.pred_ts])

    # Pad 0 for items that have test_len < 12
    pad_pred_pivot = pred_pivot_df.copy()
    pad_pred_pivot.pred_ts = pad_pred_pivot.pred_ts.apply(
        lambda x: np.pad(x, (pred_len - len(x), 0), "constant", constant_values=0)
    )

    # Aggregate results
    if aggregate_type == "top_down":
        agg_result_df = aggregate_top_down(pad_pred_pivot)
    if aggregate_type == "bottom_up":
        agg_result_df = aggregate_bottom_up(pad_pred_pivot)

    # Create dictionary of id:test_length
    id_list = pred_pivot_df.id.tolist()
    test_len_list = [
        len(pred_pivot_df.iloc[i].test_ts) for i in range(pred_pivot_df.shape[0])
    ]
    test_len_dict = dict(zip(id_list, test_len_list))

    agg_result_df = agg_result_df.set_index("id")
    # Unpad pred_ts and test_ts
    pred_list = [
        agg_result_df.loc[item].pred_ts[-test_len_dict[item] :]
        for item in agg_result_df.index
    ]
    agg_result_df.pred_ts = pred_list
    agg_result_df = agg_result_df.reset_index()

    return agg_result_df


def aggregate_weekly_ts_to_daily(weekly_ts):
    """Convert a weekly time series to daily by dividing each value equally across 7 days."""
    daily_value_list = [
        np.repeat(round(value / 7, 5), 7).tolist() for value in weekly_ts
    ]
    daily_ts = list(itertools.chain(*daily_value_list))
    return daily_ts


def aggregate_daily_ts_to_monthly(daily_ts, first_date):
    """Aggregate daily time series to monthly using sum."""
    horizon = pd.date_range(first_date, periods=len(daily_ts))
    daily_df = pd.DataFrame({"date": horizon, "value": daily_ts})
    monthly_df = daily_df.resample("M", on="date").sum()
    monthly_ts = list(np.round(monthly_df.value.values.tolist(), 7))
    return monthly_ts


def aggregate_daily_ts_to_weekly(daily_ts, first_date):
    """Aggregate daily time series to weekly (starting on Monday) using sum."""
    horizon = pd.date_range(first_date, periods=len(daily_ts))
    daily_df = pd.DataFrame({"date": horizon, "value": daily_ts})
    weekly_df = daily_df.resample("W-MON", on="date", closed="left", label="left").sum()
    weekly_df = weekly_df.loc[weekly_df.index >= first_date]
    weekly_ts = list(np.round(weekly_df.value.values.tolist(), 7))
    return weekly_ts


def aggregate_weekly_ts_to_monthly(weekly_ts, first_date):
    """Convert a weekly time series to monthly through daily interpolation."""
    daily_ts = aggregate_weekly_ts_to_daily(weekly_ts)
    monthly_ts = aggregate_daily_ts_to_monthly(daily_ts, first_date)
    return monthly_ts


def aggregate_monthly_ts_to_daily(monthly_ts, first_date):
    """Convert a monthly time series to daily by dividing each value equally across days of month."""
    first_date = pd.to_datetime(first_date)
    len_month = len(monthly_ts)
    month_range = pd.date_range(first_date, periods=len_month, freq="M").tolist()

    number_days_of_months = [
        ((month_range[i] - first_date).days + 1) if i == 0 else month_range[i].day
        for i in range(len_month)
    ]
    adjust_daily_value_list = [
        np.repeat(
            round(monthly_ts[i] / number_days_of_months[i], 5), number_days_of_months[i]
        ).tolist()
        for i in range(len_month)
    ]
    adjust_daily_ts = list(itertools.chain(*adjust_daily_value_list))

    # final_daily_ts = adjust_daily_ts + daily_ts[len(adjust_daily_ts):]
    final_daily_ts = adjust_daily_ts
    return final_daily_ts


def aggregate_monthly_ts_to_weekly(monthly_ts, first_date):
    """Convert a monthly time series to weekly (starting on Monday) through daily interpolation."""
    daily_ts = aggregate_monthly_ts_to_daily(monthly_ts, first_date)
    weekly_ts = aggregate_daily_ts_to_weekly(daily_ts, first_date)
    return weekly_ts


def clip_channel_pred_smaller_than_all_channel_pred(full_df):
    brand_list = full_df.brand_name.unique().tolist()
    check_brand_list = [
        brand
        for brand in brand_list
        if "0" in full_df[full_df.brand_name == brand].channel_id.unique()
        and len(full_df[full_df.brand_name == brand].channel_id.unique()) > 1
    ]
    no_check_df = full_df[~full_df.brand_name.isin(check_brand_list)]
    check_df = full_df[full_df.brand_name.isin(check_brand_list)]

    if check_df.shape[0] > 0:
        daily_forecast_length = len(check_df.daily_pred_ts.values[0])
        all_channel_df = check_df[check_df.channel_id == "0"].set_index(
            ["brand_name", "platform", "product_id", "variant_id"]
        )

        check_df["all_channel_daily_pred_ts"] = check_df.apply(
            lambda x: (
                all_channel_df.loc[
                    (x["brand_name"], x["platform"], x["product_id"], x["variant_id"])
                ].daily_pred_ts
                if x["channel_id"] != "0"
                and (x["brand_name"], x["platform"], x["product_id"], x["variant_id"])
                in all_channel_df.index
                else np.nan
            ),
            axis=1,
        )

        check_df["daily_pred_ts"] = check_df.apply(
            lambda x: (
                [
                    (
                        x["daily_pred_ts"][i]
                        if x["daily_pred_ts"][i] <= x["all_channel_daily_pred_ts"][i]
                        else x["all_channel_daily_pred_ts"][i]
                    )
                    for i in range(daily_forecast_length)
                ]
                if x["channel_id"] != "0"
                and (x["brand_name"], x["platform"], x["product_id"], x["variant_id"])
                in all_channel_df.index
                else x["daily_pred_ts"]
            ),
            axis=1,
        )
    final_df = pd.concat([no_check_df, check_df]).reset_index(drop=True)
    return final_df
