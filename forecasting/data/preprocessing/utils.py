from typing import List, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize
from tqdm import tqdm

tqdm.pandas()


def find_similar_items(
    all_item_df: pd.DataFrame,
    old_item_df: pd.DataFrame,
    new_item_df: pd.DataFrame,
    item_texts: List,
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 5,
):
    # Load SentenceTransformers model
    sentence_model = SentenceTransformer(model_name)

    # Encode the texts into vectors
    vectors = sentence_model.encode(item_texts)

    # Create a list for the ids of all items available
    ids = all_item_df["id"].unique().tolist()

    # Create a dictionary to store the vectors by each id
    vector_dict = dict(zip(ids, vectors))

    # Get list of ids for new and old items
    new_item_ids = new_item_df["id"].unique().tolist()
    old_item_ids = old_item_df["id"].unique().tolist()

    # Create a list containing the vector of each new items from the vector_dict
    new_vectors = [vector_dict.get(i) for i in new_item_ids]
    # Create a list containing the vector of each old items from the vector_dict
    old_vectors = [vector_dict.get(i) for i in old_item_ids]

    # Normalize the vectors
    new_vectors = normalize(new_vectors, axis=1)
    old_vectors = normalize(old_vectors, axis=1)

    # Calculate cosine similarity between new items and old items vectors
    cos_sim = linear_kernel(new_vectors, old_vectors)

    # Get top k old items with highest cosine sim for every new item
    top_results = np.argsort(-cos_sim, axis=1)[:, :top_k]

    # Create a dictionary to store the results
    similar_item_dict = {}

    # Iterate through every new item
    for i, new_item_id in enumerate(new_item_ids):
        # Get the ids of top k old items from the result matrix
        similar_items = [old_item_ids[j] for j in top_results[i]]

        # Insert into the dictionary the key is the id of the new item and value is the list of old items
        similar_item_dict[new_item_id] = similar_items

    return similar_item_dict


def extend_df(
    df: pd.DataFrame,
    value_col: str,
    avg_sim_df: Optional[pd.DataFrame],
    required_len: int,
    direction: str = "past",
    freq: str = "W-MON",
    fill_method: str = "fill_avg_sim",
):
    if isinstance(value_col, str):
        value_col = [value_col]
    meta_cols = [col for col in df.columns.tolist() if col not in value_col + ["date"]]
    last_date = df["date"].max()
    date_range_df = (
        df.groupby("id")["date"]
        .min()
        .to_frame(name="start_date")
        .join(df.groupby("id")["date"].count().to_frame(name="periods") + required_len)
    )
    id_list = df["id"].unique().tolist()

    if direction == "past":
        id_date_list = [
            (item_id, date)
            for item_id in id_list
            for date in pd.date_range(end=last_date, periods=required_len, freq=freq)
        ]
    else:
        id_date_list = [
            (item_id, date)
            for item_id in id_list
            for date in pd.date_range(
                start=date_range_df.loc[item_id]["start_date"],
                periods=date_range_df.loc[item_id]["periods"],
                freq=freq,
            )
        ]

    index_list = pd.MultiIndex.from_tuples(id_date_list, names=["id", "date"])
    df_extended = df.set_index(["id", "date"]).reindex(index_list).reset_index()

    if direction == "past":
        if fill_method == "fill_avg_sim":
            df_extended.set_index(["id", "date"], inplace=True)
            df_extended[value_col] = df_extended[value_col].fillna(
                avg_sim_df.set_index(["id", "date"])[value_col]
            )
            df_extended.reset_index(inplace=True)
        elif fill_method == "fill_zero":
            df_extended[value_col] = df_extended[value_col].fillna(0)
        df_extended[meta_cols] = df_extended[meta_cols].bfill()
    else:
        df_extended[value_col] = df_extended[value_col].ffill()
        df_extended[meta_cols] = df_extended[meta_cols].ffill()

    return df_extended
