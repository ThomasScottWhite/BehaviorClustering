# %%%
import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


def cluster_videos(
    meta_data: dict,
    bout_frames,
    append_results_to_df=True,
) -> dict:
    start_index = 0
    tsne_input = []
    for video in meta_data["videos"].values():
        video["tsne_start"] = start_index
        video["tsne_end"] = start_index + video["df"]["Group"].nunique()
        start_index = video["tsne_end"]

        df_xy = video["df"].filter(
            regex="(_x|_y)$"
        )  # This needs to be done differently

        new_values = df_xy.values.reshape(-1, len(df_xy.columns) * bout_frames)
        tsne_input.append(new_values)
    tsne_input = np.vstack(tsne_input)

    tsne_results = TSNE(
        n_components=2, perplexity=30, method="barnes_hut", random_state=42
    ).fit_transform(tsne_input)
    kmeans_labels = KMeans(n_clusters=8, random_state=42).fit_predict(tsne_results)

    # Create DataFrame for t-SNE results and assign cluster labels
    tsne_df = pd.DataFrame(tsne_results, columns=["TSNE_1", "TSNE_2"])
    tsne_df["Cluster"] = kmeans_labels

    for video in meta_data["videos"].values():
        for index, group in video["df"].groupby("Group"):
            if "Seconds" in group.columns:
                tsne_df.loc[video["tsne_start"] + index, "Seconds"] = group.iloc[0][
                    "Seconds"
                ]
                tsne_df.loc[video["tsne_start"] + index, "Video_Frame"] = group[
                    "Frame"
                ].iloc[0]

    # Adds clustering results to each group in the dataframe
    for video_name, video in meta_data["videos"].items():
        tsne_df.loc[video["tsne_start"] : video["tsne_end"], "video"] = video_name
        tsne_df.loc[video["tsne_start"] : video["tsne_end"], "trial"] = video["trial"]
        if append_results_to_df:
            video_results = kmeans_labels[video["tsne_start"] : video["tsne_end"]]
            video["df"]["Cluster"] = np.repeat(video_results, bout_frames)

    meta_data["tsne_results"] = tsne_df

    return meta_data
