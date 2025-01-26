# %%%
import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


def cluster_videos_with_frame_recluster(
    meta_data: dict,
    bout_frames,
    append_results_to_df=True,
) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    import numpy as np
    import pandas as pd

    start_index = 0
    tsne_input = []
    frame_clusters = []  # To store frame-wise clustering results

    for video in meta_data["videos"].values():
        video["tsne_start"] = start_index
        video["tsne_end"] = start_index + video["df"]["Group"].nunique()
        start_index = video["tsne_end"]

        # Extract frame-wise feature data
        df_xy = video["df"][meta_data["data_columns"]]
        frame_values = df_xy.values

        # Perform clustering on individual frames
        kmeans_frame = KMeans(n_clusters=8, random_state=42)
        frame_labels = kmeans_frame.fit_predict(frame_values)
        frame_clusters.append(frame_labels)

        # Reshape for t-SNE input (combine frames into bout-level data)
        bout_values = frame_values.reshape(-1, len(df_xy.columns) * bout_frames)
        tsne_input.append(bout_values)

    # Combine all bout-level data for t-SNE
    tsne_input = np.vstack(tsne_input)

    # Apply t-SNE on bout-level data
    tsne_results = TSNE(
        n_components=2, perplexity=30, method="barnes_hut", random_state=42
    ).fit_transform(tsne_input)

    # Perform clustering on t-SNE results
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

            # Append frame-wise clusters to the dataframe
            video["df"]["Frame_Cluster"] = frame_clusters.pop(0)

    meta_data["tsne_results"] = tsne_df

    return meta_data


def cluster_videos_pre_group(
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
        df_xy = video["df"][meta_data["data_columns"]]

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


def cluster_videos_with_pca(
    meta_data: dict,
    bout_frames,
    append_results_to_df=True,
    pca_variance=0.95,  # Fraction of variance to retain in PCA
) -> dict:
    start_index = 0
    tsne_input = []

    # Prepare t-SNE input data
    for video in meta_data["videos"].values():
        video["tsne_start"] = start_index
        video["tsne_end"] = start_index + video["df"]["Group"].nunique()
        start_index = video["tsne_end"]

        # Extract (_x, _y) columns and reshape per bout_frames
        df_xy = video["df"][meta_data["data_columns"]]

        new_values = df_xy.values.reshape(-1, len(df_xy.columns) * bout_frames)
        tsne_input.append(new_values)

    tsne_input = np.vstack(tsne_input)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=pca_variance, svd_solver="full")
    pca_input = pca.fit_transform(tsne_input)
    print(f"PCA reduced dimensions from {tsne_input.shape[1]} to {pca_input.shape[1]}")

    # Apply t-SNE
    tsne_results = TSNE(
        n_components=2, perplexity=30, method="barnes_hut", random_state=42
    ).fit_transform(pca_input)

    # Cluster using k-means
    kmeans_labels = KMeans(n_clusters=8, random_state=42).fit_predict(tsne_results)

    # Create DataFrame for t-SNE results and assign cluster labels
    tsne_df = pd.DataFrame(tsne_results, columns=["TSNE_1", "TSNE_2"])
    tsne_df["Cluster"] = kmeans_labels

    # Add time and frame data for interpretability
    for video in meta_data["videos"].values():
        for index, group in video["df"].groupby("Group"):
            if "Seconds" in group.columns:
                tsne_df.loc[video["tsne_start"] + index, "Seconds"] = group.iloc[0][
                    "Seconds"
                ]
                tsne_df.loc[video["tsne_start"] + index, "Video_Frame"] = group[
                    "Frame"
                ].iloc[0]

    # Add clustering results to each video's DataFrame
    for video_name, video in meta_data["videos"].items():
        tsne_df.loc[video["tsne_start"] : video["tsne_end"], "video"] = video_name
        tsne_df.loc[video["tsne_start"] : video["tsne_end"], "trial"] = video["trial"]
        if append_results_to_df:
            video_results = kmeans_labels[video["tsne_start"] : video["tsne_end"]]
            video["df"]["Cluster"] = np.repeat(video_results, bout_frames)

    # Store results in meta_data
    meta_data["tsne_results"] = tsne_df

    return meta_data
