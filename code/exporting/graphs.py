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


def tsne_plot(meta_data):
    tsne_df = meta_data["tsne_results"]

    plt.figure(figsize=(10, 8))
    for trial in tsne_df["trial"].unique():
        cluster_data = tsne_df[tsne_df["trial"] == trial]
        plt.scatter(
            cluster_data["TSNE_1"], cluster_data["TSNE_2"], label=trial, alpha=0.6
        )

    plt.xlabel("TSNE_1")
    plt.ylabel("TSNE_2")
    plt.title("t-SNE with K-means Clustering")
    plt.legend()
    plt.savefig(f"{meta_data['output_path']}/tsne_results_by_video.png")

    plt.figure(figsize=(10, 8))
    for cluster in range(tsne_df["Cluster"].nunique()):
        cluster_data = tsne_df[tsne_df["Cluster"] == cluster]
        plt.scatter(
            cluster_data["TSNE_1"],
            cluster_data["TSNE_2"],
            label=f"Cluster {cluster}",
            alpha=0.6,
        )

    plt.xlabel("TSNE_1")
    plt.ylabel("TSNE_2")
    plt.title("t-SNE with K-means Clustering")
    plt.legend()
    plt.savefig(f"{meta_data['output_path']}/tsne_results_by_cluster.png")


def create_heatmap_plot(meta_data):

    event_dict = {"Cluster": "mean"}
    for event in meta_data["event_columns"]:
        event_dict[event] = "max"

    # if "Seconds" in meta_data["videos"].values[0]["df"]:
    #     event_dict["Seconds"] = "mean"

    event_dict["Cluster"] = "mean"

    print(event_dict)
    grouped_dfs = []
    for video_name, video in meta_data["videos"].items():

        if not event_dict:  # Check if event_dict is empty
            # Perform groupby without aggregation
            video_df = (
                video["df"]
                .groupby("Group", as_index=False)
                .apply(lambda x: x)  # No aggregation, just keep the grouped data
                .assign(Index=lambda x: x.index)
            )
        else:
            # Perform groupby with aggregation
            video_df = (
                video["df"]
                .groupby("Group", as_index=False)
                .agg(event_dict)
                .assign(Index=lambda x: x.index)
            )

        video_df["Video"] = video_name

        grouped_dfs.append(video_df)

    combined_df = pd.concat(grouped_dfs, ignore_index=True)

    print(combined_df)
    try:
        heatmap_data = combined_df.pivot_table(
            index="Video",  # Combined Trial and Video as the row index
            columns="Group",  # Group as the x-axis
            values="Cluster",  # Cluster value for the heatmap
            aggfunc="mean",  # Aggregating by mean (or other suitable method)
        ).fillna(
            0
        )  # Fill missing values with 0 if necessary
        plt.figure(figsize=(30, 8))
    except Exception as e:
        print(e)
        return

    ax = sns.heatmap(
        heatmap_data,
        cmap="viridis",
        linewidths=0.1,
        linecolor="gray",
        cbar_kws={"label": "Cluster"},
    )
    if meta_data["experiment"] == "Fang":
        for _, row in combined_df.iterrows():
            trial_video_idx = heatmap_data.index.tolist().index(row["Video"])
            group = row["Group"]  # Get the group (x-axis)

            if row["Light On"]:
                ax.plot(
                    group + 0.5,
                    trial_video_idx + 0.5,
                    "rx",
                    markersize=12,
                    label="Light",
                )

    if meta_data["experiment"] == "fear_voiding":
        for _, row in combined_df.iterrows():
            trial_video_idx = heatmap_data.index.tolist().index(row["Video"])
            group = row["Group"]  # Get the group (x-axis)

            if row["Is_Voiding"]:
                ax.plot(
                    group + 0.5,
                    trial_video_idx + 0.5,
                    "rx",
                    markersize=12,
                    label="Is_Voiding",
                )
            if row["Shock_Start"]:
                ax.plot(
                    group + 0.5,
                    trial_video_idx + 0.5,
                    "bx",
                    markersize=12,
                    label="Shock_Start",
                )
            if row["Shock_End"]:
                ax.plot(
                    group + 0.5,
                    trial_video_idx + 0.5,
                    "yx",
                    markersize=12,
                    label="Shock_End",
                )
            if row["Tone_Start"]:
                ax.plot(
                    group + 0.5,
                    trial_video_idx + 0.5,
                    "gx",
                    markersize=12,
                    label="Tone_Start",
                )
            if row["Tone_End"]:
                ax.plot(
                    group + 0.5,
                    trial_video_idx + 0.5,
                    "mx",
                    markersize=12,
                    label="Tone_End",
                )

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))

    ax.legend(
        unique_labels.values(),
        unique_labels.keys(),
        bbox_to_anchor=(1.10, 1),
        loc="upper left",
        borderaxespad=0,
        title="Events",
    )
    # Adjust labels and title
    plt.xlabel("Group")
    plt.ylabel("Trial - Video")
    plt.title("Heatmap of Clusters with Event Markers Across Trial and Video")
    plt.savefig(f"{meta_data['output_path']}/cluster_heatmap.png")
