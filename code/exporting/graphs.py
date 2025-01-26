import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import pickle


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
        plt.figure(figsize=(60, 15))
    except Exception as e:
        print(e)
        return

    ax = sns.heatmap(
        heatmap_data,
        cmap="viridis",
        linewidths=0,
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


def graph_pie(meta_data):

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
        video_df["Trial"] = video["trial"]
        grouped_dfs.append(video_df)

    df = pd.concat(grouped_dfs, ignore_index=True)

    for group_name, group_df in df.groupby("Trial"):
        for video_name, video_df in group_df.groupby("Video"):

            cluster_counts = video_df["Cluster"].value_counts(normalize=True) * 100

            def autopct_format(pct):
                total = sum(cluster_counts)
                count = int(round(pct * total / 100))
                return f"{count}%"

            labels = [f"Cluster {int(label)}" for label in cluster_counts.index]

            # Plotting the updated pie chart
            plt.figure(figsize=(8, 8))
            plt.pie(
                cluster_counts, labels=labels, autopct=autopct_format, startangle=140
            )
            plt.title(f"Percentage of Each Cluster in {video_name}")

            graph_path = Path(meta_data["output_path"]) / "graphs" / group_name
            os.makedirs(graph_path, exist_ok=True)
            plt.savefig(graph_path / f"{video_name}.png")

    for group_name, group_df in df.groupby("Trial"):

        cluster_counts = group_df["Cluster"].value_counts(normalize=True) * 100

        def autopct_format(pct):
            total = sum(cluster_counts)
            count = int(round(pct * total / 100))
            return f"{count}%"

        labels = [f"Cluster {int(label)}" for label in cluster_counts.index]

        # Plotting the updated pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(cluster_counts, labels=labels, autopct=autopct_format, startangle=140)
        plt.title(f"Percentage of Each Cluster in {group_name}")
        graph_path = Path(meta_data["output_path"]) / "graphs" / group_name
        os.makedirs(graph_path, exist_ok=True)
        plt.savefig(graph_path / f"{group_name}.png")


if __name__ == "__main__":
    # Path to the file
    file_path = "/home/thomas/washu/behavior_clustering/outputs/fear_voiding_8_frames_reduced4x_pre_group_rotated/meta_data.pkl"

    # Open the file
    with open(file_path, "rb") as file:
        meta_data = pickle.load(file)
