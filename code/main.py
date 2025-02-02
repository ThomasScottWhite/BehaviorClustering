# %%%
import json
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import shutil
import seaborn as sns
from clustering import tsne
from exporting import graphs, videos
import cv2
import pandas as pd
import cv2
import argparse
import pickle


def load_metadata(file_path):
    with open(file_path, "r") as file:
        meta_data = json.load(file)

    for video in meta_data["videos"].values():
        video["df"] = pd.read_csv(video["csv_path"], index_col=0)
    return meta_data


def rotate_points_global(df, ref_points=["Nose", "Spine1", "Hipbone"]):
    """
    Rotate data to align the plane defined by 3 points (e.g., nose, spine, hip) with the x-axis.
    """
    # Extract coordinates for the 3 reference points
    p1 = df[[f"{ref_points[0]}_x", f"{ref_points[0]}_y"]].values
    p2 = df[[f"{ref_points[1]}_x", f"{ref_points[1]}_y"]].values
    p3 = df[[f"{ref_points[2]}_x", f"{ref_points[2]}_y"]].values

    # Compute the primary axis using PCA on the 3 points
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca.fit(np.vstack([p1, p2, p3]))
    main_axis = pca.components_[0]  # Direction of maximum variance

    # Calculate rotation angle to align main_axis with x-axis
    angle = np.arctan2(main_axis[1], main_axis[0])
    cos_theta, sin_theta = np.cos(-angle), np.sin(-angle)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    # Rotate all points
    rotated_data = {}
    for col in df.columns:
        if "_x" in col:
            y_col = col.replace("_x", "_y")
            x_vals = df[col].values
            y_vals = df[y_col].values
            rotated_coords = rotation_matrix @ np.vstack([x_vals, y_vals])
            rotated_data[col] = rotated_coords[0]
            rotated_data[y_col] = rotated_coords[1]
        else:
            rotated_data[col] = df[col]

    return pd.DataFrame(rotated_data)


def rotate_all(meta_data, ref_points=["Nose", "Spine1", "Hipbone"]):
    for video in meta_data["videos"].values():
        video["df"] = rotate_points_global(video["df"], ref_points)
    return meta_data


def add_bouts(meta_data, bout_frames=4):
    for video in meta_data["videos"].values():

        total_frames = len(video["df"])
        complete_bouts = total_frames // bout_frames
        cutoff = complete_bouts * bout_frames
        video["df"] = video["df"].iloc[:cutoff].reset_index(drop=True)

        video["df"]["Group"] = video["df"].index // bout_frames

    return meta_data


def save_csvs(meta_data):
    for video_name, video in meta_data["videos"].items():
        os.makedirs(f'{meta_data['output_path']}/csvs/{video["trial"]}', exist_ok=True)
        video["df"].to_csv(
            f'{meta_data['output_path']}/csvs/{video["trial"]}/{video_name}.csv'
        )


def save_tsne_results(meta_data):
    meta_data["tsne_results"].to_csv(
        f"{meta_data['output_path']}/csvs/tsne_results.csv"
    )


def reduce_dfs(meta_data, factor=4):
    event_columns = meta_data["event_columns"]
    for video in meta_data["videos"].values():
        df = video["df"]

        # Create a new DataFrame to store the reduced data
        reduced_df = df.iloc[::factor].copy()

        # Iterate over each event column and apply the logic
        for col in event_columns:
            # Create a boolean mask for each chunk of `factor` rows
            mask = (
                df[col]
                .rolling(window=factor, min_periods=1)
                .max()
                .iloc[::factor]
                .astype(bool)
            )
            # Update the event column in the reduced DataFrame
            reduced_df[col] = mask.values

        # Update the video's DataFrame
        video["df"] = reduced_df

    return meta_data


def increase_dfs(meta_data, factor=4):
    for video in meta_data["videos"].values():
        video["df"] = (
            video["df"].loc[np.repeat(video["df"].index, factor)].reset_index(drop=True)
        )
    return meta_data


def make_output_directory(meta_data, metadatasource, output_path_name):
    output_path = f"./../outputs/{output_path_name}/"

    counter = 0
    while os.path.exists(output_path):
        counter += 1
        output_path = f"./../outputs/{output_path_name}_{counter}/"

    meta_data["output_path"] = output_path
    os.makedirs(meta_data["output_path"], exist_ok=True)
    shutil.copy(metadatasource, f'{meta_data["output_path"]}metadata.json')
    return meta_data


def pickle_meta_data(meta_data):

    with open(f"{meta_data['output_path']}/meta_data.pkl", "wb") as f:
        pickle.dump(meta_data, f)


import json
import os


def main(
    metadatasource,
    bout_frames=16,
    reduction_factor=4,
    clustering="tsne",
    rotation=True,
    video=False,
):
    # Metadata source

    print("Step 1: Load and process metadata")
    # Load and process metadata
    meta_data = load_metadata(metadatasource)
    meta_data = reduce_dfs(meta_data, reduction_factor)

    if rotation:
        if meta_data["experiment"] == "fear_voiding":
            meta_data = rotate_all(meta_data)
        else:
            meta_data = rotate_all(meta_data)

    output_path_name = f"{meta_data["experiment"]}_{bout_frames}_frames"
    if reduction_factor != 1:
        output_path_name += f"_reduced{reduction_factor}x"
    output_path_name += f"_{clustering}"
    if rotation:
        output_path_name += "_rotated"

    meta_data = make_output_directory(meta_data, metadatasource, output_path_name)

    # Save parameters to a config file in the output directory
    config_path = os.path.join(meta_data["output_path"], "args.json")
    config = {
        "bout_frames": bout_frames,
        "reduction_factor": reduction_factor,
        "clustering": clustering,
        "rotation": rotation,
    }
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)

    print("Step 2: TSNE Clustering")
    # Add bouts and cluster videos
    meta_data = add_bouts(meta_data, bout_frames=bout_frames)

    if clustering == "tsne":
        meta_data = tsne.cluster_videos(meta_data, bout_frames=bout_frames)
    elif clustering == "pca":
        meta_data = tsne.cluster_videos_with_pca(meta_data, bout_frames=bout_frames)
    elif clustering == "pre_group":
        meta_data = tsne.cluster_videos_pre_group(meta_data, bout_frames=bout_frames)

    # if pca:
    #     meta_data = tsne.cluster_videos_with_pca(meta_data, bout_frames=bout_frames)
    # else:
    #     meta_data = tsne.cluster_videos_pre_group(meta_data, bout_frames=bout_frames)

    print("Step 3: Generate outputs")
    # Revert dataframes and save outputs
    meta_data = increase_dfs(meta_data, reduction_factor)
    save_csvs(meta_data)
    save_tsne_results(meta_data)
    pickle_meta_data(meta_data)

    # print("Step 4: Generate graphs and videos")
    # # Generate graphs and videos
    # graphs.graph_all(meta_data)

    # if video:
    #     videos.generate_videos(meta_data["output_path"])


if __name__ == "__main__":

    metadatasource = "/home/thomas/washu/behavior_clustering/data/fear_voiding_velocity_reduced/processed_csvs/metadata.json"

    bout_frames = 8
    reduction_factor = 4
    clustering = "pca"
    rotation = True
    video = False

    main(
        metadatasource=metadatasource,
        bout_frames=bout_frames,
        reduction_factor=reduction_factor,
        clustering=clustering,
        rotation=rotation,
        video=video,
    )

    # # Parse command-line arguments for bout_frames and reduction_factor
    # parser = argparse.ArgumentParser(
    #     description="Process metadata and generate outputs."
    # )
    # parser.add_argument(
    #     "--bout_frames", type=int, default=16, help="Number of frames in a bout"
    # )
    # parser.add_argument(
    #     "--reduction_factor",
    #     type=int,
    #     default=4,
    #     help="Factor by which dataframes are reduced",
    # )
    # parser.add_argument(
    #     "--pca",
    #     type=bool,
    #     default=True,
    #     help="Factor by which dataframes are reduced",
    # )
    # args = parser.parse_args()
    # main(
    #     bout_frames=args.bout_frames,
    #     reduction_factor=args.reduction_factor,
    #     pca=args.pca,
    # )

# %%
