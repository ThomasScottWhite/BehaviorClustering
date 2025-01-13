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


def load_metadata(file_path):
    with open(file_path, "r") as file:
        meta_data = json.load(file)

    for video in meta_data["videos"].values():
        video["df"] = pd.read_csv(video["csv_path"], index_col=0)
    return meta_data


def rotate_points_global(df, ref_point1, ref_point2):
    """
    Rotate the positional data so that the vector from ref_point1 to ref_point2 is aligned with the x-axis.

    Args:
        df (pd.DataFrame): DataFrame containing positional data normalized such that the nose is at (0, 0).
        ref_point1 (str): The name of the first reference point (e.g., 'spinal_front').
        ref_point2 (str): The name of the second reference point (e.g., 'spinal_low').

    Returns:
        pd.DataFrame: DataFrame with rotated coordinates.
    """
    # Extract the reference vector
    vec_x = df[f"{ref_point2}_x"] - df[f"{ref_point1}_x"]
    vec_y = df[f"{ref_point2}_y"] - df[f"{ref_point1}_y"]

    # Calculate the angle to rotate (relative to the x-axis)
    angles = np.arctan2(vec_y, vec_x)
    mean_angle = np.mean(
        angles
    )  # Use the average angle if working with multiple frames

    # Define the rotation matrix
    cos_theta = np.cos(-mean_angle)
    sin_theta = np.sin(-mean_angle)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    # Rotate all points
    rotated_data = {}
    for col in df.columns:
        if "_x" in col:
            # Get y-coordinate pair
            y_col = col.replace("_x", "_y")

            # Extract x and y coordinates
            x_vals = df[col]
            y_vals = df[y_col]

            # Apply the rotation
            rotated_coords = np.dot(rotation_matrix, np.vstack([x_vals, y_vals]))
            rotated_data[col] = rotated_coords[0, :]
            rotated_data[y_col] = rotated_coords[1, :]
        else:
            # Keep non-coordinate data unchanged
            rotated_data[col] = df[col]

    return pd.DataFrame(rotated_data)


def rotate_all(meta_data, ref_point1="spinal_front", ref_point2="spinal_low"):
    for video in meta_data["videos"].values():
        video["df"] = rotate_points_global(video["df"], ref_point1, ref_point2)

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
        os.makedirs(f'{meta_data['output_path']}/{video["trial"]}', exist_ok=True)
        video["df"].to_csv(
            f'{meta_data['output_path']}/{video["trial"]}/{video_name}.csv'
        )


def save_tsne_results(meta_data):
    meta_data["tsne_results"].to_csv(f"{meta_data['output_path']}/tsne_results.csv")


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


def make_output_directory(meta_data, metadatasource):
    output_path = f'./../outputs/{meta_data["experiment"]}/'
    counter = 0
    while os.path.exists(output_path):
        counter += 1
        output_path = f'./../outputs/{meta_data["experiment"]}_{counter}/'

    meta_data["output_path"] = output_path
    os.makedirs(meta_data["output_path"], exist_ok=True)
    shutil.copy(metadatasource, f'{meta_data["output_path"]}metadata.json')
    return meta_data


import json
import os


def main(bout_frames=16, reduction_factor=4):
    # Metadata source
    metadatasource = (
        "/home/thomas/washu/behavior_clustering/data/Fang/processed_csvs/metadata.json"
    )

    print("Step 1: Load and process metadata")
    # Load and process metadata
    meta_data = load_metadata(metadatasource)
    meta_data = reduce_dfs(meta_data, reduction_factor)
    if meta_data["experiment"] == "fear_voiding":
        meta_data = rotate_all(meta_data, ref_point1="shoulder", ref_point2="Hipbone")
    else:
        meta_data = rotate_all(meta_data)

    meta_data = make_output_directory(meta_data, metadatasource)

    # Save parameters to a config file in the output directory
    config_path = os.path.join(meta_data["output_path"], "args.json")
    config = {
        "bout_frames": bout_frames,
        "reduction_factor": reduction_factor,
    }
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)

    print("Step 2: TSNE Clustering")
    # Add bouts and cluster videos
    meta_data = add_bouts(meta_data, bout_frames=bout_frames)
    meta_data = tsne.cluster_videos_with_pca(meta_data, bout_frames=bout_frames)

    print("Step 3: Generate outputs")
    # Revert dataframes and save outputs
    meta_data = increase_dfs(meta_data, reduction_factor)
    save_csvs(meta_data)
    save_tsne_results(meta_data)

    print("Step 4: Generate graphs and videos")
    # Generate graphs and videos
    graphs.tsne_plot(meta_data)
    graphs.create_heatmap_plot(meta_data)
    videos.generate_videos(meta_data["output_path"])


if __name__ == "__main__":

    # Parse command-line arguments for bout_frames and reduction_factor
    parser = argparse.ArgumentParser(
        description="Process metadata and generate outputs."
    )
    parser.add_argument(
        "--bout_frames", type=int, default=16, help="Number of frames in a bout"
    )
    parser.add_argument(
        "--reduction_factor",
        type=int,
        default=4,
        help="Factor by which dataframes are reduced",
    )

    args = parser.parse_args()
    main(bout_frames=args.bout_frames, reduction_factor=args.reduction_factor)
