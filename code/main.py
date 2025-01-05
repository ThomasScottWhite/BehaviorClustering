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
from exporting import graphs
import cv2
import pandas as pd
import cv2

bout_frames = 4


def load_metadata(file_path):
    with open(file_path, "r") as file:
        meta_data = json.load(file)

    for video in meta_data["videos"].values():
        video["df"] = pd.read_csv(video["csv_path"], index_col=0)
    return meta_data


def add_bouts(meta_data):
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
    for video in meta_data["videos"].values():
        video["df"] = video["df"].iloc[::factor]
    return meta_data

def increase_dfs(meta_data, factor=4):
    for video in meta_data["videos"].values():
        video["df"] = video["df"].loc[np.repeat(video["df"].index, factor)].reset_index(drop=True)
    return meta_data

def main():
    metadatasource = (
        "/home/thomas/washu/behavior_clustering/data/Fang/processed_csvs/metadata.json"
    )
    meta_data = load_metadata(metadatasource)
    meta_data = reduce_dfs(meta_data)
    meta_data["output_path"] = f'./../outputs/{meta_data["experiment"]}/'

    if os.path.exists(meta_data["output_path"]):
        shutil.rmtree(meta_data["output_path"])
    os.makedirs(meta_data["output_path"], exist_ok=True)
    shutil.copy(metadatasource, f'{meta_data["output_path"]}metadata.json')


    meta_data = add_bouts(meta_data)
    meta_data = tsne.cluster_videos(meta_data, bout_frames=bout_frames)

    meta_data = increase_dfs(meta_data)
    save_csvs(meta_data)
    save_tsne_results(meta_data)

    graphs.tsne_plot(meta_data)
    graphs.create_heatmap_plot(meta_data)

    return meta_data


meta_data = main()
