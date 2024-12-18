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

bout_frames = 120

# # %%%

# for video in fear_voiding["videos"].values():
#     video["df"] = pd.read_csv(video["csv_path"])

# for video in fear_voiding["videos"].values():

#     total_frames = len(video["df"]) 
#     complete_bouts = total_frames // bout_frames
#     cutoff = complete_bouts * bout_frames
#     video["df"] = video["df"].iloc[:cutoff].reset_index(drop=True)
    
#     video["df"]['Group'] = video["df"].index // bout_frames


def cluster_videos(meta_data : dict, append_results_to_df = True) -> dict :
    start_index = 0
    tsne_input = []
    for video in meta_data["videos"].values():
        video["tsne_start"] = start_index
        video["tsne_end"] = start_index + video["df"]["Group"].nunique()
        start_index = video["tsne_end"]

        df_xy = video["df"].filter(regex='(_x|_y)$') # This needs to be done differently
        
        new_values = df_xy.values.reshape(-1, len(df_xy.columns) * bout_frames)
        tsne_input.append(new_values)
    tsne_input = np.vstack(tsne_input)

    tsne_results = TSNE(n_components=2, perplexity=30, method='barnes_hut', random_state=42).fit_transform(tsne_input)
    kmeans_labels = KMeans(n_clusters=8, random_state=42).fit_predict(tsne_results)

    # Create DataFrame for t-SNE results and assign cluster labels
    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE_1', 'TSNE_2'])
    tsne_df['Cluster'] = kmeans_labels

    # Adds clustering information to metadata for results

    # Adds clustering results to each group in the dataframe
    for video_name, video in meta_data["videos"].items():
        tsne_df.loc[video["tsne_start"]:video["tsne_end"], "video"] = video_name
        tsne_df.loc[video["tsne_start"]:video["tsne_end"], "trial"] = video["trial"]
        if append_results_to_df:
            video_results = kmeans_labels[video["tsne_start"]:video["tsne_end"]]
            video["df"]["Cluster"] = np.repeat(video_results, bout_frames) 

    meta_data["tsne_results"] = tsne_df

    return meta_data

def load_metadata(file_path):
    with open(file_path, 'r') as file:
        meta_data = json.load(file)

    for video in meta_data["videos"].values():
        video["df"] = pd.read_csv(video["csv_path"])
    return meta_data

def add_bouts(meta_data):
    for video in meta_data["videos"].values():

        total_frames = len(video["df"]) 
        complete_bouts = total_frames // bout_frames
        cutoff = complete_bouts * bout_frames
        video["df"] = video["df"].iloc[:cutoff].reset_index(drop=True)
        
        video["df"]['Group'] = video["df"].index // bout_frames

    return meta_data
# %%%
def main():
    meta_data = load_metadata('/home/thomas/washu/behavior_clustering/data/fear_voiding/metadata.json')
    
    meta_data = add_bouts(meta_data)
    meta_data = cluster_videos(meta_data)


    return meta_data

# %%
meta_data = main()
# %%
output_path = f'./../outputs/{meta_data["experiment"]}/' # Fix
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path, exist_ok=True)

# os.makedirs(output_path+"csvs/", exist_ok=True)
# %%%
for video_name, video in meta_data["videos"].items():
    os.makedirs(f'{output_path}/{video["trial"]}', exist_ok=True)
    video["df"].to_csv(f'{output_path}/{video["trial"]}/{video_name}.csv')
    meta_data["tsne_results"].to_csv(f'{output_path}/tsne_results.csv')
# %%
