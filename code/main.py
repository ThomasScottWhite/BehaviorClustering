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

bout_frames = 120

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

    for video in meta_data["videos"].values():
        for index, group in video["df"].groupby("Group"):
            tsne_df.loc[video["tsne_start"] + index, "Seconds"] = group.iloc[0]["Seconds"]
            tsne_df.loc[video["tsne_start"] + index, "Video_Frame"] = group["Frame"].iloc[0]


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
        video["df"] = pd.read_csv(video["csv_path"], index_col=0)
    return meta_data

def add_bouts(meta_data):
    for video in meta_data["videos"].values():

        total_frames = len(video["df"]) 
        complete_bouts = total_frames // bout_frames
        cutoff = complete_bouts * bout_frames
        video["df"] = video["df"].iloc[:cutoff].reset_index(drop=True)
        
        video["df"]['Group'] = video["df"].index // bout_frames

    return meta_data

def save_csvs(meta_data):
    for video_name, video in meta_data["videos"].items():
        os.makedirs(f'{meta_data['output_path']}/{video["trial"]}', exist_ok=True)
        video["df"].to_csv(f'{meta_data['output_path']}/{video["trial"]}/{video_name}.csv')

def save_tsne_results(meta_data):
    meta_data["tsne_results"].to_csv(f'{meta_data['output_path']}/tsne_results.csv')

def tsne_plot(meta_data):
    tsne_df = meta_data["tsne_results"]

    plt.figure(figsize=(10, 8))
    for trial in tsne_df['trial'].unique():
        cluster_data = tsne_df[tsne_df['trial'] == trial]
        plt.scatter(cluster_data['TSNE_1'], cluster_data['TSNE_2'], label=trial, alpha=0.6)

    plt.xlabel('TSNE_1')
    plt.ylabel('TSNE_2')
    plt.title('t-SNE with K-means Clustering')
    plt.legend()
    plt.savefig(f'{meta_data['output_path']}/tsne_results_by_video.png')

    plt.figure(figsize=(10, 8))
    for cluster in range(tsne_df['Cluster'].nunique()):
        cluster_data = tsne_df[tsne_df['Cluster'] == cluster]
        plt.scatter(cluster_data['TSNE_1'], cluster_data['TSNE_2'], label=f'Cluster {cluster}', alpha=0.6)

    plt.xlabel('TSNE_1')
    plt.ylabel('TSNE_2')
    plt.title('t-SNE with K-means Clustering')
    plt.legend()
    plt.savefig(f'{meta_data['output_path']}/tsne_results_by_cluster.png')

def create_heatmap_plot(meta_data):

    event_dict = {}
    for event in meta_data['event_columns']:
        event_dict[event] = 'max'
    event_dict["Seconds"] = 'mean'
    event_dict['Cluster'] = 'mean'

    grouped_dfs = []
    for video_name, video in meta_data["videos"].items():
        video_df = video["df"].groupby('Group', as_index=False).agg(event_dict).assign(Index=lambda x: x.index)
        video_df['Video'] = video_name

        grouped_dfs.append(video_df)


    combined_df = pd.concat(grouped_dfs, ignore_index=True)

    heatmap_data = combined_df.pivot_table(
        index='Video',  # Combined Trial and Video as the row index
        columns='Group',      # Group as the x-axis
        values='Cluster',     # Cluster value for the heatmap
        aggfunc='mean'        # Aggregating by mean (or other suitable method)
    ).fillna(0)  # Fill missing values with 0 if necessary
    plt.figure(figsize=(30, 8))

    ax = sns.heatmap(
        heatmap_data, 
        cmap='viridis', 
        linewidths=0.1, 
        linecolor='gray', 
        cbar_kws={'label': 'Cluster'}
    )

    for _, row in combined_df.iterrows():
        # Get the row index for the combined Trial and Video
        trial_video_idx = heatmap_data.index.tolist().index(row['Video'])
        group = row['Group']  # Get the group (x-axis)
        
        # Add 'X' markers at the respective trial-video and group positions
        if row['Is_Voiding']:
            ax.plot(group + 0.5, trial_video_idx + 0.5, 'rx', markersize=12, label='Is_Voiding')
        if row['Shock_Start']:
            ax.plot(group + 0.5, trial_video_idx + 0.5, 'bx', markersize=12, label='Shock_Start')
        if row['Shock_End']:
            ax.plot(group + 0.5, trial_video_idx + 0.5, 'yx', markersize=12, label='Shock_End')
        if row['Tone_Start']:
            ax.plot(group + 0.5, trial_video_idx + 0.5, 'gx', markersize=12, label='Tone_Start')
        if row['Tone_End']:
            ax.plot(group + 0.5, trial_video_idx + 0.5, 'mx', markersize=12, label='Tone_End')

    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))

    ax.legend(
        unique_labels.values(), 
        unique_labels.keys(), 
        bbox_to_anchor=(1.10, 1),  
        loc='upper left', 
        borderaxespad=0, 
        title='Events'
    )
    # Adjust labels and title
    plt.xlabel('Group')
    plt.ylabel('Trial - Video')
    plt.title('Heatmap of Clusters with Event Markers Across Trial and Video')
    plt.savefig(f'{meta_data['output_path']}/cluster_heatmap.png')

def main():
    meta_data = load_metadata('/home/thomas/washu/behavior_clustering/data/fear_voiding/processed_csvs/metadata.json')
    meta_data['output_path'] = f'./../outputs/{meta_data["experiment"]}/'

    if os.path.exists(meta_data['output_path']):
        shutil.rmtree(meta_data['output_path'])
    os.makedirs(meta_data['output_path'], exist_ok=True)

    meta_data = add_bouts(meta_data)
    meta_data = cluster_videos(meta_data)
    save_csvs(meta_data)
    save_tsne_results(meta_data)

    tsne_plot(meta_data)
    create_heatmap_plot(meta_data)
    return meta_data

meta_data = main()