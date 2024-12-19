# %%%
import os
import shutil
import re
import pandas as pd
from pathlib import Path
import json
# Define source and destination directories
file_path = __file__

script_path = Path(__file__).resolve()
project_root = script_path.parents[2]  # Go up two levels to 'behavior_clustering'
data_dir = project_root / 'data' / 'fear_voiding'

src_dir = data_dir / "unprocessed_csvs" 
dst_dir = data_dir / "processed_csvs"

# %%%
# Ensure the destination directory exists
os.makedirs(dst_dir, exist_ok=True)

# Loop through all files in the source directory

for folder in os.listdir(src_dir):
    folder_path = os.path.join(src_dir, folder)
    folder_dst_dir = os.path.join(dst_dir, folder)

    os.makedirs(folder_dst_dir, exist_ok=True)

    if not os.path.isdir(folder_path):
        continue

    for filename in os.listdir(folder_path):
        if "filtered" in filename:
            continue

        if 'Ai213' in filename and ('.CSV' in filename or '.csv' in filename):

            start_index = filename.find('Ai213')
            # Remove all characters before 'Ai213' and change .CSV to .csv
            new_filename = filename[start_index:].replace('.CSV', '.csv')

            # Extract the grouping pattern Ai213_x=x_#x using regex
            match = re.search(r'Ai213_\d+-\d+_#\d+', new_filename)
            if match:
                if "Side_viewDLC_Resnet50" in filename:
                    new_filename = match.group(0) + "_Pose_Data.csv"
                
                # Create a subdirectory based on the matched pattern
                subfolder_name = match.group(0)
                subfolder_path = os.path.join(folder_dst_dir, subfolder_name)
                os.makedirs(subfolder_path, exist_ok=True)

                # Construct full file paths
                old_filepath = os.path.join(folder_path, filename)
                new_filepath = os.path.join(subfolder_path, new_filename)

                # Move and rename the file
                shutil.copy(old_filepath, new_filepath)

    print("Files have been renamed, grouped into folders, and moved to the new directory.")


def reduce_df(df_path):
    df = pd.read_csv(df_path, index_col=0)
    likelihood_cols = [col for col in df.columns if 'likelihood' in col]
    df['total_likelihood'] = df[likelihood_cols].sum(axis=1)

    # reduced_df = df.groupby(df.index // 8).first().reset_index(drop=True)
    # Group rows into groups of 8 and select the one with the highest total likelihood in each group
    grouped = df.groupby(df.index // 8)

    reduced_df = grouped.apply(lambda group: group.loc[group['total_likelihood'].idxmax()])

    bool_columns = ['Is_Voiding', 'Shock_Start', 'Shock_End', 'Tone_Start', 'Tone_End']
    for col in bool_columns:
        reduced_df[col] = grouped[col].any()

    # Drop the 'total_likelihood' column used for selection
    reduced_df = reduced_df.drop(columns=['total_likelihood'])
    reduced_df = reduced_df.reset_index(drop=True)
    reduced_df.to_csv(df_path)

def fix_void_timing(df_path):
    df = pd.read_csv(df_path)
    df['seconds'] = df['Var4'].str.extract(r'(\d+\.\d+)')
    df = df[['seconds']]
    df.to_csv(df_path)

def fix_pose_data(df_path):
    df = pd.read_csv(df_path)

    body_parts = df.iloc[0, 1:] 
    coords = df.iloc[1, 1:] 

    new_columns = []
    new_columns.append(f'Image')
    for part, coord in zip(body_parts, coords):
        new_columns.append(f'{part}_{coord}')
    df.columns = new_columns
    df = df[2:]
    df = df.reset_index(drop=True)
    df = df.drop(['Image'], axis=1)

    df = df.apply(pd.to_numeric, errors='coerce')


    likelihood_threshold = 0.75
    for column_group in df.columns[::3]:
        base_name = column_group[:-2]
        x_col = f"{base_name}_x"
        y_col = f"{base_name}_y"
        likelihood_col = f"{base_name}_likelihood"
        mask = df[likelihood_col] < likelihood_threshold
        df.loc[mask, x_col] = pd.NA
        df.loc[mask, y_col] = pd.NA

    df.interpolate(method='linear', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    # df.to_csv(df_path)
    reference_part = 'Nose'
    x_cols = [col for col in df.columns if col.endswith('_x')]
    y_cols = [col for col in df.columns if col.endswith('_y')]
    x_df = df[x_cols]
    y_df = df[y_cols]
    reference_x = x_df[f'{reference_part}_x']
    reference_y = y_df[f'{reference_part}_y']
    relative_x_df = x_df.subtract(reference_x, axis=0)
    relative_y_df = y_df.subtract(reference_y, axis=0)
    relative_coordinates = pd.concat([relative_x_df, relative_y_df], axis=1)
    relative_coordinates.to_csv(df_path)

    # I Cannot Get this to work, I dont know if it is even needed honestly, this code seems really strange?
    tail_root_y = f'TailRoot_y'
    body_center_y = f'BodyCenter_y'

    if tail_root_y in relative_coordinates.columns and body_center_y in relative_coordinates.columns:
        if relative_coordinates.loc[0, tail_root_y] < relative_coordinates.loc[0, body_center_y]:
            relative_coordinates.loc[0, tail_root_y] = relative_coordinates.loc[0, body_center_y]

        for k in range(1, len(relative_coordinates)):
            if relative_coordinates.loc[k, tail_root_y] < relative_coordinates.loc[k, body_center_y]:
                relative_coordinates.loc[k, tail_root_y] = relative_coordinates.loc[k - 1, tail_root_y]

    # Save the updated data
    relative_coordinates.to_csv(df_path)

def fix_time_df(df_path):
    df = pd.read_csv(df_path, header=None, names=['DateTime', 'Seconds'])
    df = df.replace(r'[()]', '', regex=True)
    df.to_csv(df_path)

def combine_dfs(pose_path, side_path, void_path, shock_on_path, shock_off_path, tone_on_path, tone_off_path, new_path):
    pose_data_df = pd.read_csv(pose_path, index_col=0)
    side_view_df = pd.read_csv(side_path , index_col=0)
    void_data_df = pd.read_csv(void_path , index_col=0)
    shock_on_df = pd.read_csv(shock_on_path, index_col=0)
    shock_off_df = pd.read_csv(shock_off_path, index_col=0)
    tone_on_df = pd.read_csv(tone_on_path, index_col=0)
    tone_off_df = pd.read_csv(tone_off_path, index_col=0)

    pose_time_df = pd.merge(pose_data_df, side_view_df, left_index=True, right_index=True)
    pose_time_df['Is_Voiding'] = False 

    for voidtime in void_data_df["seconds"]:
        pose_time_df['difference'] = (pose_time_df['Seconds'] - voidtime).abs()

        closest_index = pose_time_df['difference'].idxmin()

        pose_time_df.loc[closest_index, 'Is_Voiding'] = True
        pose_time_df = pose_time_df.drop(columns=["difference"])

    pose_time_df['Shock_Start'] = False 
    for shock_on in shock_on_df["side_Shock_frame"]:

        pose_time_df.loc[shock_on, 'Shock_Start'] = True
        
    pose_time_df['Shock_End'] = False 
    for shock_off in shock_off_df["side_Shock_frame"]:

        pose_time_df.loc[shock_off, 'Shock_End'] = True

    pose_time_df['Tone_Start'] = False 
    for tone_on in tone_on_df["side_Tone_frame"]:

        pose_time_df.loc[tone_on, 'Tone_Start'] = True

    pose_time_df['Tone_End'] = False 
    for tone_off in tone_off_df["side_Tone_frame"]:

        pose_time_df.loc[tone_off, 'Tone_End'] = True

    pose_time_df["Frame"] = pose_time_df.index
    
    pose_time_df.to_csv(new_path)

# %%%   
def main():
    metadata_json = {
        "videos" : {},
        "experiment" : "fear_voiding"    }
    for trial in os.listdir(dst_dir):
        if not os.path.isdir(os.path.join(dst_dir, trial)):
            continue


        trail_path = os.path.join(dst_dir, trial)
        
        folders = [os.path.join(trail_path, d) for d in os.listdir(trail_path) if os.path.isdir(os.path.join(trail_path, d))]
        for folder in folders:

            
            for filename in os.listdir(folder):
                if "Bottom_camera" in filename:
                    bottom_path = os.path.join(folder,filename)

                if "Pose_Data" in filename:
                    pose_path = os.path.join(folder,filename)

                if "ShockOffset" in filename:
                    shock_off_path = os.path.join(folder,filename)

                if "ShockONset" in filename:
                    shock_on_path = os.path.join(folder,filename)

                if "Side_view" in filename:
                    video_path = (data_dir/'videos'/f'{filename[:-4]}.AVI').as_posix() 
                    side_path = os.path.join(folder,filename)

                if "ToneOffset" in filename:
                    tone_off_path = os.path.join(folder,filename)

                if "ToneONset" in filename:
                    tone_on_path = os.path.join(folder,filename)

                if "VoidTiming" in filename:
                    void_path = os.path.join(folder,filename)


            fix_pose_data(pose_path)
            fix_void_timing(void_path)
            fix_time_df(bottom_path)
            fix_time_df(side_path)

            new_path = os.path.join(folder, "pose_void_tone_shock_combined.csv")
            combine_dfs(pose_path, side_path, void_path, shock_on_path, shock_off_path, tone_on_path, tone_off_path, new_path)
            # reduce_df(new_path)

            metadata_json['videos'][trial + '_'+ folder[-12:]] = {
                "csv_path" : new_path,
                "trial" : trial,
                "video_path" : str(data_dir / 'videos' / f'{trial + '_'+ folder[-12:] + '_Side_view'}.AVI'), #/home/thomas/washu/behavior_clustering/data/fear_voiding/videos/Evaluation_Session1 _ Ai213_7-6_#2 _Side_view.AVI
            }

    df = pd.read_csv(new_path)
    regex = r"(_x|_y)$"
    matching_column_names = [col for col in df.columns if pd.Series(col).str.contains(regex).any()]
    metadata_json["data_columns"] = matching_column_names
    metadata_json["event_columns"] = ["Is_Voiding", "Shock_Start", "Shock_End", "Tone_Start", "Tone_End"]
    
    with open(dst_dir / 'metadata.json', 'w') as f:
        json.dump(metadata_json, f)

    print(metadata_json)
    return metadata_json
if __name__ == "__main__":
    main()
    #%%%