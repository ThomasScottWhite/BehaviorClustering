import glob
import re
import os
import shutil
import pandas as pd
import json

src = "/home/thomas/washu/behavior_clustering/data/Fang/unprocessed_csvs"
target = "/home/thomas/washu/behavior_clustering/data/Fang/processed_csvs"


def fix_pose_data(df_path):
    df = pd.read_csv(df_path)

    body_parts = df.iloc[0, 1:]
    coords = df.iloc[1, 1:]

    new_columns = []
    new_columns.append(f"Image")
    for part, coord in zip(body_parts, coords):
        new_columns.append(f"{part}_{coord}")
    df.columns = new_columns
    df = df[2:]
    df = df.reset_index(drop=True)
    df = df.drop(["Image"], axis=1)

    df = df.apply(pd.to_numeric, errors="coerce")

    likelihood_threshold = 0.75
    for column_group in df.columns[::3]:
        base_name = column_group[:-2]
        x_col = f"{base_name}_x"
        y_col = f"{base_name}_y"
        likelihood_col = f"{base_name}_likelihood"
        mask = df[likelihood_col] < likelihood_threshold
        df.loc[mask, x_col] = pd.NA
        df.loc[mask, y_col] = pd.NA

    df.interpolate(method="linear", inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)
    # df.to_csv(df_path)
    reference_part = "nose"
    x_cols = [col for col in df.columns if col.endswith("_x")]
    y_cols = [col for col in df.columns if col.endswith("_y")]
    x_df = df[x_cols]
    y_df = df[y_cols]
    reference_x = x_df[f"{reference_part}_x"]
    reference_y = y_df[f"{reference_part}_y"]
    relative_x_df = x_df.subtract(reference_x, axis=0)
    relative_y_df = y_df.subtract(reference_y, axis=0)
    relative_coordinates = pd.concat([relative_x_df, relative_y_df], axis=1)
    relative_coordinates.to_csv(df_path)

    # I Cannot Get this to work, I dont know if it is even needed honestly, this code seems really strange?
    tail_root_y = f"TailRoot_y"
    body_center_y = f"BodyCenter_y"

    if (
        tail_root_y in relative_coordinates.columns
        and body_center_y in relative_coordinates.columns
    ):
        if (
            relative_coordinates.loc[0, tail_root_y]
            < relative_coordinates.loc[0, body_center_y]
        ):
            relative_coordinates.loc[0, tail_root_y] = relative_coordinates.loc[
                0, body_center_y
            ]

        for k in range(1, len(relative_coordinates)):
            if (
                relative_coordinates.loc[k, tail_root_y]
                < relative_coordinates.loc[k, body_center_y]
            ):
                relative_coordinates.loc[k, tail_root_y] = relative_coordinates.loc[
                    k - 1, tail_root_y
                ]

    # Save the updated data
    relative_coordinates["Frame"] = relative_coordinates.index
    relative_coordinates.to_csv(df_path)


trials = ["Spine_Trial"]
for trial in trials:
    shutil.rmtree(f"{target}/{trial}/")
    os.makedirs(f"{target}/{trial}/")

metadata_json = {"videos": {}, "experiment": "Fang"}

file_path = "/home/thomas/washu/behavior_clustering/data/Fang/unprocessed_csvs"
avi_files = glob.glob(f"{file_path}/*.avi")
for file in glob.glob(f"{file_path}/*filtered.csv"):
    file_stem = file[file.index("Pde1c(+)") :]
    pattern = r"^(.*?)DLC_Resnet"

    # Apply the regex
    match = re.match(pattern, file_stem)
    if match:
        result = match.group(1)

    sucess = False
    for avi in avi_files:
        if result in avi:
            sucess = True
    if sucess:
        print("yea")
    else:
        break
    trial = trials[0]
    new_directory = f"{target}/{trial}/{result}"
    os.makedirs(f"{new_directory}/", exist_ok=True)
    new_csv_file_path = f"{new_directory}/{result}_Pose_Data.csv"
    shutil.copy(file, new_csv_file_path)
    #     "Tonehabituation_Day1_Ai213_7-6_#3": {
    #     "csv_path": "/home/thomas/washu/behavior_clustering/data/fear_voiding/processed_csvs/Tonehabituation_Day1/Ai213_7-6_#3/pose_void_tone_shock_combined.csv",
    #     "trial": "Tonehabituation_Day1",
    #     "video_path": "/home/thomas/washu/behavior_clustering/data/fear_voiding/videos/Tonehabituation_Day1_Ai213_7-6_#3_Side_view.AVI"
    # },

    metadata_json["videos"][result] = {
        "csv_path": new_csv_file_path,
        "video_path": avi,
        "trial": result,
    }

for video, data in metadata_json["videos"].items():
    fix_pose_data(data["csv_path"])

df = pd.read_csv(metadata_json["videos"]["Pde1c(+) SDGC #21 (1)"]["csv_path"])
regex = r"(_x|_y)$"
matching_column_names = [
    col for col in df.columns if pd.Series(col).str.contains(regex).any()
]
metadata_json["data_columns"] = matching_column_names
metadata_json["event_columns"] = []

with open(f"{target}/metadata.json", "w") as f:
    json.dump(metadata_json, f)
