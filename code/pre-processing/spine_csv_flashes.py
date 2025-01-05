import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import json


def open_point_window(video_path):
    # Create a hidden Tkinter root window
    root = tk.Tk()
    root.withdraw()

    # Get a single frame from the video
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    if not success:
        print("Error: Could not read the video file.")
        return None

    # Nonlocal variable to store the selected point
    selected_point = None

    def select_point(event, x, y, flags, param):
        nonlocal selected_point  # Declare as nonlocal to modify it inside the callback
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_point = (x, y)
            print(f"Point selected: {selected_point}")

    # Display the frame and set up mouse callback
    cv2.namedWindow("Select Point")
    cv2.setMouseCallback("Select Point", select_point)

    while True:
        cv2.imshow("Select Point", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or selected_point is not None:
            break

    cv2.destroyAllWindows()
    return selected_point


# with open("/home/thomas/washu/behavior_clustering/data/Fang/processed_csvs/metadata.json", 'r') as file:
#     data = json.load(file)

# for i,k in data["videos"].items():
#     video_path = k["video_path"]
#     csv_path = k["csv_path"]
#     point = open_point_window(video_path)
#     data["videos"][i]["light_source_point"] = point

# with open("/home/thomas/washu/behavior_clustering/data/Fang/processed_csvs/metadata.json", 'w') as file:
#     file.write(json.dumps(data))

# with open("/home/thomas/washu/behavior_clustering/data/Fang/processed_csvs/metadata.json", 'r') as file:
#     data = json.load(file)
# data["event_columns"] = ["Light On"]
# with open("/home/thomas/washu/behavior_clustering/data/Fang/processed_csvs/metadata.json", 'w') as file:
#     file.write(json.dumps(data))

with open(
    "/home/thomas/washu/behavior_clustering/data/Fang/processed_csvs/metadata.json", "r"
) as file:
    metajson = json.load(file)


for i, k in metajson["videos"].items():
    selected_point = k["light_source_point"]
    light_on_rgb = np.array([255, 243, 197])
    light_off_rgb = np.array([250, 160, 135])

    def is_light_on(selected_point, frame):
        x, y = selected_point
        pixel_rgb = frame[y, x]
        pixel_rgb = pixel_rgb[::-1]
        # Calculate the Euclidean distance to both the "on" and "off" reference colors
        dist_to_on = np.linalg.norm(pixel_rgb - light_on_rgb)
        dist_to_off = np.linalg.norm(pixel_rgb - light_off_rgb)

        # Return True if the light is closer to "on" color, otherwise False
        if dist_to_on < dist_to_off:
            return True
        else:
            return False

    # Create dataframe
    data = []

    cap = cv2.VideoCapture(k["video_path"])
    success, frame = cap.read()
    frame_count = 1
    while success:
        # print(f"Processing frame {frame_count}")
        frame_count += 1
        light_status = is_light_on(selected_point, frame)
        data.append(light_status)
        success, frame = cap.read()
    cap.release()
    df = pd.DataFrame(data)
    csv = pd.read_csv(k["csv_path"], index_col=0)

    csv["Light On"] = data
    progressive = 0
    onoff = [False] * len(csv)
    on = csv[csv["Light On"] == True]
    for i, row in on.iterrows():
        if progressive == 0:
            progressive = i
            onoff[i] = True

        if i - progressive < 10:
            progressive = i
        else:
            onoff[i] = True
            progressive = i

    csv["Light On"] = onoff

    print(len(df), len(pd.read_csv(k["csv_path"], index_col=0)))
    csv.to_csv(k["csv_path"])
