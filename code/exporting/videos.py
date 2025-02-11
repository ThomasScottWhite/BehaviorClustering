import glob
from pathlib import Path
import json
import os
import cv2
import pandas as pd
from tqdm import tqdm


def put_outlined_text(
    frame, text, position, font, font_scale, font_color, outline_color, thickness
):
    # Draw outline (4 passes)
    x, y = position
    cv2.putText(
        frame,
        text,
        (x - 1, y - 1),
        font,
        font_scale,
        outline_color,
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x + 1, y - 1),
        font,
        font_scale,
        outline_color,
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x - 1, y + 1),
        font,
        font_scale,
        outline_color,
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x + 1, y + 1),
        font,
        font_scale,
        outline_color,
        thickness + 2,
        cv2.LINE_AA,
    )
    # Draw main text
    cv2.putText(
        frame, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA
    )
    return frame


def generate_videos(output_path):
    with open(f"{output_path}/metadata.json", "r") as file:
        meta_data = json.load(file)

    for path in glob.glob(f"{output_path}/*/*.csv"):

        csv_path = Path(path)
        os.makedirs(f"{output_path}/videos/{csv_path.stem}", exist_ok=True)

        video_data = meta_data["videos"][csv_path.stem]

        video_path = video_data["video_path"]
        if "Tonehabituation" in video_path:
            video_path = video_path.replace("Tonehabituation", "Habituation")
        if "Evaluation_Session1" in video_path:
            video_path = video_path.replace(
                "Evaluation_Session1", "ConditinedTone_Session1"
            )

        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue

        csv = pd.read_csv(csv_path, index_col=0)

        event_dicts = {}
        for event in meta_data["event_columns"]:
            mask = csv[event]
            results = csv[mask]
            event_dicts[event] = list(results.index)

    os.makedirs(f"{output_path}/videos", exist_ok=True)

    for path in tqdm(glob.glob(f"{output_path}/*/*.csv")):

        csv_path = Path(path)
        os.makedirs(f"{output_path}/videos/{csv_path.stem}", exist_ok=True)

        video_data = meta_data["videos"][csv_path.stem]

        video_path = video_data["video_path"]
        if "Tonehabituation" in video_path:
            video_path = video_path.replace("Tonehabituation", "Habituation")
        if "Evaluation_Session1" in video_path:
            video_path = video_path.replace(
                "Evaluation_Session1", "ConditinedTone_Session1"
            )

        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue

        csv = pd.read_csv(csv_path, index_col=0)

        event_dicts = {}
        for event in meta_data["event_columns"]:
            mask = csv[event]
            results = csv[mask]
            event_dicts[event] = list(results.index)

        # Input parameters
        bouts = csv["Group"]
        numbers = csv["Cluster"]

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine the number of frames to process
        frames_to_process = min(len(numbers), frame_count)

        print(f"Processing {frames_to_process} frames out of {frame_count}.")
        # Set up the video writer
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec for AVI files
        out = cv2.VideoWriter(
            f"{output_path}/videos/{csv_path.stem}/output_video.avi",
            fourcc,
            fps,
            (frame_width, frame_height),
        )

        videos = [0] * 8
        for i in range(8):
            videos[i] = cv2.VideoWriter(
                f"{output_path}/videos/{csv_path.stem}/{i}.avi",
                fourcc,
                fps,
                (frame_width, frame_height),
            )

        frame_index = 0
        while cap.isOpened() and frame_index < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 255, 255)  # Main text color
            thickness = 3
            position_cluster = (50, 60)  # Position for Cluster text
            position_bout = (50, 30)  # Position for Bout text
            shadow_offset = 2  # Offset for the shadow

            # Shadow color (black)
            outline_color = (0, 0, 0)

            frame = put_outlined_text(
                frame,
                f"Cluster {numbers[frame_index]}",
                (50, 60),
                font,
                font_scale,
                font_color,
                outline_color,
                thickness,
            )
            frame = put_outlined_text(
                frame,
                f"Bout {bouts[frame_index]}",
                (50, 30),
                font,
                font_scale,
                font_color,
                outline_color,
                thickness,
            )

            # Logic for determining next and previous times
            event_counter = 0
            for event_name, event_array in event_dicts.items():
                event_counter += 1
                previous = 0
                next_index = 0

                for idx in event_array:
                    if frame_index < idx:
                        next_index = idx
                        break
                    else:
                        previous = idx

                if next_index == 0:
                    next_string = f"Next {event_name}: NA"
                else:
                    next_string = f"Next {event_name} {int((next_index - frame_index)/fps)} Seconds"

                if previous == 0:
                    previous_string = f"Previous {event_name}: NA"
                else:
                    previous_string = f"Previous {event_name} in {int((frame_index - previous)/fps)} Seconds Ago"

                frame = put_outlined_text(
                    frame,
                    next_string,
                    (50, (90 + event_counter * 60)),
                    font,
                    font_scale,
                    font_color,
                    outline_color,
                    thickness,
                )
                frame = put_outlined_text(
                    frame,
                    previous_string,
                    (50, (120 + event_counter * 60)),
                    font,
                    font_scale,
                    font_color,
                    outline_color,
                    thickness,
                )

            # Write the frame to the output video
            videos[numbers[frame_index]].write(frame)
            out.write(frame)

            frame_index += 1

        # Release resources
        cap.release()
        out.release()

        print(f"Video saved to {output_path}")

    for video in videos:
        video.release()


if __name__ == "__main__":
    output_path = "/home/thomas/washu/behavior_clustering/outputs/0.5 second bouts Fang"
    generate_videos(output_path)
