import numpy as np
import pandas as pd


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


# Example Usage
# Assuming `data` is a DataFrame with columns like ['nose_x', 'nose_y', ..., 'spinal_front_x', 'spinal_front_y', etc.]
# rotated_data = rotate_points_global(
#     data, ref_point1="spinal_front", ref_point2="spinal_low"
# )
