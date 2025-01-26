# Behavior Cluster Project

This project aims to cluster behavioral data for various analyses, such as creating heatmaps and rendering annotated videos. The overall workflow can be divided into three stages: **Pre-processing**, **Main Processing**, and **Post-processing**.

## Table of Contents
1. [Overview](#overview)
2. [Pre-processing Steps](#pre-processing-steps)
3. [Main Processing Steps](#main-processing-steps)
4. [Post-processing Steps](#post-processing-steps)

---

## Overview

The goal of this project is to take raw behavioral tracking data (for example, from DeepLabCut) and transform it into meaningful clusters of behavior. These clusters can then be used to generate visualizations (like heatmaps) and videos that help interpret and understand the data.

The pipeline is designed to:
- **Clean** and **standardize** raw tracking data.
- **Reduce complexity** through techniques such as PCA, t-SNE, or multi-pass clustering.
- **Identify** distinct behavior clusters for further analysis and visualization.

---

## Pre-processing Steps

Pre-processing focuses on preparing the raw data before the main clustering workflow. This includes organizing the data, filtering out low-likelihood points, and making all points relative to a consistent reference (e.g., the nose of the mouse).

1. **Reorganize DeepLabCut Data**  
   - **Description**: DeepLabCut often produces multiple CSV or HDF files with a specific naming and column format. Here, you restructure these files into a standardized format that can be easily processed later.

2. **Create Event Markers (e.g., Foot Shocks) from CSV/Other Files**  
   - **Description**: Align additional events (like foot shocks) into the dataset. These events will appear on the final heatmap and can be used to correlate behavior with specific stimuli.

3. **Remove or Filter Low-Likelihood Data**  
   - **Description**: DeepLabCut provides a likelihood (or confidence) metric for each tracked point. Data with very low likelihood can introduce noise. Filtering out or imputing these data points improves clustering quality.

4. **Convert Points to Relative Coordinates**  
   - **Description**: Often, it is helpful to make data relative to a stable reference point, such as the mouse’s nose. This helps standardize orientation and position across subjects or sessions.

---

## Main Processing Steps

After pre-processing, the data is ready for clustering and dimensionality reduction. 

1. **Rearrange Data into Bouts**  
   - **Description**: Group continuous frames into short “bouts” of behavior, typically in the range of 0.5 to 10 seconds. This helps capture short bursts of behavior that can be more meaningful than individual frames.

2. **Reduce Video Frame Rate**  
   - **Description**: Often, high framerates contain redundant information. Downsampling frames (e.g., from 30 fps to 10 fps) can reduce noise and computational load while retaining key behavioral features.

3. **Rotation Normalization**  
   - **Description**: Align the subject’s orientation to a standard angle (e.g., 0°) so that behaviors are comparable across different trials or subjects. This might involve rotating coordinate data to standardize the mouse’s body orientation.

4. **Clustering**  
   - **Description**: Apply clustering techniques to the dimensionality-reduced data. Common approaches include:  
     - **PCA + t-SNE**: Use PCA for initial dimensionality reduction, then apply t-SNE for further dimensional compression and visualization.  
     - **t-SNE Only**: Directly apply t-SNE to high-dimensional data.  
     - **Pre-Clustering + t-SNE**: First cluster each frame individually (e.g., using K-means or Gaussian Mixture Models), then cluster the resulting labels or centroids again with t-SNE.

---

## Post-processing Steps

Once the data has been clustered, the final step is to visualize and interpret the results.

1. **Create Heatmaps**  
   - **Description**: Generate heatmaps that show how different behaviors (clusters) are distributed over time or spatial locations. These might also overlay additional events (like foot shocks) to correlate cluster changes with stimuli.

2. **Render Annotated Videos**  
   - **Description**: Produce videos that display the current cluster label on each frame. This makes it easier to see how the subject’s behavior transitions over time. You can also add time-to-event markers to indicate how close (in time) a given frame is to an external event (e.g., a foot shock).

## Pre-processing Steps

Pre-processing focuses on preparing the raw data before the main clustering workflow. This includes organizing the data, filtering out low-likelihood points, and making all points relative to a consistent reference (e.g., the nose of the mouse).

1. **Reorganize DeepLabCut Data**  
   - **Description**: DeepLabCut often produces multiple CSV or HDF files with a specific naming and column format. Here, you restructure these files into a standardized format that can be easily processed later.

2. **Create Event Markers (e.g., Foot Shocks) from CSV/Other Files**  
   - **Description**: Align additional events (like foot shocks) into the dataset. These events will appear on the final heatmap and can be used to correlate behavior with specific stimuli.

3. **Remove or Filter Low-Likelihood Data**  
   - **Description**: DeepLabCut provides a likelihood (or confidence) metric for each tracked point. Data with very low likelihood can introduce noise. Filtering out or imputing these data points improves clustering quality.

4. **Convert Points to Relative Coordinates**  
   - **Description**: Often, it is helpful to make data relative to a stable reference point, such as the mouse’s nose. This helps standardize orientation and position across subjects or sessions.

---

## Main Processing Steps

After pre-processing, the data is ready for clustering and dimensionality reduction. 

1. **Rearrange Data into Bouts**  
   - **Description**: Group continuous frames into short “bouts” of behavior, typically in the range of 0.5 to 10 seconds. This helps capture short bursts of behavior that can be more meaningful than individual frames.

2. **Reduce Video Frame Rate**  
   - **Description**: Often, high framerates contain redundant information. Downsampling frames (e.g., from 30 fps to 10 fps) can reduce noise and computational load while retaining key behavioral features.

3. **Rotation Normalization**  
   - **Description**: Align the subject’s orientation to a standard angle (e.g., 0°) so that behaviors are comparable across different trials or subjects. This might involve rotating coordinate data to standardize the mouse’s body orientation.

4. **Clustering**  
   - **Description**: Apply clustering techniques to the dimensionality-reduced data. Common approaches include:  
     - **PCA + t-SNE**: Use PCA for initial dimensionality reduction, then apply t-SNE for further dimensional compression and visualization.  
     - **t-SNE Only**: Directly apply t-SNE to high-dimensional data.  
     - **Pre-Clustering + t-SNE**: First cluster each frame individually (e.g., using K-means or Gaussian Mixture Models), then cluster the resulting labels or centroids again with t-SNE.

---

## Post-processing Steps

Once the data has been clustered, the final step is to visualize and interpret the results.

1. **Create Heatmaps**  
   - **Description**: Generate heatmaps that show how different behaviors (clusters) are distributed over time or spatial locations. These might also overlay additional events (like foot shocks) to correlate cluster changes with stimuli.

2. **Render Annotated Videos**  
   - **Description**: Produce videos that display the current cluster label on each frame. This makes it easier to see how the subject’s behavior transitions over time. You can also add time-to-event markers to indicate how close (in time) a given frame is to an external event (e.g., a foot shock).
