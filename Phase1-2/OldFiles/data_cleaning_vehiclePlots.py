# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:50:09 2025

@author: Hannah Cimene
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('cleaned_data.csv', header=0, index_col=0)
values = df.values

# Group the DataFrame by 'vehicle_id'
grouped_data = df.groupby('vehicle_id')
vehicle_dataframes = {vid: group_df.copy() for vid, group_df in grouped_data}

def plot_vehicle_dataframes_fixed_scale(vehicle_dataframes, cols=3):
    """Plots vehicle dataframes with latitude as y and longitude as x, using a fixed scale."""

    num_vehicles = len(vehicle_dataframes)
    rows = (num_vehicles + cols - 1) // cols

    # Determine the overall min and max for latitude and longitude
    min_lon = min(df["longitude"])
    max_lon = max(df["longitude"])
    min_lat = min(df["latitude"])
    max_lat = max(df["latitude"])

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    for i, (vehicle_id, vehicle_df) in enumerate(vehicle_dataframes.items()):
        row = i // cols
        col = i % cols

        # Ensure latitude is y and longitude is x
        x = vehicle_df["longitude"]
        y = vehicle_df["latitude"]

        if rows == 1 and cols == 1:
            axes.scatter(x, y)
            axes.set_title(f"Vehicle ID: {vehicle_id}")
            axes.set_xlabel("Longitude")
            axes.set_ylabel("Latitude")
            axes.set_xlim(min_lon, max_lon)  # Set fixed x-axis limits
            axes.set_ylim(min_lat, max_lat)  # Set fixed y-axis limits
        elif rows == 1:
            axes[col].scatter(x, y)
            axes[col].set_title(f"Vehicle ID: {vehicle_id}")
            axes[col].set_xlabel("Longitude")
            axes[col].set_ylabel("Latitude")
            axes[col].set_xlim(min_lon, max_lon)  # Set fixed x-axis limits
            axes[col].set_ylim(min_lat, max_lat)  # Set fixed y-axis limits
        else:
            axes[row, col].scatter(x, y)
            axes[row, col].set_title(f"Vehicle ID: {vehicle_id}")
            axes[row, col].set_xlabel("Longitude")
            axes[row, col].set_ylabel("Latitude")
            axes[row, col].set_xlim(min_lon, max_lon)  # Set fixed x-axis limits
            axes[row, col].set_ylim(min_lat, max_lat)  # Set fixed y-axis limits

    # Remove empty subplots
    if num_vehicles < rows * cols:
        for j in range(num_vehicles, rows * cols):
            row = j // cols
            col = j % cols
            if rows == 1 and cols == 1:
                fig.delaxes(axes)
            elif rows == 1:
                fig.delaxes(axes[col])
            else:
                fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()

# Example usage:
plot_vehicle_dataframes_fixed_scale(vehicle_dataframes, cols=3)

for vehicle_id, vehicle_df in vehicle_dataframes.items():
    x = vehicle_df["longitude"]
    y = vehicle_df["latitude"]

    plt.figure()  # Create a new figure for each vehicle
    plt.scatter(x, y)
    plt.title(f"Vehicle ID: {vehicle_id}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.xlim(min(df["longitude"]), max(df["longitude"]))  # Set fixed x-axis limits
    plt.ylim(min(df["latitude"]), max(df["latitude"]))  # Set fixed y-axis limits
    plt.show()