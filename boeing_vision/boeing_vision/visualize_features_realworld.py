import pandas as pd
import sys, os
import matplotlib.pyplot as plt
import numpy as np
import math



def normalize(x):
    if x <= 0:
        return 0.0  # handle non-positive values if they somehow appear
    log_x = math.log(x)
    return 1 / (1 + math.exp(-log_x))


import csv
from matplotlib import cm

def read_jagged_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        return list(reader)  # Returns list of lists
    

def plot_features(data, column_indices, column_names, name):
 
    

    x_values = np.arange(1, len(data) + 1)

    # Create the plot
    plt.figure(figsize=(8, 5))  # Optional: set figure size
    for idx in column_indices:
        column = data[:, idx]
        if idx == 7:
            column *= 10**12

        if idx == 6:
            column *= 10**4

        plt.scatter(x_values, column, color='b', marker='o')
        
        mean_value = np.mean(column)
        if idx == 8:
            mean_value = np.degrees(mean_value)

        plt.axhline(y=mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')
        plt.legend()
    

    # Add labels and title
    plt.xlabel('Sample')
    plt.ylabel(", ".join(column_names))
    plt.title(name)

    # Optional: Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()
    """
    gjs = []

    delta = 0.1
    
    for index, row in data.iterrows():
        volume = row.iloc[6]
        shape = row.iloc[7] * 10**6
        theta = row.iloc[8]

        g_j = (delta / (volume * shape)) * (1 - (theta / np.pi))

        gj_normalized = normalize(g_j)

        gjs.append(gj_normalized)

    # Create x-axis values (1 incremental)
    x_values = np.arange(1, len(gjs) + 1)

    # Create the plot
    plt.figure(figsize=(8, 5))  # Optional: set figure size
    plt.scatter(x_values.reshape(-1), np.asarray(gjs).reshape(-1), marker='o', color='b')

    # Add labels and title
    plt.xlabel('Index (1 incremental)')
    plt.ylabel('Array Values')
    plt.title('Plot of 1D Array Values')

    # Optional: Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()
    """

def plot_combined_features(datasets, column_index, column_name, title):
    """
    Plots the same column from multiple datasets in a single plot.
    
    Args:
        datasets (list of np.ndarray): List of datasets to plot.
        column_index (int): Index of the column to plot.
        column_name (str): Name of the column to use as the label.
        title (str): Title of the plot.
    """
    # Define a colormap for the plots
    colors = [plt.cm.tab10(i) for i in range(5)]
    plt.figure(figsize=(10, 6))  # Set figure size

    # Downsample all datasets to match the shortest dataset length
    min_length = min(len(data) for data in datasets if len(data) > 0)
    datasets = [data[np.linspace(0, len(data) - 1, min_length, dtype=int)] for data in datasets if len(data) >= min_length]

    for i, data in enumerate(datasets):
        if data.shape[0] == 0:  # Skip empty datasets
            continue

        x_values = np.arange(1, len(data) + 1)
        y_values = data[:, column_index]

        # Adjust scaling for specific columns if needed
        if column_index == 7:
            y_values *= 10**12
        elif column_index == 6:
            y_values /= 10**4
            #y_values = y_values[y_values <= 200]
            #x_values = x_values[:len(y_values)]
            size_label = ["Largest", "Large", "Medium", "Small", "Smallest"][i]  # Assign size label based on iteration
        if column_index == 8:
            size_label = ["Uphill", "Flat", "Downhill"][i]  # Assign size label based on iteration

        mean_value = np.mean(y_values)

        #plt.fill_between(x_values, y_values - smoothed_std_values, y_values + smoothed_std_values, color=colors[i], alpha=0.4)
        #plt.plot(x_values, y_values, marker='o', label=f'Dataset {i + 1}')
        #std_value = np.std(y_values)

        #plt.axhline(y=mean_value, linestyle='--', color=colors[i], label=f'{size_label} - Mean: {mean_value:.2f}')
        #plt.fill_between(x_values, mean_value - std_value, mean_value + std_value, alpha=0.2)

        # Calculate running mean and running standard deviation
        window_size = 50  # Define the window size for running calculations
        running_mean = np.convolve(y_values, np.ones(window_size) / window_size, mode='valid')
        running_std = np.sqrt(np.convolve((y_values - np.mean(y_values))**2, np.ones(window_size) / window_size, mode='valid'))

        # Adjust x_values to match the length of running_mean and running_std
        adjusted_x_values = x_values[:len(running_mean)]

        # Plot running mean and running standard deviation
        plt.plot(adjusted_x_values, running_mean, color=colors[i], label=f'{size_label} - Mean: {mean_value:.2f}')
        plt.fill_between(adjusted_x_values, running_mean - running_std, running_mean + running_std, color=colors[i], alpha=0.2)

        #plt.scatter(x_values, y_values, marker='o', label=f'Dataset {i + 1}')

    # Increase font sizes for the plot
    plt.rc('font', size=14)  # Default text size
    plt.rc('axes', titlesize=18)  # Title size
    plt.rc('axes', labelsize=16)  # Axis label size
    plt.rc('xtick', labelsize=14)  # X-axis tick size
    plt.rc('ytick', labelsize=14)  # Y-axis tick size
    plt.rc('legend', fontsize=10)  # Legend font size
    # Add labels, title, and legend
    #plt.xlabel('Sample')
    #plt.ylabel(column_name)
    plt.title(f"{title}", fontdict={'fontsize': 18})
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Show the plot
    # Save the plot to a file
    output_folder = "/home/students/girgine/ros2_ws/src/boeing_vision/plots/"
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
    output_file = os.path.join(output_folder, f"{title.replace(' ', '_').replace('(m³)', '')}_realworld.png")
    plt.savefig(output_file, dppi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    plt.show()


if __name__ == "__main__":
    # Define column names for reference
    """
    Each record contains the following elements:
            - Obstacle position (x, y, z)
            - Bounding box dimensions (width, height, length)
            - Bounding box volume
            - Obstacle shape
            - Ground point coordinates (x, y, z) for each normal
            - Normal vector components (x, y, z) for each normal
            - Angle (theta) between the normal and the reference vector
    """

    column_names = ["x", "y", "z", "width", "height", "length", "volume", "shape", "theta"]
    csv_folder = "/home/students/girgine/ros2_ws/src/boeing_vision/dataset_realworld/"
    column_indices = [6]  # Replace with the desired column indices (starting from 0)
    

    data_downhill_rock_large = np.zeros((0, 9))
    data_downhill_rock_mediumlarge = np.zeros((0, 9))
    data_downhill_rock_medium = np.zeros((0, 9))
    data_downhill_rock_mediumsmall = np.zeros((0, 9))
    data_downhill_rock_small = np.zeros((0, 9))
    data_flat_rock_large = np.zeros((0, 9))
    data_flat_rock_mediumlarge = np.zeros((0, 9))
    data_flat_rock_medium = np.zeros((0, 9))
    data_flat_rock_mediumsmall = np.zeros((0, 9))
    data_flat_rock_small = np.zeros((0, 9))
    data_uphill_rock_large = np.zeros((0, 9))
    data_uphill_rock_mediumlarge = np.zeros((0, 9))
    data_uphill_rock_medium = np.zeros((0, 9))
    data_uphill_rock_mediumsmall = np.zeros((0, 9))
    data_uphill_rock_small = np.zeros((0, 9))

    for csv_file in os.listdir(csv_folder):

        csv_file_dropped = csv_file.split('.')[-2]

        #if '_alt.' in csv_file:
        #    continue
        slope = csv_file_dropped.split("_")[1]
        rock_id = csv_file_dropped.split("_")[0]

        #print(slope)
        print(slope, rock_id)

        # Separate data into numpy arrays based on rock_id and slope
        if "downhill" == slope and "large" == rock_id:
            data_downhill_rock_large = np.vstack((data_downhill_rock_large, np.genfromtxt(csv_folder + csv_file, delimiter=',', skip_header=1)))
        elif "downhill" == slope and "largemedium" == rock_id:
            data_downhill_rock_mediumlarge = np.vstack((data_downhill_rock_mediumlarge, np.genfromtxt(csv_folder + csv_file, delimiter=',', skip_header=1)))
        elif "downhill" == slope and "medium" == rock_id:
            data_downhill_rock_medium = np.vstack((data_downhill_rock_medium, np.genfromtxt(csv_folder + csv_file, delimiter=',', skip_header=1)))
        elif "downhill" == slope and "smallmedium" == rock_id:
            data_downhill_rock_mediumsmall = np.vstack((data_downhill_rock_mediumsmall, np.genfromtxt(csv_folder + csv_file, delimiter=',', skip_header=1)))
        elif "downhill" == slope and "small" == rock_id:
            data_downhill_rock_small = np.vstack((data_downhill_rock_small, np.genfromtxt(csv_folder + csv_file, delimiter=',', skip_header=1)))
        elif "flat" == slope and "large" == rock_id:
            data_flat_rock_large = np.vstack((data_flat_rock_large, np.genfromtxt(csv_folder + csv_file, delimiter=',', skip_header=1)))
        elif "flat" == slope and "largemedium" == rock_id:
            data_flat_rock_mediumlarge = np.vstack((data_flat_rock_mediumlarge, np.genfromtxt(csv_folder + csv_file, delimiter=',', skip_header=1)))
        elif "flat" == slope and "medium" == rock_id:
            data_flat_rock_medium = np.vstack((data_flat_rock_medium, np.genfromtxt(csv_folder + csv_file, delimiter=',', skip_header=1)))
        elif "flat" == slope and "smallmedium" == rock_id:
            data_flat_rock_mediumsmall = np.vstack((data_flat_rock_mediumsmall, np.genfromtxt(csv_folder + csv_file, delimiter=',', skip_header=1)))
        elif "flat" == slope and "small" == rock_id:
            data_flat_rock_small = np.vstack((data_flat_rock_small, np.genfromtxt(csv_folder + csv_file, delimiter=',', skip_header=1)))
        elif "uphill" == slope and "large" == rock_id:
            data_uphill_rock_large = np.vstack((data_uphill_rock_large, np.genfromtxt(csv_folder + csv_file, delimiter=',', skip_header=1)))
        elif "uphill" == slope and "largemedium" == rock_id:
            data_uphill_rock_mediumlarge = np.vstack((data_uphill_rock_mediumlarge, np.genfromtxt(csv_folder + csv_file, delimiter=',', skip_header=1)))
        elif "uphill" == slope and "medium" in rock_id:
            data_uphill_rock_medium = np.vstack((data_uphill_rock_medium, np.genfromtxt(csv_folder + csv_file, delimiter=',', skip_header=1)))
        elif "uphill" == slope and "smallmedium" == rock_id:
            data_uphill_rock_mediumsmall = np.vstack((data_uphill_rock_mediumsmall, np.genfromtxt(csv_folder + csv_file, delimiter=',', skip_header=1)))
        elif "uphill" == slope and "small" == rock_id:
            data_uphill_rock_small = np.vstack((data_uphill_rock_small, np.genfromtxt(csv_folder + csv_file, delimiter=',', skip_header=1)))


    print("Size of data_downhill_rock_large:", data_downhill_rock_large.shape)
    print("Size of data_downhill_rock_mediumlarge:", data_downhill_rock_mediumlarge.shape)
    print("Size of data_downhill_rock_medium:", data_downhill_rock_medium.shape)
    print("Size of data_downhill_rock_mediumsmall:", data_downhill_rock_mediumsmall.shape)
    print("Size of data_downhill_rock_small:", data_downhill_rock_small.shape)
    print("Size of data_flat_rock_large:", data_flat_rock_large.shape)
    print("Size of data_flat_rock_mediumlarge:", data_flat_rock_mediumlarge.shape)
    print("Size of data_flat_rock_medium:", data_flat_rock_medium.shape)
    print("Size of data_flat_rock_mediumsmall:", data_flat_rock_mediumsmall.shape)
    print("Size of data_flat_rock_small:", data_flat_rock_small.shape)
    print("Size of data_uphill_rock_large:", data_uphill_rock_large.shape)
    print("Size of data_uphill_rock_mediumlarge:", data_uphill_rock_mediumlarge.shape)
    print("Size of data_uphill_rock_medium:", data_uphill_rock_medium.shape)
    print("Size of data_uphill_rock_mediumsmall:", data_uphill_rock_mediumsmall.shape)
    print("Size of data_uphill_rock_small:", data_uphill_rock_small.shape)

    # Merge datasets having the same rock size
    data_rock_large = np.vstack((data_downhill_rock_large, data_flat_rock_large, data_uphill_rock_large))
    data_rock_mediumlarge = np.vstack((data_downhill_rock_mediumlarge, data_flat_rock_mediumlarge, data_uphill_rock_mediumlarge))
    data_rock_medium = np.vstack((data_downhill_rock_medium, data_flat_rock_medium, data_uphill_rock_medium))
    data_rock_mediumsmall = np.vstack((data_downhill_rock_mediumsmall, data_flat_rock_mediumsmall, data_uphill_rock_mediumsmall))
    data_rock_small = np.vstack((data_downhill_rock_small, data_flat_rock_small, data_uphill_rock_small))

    # Plot combined features for each rock size
    plot_combined_features([data_rock_large, data_rock_mediumlarge, data_rock_medium, data_rock_mediumsmall, data_rock_small], column_indices[0], column_names[column_indices[0]], "Volume (m³)")

    # Merge datasets having the same slope type
    data_downhill = np.vstack((data_downhill_rock_large, data_downhill_rock_mediumlarge, data_downhill_rock_medium, data_downhill_rock_mediumsmall, data_downhill_rock_small))
    data_flat = np.vstack((data_flat_rock_large, data_flat_rock_mediumlarge, data_flat_rock_medium, data_flat_rock_mediumsmall, data_flat_rock_small))
    data_uphill = np.vstack((data_uphill_rock_large, data_uphill_rock_mediumlarge, data_uphill_rock_medium, data_uphill_rock_mediumsmall, data_uphill_rock_small))

    # Plot combined features for each slope type
    #plot_combined_features([data_uphill, data_flat, data_downhill], column_indices[0], column_names[column_indices[0]], "Surface Slope (θ)")
    """
    plot_features(data_downhill_rock_0, column_indices, [column_names[idx] for idx in column_indices], "Downhill Rock 0")
    plot_features(data_downhill_rock_1, column_indices, [column_names[idx] for idx in column_indices], "Downhill Rock 1")
    plot_features(data_downhill_rock_8, column_indices, [column_names[idx] for idx in column_indices], "Downhill Rock 8")
    plot_features(data_flat_rock_0, column_indices, [column_names[idx] for idx in column_indices], "Flat Rock 0")
    plot_features(data_flat_rock_1, column_indices, [column_names[idx] for idx in column_indices], "Flat Rock 1")
    plot_features(data_flat_rock_8, column_indices, [column_names[idx] for idx in column_indices], "Flat Rock 8")
    plot_features(data_uphill_rock_0, column_indices, [column_names[idx] for idx in column_indices], "Uphill Rock 0")
    plot_features(data_uphill_rock_1, column_indices, [column_names[idx] for idx in column_indices], "Uphill Rock 1")
    plot_features(data_uphill_rock_8, column_indices, [column_names[idx] for idx in column_indices], "Uphill Rock 8")
    """