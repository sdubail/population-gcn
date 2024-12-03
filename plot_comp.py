import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import seaborn as sns


def create_training_results_boxplot(filepaths):
    """
    Creates a boxplot comparing accuracy distributions across different training result files.

    Args:
        filepaths (list): List of paths to .mat files containing training results
    """
    # Set the plotting parameters
    rcParams = {
        "font.size": 12,  # General font size
        "axes.titlesize": 20,  # Title font size of the plot
        "axes.labelsize": 18,  # Font size of x and y labels
        "xtick.labelsize": 12,  # Font size of x tick labels
        "ytick.labelsize": 14,  # Font size of y tick labels
        "legend.fontsize": 12,  # Font size of legend text
        "figure.titlesize": 20,  # Font size of the figure title
    }
    plt.rcParams.update(rcParams)

    # Initialize list to store results
    all_results = []

    for filepath in filepaths:
        try:
            # Load the data
            data = sio.loadmat(filepath)
            filename = os.path.basename(filepath)

            # Parse the filename to determine the label
            if "expo_top_k" in filename:
                # Extract the k value
                parts = filename.split("_")
                k_idx = parts.index("k") + 1
                k_value = str(parts[k_idx]).split(".")[0]
                label = f"{k_value}th closest nodes"
            elif "expo_threshold" in filename:
                label = "Full similarity"
            else:
                label = filename  # fallback to filename if pattern not found

            # Get all accuracy values and normalize them
            accuracies = (data["acc"] / 87).flatten() * 100

            # Add each accuracy value with its corresponding label
            for acc in accuracies:
                all_results.append({"Label": label, "Accuracy": acc})

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Create the plot with increased figure size for better readability
    plt.figure(figsize=(12, 8))  # Increased figure size

    # Create boxplot with customized style
    sns.boxplot(
        data=df, x="Label", y="Accuracy", linewidth=2
    )  # Increased linewidth for better visibility

    # Customize the plot
    plt.title(
        "Influence of similarity cutoff on prediction performances", pad=20
    )  # Added padding for larger title
    plt.xlabel("")  # Empty xlabel as per original
    plt.ylabel("Accuracy (%)", labelpad=15)  # Added labelpad for better spacing

    # Set y-axis limits to focus on the 50-70% range
    plt.ylim(45, 75)  # slightly wider range to show potential outliers

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Add grid for better readability
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Adjust layout to prevent label cutoff
    plt.tight_layout(pad=2.0)  # Increased padding to accommodate larger fonts

    # Save figure with high resolution
    plt.savefig("similarity_comp.png", dpi=300, bbox_inches="tight")

    # Close the figure to free memory
    plt.close()


# Example usage:
filepaths = [
    "/Users/simondubail/Documents/MVA/Geometric/gcn_perso/population-gcn/results/ABIDE_classification_gcn_cheby_0_3_expo_threshold.mat",
    "/Users/simondubail/Documents/MVA/Geometric/gcn_perso/population-gcn/results/ABIDE_classification_gcn_cheby_0_3_expo_top_k_4.mat",
    "/Users/simondubail/Documents/MVA/Geometric/gcn_perso/population-gcn/results/ABIDE_classification_gcn_cheby_0_3_expo_top_k_10.mat",
    "/Users/simondubail/Documents/MVA/Geometric/gcn_perso/population-gcn/results/ABIDE_classification_gcn_cheby_0_3_expo_top_k_20.mat",
    "/Users/simondubail/Documents/MVA/Geometric/gcn_perso/population-gcn/results/ABIDE_classification_gcn_cheby_0_3_expo_top_k_30.mat",
]
create_training_results_boxplot(filepaths)
