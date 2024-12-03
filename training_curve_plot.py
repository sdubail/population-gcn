import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_training_comparison(
    csv_paths: List[str],
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    colors: Optional[List[str]] = None,
):
    """
    Compare training curves from multiple experiments.

    Args:
        csv_paths: List of paths to CSV files containing training metrics
        labels: List of labels for each experiment (defaults to filenames if None)
        save_path: Path to save the resulting plot
        colors: List of colors for different experiments (defaults to seaborn color palette)
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

    if labels is None:
        labels = [os.path.splitext(os.path.basename(path))[0] for path in csv_paths]

    if colors is None:
        colors = sns.color_palette("husl", n_colors=len(csv_paths))

    # Create figure with two subplots - increased height for better legend placement
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot each experiment
    for i, (path, label, color) in enumerate(zip(csv_paths, labels, colors)):
        df = pd.read_csv(path)

        # Plot Loss
        sns.lineplot(
            data=df,
            x="epoch",
            y="train_loss",
            label=f"{label} (Train)",
            ax=ax1,
            color=color,
            linestyle="-",
            errorbar=("ci", 95),
            alpha=0.8,
            linewidth=2,  # Increased line width for better visibility
        )
        sns.lineplot(
            data=df,
            x="epoch",
            y="val_loss",
            label=f"{label} (Val)",
            ax=ax1,
            color=color,
            linestyle="--",
            errorbar=("ci", 95),
            alpha=0.8,
            linewidth=2,  # Increased line width for better visibility
        )

        # Plot Accuracy
        sns.lineplot(
            data=df,
            x="epoch",
            y="train_acc",
            label=f"{label} (Train)",
            ax=ax2,
            color=color,
            linestyle="-",
            errorbar=("ci", 95),
            alpha=0.8,
            linewidth=2,  # Increased line width for better visibility
        )
        sns.lineplot(
            data=df,
            x="epoch",
            y="val_acc",
            label=f"{label} (Val)",
            ax=ax2,
            color=color,
            linestyle="--",
            errorbar=("ci", 95),
            alpha=0.8,
            linewidth=2,  # Increased line width for better visibility
        )

    # Customize plots
    ax1.set_title("Training and Validation Loss Comparison", pad=20)  # Added padding
    ax1.set_xlabel("Epoch", labelpad=15)  # Added labelpad
    ax1.set_ylabel("Loss", labelpad=15)  # Added labelpad
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="both", which="major", pad=8)  # Increased tick padding

    ax2.set_title(
        "Training and Validation Accuracy Comparison", pad=20
    )  # Added padding
    ax2.set_xlabel("Epoch", labelpad=15)  # Added labelpad
    ax2.set_ylabel("Accuracy", labelpad=15)  # Added labelpad
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="both", which="major", pad=8)  # Increased tick padding

    # Modified legend placement and style
    # legend = ax2.legend(
    #     bbox_to_anchor=(1.05, 1),
    #     loc="upper left",
    #     borderaxespad=0,
    #     frameon=True,
    #     fancybox=True,
    #     shadow=True,
    # )

    # Adjust layout with extra padding
    plt.tight_layout(pad=3.0, w_pad=4.0)  # Increased padding between subplots

    # Save if path is provided
    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            # bbox_extra_artists=[legend],  # Ensure legend is included in saved figure
            pad_inches=0.5,  # Add padding around the figure
        )

    return fig, (ax1, ax2)


def plot_all_metrics_comparison(
    csv_paths: List[str],
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    colors: Optional[List[str]] = None,
):
    """
    Compare all metrics from multiple experiments.
    """
    if labels is None:
        labels = [os.path.splitext(os.path.basename(path))[0] for path in csv_paths]

    if colors is None:
        colors = sns.color_palette("husl", n_colors=len(csv_paths))

    # Create figure with four subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot each experiment
    for i, (path, label, color) in enumerate(zip(csv_paths, labels, colors)):
        df = pd.read_csv(path)

        # Plot Loss
        sns.lineplot(
            data=df,
            x="epoch",
            y="train_loss",
            label=f"{label} (Train)",
            ax=ax1,
            color=color,
            linestyle="-",
            errorbar=("ci", 95),
            alpha=0.8,
        )
        sns.lineplot(
            data=df,
            x="epoch",
            y="val_loss",
            label=f"{label} (Val)",
            ax=ax1,
            color=color,
            linestyle="--",
            errorbar=("ci", 95),
            alpha=0.8,
        )

        # Plot Accuracy
        sns.lineplot(
            data=df,
            x="epoch",
            y="train_acc",
            label=f"{label} (Train)",
            ax=ax2,
            color=color,
            linestyle="-",
            errorbar=("ci", 95),
            alpha=0.8,
        )
        sns.lineplot(
            data=df,
            x="epoch",
            y="val_acc",
            label=f"{label} (Val)",
            ax=ax2,
            color=color,
            linestyle="--",
            errorbar=("ci", 95),
            alpha=0.8,
        )

        # Plot AUC
        sns.lineplot(
            data=df,
            x="epoch",
            y="train_auc",
            label=f"{label} (Train)",
            ax=ax3,
            color=color,
            linestyle="-",
            errorbar=("ci", 95),
            alpha=0.8,
        )
        sns.lineplot(
            data=df,
            x="epoch",
            y="val_auc",
            label=f"{label} (Val)",
            ax=ax3,
            color=color,
            linestyle="--",
            errorbar=("ci", 95),
            alpha=0.8,
        )

        # Plot Training Time
        sns.lineplot(
            data=df,
            x="epoch",
            y="time",
            label=label,
            ax=ax4,
            color=color,
            errorbar=("ci", 95),
            alpha=0.8,
        )

    # Customize plots
    ax1.set_title("Loss comparison")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    # ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    ax2.set_title("Accuracy comparison")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    ax3.set_title("AUC Score comparison")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("AUC")
    ax3.grid(True, alpha=0.3)
    # ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    ax4.set_title("Training time comparison")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Time (seconds)")
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Adjust layout
    plt.tight_layout()

    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ((ax1, ax2), (ax3, ax4))


# Helper function to compare final metrics across experiments
def compare_final_metrics(csv_paths: List[str], labels: Optional[List[str]] = None):
    """
    Prints comparison of final metrics across experiments
    """
    if labels is None:
        labels = [os.path.splitext(os.path.basename(path))[0] for path in csv_paths]

    metrics = ["train_loss", "val_loss", "train_acc", "val_acc", "train_auc", "val_auc"]

    print("Final Metrics Comparison:")
    print("-" * 50)

    for metric in metrics:
        print(f"\n{metric}:")
        for path, label in zip(csv_paths, labels):
            df = pd.read_csv(path)
            final_epoch = df["epoch"].max()
            final_stats = df[df["epoch"] == final_epoch][metric].describe()
            print(f"\n{label}:")
            print(f"  Mean: {final_stats['mean']:.4f}")
            print(f"  Std:  {final_stats['std']:.4f}")


if __name__ == "__main__":
    # Example usage
    csv_paths = [
        "training_curve_gcn_cheby_0_4_expo_threshold.csv",
        "training_curve_gcn_cheby_0_20_expo_threshold.csv",
    ]

    labels = ["Chebychev order = 4", "Chebychev order = 20"]

    # Plot basic comparison
    fig, axes = plot_training_comparison(
        csv_paths=csv_paths, labels=labels, save_path="training_comparison.png"
    )

    # # Plot all metrics comparison
    # fig, axes = plot_all_metrics_comparison(
    #     csv_paths=csv_paths, labels=labels, save_path="all_metrics_comparison.png"
    # )

    # # Compare final metrics
    # compare_final_metrics(csv_paths, labels)
