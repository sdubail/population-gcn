import argparse
import os
import subprocess
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio


def run_parameter_sweep(base_args=None):
    """
    Run GCN training with different hyperparameter combinations

    Args:
        base_args: Base argument namespace to use for non-varying parameters
    """
    if base_args is None:
        # Default base arguments
        base_args = {
            # "dropout": 0.3,
            # "decay": 5e-4,
            # "hidden": 16,
            # "lrate": 0.005,
            # "atlas": "ho",
            # "epochs": 150,
            # "num_features": 2000,
            # "num_training": 1.0,
            # "seed": 123,
            # "folds": 11,
            # "save": 1,
            # "connectivity": "correlation",
        }

    # Define parameter combinations to test
    depths = [0, 1, 2, 3]
    models = ["gcn", "gcn_cheby", "dense"]
    max_degrees = [1, 2, 3, 4]  # Only for gcn_cheby

    # Create parameter combinations based on model type
    param_combinations = []
    for model in models:
        for depth in depths:
            if model == "gcn_cheby":
                # For gcn_cheby, include max_degree variations
                for max_degree in max_degrees:
                    param_combinations.append((depth, max_degree, model))
            else:
                # For other models, use default max_degree (doesn't affect the model)
                param_combinations.append((depth, 1, model))

    # Create results directory if it doesn't exist
    results_dir = "parameter_sweep_results"
    os.makedirs(results_dir, exist_ok=True)

    # Store metadata about the sweep
    sweep_metadata = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "depths": depths,
        "models": models,
        "max_degrees": max_degrees,
        "base_args": base_args,
    }

    # Run training for each combination
    for depth, max_degree, model in param_combinations:
        print(
            f"\nRunning with parameters: depth={depth}, max_degree={max_degree}, model={model}"
        )

        # Construct command line arguments
        cmd = ["python", "main_ABIDE.py"]

        # Add base arguments
        for arg_name, arg_value in base_args.items():
            cmd.extend([f"--{arg_name}", str(arg_value)])

        # Add varying parameters
        cmd.extend(
            ["--depth", str(depth), "--max_degree", str(max_degree), "--model", model]
        )

        try:
            # Run the command
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                print(f"Error running combination: {stderr.decode()}")
                continue

            print(stdout.decode())

        except Exception as e:
            print(f"Error running combination: {e}")
            continue

    return sweep_metadata


def analyze_results(results_dir="parameter_sweep_results", metadata=None):
    """
    Analyze results from parameter sweep and create visualizations focusing on
    accuracy vs order and accuracy vs depth relationships
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

    results = []

    # Load all result files
    for filename in os.listdir(results_dir):
        if filename.endswith(".mat"):
            filepath = os.path.join(results_dir, filename)
            try:
                # Parse parameters from filename
                parts = (
                    filename.replace("ABIDE_classification_", "")
                    .replace(".mat", "")
                    .split("_")
                )

                if "gcn" in parts and "cheby" in parts:
                    parts = [p for p in parts if p not in ["gcn", "cheby"]]
                    parts.insert(0, "gcn_cheby")

                # Replace model names
                model = parts[0]
                if model == "gcn_cheby":
                    model = "Chebyshev filters"
                elif model == "gcn":
                    model = "Linear filters"

                depth = int(parts[1])
                max_degree = int(parts[2])

                # Load results
                data = sio.loadmat(filepath)

                results.append(
                    {
                        "model": model,
                        "depth": depth,
                        "max_degree": max_degree,
                        "accuracy": np.mean(data["acc"] / 87) * 100,
                        "auc": np.mean(data["auc"]),
                        "linear_acc": np.mean(data["lin"] / 87) * 100,
                        "linear_auc": np.mean(data["lin_auc"]),
                    }
                )
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)

    # Create analysis plots with vertical stacking
    plt.figure(figsize=(12, 8))

    # # Plot 1: Depth vs Accuracy by Model
    # plt.subplot(2, 1, 1)
    # for model in df["model"].unique():
    #     model_data = df[df["model"] == model]
    #     # For linear filters, take the single max_degree value
    #     if model != "Chebyshev filters":
    #         plt.plot(model_data["depth"] + 1, model_data["accuracy"], "o-", label=model)
    #     else:
    #         # For Chebyshev filters, show the mean and std across max_degrees
    #         depth_stats = (
    #             model_data.groupby("depth")["accuracy"]
    #             .agg(["mean", "std"])
    #             .reset_index()
    #         )
    #         plt.errorbar(
    #             depth_stats["depth"] + 1,
    #             depth_stats["mean"],
    #             yerr=depth_stats["std"],
    #             fmt="o-",
    #             label=model,
    #         )
    # plt.title("Network Depth vs Accuracy", pad=20)  # Added padding for the larger title
    # plt.xlabel("Number of layers")
    # plt.ylabel("Accuracy (%)")
    # plt.legend()
    # plt.grid(True, linestyle="--", alpha=0.7)

    # # Plot 2: max_degree vs Accuracy (only for Chebyshev filters)
    # plt.subplot(2, 1, 2)
    cheby_data = df[df["model"] == "Chebyshev filters"]
    if not cheby_data.empty:
        for depth in cheby_data["depth"].unique():
            depth_data = cheby_data[cheby_data["depth"] == depth]
            # Sort the data by max_degree before plotting
            depth_data = depth_data.sort_values("max_degree")
            plt.plot(
                depth_data["max_degree"],
                depth_data["accuracy"],
                "o-",
                label=f"layers={depth + 1}",
            )
        plt.title(
            "Polynomial Order vs Accuracy\n(Chebyshev filters)", pad=20
        )  # Added padding for the larger title
        plt.xlabel("Polynomial order")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
    else:
        plt.text(
            0.5,
            0.5,
            "No Chebyshev filters data available",
            horizontalalignment="center",
            verticalalignment="center",
        )

    plt.tight_layout(pad=3.0)  # Increased padding to accommodate larger fonts

    # Save plots
    plt.savefig(
        os.path.join(results_dir, "parameter_sweep_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Save numerical results
    summary_file = os.path.join(results_dir, "parameter_sweep_summary.csv")
    df.to_csv(summary_file, index=False)

    # Print best configurations
    print("\nTop 5 configurations by accuracy:")
    print(
        df.nlargest(5, "accuracy")[["model", "depth", "max_degree", "accuracy", "auc"]]
    )

    # Print best configuration for each model
    print("\nBest configuration per model:")
    for model in df["model"].unique():
        best = df[df["model"] == model].nlargest(1, "accuracy")
        print(f"\n{model}:")
        print(best[["depth", "max_degree", "accuracy", "auc"]].to_string(index=False))

    return df


def main():
    # Setup base arguments
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweep for GCN ABIDE classification"
    )
    parser.add_argument(
        "--base_config", type=str, help="Path to base configuration file"
    )
    args = parser.parse_args()

    # Load base configuration if provided
    base_args = None
    if args.base_config:
        import json

        with open(args.base_config, "r") as f:
            base_args = json.load(f)

    # Run parameter sweep
    print("Starting parameter sweep...")
    # sweep_metadata = run_parameter_sweep(base_args)

    # Analyze results
    print("\nAnalyzing results...")
    results_df = analyze_results()

    print("\nParameter sweep complete. Results saved in parameter_sweep_results/")


if __name__ == "__main__":
    main()
