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


def analyze_results(results_dir="results", metadata=None):
    """
    Analyze results from parameter sweep and create visualizations

    Args:
        results_dir: Directory containing the .mat result files
        metadata: Metadata about the parameter sweep
    """
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

                model = parts[0]
                depth = int(parts[1])
                max_degree = int(parts[2])

                # Load results
                data = sio.loadmat(filepath)

                results.append(
                    {
                        "model": model,
                        "depth": depth,
                        "max_degree": max_degree,
                        # be careful 'acc' is not a percentage, need to divide by ~87 (batch size) and take mean
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

    # Create analysis plots
    plt.figure(figsize=(15, 10))

    # Plot 1: Model comparison
    plt.subplot(2, 2, 1)
    model_stats = df.groupby("model")["accuracy"].agg(["mean", "std"]).reset_index()
    plt.bar(model_stats["model"], model_stats["mean"], yerr=model_stats["std"])
    plt.title("Model Performance Comparison")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45)

    # Plot 2: Depth vs Accuracy by Model
    plt.subplot(2, 2, 2)
    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        # For non-cheby models, take the single max_degree value
        if model != "gcn_cheby":
            plt.plot(model_data["depth"], model_data["accuracy"], "o-", label=model)
        else:
            # For gcn_cheby, show the mean and std across max_degrees
            depth_stats = (
                model_data.groupby("depth")["accuracy"]
                .agg(["mean", "std"])
                .reset_index()
            )
            plt.errorbar(
                depth_stats["depth"],
                depth_stats["mean"],
                yerr=depth_stats["std"],
                fmt="o-",
                label=model,
            )
    plt.title("Depth vs Accuracy")
    plt.xlabel("Depth")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    # Plot 3: max_degree vs Accuracy (only for gcn_cheby)
    plt.subplot(2, 2, 3)
    cheby_data = df[df["model"] == "gcn_cheby"]
    if not cheby_data.empty:
        for depth in cheby_data["depth"].unique():
            depth_data = cheby_data[cheby_data["depth"] == depth]
            # Sort the data by max_degree before plotting
            depth_data = depth_data.sort_values("max_degree")
            plt.plot(
                depth_data["max_degree"],
                depth_data["accuracy"],
                "o-",
                label=f"depth={depth}",
            )
        plt.title("Chebyshev Degree vs Accuracy\n(gcn_cheby only)")
        plt.xlabel("max_degree")
        plt.ylabel("Accuracy (%)")
        plt.legend()
    else:
        plt.text(
            0.5,
            0.5,
            "No gcn_cheby data available",
            horizontalalignment="center",
            verticalalignment="center",
        )

    # Plot 4: AUC comparison
    plt.subplot(2, 2, 4)
    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        plt.scatter(model_data["accuracy"], model_data["auc"], label=model)
    plt.xlabel("Accuracy (%)")
    plt.ylabel("AUC")
    plt.title("Accuracy vs AUC")
    plt.legend()

    plt.tight_layout()

    # Save plots
    plt.savefig(os.path.join(results_dir, "parameter_sweep_analysis.png"))

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
    sweep_metadata = run_parameter_sweep(base_args)

    # Analyze results
    print("\nAnalyzing results...")
    results_df = analyze_results()

    print("\nParameter sweep complete. Results saved in parameter_sweep_results/")


if __name__ == "__main__":
    main()
