import argparse
import os
import subprocess
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio


def run_adj_matrix_sweep(base_args=None):
    """
    Build adj matrix with different hyperparameter combinations

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
    phenotypic_graph_types = ['classic', 'random', 'G', 'As', 'worst', 'all']
    similarity_support_types = ['classic']#'random', 'worst', '1']

    # Create parameter combinations based on model type
    param_combinations = []
    for phenotypic_graph_type in phenotypic_graph_types:
        for similarity_support_type in similarity_support_types:
            param_combinations.append((phenotypic_graph_type, similarity_support_type))

    # Create results directory if it doesn't exist
    results_dir = "results/adj_sweep_results/"
    os.makedirs(results_dir, exist_ok=True)

    # Store metadata about the sweep
    sweep_metadata = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "similarity_support_types": similarity_support_types,
        "similarity_support_types": similarity_support_types,
        "base_args": base_args,
    }

    # Run training for each combination
    for phenotypic_graph_type, similarity_support_type in param_combinations:
        print(
            f"\nRunning with parameters: phenotypic_graph_type={phenotypic_graph_type}, similarity_support_type={similarity_support_type}"
        )

        # Construct command line arguments
        cmd = ["python", "adj_matrix_construction_visualization.py"]

        # Add base arguments
        for arg_name, arg_value in base_args.items():
            cmd.extend([f"--{arg_name}", str(arg_value)])

        # Add varying parameters
        cmd.extend(
            ["--phenotypic_graph_type", phenotypic_graph_type, "--similarity_support_type", similarity_support_type]
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



def main():
    # Setup base arguments
    parser = argparse.ArgumentParser(
        description="Run adj matrix sweep for GCN ABIDE classification"
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
    sweep_metadata = run_adj_matrix_sweep(base_args)

    print("\nParameter sweep complete. Results saved in parameter_sweep_results/")


if __name__ == "__main__":
    main()