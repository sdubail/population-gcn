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
            "folder_name_for_saving":"nb_feats_sens/",
            # "dropout": 0.3,
            # "decay": 5e-4,
            # "hidden": 16,
            # "lrate": 0.005,
            # "atlas": "ho",
            "epochs": 150,
            # "num_features": 2000,
            # "num_training": 1.0,
            # "seed": 123,
            "folds": 11,
            # "save": 1,
            # "connectivity": "correlation",
            "n_splits": 10
        }

    # Define parameter combinations to test
    num_features_list = ['250', '750', '1500', '5000', '6000']#['3', '5', '10', '50', '100', '500', '1000', '2000', '3000', '4000']
    seeds = ['1', '2', '3',]# '4', '5', '6', '7', '8', '9', '10']
    seed_cv_folds = ['31', '32',]# '33', '34', '35'] #'36', '37', '38', '39', '40']
    # Create parameter combinations based on model type
    param_combinations = []
    for num_features in num_features_list:
        for seed in seeds:
            for seed_cv_fold in seed_cv_folds:
                param_combinations.append((num_features, seed, seed_cv_fold))

    # Store metadata about the sweep
    sweep_metadata = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "similarity_support_types": num_features_list,
        "seeds": seeds,
        "seed_cv_folds": seed_cv_folds,
        "base_args": base_args,
    }

    # Run training for each combination
    for num_features, seed, seed_cv_fold in param_combinations:
        print(
            f"\nRunning with parameters: num_features={num_features}, seed={seed}, seed_cv_fold={seed_cv_fold}"
        )

        # Construct command line arguments
        cmd = ["python", "main_ABIDE.py"]

        # Add base arguments
        for arg_name, arg_value in base_args.items():
            cmd.extend([f"--{arg_name}", str(arg_value)])


        # Add varying parameters
        cmd.extend(
            ["--num_features", num_features, '--seed', seed, '--seed_cv_fold', seed_cv_fold]
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
    pass

def main():
    # Setup base arguments
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweep for GCN ABIDE classification"
    )
    parser.add_argument(
        "--base_config", type=str, help="Path to base configuration file"
    )
    args = parser.parse_args()

    # Run parameter sweep
    print("Starting parameter sweep...")
    sweep_metadata = run_parameter_sweep()

    # Analyze results
    print("\nAnalyzing results...")
    results_df = analyze_results()


if __name__ == "__main__":
    main()