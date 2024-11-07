import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from gcn import utils


def plot_chebyshev_filters(adj, k, plot_file, learned_coeffs=None, num_points=1000):
    """
    Plot Chebyshev polynomial filters and their learned combination against the spectrum
    of the normalized Laplacian.

    Parameters:
    -----------
    adj : scipy.sparse matrix
        Adjacency matrix of the graph
    k : int
        Maximum order of Chebyshev polynomials
    plot_file : str
        Name of the output file.
    learned_coeffs : array-like, optional
        Learned coefficients for the linear combination of Chebyshev polynomials.
        Shape should be (output_features, input_features, k+1)
    num_points : int
        Number of points to evaluate filters
    """

    adj_normalized = utils.normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    eigenvals, _ = eigsh(laplacian, k=min(100, adj.shape[0] - 1), which="LM")
    lambda_max = max(eigenvals)

    # Scale eigenvalues to [-1, 1] as done in chebyshev_polynomials()
    scaled_eigenvals = (2.0 * eigenvals / lambda_max) - 1

    # Create x-axis points in [-1, 1]
    x = np.linspace(-1, 1, num_points)

    # Evaluate Chebyshev polynomials for each x
    T = np.zeros((k + 1, num_points))
    T[0] = np.ones(num_points)  # T_0(x) = 1
    if k > 0:
        T[1] = x  # T_1(x) = x
        for n in range(2, k + 1):
            T[n] = 2 * x * T[n - 1] - T[n - 2]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[1, 1])
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink"]

    # plotting eigenvalues
    for eval in scaled_eigenvals:
        ax1.axvline(x=eval, color="gray", alpha=0.2)

    # Plot 1: Base polynomials

    for i in range(k + 1):
        ax1.plot(
            x,
            T[i],
            label=f"T_{i}(x)",
            color=colors[i % len(colors)],
            linewidth=2,
        )

    ax1.set_xlabel("λ")
    ax1.set_ylabel("|T_k(λ)|")
    ax1.set_title("Chebyshev Polynomial Basis")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: All learned filters
    if learned_coeffs is not None:
        in_features = learned_coeffs[0].shape[0]
        out_features = learned_coeffs[0].shape[1]

        # Plot eigenvalues
        for eval in scaled_eigenvals:
            ax2.axvline(x=eval, color="gray", alpha=0.2)

        # Generate colors for filters
        n_filters = in_features * out_features
        filter_colors = plt.cm.viridis(np.linspace(0, 1, n_filters))

        # Plot each filter (each input-output feature pair)
        for in_feat in range(in_features):
            for out_feat in range(out_features):
                # Get coefficients for this filter
                coeffs = [w[in_feat, out_feat] for w in learned_coeffs]

                # Compute filter
                filt = np.zeros(num_points)
                for i in range(k + 1):
                    filt += coeffs[i] * T[i]

                color_idx = in_feat * out_features + out_feat
                ax2.plot(
                    x,
                    filt,
                    color=filter_colors[color_idx],
                    linewidth=2,
                    alpha=0.7,
                )

                print(
                    f"Coefficients for filter (in={in_feat}, out={out_feat}):",
                    [f"{c:.3f}" for c in coeffs],
                )

    ax2.set_xlabel("λ")
    ax2.set_ylabel("g(λ)")
    ax2.set_title("All Learned Spectral Filters")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        plot_file,
        bbox_inches="tight",
        dpi=300,
    )

    return None


def extract_chebyshev_coeffs(sess, model):
    """
    Extract learned Chebyshev coefficients from the model.
    Returns list of weight matrices, one per Chebyshev order.
    Each matrix has shape (input_features, output_features)
    """
    weights = []
    for i in range(len(model.layers[0].support)):
        weights.append(sess.run(model.layers[0].vars[f"weights_{i}"]))
    return weights
