import ABIDEParser as Reader
import numpy as np
import argparse
import time
from enum import Enum
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from scipy.spatial import distance
import scipy.io as sio
import plotting
import os
import utils

class SimMethod(Enum):
    expo_threshold = "expo_threshold"
    expo_topk = "expo_top_k"
    cosine = "cosine"
    linear = "linear"
    polynomial_decay = "polynomial_decay"


def similarity(method: SimMethod, dist=None, x_data=None, threshold=0, top_k=10):
    if method == SimMethod.cosine:
        sparse_graph = cosine_similarity(x_data)
    elif method == SimMethod.expo_threshold:
        sigma = np.mean(dist)
        sparse_graph = np.where(
            np.exp(-(dist**2) / (2 * sigma**2)) > threshold,
            np.exp(-(dist**2) / (2 * sigma**2)),
            0,
        )
    elif method == SimMethod.expo_topk:
        # Keep only top k strongest connections per node
        k = top_k  # or another value
        sigma = np.mean(dist)
        sparse_graph = np.zeros_like(dist)
        for i in range(len(dist)):
            indices = np.argsort(dist[i])[:k]
            sparse_graph[i, indices] = np.exp(-(dist[i, indices] ** 2) / (2 * sigma**2))
    elif method == SimMethod.linear:
        # Linear kernel
        sparse_graph = 1 - (dist / np.max(dist))
    elif method == SimMethod.polynomial_decay:
        # Or polynomial decay
        sigma = np.mean(dist)
        sparse_graph = 1 / (1 + (dist / sigma) ** 2)

    return sparse_graph

def features_similarity_graph(similarity_support_type, sim_method, dist, x_data, sim_threshold, sim_top_k):
    sparse_graph =  similarity(
                    method=SimMethod(sim_method),
                    dist=dist,
                    x_data=x_data,
                    threshold=sim_threshold,
                    top_k=sim_top_k,
                )
    if similarity_support_type == 'classic':
        return sparse_graph
    elif similarity_support_type == '1':
        return np.ones(sparse_graph.shape)
    elif similarity_support_type == 'random':
        return Reader.random_affinity_graph_with_same_density(sparse_graph)
    elif similarity_support_type == 'worst':
        return Reader.create_worst_affinity_graph_from_initial_graph(sparse_graph) 

def get_phenotypic_graph(phenotypic_graph_type:str, subject_IDs:np.ndarray):
    if phenotypic_graph_type == 'classic':
        return Reader.create_affinity_graph_from_scores(["SEX", "SITE_ID"], subject_IDs)
    if phenotypic_graph_type == 'random':
        graph =  Reader.create_affinity_graph_from_scores(["SEX", "SITE_ID"], subject_IDs)
        return Reader.random_affinity_graph_with_same_density(graph)
    if phenotypic_graph_type == 'G':
        return Reader.create_affinity_graph_from_scores(["SEX"], subject_IDs)
    if phenotypic_graph_type == 'As':
        return Reader.create_affinity_graph_from_scores(["SITE_ID"], subject_IDs)
    if phenotypic_graph_type == 'all':
        return Reader.create_fully_connected_graph(subject_IDs)
    if phenotypic_graph_type == 'worst':
        graph = Reader.create_affinity_graph_from_scores(["SEX", "SITE_ID"], subject_IDs)
        return Reader.create_worst_affinity_graph_from_initial_graph(graph)     
    
def get_adj_matrix(train_ind, graph_feat, features, y, params, subject_IDs, similarity_support_type):
    
    # selection of a subset of data if running experiments with a subset of the training set
    labeled_ind = Reader.site_percentage(train_ind, params["num_training"], subject_IDs)

    # feature selection/dimensionality reduction step
    x_data = Reader.feature_selection(features, y, labeled_ind, params["num_features"], params["method_feature_selection"])

    # Calculate all pairwise distances
    distv = distance.pdist(x_data, metric="correlation")
    # Convert to a square symmetric distance matrix
    dist = distance.squareform(distv)
    
    
    sim_method = params["sim_method"]
    sim_threshold = params["sim_threshold"]
    sim_top_k = params["sim_top_k"]
    
    sparse_graph = features_similarity_graph(similarity_support_type, sim_method, dist, x_data, sim_threshold, sim_top_k)
    
    final_graph = graph_feat * sparse_graph

    return final_graph, sparse_graph



def adj_matrix_visualization():
    parser = argparse.ArgumentParser(
        description="Graph CNNs for population graphs: "
        "classification of the ABIDE dataset"
    )
    
    parser.add_argument(
        "--atlas",
        default="ho",
        help="atlas for network construction (node definition) (default: ho, "
        "see preprocessed-connectomes-project.org/abide/Pipelines.html "
        "for more options )",
    )

    parser.add_argument(
        "--num_features",
        default=2000,
        type=int,
        help="Number of features to keep for "
        "the feature selection step (default: 2000)",
    )
    parser.add_argument(
        "--num_training",
        default=1.0,
        type=float,
        help="Percentage of training set used for " "training (default: 1.0)",
    )

    parser.add_argument(
        "--seed",
        default=34,
        type=int,
        help="Seed for K-fold initialization (default: 34)",
    )

    parser.add_argument(
        "--connectivity",
        default="correlation",
        help="Type of connectivity used for network "
        "construction (default: correlation, "
        "options: correlation, partial correlation, "
        "tangent)",
    )

    parser.add_argument(
        "--sim_method",
        default=SimMethod.expo_threshold.value,
        type=str,
        help="Method for graph similarity computation.",
    )
    parser.add_argument(
        "--sim_threshold",
        default=0,
        type=int,
        help="Threshold for graph similarity computation. Default : 0",
    )
    parser.add_argument(
        "--sim_top_k",
        default=10,
        type=int,
        help="Top_k for graph similarity computation. Default : 10",
    )
    parser.add_argument(
        "--save",
        default=1,
        type=int,
        help="Parameter that specifies if results have to be saved. "
        "Results will be saved if set to 1 (default: 1)",
    )
    parser.add_argument(
        "--phenotypic_graph_type",
        default='classic',
        type=str,
        help="Parameters for choosing the type of phenotypic graph. (default: classic). "
        "Options : 'random', 'G' (only Gender), 'As' (only Acquisition Site), 'worst' (worst case), 'all' (a fully connected graph)",
    )
    parser.add_argument(
        "--similarity_support_type",
        default='classic',
        type=str,
        help="Parameters for choosing the type of features' similarity graph (default: classic)."
        "Options: 'random', 'worst', '1'"
        
    )

    args = parser.parse_args()
    start_time = time.time()

    # GCN Parameters
    params = dict()
    params["seed"] = args.seed  # seed for random initialization

    # GCN Parameters
    params["num_features"] = (
        args.num_features
    )  # number of features for feature selection step
    params["num_training"] = (
        args.num_training
    )  # percentage of training set used for training
    params["sim_method"] = args.sim_method
    params["sim_threshold"] = args.sim_threshold
    params["sim_top_k"] = args.sim_top_k
    atlas = args.atlas  # atlas for network construction (node definition)
    connectivity = (
        args.connectivity
    )
    
    phenotypic_graph_type = args.phenotypic_graph_type
    similarity_support_type = args.similarity_support_type
    
    #   Compute population graph using gender and acquisition site
    subject_IDs = Reader.get_ids()
    phenotypic_graph = get_phenotypic_graph(phenotypic_graph_type, subject_IDs)

    num_classes = 2
    num_nodes = len(subject_IDs)

    # Initialise variables for class labels and acquisition sites
    y_data = np.zeros([num_nodes, num_classes])
    y = np.zeros([num_nodes, 1])
    
    features = Reader.get_networks(subject_IDs, kind=connectivity, atlas_name=atlas)

    # Folds for cross validation experiments
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
      
    cv_splits = list(skf.split(features, np.squeeze(y)))

    train = cv_splits[0][0]
            
    final_adj_matrices, feature_similarity_graph = get_adj_matrix(
            train, phenotypic_graph, features, y, params, subject_IDs, similarity_support_type
            )

      
    
    if args.save == 1:
        
        result_name = f"AdjMatr_pheno_{phenotypic_graph_type}_simi_{similarity_support_type}_nbfeats_{args.num_features}_SimMeth_{args.sim_method}_SimThre_{args.sim_threshold}_SimTopk_{args.sim_top_k}"
        folder_path = "results/AdjMaExp/" + result_name + '/'
        os.makedirs(os.path.dirname(folder_path), exist_ok=True)

        np.save(folder_path + "phenotypic_graph.npy", phenotypic_graph)
        np.save(folder_path +  "adj_matrix.npy", final_adj_matrices)
        np.save(folder_path +  "feature_similarity_graph.npy", feature_similarity_graph)

        plotting.plot_graph_connectivity(phenotypic_graph, subtitle=" - phenotypic graph", saving_filename = folder_path + 'phenotypic_graph.png')
        plotting.plot_graph_connectivity(final_adj_matrices, subtitle=" - final adj matrix", saving_filename = folder_path + 'adj_matrix.png')
        plotting.plot_graph_connectivity(feature_similarity_graph, subtitle=" - features similarity", saving_filename = folder_path + 'feature_similarity_graph.png') 
        plotting.plot_summary_adj_matrix_construction(plotting.build_dic_for_plot_summary_adj_matrix_construction(
                                                        phenotypic_graph, feature_similarity_graph, final_adj_matrices), 
                                                      saving_filename = folder_path + 'summary_construction_adj_mat.png')
        
        print("graph density: ", utils.get_graph_density(phenotypic_graph))
        print("final_adj_matrices density: ", utils.get_graph_density(final_adj_matrices))
        print("graph sparse density: ", utils.graph_weights_density(phenotypic_graph))
        print("final_adj_matrices sparse density: ", utils.graph_weights_density(final_adj_matrices))

if __name__ == "__main__":
    adj_matrix_visualization()