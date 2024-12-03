# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import argparse
import time
from enum import Enum

import ABIDEParser as Reader
import numpy as np
import scipy.io as sio
import sklearn.metrics
import train_GCN as Train
from joblib import Parallel, delayed
from scipy import sparse
from scipy.spatial import distance
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold


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


# Prepares the training/test data for each cross validation fold and trains the GCN
def train_fold(
    train_ind, test_ind, val_ind, graph_feat, features, y, y_data, params, subject_IDs, test_with_other_classifiers:bool=False
):
    """
        train_ind       : indices of the training samples
        test_ind        : indices of the test samples
        val_ind         : indices of the validation samples
        graph_feat      : population graph computed from phenotypic measures num_subjects x num_subjects
        features        : feature vectors num_subjects x num_features
        y               : ground truth labels (num_subjects x 1)
        y_data          : ground truth labels - different representation (num_subjects x 2)
        params          : dictionnary of GCNs parameters
        subject_IDs     : list of subject IDs

    returns:

        test_acc    : average accuracy over the test samples using GCNs
        test_auc    : average area under curve over the test samples using GCNs
        lin_acc     : average accuracy over the test samples using the linear classifier
        lin_auc     : average area under curve over the test samples using the linear classifier
        fold_size   : number of test samples
    """

    print(len(train_ind))

    # selection of a subset of data if running experiments with a subset of the training set
    labeled_ind = Reader.site_percentage(train_ind, params["num_training"], subject_IDs)

    # feature selection/dimensionality reduction step
    x_data = Reader.feature_selection(features, y, labeled_ind, params["num_features"])

    fold_size = len(test_ind)

    # Calculate all pairwise distances
    distv = distance.pdist(x_data, metric="correlation")
    # Convert to a square symmetric distance matrix
    dist = distance.squareform(distv)

    # Get affinity from similarity matrix

    sim_method = params["sim_method"]
    sim_threshold = params["sim_threshold"]
    sim_top_k = params["sim_top_k"]

    sparse_graph = similarity(
        method=SimMethod(sim_method),
        dist=dist,
        x_data=x_data,
        threshold=sim_threshold,
        top_k=sim_top_k,
    )

    final_graph = graph_feat * sparse_graph

    # Linear classifier
    clf = RidgeClassifier()
    clf.fit(x_data[train_ind, :], y[train_ind].ravel())
    # Compute the accuracy
    lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
    # Compute the AUC
    pred = clf.decision_function(x_data[test_ind, :])
    lin_auc = sklearn.metrics.roc_auc_score(y[test_ind] - 1, pred)

    print("Linear Accuracy: " + str(lin_acc))

    # Other Classifiers
    if test_with_other_classifiers:
        logistic_clf = LogisticRegression(max_iter=1000, random_state=42)
        svc_clf = SVC(kernel='linear', random_state=42)
        models = {
            "Logistic Regression": logistic_clf,
            "Linear SVC": svc_clf
        }
        otherResults = {}

        for name, model in models.items():           
            # Fit the model and evaluate on test data
            model.fit(x_data[train_ind, :], y[train_ind].ravel())
            # Compute the accuracy
            model_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
            # Compute the AUC
            pred = clf.decision_function(x_data[test_ind, :])
            model_auc = sklearn.metrics.roc_auc_score(y[test_ind] - 1, pred)
            otherResults[f'{name}_acc'] = model_acc
            otherResults[f'{name}_auc'] = model_auc
    
    # print("Y_LABEL BEFORE EVERYTHIN", np.unique(y_data, axis=0))
    # Classification with GCNs
    test_acc, test_auc = Train.run_training(
        final_graph,
        sparse.coo_matrix(x_data).tolil(),
        y_data,
        train_ind,
        val_ind,
        test_ind,
        params,
    )

    print(test_acc)

    # return number of correctly classified samples instead of percentage
    test_acc = int(round(test_acc * len(test_ind)))
    lin_acc = int(round(lin_acc * len(test_ind)))
    if test_with_other_classifiers:
        return test_acc, test_auc, lin_acc, lin_auc, fold_size, otherResults
    else:
        return test_acc, test_auc, lin_acc, lin_auc, fold_size



def main():
    parser = argparse.ArgumentParser(
        description="Graph CNNs for population graphs: "
        "classification of the ABIDE dataset"
    )
    parser.add_argument(
        "--dropout",
        default=0.3,
        type=float,
        help="Dropout rate (1 - keep probability) (default: 0.3)",
    )
    parser.add_argument(
        "--decay",
        default=5e-4,
        type=float,
        help="Weight for L2 loss on embedding matrix (default: 5e-4)",
    )
    parser.add_argument(
        "--hidden",
        default=16,
        type=int,
        help="Number of filters in hidden layers (default: 16)",
    )
    parser.add_argument(
        "--lrate",
        default=0.005,
        type=float,
        help="Initial learning rate (default: 0.005)",
    )
    parser.add_argument(
        "--atlas",
        default="ho",
        help="atlas for network construction (node definition) (default: ho, "
        "see preprocessed-connectomes-project.org/abide/Pipelines.html "
        "for more options )",
    )
    parser.add_argument(
        "--epochs", default=150, type=int, help="Number of epochs to train"
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
        "--depth",
        default=0,
        type=int,
        help="Number of additional hidden layers in the GCN. "
        "Total number of hidden layers: 1+depth (default: 0)",
    )
    parser.add_argument(
        "--jacobi_iteration",
        default=15,
        type=int,
        help="Number of iteration for the jacobi algo in CayleyNets. ",
    )
    parser.add_argument(
        "--max_degree",
        default=3,
        type=int,
        help="Maximal degree of the Chebychev polynom used to approximate the filter.",
    )
    parser.add_argument(
        "--model",
        default="gcn_cheby",
        help="gcn model used (default: gcn_cheby, "
        "uses chebyshev polynomials, "
        "options: gcn, gcn_cheby, dense )",
    )
    parser.add_argument(
        "--seed",
        default=123,
        type=int,
        help="Seed for random initialisation (default: 123)",
    )
    parser.add_argument(
        "--folds",
        default=11,
        type=int,
        help="For cross validation, specifies which fold will be "
        "used. All folds are used if set to 11 (default: 11)",
    )
    parser.add_argument(
        "--save",
        default=1,
        type=int,
        help="Parameter that specifies if results have to be saved. "
        "Results will be saved if set to 1 (default: 1)",
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
        "--spectral_analysis",
        default=False,
        type=bool,
        help="Compute spectral analysis or not. Default False",
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
        help="Threshold for graph similarity computation.",
    )
    parser.add_argument(
        "--sim_top_k",
        default=10,
        type=int,
        help="Top_k for graph similarity computation.",
    )
    parser.add_argument(
        "--Random_connectivity",
        default='No',
        type=str,
        help="Model with a random graph support with same density. Default: 'No' (options: 'Random', 'Worst')",
    )
    parser.add_argument(
        "--test_with_other_classifiers",
        default=False,
        type=bool,
        help="Test with other linear classifiers if True. Default: False.",
    )

    args = parser.parse_args()
    start_time = time.time()

    # GCN Parameters
    params = dict()
    params["model"] = args.model  # gcn model using chebyshev polynomials
    params["lrate"] = args.lrate  # Initial learning rate
    params["epochs"] = args.epochs  # Number of epochs to train
    params["dropout"] = args.dropout  # Dropout rate (1 - keep probability)
    params["hidden"] = args.hidden  # Number of units in hidden layers
    params["decay"] = args.decay  # Weight for L2 loss on embedding matrix.
    params["early_stopping"] = params[
        "epochs"
    ]  # Tolerance for early stopping (# of epochs). No early stopping if set to param.epochs
    params["max_degree"] = args.max_degree  # Maximum Chebyshev polynomial degree.
    params["jacobi_iteration"] = (
        args.jacobi_iteration
    )  # Jacobi algo iteration number for CayleyNets only
    params["depth"] = (
        args.depth
    )  # number of additional hidden layers in the GCN. Total number of hidden layers: 1+depth
    params["seed"] = args.seed  # seed for random initialisation

    # GCN Parameters
    params["num_features"] = (
        args.num_features
    )  # number of features for feature selection step
    params["num_training"] = (
        args.num_training
    )  # percentage of training set used for training
    params["spectral_analysis"] = args.spectral_analysis
    params["sim_method"] = args.sim_method
    params["sim_threshold"] = args.sim_threshold
    params["sim_top_k"] = args.sim_top_k
    atlas = args.atlas  # atlas for network construction (node definition)
    connectivity = (
        args.connectivity
    )  # type of connectivity used for network construction
    RandomConnectivityType = args.Random_connectivity
    test_with_other_classifiers = args.test_with_other_classifiers
    # Get class labels
    subject_IDs = Reader.get_ids()
    labels = Reader.get_subject_score(subject_IDs, score="DX_GROUP")

    # Get acquisition site
    sites = Reader.get_subject_score(subject_IDs, score="SITE_ID")
    unique = np.unique(list(sites.values())).tolist()

    num_classes = 2
    num_nodes = len(subject_IDs)

    # Initialise variables for class labels and acquisition sites
    y_data = np.zeros([num_nodes, num_classes])
    y = np.zeros([num_nodes, 1])
    site = np.zeros([num_nodes, 1], dtype=int)

    # Get class labels and acquisition site for all subjects
    for i in range(num_nodes):
        y_data[i, int(labels[subject_IDs[i]]) - 1] = 1
        y[i] = int(labels[subject_IDs[i]])
        site[i] = unique.index(sites[subject_IDs[i]])

    # Compute feature vectors (vectorised connectivity networks)
    features = Reader.get_networks(subject_IDs, kind=connectivity, atlas_name=atlas)

    # Compute population graph using gender and acquisition site
    graph = Reader.create_affinity_graph_from_scores(["SEX", "SITE_ID"], subject_IDs)
    # shuffle the graph depending if it is a random connectivity experience
    if RandomConnectivityType == 'Random':
        graph = Reader.random_affinity_graph_with_same_density(graph)
    if RandomConnectivityType == 'Worst':
        graph = Reader.create_worst_affinity_graph_from_scores(["SEX", "SITE_ID"], subject_IDs)

    # Folds for cross validation experiments
    skf = StratifiedKFold(n_splits=10)

    if args.folds == 11:  # run cross validation on all folds
        scores = Parallel(n_jobs=10)(
            delayed(train_fold)(
                train_ind,
                test_ind,
                test_ind,
                graph,
                features,
                y,
                y_data,
                params,
                subject_IDs,
                test_with_other_classifiers
            )
            for train_ind, test_ind in reversed(
                list(skf.split(np.zeros(num_nodes), np.squeeze(y)))
            )
        )

        print(scores)

        scores_acc = [x[0] for x in scores]
        scores_auc = [x[1] for x in scores]
        scores_lin = [x[2] for x in scores]
        scores_auc_lin = [x[3] for x in scores]
        fold_size = [x[4] for x in scores]
        if test_with_other_classifiers:
            scores_other_classifiers = {}
            for k in scores[0][5].keys():
                scores_other_classifiers[k] = [x[5][k] for x in scores]

        print("overall linear accuracy %f" + str(np.sum(scores_lin) * 1.0 / num_nodes))
        print("overall linear AUC %f" + str(np.mean(scores_auc_lin)))
        print("overall accuracy %f" + str(np.sum(scores_acc) * 1.0 / num_nodes))
        print("overall AUC %f" + str(np.mean(scores_auc)))

    else:  # compute results for only one fold
        cv_splits = list(skf.split(features, np.squeeze(y)))

        train = cv_splits[args.folds][0]
        test = cv_splits[args.folds][1]

        val = test

        scores = train_fold(
            train, test, val, graph, features, y, y_data, params, subject_IDs, test_with_other_classifiers
        )
        if not test_with_other_classifiers:
            scores_acc, scores_auc, scores_lin, scores_auc_lin, fold_size = scores
        else:
            scores_other_classifiers = {}
            for k in scores[0][5].keys():
                scores_other_classifiers[k] = scores[5][k]

        print("overall linear accuracy %f" + str(np.sum(scores_lin) * 1.0 / fold_size))
        print("overall linear AUC %f" + str(np.mean(scores_auc_lin)))
        print("overall accuracy %f" + str(np.sum(scores_acc) * 1.0 / fold_size))
        print("overall AUC %f" + str(np.mean(scores_auc)))

    if args.save == 1:
        result_name = f"ABIDE_classification_{args.model}_{args.depth}_{args.max_degree}_{args.sim_method}_RdGraph_{args.Random_connectivity}"
        if args.sim_threshold > 0:
            result_name += f"_{args.sim_threshold}"
        if args.sim_method == SimMethod.expo_topk.value:
            result_name += f"_{args.sim_top_k}"
            
        data = {
                "lin": scores_lin,
                "lin_auc": scores_auc_lin,
                "acc": scores_acc,
                "auc": scores_auc,
                "folds": fold_size,
            }
        if test_with_other_classifiers:
            for k,v in scores_other_classifiers.items():
                data[k] = v
                
        sio.savemat(
            "results/" + result_name + ".mat",
            data,
        )
        np.save("results/" + f"graphConnectivity_{args.model}_{args.depth}_{args.max_degree}_{args.sim_method}_RdGraph_{args.Random_connectivity}.npy", graph)


if __name__ == "__main__":
    main()
