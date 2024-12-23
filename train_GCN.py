# Copyright (c) 2016 Thomas Kipf
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from __future__ import division, print_function

import csv
import os
import random
import time

import numpy as np
import sklearn.metrics
import tensorflow as tf
from spectral_analysis import extract_chebyshev_coeffs, plot_chebyshev_filters

from gcn.models import MLP, Deep_Cayley_GCN, Deep_GCN
from gcn.utils import *

tf.compat.v1.disable_eager_execution()


def get_train_test_masks(labels, idx_train, idx_val, idx_test):
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask


def run_training(adj, features, labels, idx_train, idx_val, idx_test, params):
    # Set random seed
    random.seed(params["seed"])
    np.random.seed(params["seed"])
    tf.compat.v1.set_random_seed(params["seed"])

    # Settings
    flags = tf.compat.v1.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string(
        "model", params["model"], "Model string."
    )  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float("learning_rate", params["lrate"], "Initial learning rate.")
    flags.DEFINE_integer("epochs", params["epochs"], "Number of epochs to train.")
    flags.DEFINE_integer(
        "hidden1", params["hidden"], "Number of units in hidden layer 1."
    )
    flags.DEFINE_float(
        "dropout", params["dropout"], "Dropout rate (1 - keep probability)."
    )
    flags.DEFINE_float(
        "weight_decay", params["decay"], "Weight for L2 loss on embedding matrix."
    )
    flags.DEFINE_integer(
        "early_stopping",
        params["early_stopping"],
        "Tolerance for early stopping (# of epochs).",
    )
    flags.DEFINE_integer(
        "max_degree", params["max_degree"], "Maximum Chebyshev polynomial degree."
    )
    flags.DEFINE_integer("depth", params["depth"], "Depth of Deep GCN")
    flags.DEFINE_bool(
        "spectral_analysis",
        params["spectral_analysis"],
        "Perform filters spectral analysis or not.",
    )
    flags.DEFINE_integer(
        "jacobi_iteration",
        params["jacobi_iteration"],
        "Number of iteration for Jacobi algorithm.",
    )
    flags.DEFINE_string("sim_method", params["sim_method"], "")
    flags.DEFINE_integer("sim_top_k", params["sim_top_k"], "")
    flags.DEFINE_float("sim_threshold", params["sim_threshold"], "")

    # Create test, val and train masked variables
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_train_test_masks(
        labels, idx_train, idx_val, idx_test
    )

    # Some preprocessing
    features = preprocess_features(features)
    if FLAGS.model == "gcn":
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = Deep_GCN
    elif FLAGS.model == "gcn_cheby":
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = Deep_GCN
    elif FLAGS.model == "gcn_cayley":
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = Deep_Cayley_GCN
    elif FLAGS.model == "dense":
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError("Invalid argument for GCN model ")

    # Define placeholders
    placeholders = {
        "support": [
            tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)
        ],
        "features": tf.compat.v1.sparse_placeholder(
            tf.float32, shape=tf.constant(features[2], dtype=tf.int64)
        ),
        "phase_train": tf.compat.v1.placeholder_with_default(False, shape=()),
        "labels": tf.compat.v1.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        "labels_mask": tf.compat.v1.placeholder(tf.int32),
        "dropout": tf.compat.v1.placeholder_with_default(0.0, shape=()),
        "num_features_nonzero": tf.compat.v1.placeholder(
            tf.int32
        ),  # helper variable for sparse dropout
    }

    # Create model
    kwargs = {}
    if FLAGS.model == "gcn_cayley":
        kwargs = {
            "jacobi_iteration": FLAGS.jacobi_iteration,
            "order": FLAGS.max_degree,
            "adj_normalized": normalize_adj(adj),
        }
    model = model_func(
        placeholders,
        input_dim=features[2][1],
        depth=FLAGS.depth,
        logging=True,
        **kwargs,
    )

    # Initialize session
    sess = tf.compat.v1.Session()

    # Define model evaluation function
    def evaluate(feats, graph, label, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(feats, graph, label, mask, placeholders)
        feed_dict_val.update(
            {placeholders["dropout"]: 0.0, placeholders["phase_train"]: False}
        )

        # Get all activations
        outputs_to_check = [
            model.loss,
            model.accuracy,
            model.predict(),
        ]

        outs_val = sess.run(outputs_to_check, feed_dict=feed_dict_val)

        pred = outs_val[2]
        pred = pred[np.squeeze(np.argwhere(mask == 1)), :]
        lab = label
        lab = lab[np.squeeze(np.argwhere(mask == 1)), :]
        auc = sklearn.metrics.roc_auc_score(np.squeeze(lab), np.squeeze(pred))

        return outs_val[0], outs_val[1], auc, (time.time() - t_test)

    # Init variables
    sess.run(tf.compat.v1.global_variables_initializer())

    cost_val = []

    # Train model
    for epoch in range(params["epochs"]):
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(
            features, support, y_train, train_mask, placeholders
        )
        feed_dict.update(
            {placeholders["dropout"]: FLAGS.dropout, placeholders["phase_train"]: True}
        )

        # Training step
        outputs_to_check = [model.opt_op, model.loss, model.accuracy, model.predict()]

        outs = sess.run(outputs_to_check, feed_dict=feed_dict)

        pred = outs[3]

        pred = pred[np.squeeze(np.argwhere(train_mask == 1)), :]
        labs = y_train
        labs = labs[np.squeeze(np.argwhere(train_mask == 1)), :]
        train_auc = sklearn.metrics.roc_auc_score(np.squeeze(labs), np.squeeze(pred))

        # Validation
        cost, acc, auc, duration = evaluate(
            features, support, y_val, val_mask, placeholders
        )
        cost_val.append(cost)

        # Print results
        print(
            "Epoch:",
            "%04d" % (epoch + 1),
            "train_loss=",
            "{:.5f}".format(outs[1]),
            "train_acc=",
            "{:.5f}".format(outs[2]),
            "train_auc=",
            "{:.5f}".format(train_auc),
            "val_loss=",
            "{:.5f}".format(cost),
            "val_acc=",
            "{:.5f}".format(acc),
            "val_auc=",
            "{:.5f}".format(auc),
            "time=",
            "{:.5f}".format(time.time() - t + duration),
        )
        # Store results
        metrics = {
            "epoch": epoch + 1,
            "train_loss": outs[1],
            "train_acc": outs[2],
            "train_auc": train_auc,
            "val_loss": cost,
            "val_acc": acc,
            "val_auc": auc,
            "time": time.time() - t + duration,
        }

        csv_path = f"training_curve_{FLAGS.model}_{FLAGS.depth}_{FLAGS.max_degree}_{FLAGS.sim_method}.csv"
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(metrics)

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(
            cost_val[-(FLAGS.early_stopping + 1) : -1]
        ):
            print("Early stopping...")
            break

    print("Optimization Finished!")

    # Testing
    sess.run(tf.compat.v1.local_variables_initializer())
    test_cost, test_acc, test_auc, test_duration = evaluate(
        features, support, y_test, test_mask, placeholders
    )
    print(
        "Test set results:",
        "cost=",
        "{:.5f}".format(test_cost),
        "accuracy=",
        "{:.5f}".format(test_acc),
        "auc=",
        "{:.5f}".format(test_auc),
    )

    if FLAGS.spectral_analysis:
        learned_coeffs = extract_chebyshev_coeffs(sess, model)
        plot_file = f"learned_filters_{FLAGS.model}_{FLAGS.depth}_{FLAGS.max_degree}_{FLAGS.sim_method}"
        if FLAGS.sim_threshold > 0:
            plot_file += f"_{FLAGS.sim_threshold}"
        if FLAGS.sim_method == "expo_top_k":
            plot_file += f"_{FLAGS.sim_top_k}"
        plot_chebyshev_filters(
            adj,
            k=FLAGS.max_degree,
            plot_file=plot_file + ".png",
            learned_coeffs=learned_coeffs,
        )

    return test_acc, test_auc
