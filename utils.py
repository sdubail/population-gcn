import networkx as nx
import numpy as np
import scipy.io
import pandas as pd

def get_graph_density(graph:np.ndarray):
    return nx.density(nx.from_numpy_array(graph))

def load_data(filenames_for_experience_with_rd_graph_support:list, filename_best_model:str, other_classifiers:bool):
    data_best_model = scipy.io.loadmat(filename_best_model)
    data_rd_graph_list = [scipy.io.loadmat(filename) for filename in filenames_for_experience_with_rd_graph_support]
    # retrieve acc and auc
    data = {
        'ridge_classifier_acc': data_best_model['lin'].flatten(),#/100,
        'ridge_classifier_auc': data_best_model['lin_auc'].flatten(),
        'GCN_acc': data_best_model['acc'].flatten(),
        'GCN_auc': data_best_model['auc'].flatten(),
        'folds': data_best_model['folds'].flatten()
    }
    if other_classifiers:
        for key in list(scipy.io.loadmat(filename_best_model).keys())[8:]:
            data[key] = data_best_model[key].flatten()
    
    for k, data_rd_graph in enumerate(data_rd_graph_list):
        data[f"rd{k+1}_GCN_acc"] = data_rd_graph['acc'].flatten()/100
        data[f"rd{k+1}_GCN_auc"] = data_rd_graph['auc'].flatten()

    df = pd.DataFrame(data)
    transform_acc_in_ratio(df)
    return df

def transform_acc_in_ratio(df):
    for col in list(df.columns):
        if "acc" in col:
            df[col] = df[col] / df['folds']