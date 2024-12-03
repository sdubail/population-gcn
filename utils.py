import networkx as nx
import numpy as np
import scipy.io
import pandas as pd
import ABIDEParser as Reader
import os

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
        data[f"rd{k+1}_GCN_acc"] = data_rd_graph['acc'].flatten()
        data[f"rd{k+1}_GCN_auc"] = data_rd_graph['auc'].flatten()

    df = pd.DataFrame(data)
    transform_acc_in_ratio(df)
    return df

def transform_acc_in_ratio(df):
    for col in list(df.columns):
        if "acc" in col:
            df[col] = df[col] / df['folds']
            

def get_label_group():
    # Get class labels
    subject_IDs = Reader.get_ids()
    labels = Reader.get_subject_score(subject_IDs, score="DX_GROUP")
    # Compute feature vectors (vectorized connectivity networks)
    # features = Reader.get_networks(subject_IDs, kind=connectivity, atlas_name=atlas)
    # Compute population graph using gender and acquisition site
    # graph = Reader.create_affinity_graph_from_scores(["SEX", "SITE_ID"], subject_IDs)
    return subject_IDs, labels

def list_folders_in_directory(directory_path):
    try:
        # Get a list of all items in the directory
        all_items = os.listdir(directory_path)
        
        # Filter the items to include only directories
        folders = [item for item in all_items if os.path.isdir(os.path.join(directory_path, item))]
        
        return folders
    except Exception as e:
        print(f"An error occurred: {e}")
        return []