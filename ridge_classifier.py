import ridge_classifier as ridc 
import numpy as np
import pandas as pd
import scipy.io as sio
import plotting
import os
import utils
import ABIDEParser as Reader
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split, GridSearchCV
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, Ridge, LogisticRegression
import sklearn
from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_networks(subject_list,  kind="correlation", atlas_name="aal", variable="connectivity"):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks


    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """
    all_networks = []
    for subject in subject_list:
        fl = os.path.join(
            Reader.data_folder, subject, subject + "_" + atlas_name + "_" + kind + ".mat"
        )
        matrix = sio.loadmat(fl)[variable]
        
        all_networks.append(matrix)
    all_networks=np.array(all_networks)
    
    idx = np.triu_indices_from(all_networks[0], 1)
    norm_networks = np.arctanh(all_networks) 
    vec_networks = [mat[idx] for mat in norm_networks]
    matrix = np.vstack(vec_networks)
    return matrix

def get_x_data():
    acc_res = []
    subject_IDs = Reader.get_ids()
    x_data = get_networks(subject_IDs, kind="correlation", atlas_name="ho", variable="connectivity")

    num_classes = 2
    num_nodes = len(subject_IDs)
    labels = Reader.get_subject_score(subject_IDs, score="DX_GROUP")

    # Initialise variables for class labels and acquisition site
    y_data = np.zeros([num_nodes, num_classes])
    y = np.zeros([num_nodes, 1])
    # x_data = feature_selection(x_data, y, train_ind, 2000)
    # Get acquisition site
    site = Reader.get_subject_score(subject_IDs, score="SITE_ID")
    unique = np.unique(list(site.values())).tolist()
    
    # G et class labels and acquisition site for all subjects
    for i in range(num_nodes):
        y_data[i, int(labels[subject_IDs[i]]) - 1] = 1
        y[i] = int(labels[subject_IDs[i]])
        site[i] = unique.index(site[subject_IDs[i]])
        
    return x_data, y, num_nodes

def apply_PCA_on_features(n_components, x_data):
    pca = PCA(n_components=n_components)
    matrix = pca.fit_transform(x_data)    
    # print(pca.explained_variance_)
    return matrix

def feature_selection(matrix, labels, train_ind, fnum):
    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum, step=100, verbose=0)
    featureX = matrix[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(matrix)
    return x_data

def sensitivity_on_PCA(alphas, n_components_list, delete_last_data, save_filename:str='df_ridge_PCA_sensitivity'):
    print("sensitivity of PCA\n")
    data = {"alpha":[], "PCA_nb_components":[], "accuracy_mean":[]}
    
    for n_components in n_components_list:
        print("n_components: ", n_components)
        x_data, y, num_nodes = get_x_data()
        x_data = apply_PCA_on_features(n_components, x_data)
        print("x_data.shape: ", x_data.shape)
        for alpha in alphas:
            averaged_acc_list = []
            for k in range(10):
                acc_list = []
                skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
                for train_ind, test_ind in reversed(list(skf.split(np.zeros(num_nodes), np.squeeze(y)))):
                    # Linear classifier
                    clf = RidgeClassifier(alpha=alpha)
                    clf.fit(x_data[train_ind,:], y[train_ind].ravel())
                    # Compute the accuracy
                    lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
                    acc_list.append(lin_acc)
                averaged_acc_list.append(np.mean(np.array(acc_list)))

            data["alpha"].append(alpha)
            data["PCA_nb_components"].append(n_components)
            data["accuracy_mean"].append(np.array(averaged_acc_list).mean())
            
    df = pd.DataFrame(data=data)
    if not delete_last_data:
        df2 = pd.read_csv(f'results/linearRegression/{save_filename}.csv') 
        df2.drop(axis=1, labels="Unnamed: 0", inplace=True)
        df3 = pd.concat([df,df2])
    else:
        df3 = df
    df3.to_csv(f'results/linearRegression/{save_filename}.csv')


def sensitivity_on_RFE(alphas, n_components_list, delete_last_data):
    print("sensitivity of RFE selection\n")

    accuracy_array  = np.zeros((len(n_components_list), len(alphas)))
    for k, n_components in enumerate(n_components_list):
        print("n_components: ", n_components)
        _, y, num_nodes = get_x_data()
        n_splits = 10
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
        # cur = 0
        for train_ind, test_ind in reversed(list(skf.split(np.zeros(num_nodes), np.squeeze(y)))):
            x_data, y, num_nodes = get_x_data()
            x_data = feature_selection(x_data, y,train_ind, n_components)
            # cur += 1
            # print("fold ", cur)
            for l, alpha in enumerate(alphas):
                acc_list = []
                for j in range(10):
                    # Linear classifier
                    clf = RidgeClassifier(alpha=alpha)
                    clf.fit(x_data[train_ind,:], y[train_ind].ravel())
                    # Compute the accuracy
                    lin_acc = clf.score(x_data[test_ind, :], y[test_ind].ravel())
                    acc_list.append(lin_acc)
                acc_mean_for_a_given_fold = np.mean(np.array(acc_list))
                accuracy_array[k,l] += acc_mean_for_a_given_fold
    accuracy_array /=  n_splits   
    
    
    data = {"alpha":[], "RFE_nb_components":[], "accuracy_mean":[]}
    for k, n_components in enumerate(n_components_list):
        for l, alpha in enumerate(alphas):
            data["alpha"].append(alpha)
            data["RFE_nb_components"].append(n_components)
            data["accuracy_mean"].append(accuracy_array[k,l])
            
    df = pd.DataFrame(data=data)
    if not delete_last_data:
        df2 = pd.read_csv('results/linearRegression/df_ridge_RFE_sensitivity.csv') 
        df2.drop(axis=1, labels="Unnamed: 0", inplace=True)
        df3 = pd.concat([df,df2])
    else: 
        df3 = df
    df3.to_csv('results/linearRegression/df_ridge_RFE_sensitivity.csv')

if __name__ == "__main__":
    
    delete_last_data = True
    alphas = range(100, 1500, 5) # np.array([0.001,0.01,0.1,1., 10, 50] + [k for k in range(100, 2000, 25)] + [3000] )
    n_components_list =  np.arange(150, 250, 5)# np.concatenate([np.arange(2,99, 10), np.arange(110,400, 20), np.array([550, 650, 750, 800, 850])])
    sensitivity_on_PCA(alphas, n_components_list, delete_last_data, save_filename='df_ridge_PCA_sensitivity_precise_tuning')

    # delete_last_data = False
    # alphas_RFE = np.array([0.001,0.01,0.1,1., 10, 50] + [k for k in range(100, 2001, 100)] )
    # n_components_list_RFE = np.arange(1250, 6001, 500) #[3, 5, 7, 10, 15, 20, 50, 100, 500, 750] #np.arange(1250, 6001, 500)# np.array([3, 5, 7, 10, 15, 20, 50, 100, 500, 750]) #np.arange(1000, 6001, 500) # np.concatenate([np.array([3, 5, 7, 10, 15, 20, 50, 100, 500, 750]), ])
    # sensitivity_on_RFE(alphas_RFE, n_components_list_RFE, delete_last_data)