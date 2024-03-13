from eli5.sklearn import PermutationImportance
import os
import numpy as np
from ModelConstruction import SC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import copy

def save_importance(feature_importance, idx, FSname, feature_num, project_name, classifier_name, dataset_name, results_dir):
    idx = idx.reshape((1, -1))
    path2 = os.path.join(results_dir, 'RQ4', dataset_name, classifier_name, FSname, project_name[
             0:len(project_name) - 4] + '_' +  FSname + '.csv')
    os.makedirs(os.path.dirname(path2), exist_ok=True)

    feature_importances_SKESD = np.full((1, feature_num), np.nan)
    a = list(feature_importance)
    print("feature_importance", feature_importance)
    print('idx', idx)
    b = list(idx[0])

    if FSname == 'NONE':
        feature_importances_SKESD[0,:] = feature_importance
    else:
        for i in range(len(b)):
            if int(b[i]) - 1 >= 0:
                feature_importances_SKESD[0, int(b[i]) - 1] = a[i]
            else:
                break

    with open(path2, 'a') as file:
        np.savetxt(file, feature_importances_SKESD, delimiter=',')

def supervised_model_interpretation(model, test_data, test_label, project_name, classifier_name, FSname, idx, feature_num,
                             dataset_name, results_dir):
    #Calculating the importance score of features
    perm = PermutationImportance(model).fit(test_data, test_label)
    feature_importance = perm.feature_importances_
    print(classifier_name, "feature_importance", feature_importance)

    if FSname != 'CFS' and FSname != 'XGBF' and FSname != 'RFF':
        save_importance(feature_importance, idx, FSname, feature_num, project_name, classifier_name, dataset_name, results_dir)

def labelCluster(data, culster):
    # label the defective and non-defective clusters
    rs = np.sum(data, axis=1)  # Summing operations on elements along axis 1 (i.e., rows)
    if len(rs[culster == 1]) == 0:
        preLabel2 = (culster == 1)
    if len(rs[culster == 0]) == 0:
        preLabel2 = (culster == 0)
    elif np.mean(rs[culster == 1]) < np.mean(rs[culster == 0]):  # culster == 1 is class 1，culster == 0 is class 2
        preLabel2 = (culster == 0)
    else:
        preLabel2 = (culster == 1)
    preLabel2 = preLabel2 + 0
    return preLabel2

def G2PC(preLabel, test_data, test_label, model):
    scaler = StandardScaler()
    Pct_Chg = []
    accurary0 = accuracy_score(test_label, preLabel)
    for i in range(test_data.shape[1]):
        acc = []
        for k in range(5):
            # X1 = X.copy()
            X1 = copy.deepcopy(test_data)
            temp = test_data[:, i]
            temp1 = np.random.permutation(temp)
            X1[:, i] = temp1  # Randomize the i+1th feature

            if model == 'KMEANS':
                culster1 = KMeans(n_clusters=2).fit_predict(scaler.fit_transform(X1))
                preLabel1 = labelCluster(scaler.fit_transform(X1), culster1)
            elif model == 'SC':
                preLabel1 = SC.SC(X1)

            accurary1 = accuracy_score(test_label, preLabel1)

            temp_acc =  accurary0 - accurary1
            acc.append(temp_acc)
        Pct_Chg.append(acc)
    # print("Pct_Chg", Pct_Chg)
    row_means = np.mean(np.array(Pct_Chg), axis=1)
    return row_means


def unsupervised_model_interpretation(predict_label, test_data, test_label, classifier_name, project_name, FSname, idx,
                                      feature_num, dataset_name, results_dir):

    feature_importance = G2PC(predict_label, test_data, test_label, classifier_name)
    print("feature_importanceKMEANS", feature_importance)

    if FSname != 'CFS' and FSname != 'XGBF' and FSname != 'RFF':
        save_importance(feature_importance, idx, FSname, feature_num, project_name, classifier_name, dataset_name, results_dir)
