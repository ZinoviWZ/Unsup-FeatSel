import os
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()
import pandas as pd
import numpy as np
sk = importr('ScottKnottESD')
import json

def scott_knott(data, lenth):
    r_sk = sk.sk_esd(data, version='np')
    a = []
    for i in range(lenth):
        a.append(int(r_sk[1][i]))

    column_order = list(r_sk[3])
    column_order = [i - 1 for i in column_order]

    ranking = pd.DataFrame(
        {
            "technique": [data.columns[i] for i in column_order],
            "rank": a,
        }
    )

    return column_order, a, ranking, ranking["technique"].tolist()

def replace_nan_with_zero(array):
    nan_mask = np.isnan(array)

    array[nan_mask] = 0

    return array

def re_ranking(tech_Index, tech_rank):
    re_list = []
    for i in range(len(tech_rank)):
        k = tech_Index.index(i)
        re_list.append(tech_rank[k])

    return re_list

def double_feature_importance_dict(classifier):
    dataset_type = 'JIRA'

    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    folder_path = os.path.join(parent_dir, "Basic_Experiment_Results", "RQ4", dataset_type, classifier)

    P = os.path.join(parent_dir, 'dataset', dataset_type)
    D = pd.read_csv(os.path.join(P, os.listdir(P)[0]))
    featute_name = list(D.columns)[:len(list(D.columns)) - 1]
    # print(len(featute_name))

    FS_list = os.listdir(folder_path)
    print(FS_list)

    for i in range(len(FS_list)):
        FSi_dataset_path = os.path.join(folder_path, FS_list[i])
        FS_dataset_list = os.listdir(FSi_dataset_path)

        #Calculate the feature importance ranking of feature selection algorithms on each dataset
        step1 = np.zeros((len(FS_dataset_list), len(featute_name)))
        for j in range(len(FS_dataset_list)):
            feature_importance_path = os.path.join(FSi_dataset_path, FS_dataset_list[j])
            feature_importance = np.loadtxt(feature_importance_path, delimiter=',')

            if dataset_type == 'AEEEM':
                feature_importance = feature_importance[:, 0:61]  # AEEEM

            #Importance scores for features that were not selected are replaced with 0
            feature_importance = replace_nan_with_zero(feature_importance)

            D = pd.DataFrame(feature_importance, columns=featute_name)
            tech_Index1, tech_rank1, ranking1, tech_name1 = scott_knott(D, len(featute_name))

            re_list1 = re_ranking(tech_Index1, tech_rank1)
            step1[j, :] = re_list1

        #Save the feature importance ranking on each dataset
        step1 = pd.DataFrame(step1.astype('int'), columns=featute_name, index=FS_dataset_list)
        step1_path = os.path.join(parent_dir, 'Data_Analysis_Results', 'RQ4', dataset_type, classifier, 'step1', classifier + '-' + FS_list[i] + '-step1.csv')
        os.makedirs(os.path.dirname(step1_path), exist_ok=True)
        step1.to_csv(step1_path)

        step1 = np.negative(step1)

        tech_Index2, tech_rank2, ranking2, tech_name2 = scott_knott(step1, len(featute_name))

        tech_dict = {}
        for key, value in zip(tech_rank2, tech_Index2):
            if key in tech_dict:
                tech_dict[key].append(value)
            else:
                tech_dict[key] = [value]

        step2_path = os.path.join(parent_dir, 'Data_Analysis_Results', 'RQ4', dataset_type, classifier, 'step2-json', classifier + '-' + FS_list[i] + '-step2.json')
        os.makedirs(os.path.dirname(step2_path), exist_ok=True)
        with open(step2_path, 'w') as json_file:
            json.dump(tech_dict, json_file)

        csv_path = os.path.join(parent_dir, 'Data_Analysis_Results', 'RQ4', dataset_type, classifier, 'step2', classifier + '-' + FS_list[i] + '-step2.csv')
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        ranking2.to_csv(csv_path)

classifier = ['LR', 'RF', 'SC', 'KMEANS']
for i in range(len(classifier)):
    double_feature_importance_dict(classifier[i])