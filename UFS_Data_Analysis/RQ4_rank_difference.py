import os
import pandas as pd
import numpy as np
import json

def rank1(FS_list, folder_path, NONE_dict, classifier):
    FS_NANE = []
    rank1_difference = []
    save_rank1_difference = np.full((len(FS_list) - 1, 65), np.nan)
    K = 0
    for i in range(len(FS_list)):
        if 'NONE' not in FS_list[i]:
            print("FS_list[i]", FS_list[i])
            FS_dict_path = os.path.join(folder_path, FS_list[i])
            with open(FS_dict_path, 'r') as json_file:
                FSi_dict = json.load(json_file)

            # Calculating Rank Difference
            FS_rank1 = FSi_dict["1"]
            rank_difference_FSi = []

            # Find the rank of each element in FS rank1 in NONE
            for j in range(len(FS_rank1)):
                keys_for_value = [key for key, value in NONE_dict.items() if FS_rank1[j] in value]
                rank_differencej = int(keys_for_value[0]) - 1
                rank_difference_FSi.append(rank_differencej)

            rank1_difference.append(rank_difference_FSi)
            save_rank1_difference[K, 0:len(rank_difference_FSi)] = rank_difference_FSi
            FS_NANE.append(FS_list[i][len(classifier) + 1: len(FS_list[i]) - 11])
            K = K + 1

    return FS_NANE, rank1_difference, save_rank1_difference

def rq4(classifier):
    dataset_type = 'AEEEM'
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    f_path = os.path.join(parent_dir, "Data_Analysis_Results", "RQ4", dataset_type)

    folder_path = os.path.join(f_path, classifier, 'step2-json')
    FS_list = os.listdir(folder_path)
    print("FS_list", FS_list)

    #The feature importance of the model constructed from the original dataset (i.e., NONE)
    NONE_dict_path = os.path.join(folder_path, classifier + '-NONE-step2.json')
    with open(NONE_dict_path, 'r') as json_file:
        NONE_dict = json.load(json_file)
    print(NONE_dict)
    NONE_dict_length = len(NONE_dict)
    if NONE_dict_length > 5: # Categorize all ranks greater than 5 as 5
        for q in range(6,NONE_dict_length + 1):
            if str(q) in NONE_dict:
                NONE_dict['5'] += NONE_dict[str(q)]
                del NONE_dict[str(q)]
    length = 5

    FS_NANE, rank_difference, save_rank1_difference = rank1(FS_list, folder_path, NONE_dict, classifier)

    rank1_percentage = np.zeros((length, len(FS_NANE)))
    x = [0, 1, 2, 3, 4]
    for p in range(len(rank_difference)):
        curr_FS_rank1_difference = rank_difference[p]
        y = [curr_FS_rank1_difference.count(i) / len(curr_FS_rank1_difference) for i in x]
        rank1_percentage[:,p] = y

    save_rank1_percentage = pd.DataFrame(rank1_percentage, index=['0', '1', '2', '3', '≥4'], columns=FS_NANE)
    save_rank1_percentage.to_csv(os.path.join(f_path, classifier + '_rank1_difference.csv'))

classifier = ['LR', 'RF', 'SC', 'KMEANS']
for i in range(len(classifier)):
    rq4(classifier[i])