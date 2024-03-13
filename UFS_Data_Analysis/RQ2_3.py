import os
import pandas as pd
import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import RQ2_3_draw

pandas2ri.activate()
sk = importr('ScottKnottESD')

def scott_knott(data):
    r_sk = sk.sk_esd(data, version = 'np')
    a = []
    for i in range(25):
        a.append(int(r_sk[1][i]))

    column_order = list(r_sk[3])
    column_order = [i - 1 for i in column_order]

    ranking = pd.DataFrame(
        {
            "technique": [data.columns[i] for i in column_order],
            "rank": a,
        }
    )

    return column_order, a, ranking

def re_ranking(tech_Index, tech_rank):
    yuanshi_list = []
    for i in range(len(tech_rank)):
        k = tech_Index.index(i)
        yuanshi_list.append(tech_rank[k])

    return yuanshi_list

# classifiers = ['LR', 'RF']
classifiers = ['SC', 'KMEANS']
for classifier in classifiers:
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    print(parent_dir)
    folder_path = os.path.join(parent_dir, "Basic_Experiment_Results", "RQ3", classifier)
    print(folder_path)

    performances = ['AUC', 'MCC']#When the XGBF and RFF algorithms internally use AUC as an evaluation metric, it applies to RQ2-1 and RQ3-1
    # performances = ['IFA_effort', 'recall_effort']#When the XGBF and RFF algorithms internally use Recall@20% as an evaluation metric, it applies to RQ2-2 and RQ3-2

    for performance in performances:
        dataset_path = os.path.join(folder_path, performance)
        file_list = os.listdir(dataset_path)
        file_list1 = [file_name.replace('.csv', '') for file_name in file_list]

        step1 = np.zeros((len(file_list), 25))
        k = 0
        for csv_file_name in file_list:
            file_path = os.path.join(dataset_path, csv_file_name)
            dataset = pd.read_csv(file_path)
            dataset = dataset.values

            # First NPSKESD
            if performance == 'IFA_effort':
                dataset = np.negative(dataset)

            D = pd.DataFrame(dataset, columns=['UFSoL', 'UDFS', 'U2FS', 'SRCFS', 'SOGFS', 'SOCFS', 'RUFS', 'NDFS',
                                               'LLCFS', 'Inf-FS2020', 'Inf-FS', 'FSASL', 'FMIUFS', 'CNAFS', 'JELSR',
                                               'GLSPFS', 'LS', 'MCFS', 'NONE', 'REFS', 'AutoSpearman', 'SPEC', 'CFS',
                                               'XGBF', 'RFF'])

            tech_Index1, tech_rank1, ranking1 = scott_knott(D)
            RElist = re_ranking(tech_Index1, tech_rank1)
            step1[k, :] = RElist

            k = k + 1

        step1 = pd.DataFrame(step1.astype('int'), columns=['UFSoL', 'UDFS', 'U2FS', 'SRCFS', 'SOGFS', 'SOCFS', 'RUFS', 'NDFS', 'LLCFS',
                                      'Inf-FS2020', 'Inf-FS', 'FSASL', 'FMIUFS', 'CNAFS', 'JELSR', 'GLSPFS',
                                      'LS', 'MCFS', 'NONE', 'REFS', 'AutoSpearman', 'SPEC', 'CFS', 'XGBF', 'RFF'], index=file_list1)
        csv_path = os.path.join(parent_dir, "Data_Analysis_Results", "RQ3", classifier, 'step1')
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        step1.to_csv(os.path.join(csv_path, classifier + '_' + performance + '_step1.csv'))

        # Second NPSKESD
        step11 = np.negative(step1)
        tech_Index, tech_rank, ranking2 = scott_knott(step11)
        csv_path = os.path.join(parent_dir, "Data_Analysis_Results", "RQ3", classifier, 'step2')
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        ranking2.to_csv(os.path.join(csv_path, classifier + '_' + performance + '_step2.csv'))

        # Draw a diagram and save it
        RQ2_3_draw.draw(step1, ranking2, classifier, performance, parent_dir)
