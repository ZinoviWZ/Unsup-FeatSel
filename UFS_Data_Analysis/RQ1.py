import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def get_set(lists):
    # Compute the union of sets
    union_set = set()
    for l in lists:
        union_set.update(l)

    # Compute intersection
    intersection_set = set(lists[0])
    for l in lists[1:]:
        intersection_set.intersection_update(l)

    rate = float(len(intersection_set)) / float(len(union_set))

    return rate

def rq1():
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    print(parent_dir)
    folder_path = os.path.join(parent_dir, "Basic_Experiment_Results\RQ1\AEEEM")
    dirs = os.listdir(folder_path)


    K = 0
    FSi_FSj_Median_all_dataset = np.zeros((len(dirs),21,21,100))
    for dir in dirs:
        dir_path = os.path.join(folder_path, dir)
        file_list = os.listdir(dir_path)
        csv_list = [file_name for file_name in file_list if file_name.endswith(".csv")]
        print("find csv file : ", csv_list)

        csv_list_name = []
        for i in range(len(csv_list)):
            # FSi_FSj_median_value[i, i] = 1
            csv_list_name.append(csv_list[i][0:len(csv_list[i]) - 4])
        if 'jelsr_lle' in csv_list_name:
            index = csv_list_name.index('jelsr_lle')
            csv_list_name[index] = 'JELSR'
        if 'Inf_FS' in csv_list_name:
            index = csv_list_name.index('Inf_FS')
            csv_list_name[index] = 'Inf-FS'
        if 'Inf_FS2020' in csv_list_name:
            index = csv_list_name.index('Inf_FS2020')
            csv_list_name[index] = 'Inf-FS2020'

        #---------------------------------
        #Compute consistency between pairwise feature selection algorithms
        for i in range(len(csv_list)):
            for j in range(i+1,len(csv_list)):
                path_i = os.path.join(dir_path, csv_list[i])
                path_j = os.path.join(dir_path, csv_list[j])

                rate_list = []
                IJ = []
                for p in range(100):
                    with open(path_i, 'r') as file:
                        reader = csv.reader(file)
                        for _ in range(p):
                            next(reader)
                        row_i = next(reader)
                    with open(path_j, 'r') as file:
                        reader = csv.reader(file)
                        for _ in range(p):
                            next(reader)
                        row_j = next(reader)
                    IJ.append(row_i)
                    IJ.append(row_j)

                    rate = get_set(IJ)
                    rate_list.append(rate)

                    IJ = []

                rate_list = np.array(rate_list)
                FSi_FSj_Median_all_dataset[K, i, j, :] = rate_list

        K = K + 1

    #Obtaining the median subset consistency of two-by-two feature selection technique metrics
    FSi_FSj_Median_all_dataset_draw = np.zeros((21,21))
    for i in range(len(csv_list)):
        for j in range(i + 1, len(csv_list)):
            median_list = []
            for k in range(len(dirs)):
                median_list.append(FSi_FSj_Median_all_dataset[k][i, j])

            a_median_list = np.array(median_list)
            median = np.median(a_median_list)
            FSi_FSj_Median_all_dataset_draw[i,j] = median


    percentage_array_2d = FSi_FSj_Median_all_dataset_draw * 100
    percentage_string_array_2d = np.array([["{:.1f}%".format(val) for val in row] for row in percentage_array_2d])

    #Remove the bottom half of the matrix
    for i in range(21):
        for j in range(0, i + 1):
            percentage_array_2d[i,j] = np.nan
            percentage_string_array_2d[i,j] = np.nan

    annot_kws = {"size": 12}

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(22, 22))
    l = percentage_array_2d.shape[0]
    ax = sns.heatmap(percentage_array_2d[:l-1,1:l], xticklabels=csv_list_name[1:], yticklabels=csv_list_name[:l-1], annot=percentage_string_array_2d[:l-1,1:l], fmt='', cmap='coolwarm', linewidths=.5, square=True, annot_kws=annot_kws, cbar_kws={"shrink": 0.2, "location": "top", "label": "Percentage of Consistency"})
    ax.tick_params(left=False, bottom=False)

    # Adjusting the font size of the color bar label
    cbar = ax.collections[0].colorbar
    # Adjust the font size of the colorbar label
    cbar.set_label("Percentage of Consistency", fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    # Adjusting the font size of the x-axis scale labels
    ax.set_xticklabels(csv_list_name[1:], size=18, rotation=90)
    ax.set_yticklabels(csv_list_name[:l-1], size=18)

    plt.subplots_adjust(left=0, right=1.1, bottom=0.1, top=1.05)
    PATH = os.path.join(parent_dir, 'Data_Analysis_Results', 'RQ1')
    os.makedirs(PATH, exist_ok=True)
    plt.savefig(os.path.join(PATH, 'AEEEM_median_dataset.pdf'))

    plt.show()
rq1()
