import warnings
from ModelConstruction.xgb_rfclassifiers import xgboost,random_forest_classifier
import pandas as pd
import numpy as np
import rankMeasure
from ModelConstruction import partition_dataset
from ModelEvalution import rankMeasure

warnings.filterwarnings('ignore')

def forward_search(dataset_train, dataset_test, classifier_name, classifier, dataset_name):
    """
    apply forkward search for wrapper-based feature subset selection
    :param dataset_train:
    :param dataset_test:
    :param classifier:
    :return:
    """
    training_data_y = dataset_train.iloc[:, -1]
    training_data_x = dataset_train.iloc[:, :-1]

    testing_data_y = dataset_test.iloc[:, -1]
    testing_data_x = dataset_test.iloc[:, :-1]

    if dataset_name == 'JIRA':
        testingcode = testing_data_x.iloc[:, 27].tolist()#JIRA
    elif dataset_name == 'AEEEM':
        testingcode = testing_data_x.iloc[:, 25].tolist()#AEEEM
    # testingcode = testing_data_x.iloc[:, 27].tolist()#JIRA
    # testingcode = testing_data_x.iloc[:, 25].tolist()#AEEEM

    best_Recall = 0
    column = -1

    for i in range(training_data_x.shape[1]):
        tmp_training_data_x = training_data_x[i].to_frame()
        tmp_testing_data_x = testing_data_x[i].to_frame()
        # print(tmp_training_data_x.columns.values)
        new_model = classifier_name[classifier](tmp_training_data_x, training_data_y)
        # new_preds = new_model.predict_proba(tmp_testing_data_x)
        # new_pofb = Pofb(new_preds, testingcode, testing_data_value)
        new_preds = new_model.predict_proba(tmp_testing_data_x)
        new_preds = new_preds[:, 1]

        #这里是工作量感知的
        new_Recall, _ = rankMeasure.rank_measure(new_preds, testingcode, testing_data_y)

        if new_Recall >= best_Recall:
            best_Recall = new_Recall
            column = i
            new_training_data_x = tmp_training_data_x
            new_testing_data_x = tmp_testing_data_x

    # print("column", column)

    old_column = column
    selected_cloumn = []
    dropped_column = []
    selected_cloumn.append(column)

    for i in range(training_data_x.shape[1] - 1):
        candidate_columns = []
        for col in training_data_x.columns.values.tolist():
            if col not in new_training_data_x.columns.values.tolist():
                candidate_columns.append(col)
        # print("candidate_columns", candidate_columns)
        print(new_training_data_x.columns.values.tolist())
        for j in candidate_columns:
            tmp_training_data_x = pd.concat([new_training_data_x, training_data_x[j].to_frame()], axis=1)
            tmp_testing_data_x = pd.concat([new_testing_data_x, testing_data_x[j].to_frame()], axis=1)
            # print(tmp_training_data_x.columns.values)
            new_model = classifier_name[classifier](tmp_training_data_x, training_data_y)
            # new_preds = new_model.predict_proba(tmp_testing_data_x)
            # new_pofb = Pofb(new_preds, testingcode, testing_data_value)
            new_preds = new_model.predict_proba(tmp_testing_data_x)
            new_preds = new_preds[:, 1]

            new_Recall, _ = rankMeasure.rank_measure(new_preds, testingcode, testing_data_y)

            if new_Recall >= best_Recall:
                best_Recall = new_Recall
                column = j

        # print("old_column", old_column)
        # print("column", column)

        if old_column != column:
            new_training_data_x = pd.concat([new_training_data_x, training_data_x[column].to_frame()], axis=1)
            new_testing_data_x = pd.concat([new_testing_data_x, testing_data_x[column].to_frame()], axis=1)
            selected_cloumn.append(column)
            old_column = column
            if i == training_data_x.shape[1] - 2:
                for p in training_data_x.columns.values.tolist():
                    if p not in selected_cloumn:
                        dropped_column.append(p)
                # print(selected_cloumn)
                return dropped_column, selected_cloumn, best_Recall
        else:
            for k in training_data_x.columns.values.tolist():
                if k not in selected_cloumn:
                    dropped_column.append(k)
            # print(selected_cloumn)
            return dropped_column, selected_cloumn, best_Recall

def XGB_RF_ADB_feature_selection(data, classifier, dataset_name):
    classifier_name = {
        'XGB': xgboost,
        'RF': random_forest_classifier,
    }

    train_data, train_label, test_data, test_label, _ = partition_dataset.partition(data, 42)
    train_label = train_label.reshape((len(train_label)), 1)
    test_label = test_label.reshape((len(test_label)), 1)

    dataset_train = np.concatenate((train_data, train_label), axis=1)
    dataset_test = np.concatenate((test_data, test_label), axis=1)

    dataset_train = pd.DataFrame(dataset_train)
    dataset_test = pd.DataFrame(dataset_test)
    # print(dataset_test.shape)

    _, selected_cloumn2, _ = forward_search(dataset_train, dataset_test, classifier_name, classifier, dataset_name)
    print("selected_cloumn2", selected_cloumn2)

    return selected_cloumn2
