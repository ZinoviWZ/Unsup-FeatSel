from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from ModelConstruction import SC
from ModelEvalution import classificationMeasure, rankMeasure
from sklearn.preprocessing import StandardScaler
from ModelConstruction import ModelInterpretation

class Classifiers:
    def __init__(self, test_label, test_data, fs_number, testLOC, project_name, fs_name, selected_idx, feature_num,
                                                                dataset_name, results_dir):
        self.test_label = test_label
        self.test_data = test_data
        self.fs_number = fs_number
        self.testLOC = testLOC
        self.project_name = project_name
        self.fs_name = fs_name
        self.selected_idx = selected_idx
        self.feature_num = feature_num
        self.dataset_name = dataset_name
        self.results_dir = results_dir

    def LR_classifier(self, train_label, train_data, evalution_measure_lr):
        scaler = StandardScaler()
        train_data1 = scaler.fit_transform(train_data)
        test_data1 = scaler.fit_transform(self.test_data)

        model = LogisticRegression()
        model.fit(train_data1, train_label)
        predict_label = model.predict(test_data1)
        predict_prob = model.predict_proba(test_data1)
        predict_prob = predict_prob[:, 1]

        measure_classifier = classificationMeasure.evaluateMeasure(self.test_label, predict_label)
        measure_effort = rankMeasure.rank_measure(predict_prob, self.testLOC, self.test_label)

        for j in range(2):
            evalution_measure_lr[j][self.fs_number] = measure_classifier[j]
        for j in range(2):
            evalution_measure_lr[j + 2][self.fs_number] = measure_effort[j]

        #Using permutetion feature importance interpret the model
        ModelInterpretation.supervised_model_interpretation(model, test_data1, self.test_label, self.project_name, 'LR', self.fs_name,
                                        self.selected_idx, self.feature_num, self.dataset_name, self.results_dir)
        return evalution_measure_lr


    def RF_classifier(self, train_label, train_data, evalution_measure_rf):
        scaler = StandardScaler()
        train_data1 = scaler.fit_transform(train_data)
        test_data1 = scaler.fit_transform(self.test_data)

        rfc = RandomForestClassifier(n_estimators=100, random_state=0)
        rfc.fit(train_data1, train_label)
        predict_label = rfc.predict(test_data1)
        predict_prob = rfc.predict_proba(test_data1)
        predict_prob = predict_prob[:, 1]

        measure_classifier = classificationMeasure.evaluateMeasure(self.test_label, predict_label)
        measure_effort = rankMeasure.rank_measure(predict_prob, self.testLOC, self.test_label)

        for j in range(2):
            evalution_measure_rf[j][self.fs_number] = measure_classifier[j]
        for j in range(2):
            evalution_measure_rf[j + 2][self.fs_number] = measure_effort[j]

        ModelInterpretation.supervised_model_interpretation(rfc, test_data1, self.test_label, self.project_name, 'RF', self.fs_name,
                                                            self.selected_idx, self.feature_num, self.dataset_name, self.results_dir)

        return evalution_measure_rf


    def Kmeans_classifier(self, evalution_measure_kmeans):
        scaler = StandardScaler()
        test_data0 = scaler.fit_transform(self.test_data)
        culster0 = KMeans(n_clusters=2).fit_predict(test_data0) # Clustered into two clusters

        # label the defective and non-defective clusters
        preLabel0 = ModelInterpretation.labelCluster(test_data0, culster0)

        measure_classifier = classificationMeasure.evaluateMeasure(self.test_label, preLabel0)
        measure_effort = rankMeasure.rank_measure(preLabel0, self.testLOC, self.test_label)

        for j in range(2):
            evalution_measure_kmeans[j][self.fs_number] = measure_classifier[j]
        for j in range(2):
            evalution_measure_kmeans[j + 2][self.fs_number] = measure_effort[j]

        ModelInterpretation.unsupervised_model_interpretation(preLabel0, self.test_data, self.test_label, 'KMEANS', self.project_name, self.fs_name,
                                                              self.selected_idx, self.feature_num, self.dataset_name, self.results_dir)

        return evalution_measure_kmeans

    def SC_classifier(self, evalution_measure_sc):
        predict_label = SC.SC(self.test_data)
        measure_classifier = classificationMeasure.evaluateMeasure(self.test_label, predict_label)
        measure_effort = rankMeasure.rank_measure(predict_label, self.testLOC, self.test_label)

        for j in range(2):
            evalution_measure_sc[j][self.fs_number] = measure_classifier[j]
        for j in range(2):
            evalution_measure_sc[j + 2][self.fs_number] = measure_effort[j]

        ModelInterpretation.unsupervised_model_interpretation(predict_label, self.test_data, self.test_label, 'SC', self.project_name,
                                                              self.fs_name, self.selected_idx, self.feature_num, self.dataset_name, self.results_dir)

        return evalution_measure_sc
