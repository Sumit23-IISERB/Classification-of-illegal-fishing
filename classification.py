import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_selection import SelectKBest, SelectFromModel, RFE
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import joblib

class Classification:
    def __init__(self, path='fishing_imputed.csv', clf_opt='lr', no_of_selected_features=None):
        self.path = path
        self.clf_opt = clf_opt
        self.no_of_selected_features = no_of_selected_features
        if self.no_of_selected_features is not None:
            self.no_of_selected_features = int(self.no_of_selected_features)

    def classification_pipeline(self):
        if self.clf_opt == 'ab':
            print('\n\t### Training AdaBoost Classifier ### \n')
            be1 = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=0)
            be2 = DecisionTreeClassifier(max_depth=50, random_state=0)
            clf = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=100, random_state=0)
            param_grids = {
                'clf__base_estimator': [be1, be2],
                'clf__base_estimator__random_state': [0],
                'clf__random_state': [0],
            }
        elif self.clf_opt == 'lr':
            print('\n\t### Training Logistic Regression Classifier ### \n')
            clf = LogisticRegression(solver='lbfgs', class_weight='balanced', random_state=0)
            param_grids = {
                'clf__random_state': [0],
                'clf__solver': ['lbfgs','newton-cg', 'sag', 'saga']
            }
        elif self.clf_opt == 'rf':
            print('\n\t ### Training Random Forest Classifier ### \n')
            clf = RandomForestClassifier(max_features=None, class_weight='balanced', random_state=0)
            param_grids = {
                'clf__criterion': ['entropy', 'gini'],
                'clf__n_estimators': [30, 100],
                'clf__max_depth': [10, 50, 200],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 4],
                'clf__random_state': [0],
            }
        elif self.clf_opt == 'svm':
            print('\n\t### Training SVM Classifier ### \n')
            clf = svm.SVC(class_weight='balanced', probability=True, random_state=0)
            param_grids = {
                'clf__C': [0.5, 5],
                'clf__kernel': ['linear'],
                'clf__random_state': [0],
            }
        elif self.clf_opt == 'dt':
            print('\n\t### Training Decision Tree Classifier ### \n')
            clf = DecisionTreeClassifier(random_state=0)
            param_grids = {
                'clf__criterion': ['gini', 'entropy'],
                'clf__max_depth': [10, 20, 30],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 4],
                'clf__random_state': [0],
            }
        else:
            print('Select a valid classifier \n')
            sys.exit(0)

        feature_selection_methods = [
            # uncomment only the method you want to use
            SelectKBest(k=self.no_of_selected_features),
            # SelectFromModel(LogisticRegression(solver='liblinear', class_weight='balanced', random_state=0)),
            # RFE(estimator=DecisionTreeClassifier(random_state=0), n_features_to_select=self.no_of_selected_features)
        ]

        return clf, param_grids, feature_selection_methods

    def get_class_statistics(self, labels):
        class_statistics = Counter(labels)
        print('\n Class \t\t Number of Instances \n')
        for item in list(class_statistics.keys()):
            print('\t' + str(item) + '\t\t\t' + str(class_statistics[item]))

    def get_data(self):
        reader = pd.read_csv(self.path)
        # reader = reader[reader['target'] != -1]
        data = reader.drop(['target'], axis=1)
        labels = reader['target']
        self.get_class_statistics(labels)
        return data, labels

    def classification(self):
        data, labels = self.get_data()
        clf, param_grids, feature_selection_methods = self.classification_pipeline()

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for feature_selection_method in feature_selection_methods:
            pipeline = Pipeline([
                ('feature_selection', feature_selection_method),
                ('clf', clf),
            ])

            for train_index, test_index in skf.split(data, labels):
                X_train, X_test = data.iloc[train_index], data.iloc[test_index]
                y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

                grid = GridSearchCV(pipeline, param_grids, scoring='f1_macro', cv=skf)
                with tqdm(total=1) as pbar:
                    pbar.set_description("Grid Search Progress")
                    grid.fit(X_train, y_train)
                    pbar.update(1)
                    clf = grid.best_estimator_
                    pbar.update(1)

                    # Save the trained model
                    joblib.dump(clf, f"{self.clf_opt}_{feature_selection_method.__class__.__name__}_model.joblib")
                    print(f"Saved the trained {self.clf_opt} model to {self.clf_opt}_{feature_selection_method.__class__.__name__}_model.joblib")

                    # Best parameters after grid search
                    print('\n Best Parameters: ')
                    print(grid.best_params_)

                    # Evaluation on the validation set
                    predicted = clf.predict(X_test)

                    print('\n *************** Confusion Matrix ***************  \n')
                    print(confusion_matrix(y_test, predicted))

                    class_names = list(Counter(y_test).keys())
                    class_names = [str(x) for x in class_names]

                    print('\n ##### Classification Report ##### \n')
                    print(classification_report(y_test, predicted, target_names=class_names))

                    pr = precision_score(y_test, predicted, average='macro')
                    print('\n Precision:\t' + str(pr))

                    rl = recall_score(y_test, predicted, average='macro')
                    print('\n Recall:\t' + str(rl))

                    fm = f1_score(y_test, predicted, average='macro')
                    print('\n F1-Score:\t' + str(fm))
