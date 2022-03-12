
import pandas as pd
import numpy as np
# import h5py
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling libraries
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# from sklearn.feature_selection import VarianceThreshold <- use to remove low variance features
# from bayes_opt import BayesianOptimization #!pip install bayesian-optimization


def print_results(accuracy, precision, recall, f1, auc):
    # print all values
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1}")
    print(f"ROC AUC: {auc}")


def read_data_file(file_path, load_mem_score=False):
    """
    Import train.csv and test.csv files from data path

    Args:
        file_path (str):        file path
        load_mem_score (bool):  if true, the mem_score is included

    Returns:
    train and test data
    """
    data = pd.read_csv(file_path + '\\train.csv')
    test = pd.read_csv(file_path + '\\test.csv')

    # preprocess data file
    has_alzheimer = data.prmdiag.isin([2, 3])
    no_alzheimer = data.prmdiag.isin([0])
    data.loc[has_alzheimer, 'target'] = 1
    data.loc[no_alzheimer, 'target'] = 0
    data.dropna(subset=['target'], axis=0, inplace=True)
    data.drop(['ConnID', 'Repseudonym', 'visdat', 'siteid', 'IDs', 'prmdiag'], axis=1, inplace=True)
    if not load_mem_score:
        data.drop('MEM_score', axis=1, inplace=True)
    features = data.drop('target', axis=1)
    labels = data['target']
    # n_features = features.shape[1]

    # preprocess test file
    has_alzheimer = test.prmdiag.isin([2, 3])
    no_alzheimer = test.prmdiag.isin([0])
    test.loc[has_alzheimer, 'target'] = 1
    test.loc[no_alzheimer, 'target'] = 0
    test.dropna(subset=['target'], axis=0, inplace=True)
    test.drop(['ConnID', 'Repseudonym', 'visdat', 'siteid', 'IDs', 'prmdiag'], axis=1, inplace=True)
    if not load_mem_score:
        test.drop('MEM_score', axis=1, inplace=True)
    x_test = test.drop('target', axis=1)
    y_test = test['target']

    # Train Test Split
    # x_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size = 0.2, random_state = 42, stratify = labels)
    x_train, y_train = features, labels

    return x_train, x_test, y_train, y_test


# create input pipeline / preprocessing
def preprocess_random_forest(file_path, load_mem_score):
    x_train, x_test, y_train, y_test = read_data_file(file_path, load_mem_score)
    features = x_train.copy()

    # impute missing values
    # print(features[features.isna().any(axis=1)])
    imp = KNNImputer(missing_values=np.nan, n_neighbors=7)
    x_train = imp.fit_transform(x_train)
    # X_valid = imp.transform(X_valid)
    x_test = imp.transform(x_test)

    # scale data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    # X_valid = scaler.transform(X_valid)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, features


# run random forest model
def run_random_forest(x_train, x_test, y_train, y_test, features):
    """
    Run Training algorithm of random forest model
    and evaluate result based on test data

    Args:
        x_train (np.array):         training input
        x_test (pd.DataFrame):      test input
        y_train (pd.Series):        training output
        y_test (pd.Series):         test output
        features (pd.DataFrame):    original training input, used for feature names

    Returns:
        returns rf model if no test data is specified,
        else returns None
    """

    # run model
    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(x_train, np.ravel(y_train))

    if x_test is None:
        # skip evaluation and pass trained model
        return rf

    # visualization
    # extract feature importance array
    feature_imp = pd.Series(rf.feature_importances_, index=features.columns).sort_values(ascending=False)
    feature_imp = feature_imp[:10]

    # plot feature importance
    sns.barplot(x=feature_imp, y=feature_imp.index)
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()

    # Evaluation
    predictions = rf.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)
    # print results
    print_results(accuracy, precision, recall, f1, auc)

    pd.DataFrame({"Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "AUC": [auc]})
    plot_confusion_matrix(rf, x_test, y_test)
    importances = list(rf.feature_importances_)
    feature_list = list(features.columns)
    feature_importances = [(feature, round(importance, 10)) for feature, importance in zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances[:10]]

    return None


def run_lin_regression(x_train, x_test, y_train, y_test):
    logreg = LogisticRegression(penalty='elasticnet', solver='saga')
    params_grid = {'l1_ratio': np.linspace(0, 1, 11)}
    grid_clf = GridSearchCV(estimator=logreg, param_grid=params_grid, n_jobs=-1, cv=10, verbose=3)
    grid_clf.fit(x_train, y_train)
    grid_clf.best_estimator_
    grid_clf.best_params_

    logreg = LogisticRegression(penalty='l2', solver='lbfgs')
    logreg.fit(x_train, y_train)
    predictions = logreg.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)
    # print results
    print_results(accuracy, precision, recall, f1, auc)

    pd.DataFrame({"Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1], "AUC": [auc]})

    plot_confusion_matrix(logreg, x_test, y_test)

    return None


def pca_pipeline_test(file_path, load_mem_score):
    x_train, x_test, y_train, y_test = read_data_file(file_path, load_mem_score)

    pipeline = Pipeline([
        ('KNN_Impute', KNNImputer(missing_values=np.nan, n_neighbors=7)),
        ('scale', StandardScaler()),
        ('PCA', PCA(n_components=.95)),
    ])

    x_train = pipeline.fit_transform(x_train)
    x_test = pipeline.transform(x_test)
    x_train.shape[1] == x_test.shape[1]
    logreg = LogisticRegression(penalty='elasticnet', solver='saga')
    params_grid = {'l1_ratio': np.linspace(0, 1, 11)}

    grd_search = GridSearchCV(estimator=logreg, param_grid=params_grid, n_jobs=-1, cv=10, verbose=3)
    grd_search.fit(x_train, y_train)
    grd_search.best_params_

    logreg = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=.9)
    logreg.fit(x_train, y_train)

    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(x_train, y_train)

    log_pred = logreg.predict(x_test)
    rf_pred = rf.predict(x_test)
    model = ['LogReg', 'RF']
    accuracy = []
    precision = []
    recall = []
    f1 = []
    auc = []
    accuracy.append(accuracy_score(y_test, log_pred))
    precision.append(precision_score(y_test, log_pred))
    recall.append(recall_score(y_test, log_pred))
    f1.append(f1_score(y_test, log_pred))
    auc.append(roc_auc_score(y_test, log_pred))
    # rf
    accuracy.append(accuracy_score(y_test, rf_pred))
    precision.append(precision_score(y_test, rf_pred))
    recall.append(recall_score(y_test, rf_pred))
    f1.append(f1_score(y_test, rf_pred))
    auc.append(roc_auc_score(y_test, rf_pred))
    # print results
    print(f"Model: {model}")
    print_results(accuracy, precision, recall, f1, auc)

    pd.DataFrame({'Model': model, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "AUC": auc})
    plot_confusion_matrix(logreg, x_test, y_test)
    plot_confusion_matrix(rf, x_test, y_test)


if __name__ == '__main__':
    # run flags
    binary_classifier_flag = True  # set flag for binary training, else regression
    run_test_flag = False   # run test files for both algorithms
    load_mem_score = False  # load dataset without MEM_score

    # file_path = r"C:\Users\likai\Desktop\My Life\Master\3. Semester\Innolabs\Connectome Git\00_Data\Results"
    file_path = r"C:\Users\katha\Downloads\Test"

    # start preprocess
    print("Start preprocessing")
    x_train, x_test, y_train, y_test, features = preprocess_random_forest(file_path, load_mem_score)

    # check classifier
    if binary_classifier_flag:
        # start random forest model
        print("Start RF model")
        run_random_forest(x_train, x_test, y_train, y_test, features)

    else:
        print("Start lr model")
        run_lin_regression(x_train, x_test, y_train, y_test)
        # start linear regression model

    if run_test_flag:
        print("Run PCA test pipeline")
        pca_pipeline_test(file_path, load_mem_score)

    print("Finished RF_pipeline")