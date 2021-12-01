import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import data_loader as dtl
import pickle

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score, plot_confusion_matrix, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold


def plot_av_data(dataset: pd.DataFrame, cols: int = 20) -> None:
    sns.heatmap(dataset.iloc[:, 0:cols].isnull())
    plt.show()


def describe_data(dataset: pd.DataFrame) -> None:
    pass


def pred_evaluation(true: np.ndarray, preds: np.ndarray,
                    model: str = "Model") -> pd.DataFrame:
    """
    evaluates the given predictions

    Args:
        true:
        preds:
        model:

    Returns:
        Pandas Dataframe with several evaluation metrics

    Raises:
        KeyError: ...
    """

    accuracy = accuracy_score(true, preds)
    precision = precision_score(true, preds)
    recall = recall_score(true, preds)
    f1 = f1_score(true, preds)
    auc = roc_auc_score(true, preds)

    res = pd.DataFrame({"Accuracy": [accuracy],
                        "Precision": [precision],
                        "Recall": [recall],
                        "F1": [f1],
                        "AUC": [auc]})
    res.index = list(model) * len(res)
    return res


def save_model(model: any, filename: str = "Model") -> None:
    filename = filename + '.sav'
    pickle.dump(model, open(filename, 'wb'))


def main() -> None:
    pass

    # todo: hyperparameter tuning
    # todo: save evaluation to csv
    # todo: save model (pickle)
    # todo: model fitting into function (option to fit or load)
    # todo: save predictions to file


if __name__ == "__main__":
    # loading and preprocessing data
    train = dtl.load_data(True)
    test = dtl.load_data(False)
    y, X = dtl.preprocess_data(train)
    y_test, X_test = dtl.preprocess_data(test)

    # repeated cv with untuned xgb
    xg_class = xgb.XGBClassifier()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
    cv_scores = cross_val_score(xg_class, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    print("CV Accuracy: {:.2f}".format(cv_scores.mean()))
    print(cv_scores)

    # saving cv results to file
    pd.DataFrame(cv_scores).to_csv("cv_untuned_result.csv", encoding='utf-8', index=False)


    # evaluation on test set
    xg_class.fit(X, y)
    save_model(xg_class, "xgb_untuned")

    y_pred = xg_class.predict(X_test)
    pd.DataFrame(y_pred).to_csv("predictions_test.csv", encoding='utf-8', index=False)

    test_eval = pred_evaluation(y_test, y_pred, "XGBoost")
    test_eval.to_csv("test_eval_xgb_untuned_result.csv", encoding='utf-8', index=False)
    plot_confusion_matrix(xg_class, X_test, y_test)
