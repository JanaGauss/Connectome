import numpy as np
import pandas as pd
import h5py
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import data_loader as dtl

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score, plot_confusion_matrix, confusion_matrix
from sklearn.pipeline import Pipeline


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


def main() -> None:
    train = dtl.load_data(True)
    test = dtl.load_data(False)

    print(train['prmdiag'].value_counts())
    print(len(train))


if __name__ == "__main__":
    main()
