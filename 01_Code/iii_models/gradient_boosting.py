import numpy as np
import pandas as pd
import h5py
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

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


def load_data(train: bool = True) -> pd.DataFrame:
    if train:
        ds = "train"
    else:
        ds = "test"

    path = input(r"Input path to {} dataset (with name + .csv): ".format(ds))

    return pd.read_csv(path)


def preprocess_data(dataset: pd.DataFrame) -> tuple:
    create_target(dataset)
    df = drop_cases(dataset)
    df = drop_cols(df)
    return split_target_data(df)


def create_target(dataset: pd.DataFrame) -> None:
    """
    This function creates the target variable based on the prmdiag column

    Args:
        dataset: The dataset for which the target variable should be created

    Returns:
        None

    Raises:
        KeyError: ...
    """
    dataset["target"] = np.where((dataset['prmdiag'] == 2) | (dataset['prmdiag'] == 3), 1, 0)


def drop_cases(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Drops the observations of variable prmdiag with value 1 or 4

    Args:
        dataset: dataset on which the obs. should be dropped

    Returns:
        None

    Raises:
        KeyError: ...
    """
    return dataset.drop(dataset[(dataset["prmdiag"] == 1) | (dataset["prmdiag"] == 4)].index)


def drop_cols(dataset: pd.DataFrame, cols: set = ('ConnID', 'Repseudonym',
                                                  'siteid', 'visdat', 'IDs', "prmdiag")) -> pd.DataFrame:
    """
    Drops the columns which are not needed for further modelling

    Args:
        dataset: dataset on which the cols should be dropped
        cols:

    Returns:
        None

    Raises:
        KeyError: ...
    """
    return dataset.drop(columns=list(cols))


def split_target_data(dataset: pd.DataFrame) -> tuple:
    """
    splits the given dataset into target variable and features

    Args:
        dataset:

    Returns:
        None
    """

    y, x = dataset["target"], dataset.drop(columns="target")
    return y, x


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
    train = load_data(True)
    test = load_data(False)

    print(train['prmdiag'].value_counts())
    print(len(train))


if __name__ == "__main__":
    main()
