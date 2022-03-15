"""
several preprocessing and data transformation helpers
"""
import numpy as np
import pandas as pd


def preprocess_data(dataset: pd.DataFrame) -> tuple:
    """
    Combines several preprocessing steps which are to be performed on the given dataset.
    Results are then returned as target and features (splitted)

    Args:
        dataset: The dataset on which the preprocessing should be performed

    Returns:
        tuple, of (target, features)
    """
    assert isinstance(dataset, pd.DataFrame), "supplied input [dataset] is no DataFrame"

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

    """
    assert isinstance(dataset, pd.DataFrame), "supplied input [dataset] is no DataFrame"

    dataset["target"] = np.where((dataset['prmdiag'] == 2) | (dataset['prmdiag'] == 3), 1., 0.)


def drop_cases(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Drops the observations of variable prmdiag with value 1 or 4

    Args:
        dataset: dataset on which the obs. should be dropped

    Returns:
        None

    """
    assert isinstance(dataset, pd.DataFrame), "supplied input [dataset] is no DataFrame"

    return dataset.drop(dataset[(dataset["prmdiag"] == 1) | (dataset["prmdiag"] == 4)].index)


def drop_cols(dataset: pd.DataFrame,
              cols: tuple = ('ConnID', 'Repseudonym',
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
    assert isinstance(dataset, pd.DataFrame), "supplied input [dataset] is no DataFrame"
    assert (isinstance(cols, tuple) & (len(cols) != 0)), "invalid input [cols]"

    return dataset.drop(columns=list(cols))


def split_target_data(dataset: pd.DataFrame) -> tuple:
    """
    splits the given dataset into target variable and features

    Args:
        dataset:

    Returns:
        None
    """
    assert isinstance(dataset, pd.DataFrame), "supplied input [dataset] is no DataFrame"

    y, x = dataset["target"], dataset.drop(columns="target")
    return y, x


def flat_to_mat(x: np.ndarray) -> np.ndarray:
    """
    - converts a flat np.array into a matrix by turning
      the values of the array into a symmetric matrix
    - excluding diagonal

    Examples:
    >>>import numpy as np
    >>>from connectome.preprocessing.data_loader import flat_to_mat
    >>>k = 50 #
    >>>m = int((k*k)/2 - k/2)
    >>>x = np.random.standard_normal(size=m)
    >>>mat = flat_to_mat(x)
    >>>print(mat)
    Args:
         x: 1D array which should be turned into symmetric matrix

    Returns:
         np.ndarray - matrix

    """

    n = len(x)
    n_a = int((1 / 2) + np.sqrt((1 / 4) + 2 * n))
    A = np.zeros(n_a * n_a).reshape(n_a, n_a)
    ind = np.triu_indices(n_a, k=1)
    A[ind] = x
    return A.T + A


def flat_to_mat_aggregation(x: np.ndarray) -> np.ndarray:
    """
    - converts a flat np.array into a matrix by turning
      the values of the array into a symmetric matrix

    Examples:
    >>>import numpy as np
    >>>from connectome.preprocessing.data_loader import flat_to_mat_aggregation
    >>>k = 8 #
    >>>m = int((k*k)/2 + k/2)
    >>>x = np.random.standard_normal(size=m)
    >>>mat = flat_to_mat_aggregation(x)
    >>>print(mat)
    Args:
         x: 1D array which should be turned into symmetric matrix

    Returns:
         np.ndarray - matrix

    """

    n = len(x)
    n_a = 8
    A = np.zeros(n_a * n_a).reshape(n_a, n_a)
    ind = np.triu_indices(n_a, k=0)
    A[ind] = x
    mat = A.T + A
    return mat - .5 * np.diag(mat) * np.identity(8)

if __name__ == "__main__":
    k = 50 #
    m = int((k*k)/2 - k/2)
    x = np.random.standard_normal(size=m)
    mat = flat_to_mat(x)
    print(mat)