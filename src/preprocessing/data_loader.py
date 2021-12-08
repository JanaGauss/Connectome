import numpy as np
import pandas as pd


def load_data(train: bool = True) -> pd.DataFrame:
    """
    loads the datasets (csv-files) from specified path

    Args:
        train:

    Returns:

    """
    assert isinstance(train, bool), "invalid input [train]"

    if train:
        ds = "train"
    else:
        ds = "test"

    path = input(r"Input path to {} dataset (with name + .csv): ".format(ds))

    return pd.read_csv(path)


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

    Raises:
        KeyError: ...
    """
    assert isinstance(dataset, pd.DataFrame), "supplied input [dataset] is no DataFrame"

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
    assert isinstance(dataset, pd.DataFrame), "supplied input [dataset] is no DataFrame"

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
    assert isinstance(dataset, pd.DataFrame), "supplied input [dataset] is no DataFrame"
    assert (isinstance(cols, set) & (len(cols) != 0)), "invalid input [cols]"

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
