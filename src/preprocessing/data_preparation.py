import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_data(data: pd.DataFrame, classification: bool = True, 
                 columns_drop: list = ["ConnID", "Repseudonym", "siteid" "visdat", "MEM_Score", "Apoe", "IDs"], 
                 target: str = "prmdiag", y_0: list = [0], y_1: list = [2, 3],
                 train_size: float = 0.8, seed: int = 123) -> list:
    """
    Function that prepares the data for modelling
    
    Args:
        data: A pd.Dataframe
        classification: boolean, if false regression task
        columns_drop: which columns should be dropped (all other columns will be used for modelling)
        target: name of the target variable
        y_0: (only relevant for classification task) which values of target should be treated as 0
        y_1: (only relevant for classification task) which values of target should be treated as 1
        train_size: size of the training data (default 0.8)
        seed: seed for reproducibility of train/test split
    Returns:
        A list containing the training data (index 0) and the test data (index 1) with a y-column and all columns that shouldn't be used for modelling are dropped
    """

    # check input types
    assert isinstance(data, pd.DataFrame), "provided data is no pd.DataFrame"
    assert isinstance(classification, bool), "classification is no boolean"
    assert isinstance(columns_drop, list), "invalid columns_drop, must be list"
    assert all(isinstance(x, str) for x in columns_drop), "invalid columns_drop, elements must be strings"
    assert isinstance(target, str), "invalid target, must be string"
    assert isinstance(y_0, list), "invalid, y_0, must be list"
    assert all(isinstance(x, int) for x in y_0), "invalid y_0, elements must be integers"
    assert isinstance(y_1, list), "invalid, y_1, must be list"
    assert all(isinstance(x, int) for x in y_1), "invalid y_1, elements must be integers"
    assert isinstance(train_size, float), "invalid train size, must be float"
    assert isinstance(seed, int), "provided seed is no integer"

    # check if inputs are valid
    assert (train_size > 0.0) & (train_size < 1.0), "invalid train size, must be > 0 and < 1" 
    assert target in data.columns, "target not found in columns of data"
    assert all(x in data.columns for x in columns_drop), "columns_drop not found in columns of data"
    if classification:
      assert all(x in data[target].unique() for x in y_0), "y_0 not found in target variable"
      assert all(x in data[target].unique() for x in y_1), "y_1 not found in target variable"


    # create y variable
    if classification:
            
      # create 0 and 1 based on target and y_0 and y_1
      target_0 = data[target].isin(y_0)
      target_1 = data[target].isin(y_1)
  
      data.loc[target_0, "y"] = 0
      data.loc[target_1, "y"] = 1

      # drop NAs in y
      data.dropna(subset = ["y"], inplace = True)

    else: # regression

      data["y"] = data[target]

    # drop target (as y was created) and other columns 
    data.drop(columns=columns_drop, inplace = True)
    data.drop(columns=[target], inplace = True)


    # reorder data so that y is the first variable
    data = pd.concat([data["y"], data.drop(columns = ["y"])], axis=1)

    # perform train/test split
    data_list = train_test_split(data, train_size=train_size, random_state=seed, shuffle=True)

    return data_list


 