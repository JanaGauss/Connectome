"""
function to create and compute a random forest classification or regression model
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


def run_random_forest(X_train, y_train,
                      classification: bool = True,
                      verbose=0,
                      **kwargs):
    """
    Function that fits a random forest model. 
    See also https://bit.ly/3Jj6fta and https://bit.ly/3CNSUXr for
    further arguments that can be specified.
  
    Examples:
    >>> rf = run_random_forest(X_train = X_train,
                                  y_train = y_train,
                                  classification = True)   
    Args:
        X_train: The training dataset
        y_train: The true labels
        classification: classification or regression task
        verbose: amount of verbosity, default = 0 
        
    Returns:
        Returns fitted model
    """

    assert isinstance(X_train, pd.DataFrame), "provided X_train is no pd.DataFrame"
    assert isinstance(y_train, pd.Series) or isinstance(y_train, np.ndarray), \
        "provided y_train is no pd.Series or numpy array"

    if classification:
        model = RandomForestClassifier(verbose=verbose, **kwargs)
    
    else:
        model = RandomForestRegressor(verbose=verbose, **kwargs)

    fit_model = model.fit(X_train, y_train)

    return fit_model  






