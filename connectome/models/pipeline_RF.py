"""
Run Random Forest
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


def run_random_forest(X_train, y_train, 
			    classification: bool = True, verbose = 1,
			    **kwargs):
    """
  Function that fits a random forest model. 
  See also https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html and https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html for further arguments that can be specified.
  
  
  Args:
        X_train: The training dataset
        y_train: The true labels
        classification: classification task -> logistic regression or regression task -> linear model
        verbose: amount of verbosity, default = 1 
        
  Returns:
  	Returns fitted model
    """

    assert isinstance(X_train, pd.DataFrame), "provided X_train is no pd.DataFrame"
    assert isinstance(y_train, pd.Series), "provided y_train is no pd.Series"


    if classification:
    
      model = RandomForestClassifier(verbose = verbose, **kwargs)
    
    else:

      model = RandomForestRegressor(verbose = verbose, **kwargs)
    

    fit_model = model.fit(X_train, y_train)

    return fit_model  






